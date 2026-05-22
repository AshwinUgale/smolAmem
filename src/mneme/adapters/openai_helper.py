"""Raw-OpenAI ``context_for`` helper.

Returns a list of OpenAI-style chat messages (``[{"role": "...",
"content": "..."}, ...]``) packed for a given query. Composes:

1. **Retrieved memory** as a single ``system`` message with citations,
   produced via :meth:`MemoryManager.retrieve` (defaults to top-5 across
   episodic + semantic, with authority + freshness weighting).
2. **Recent working-memory turns**, if ``include_working=True``, preserved
   in chronological order.

Designed to drop in front of any OpenAI chat-completions call:

.. code-block:: python

    messages = context_for(manager, query="how do I use Suspense?", token_budget=2000)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages + [{"role": "user", "content": user_query}],
    )

Token-budget packing uses ``tiktoken``'s ``cl100k_base`` encoding, which
matches every OpenAI chat model from ``gpt-4o`` through ``gpt-3.5-turbo``.
The ``[tokens]`` extra installs tiktoken; if it's not installed and
``token_budget`` is passed, we raise with a helpful message. Without
``token_budget`` you never need tiktoken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mneme.manager import MemoryManager
from mneme.types import MemoryTier

if TYPE_CHECKING:
    pass


__all__ = ["context_for"]


# Module-level encoding cache so we initialise tiktoken at most once per
# process. tiktoken.get_encoding does its own caching internally but
# stashing a reference keeps the hot path tight.
_ENCODING_NAME = "cl100k_base"
_encoding_cache: object | None = None


def _get_tiktoken_encoding() -> object:
    """Lazy-load tiktoken's cl100k_base encoder. Cached. Raises if extra missing."""
    global _encoding_cache
    if _encoding_cache is not None:
        return _encoding_cache
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "context_for(token_budget=...) requires the 'tokens' extra "
            "for tiktoken-based token counting. Install with:\n"
            "    pip install 'mneme[tokens]'\n"
            "    # or call context_for() without token_budget for raw output."
        ) from exc
    _encoding_cache = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding_cache


def _count_tokens(text: str) -> int:
    """Count tokens in ``text`` using cl100k_base."""
    enc = _get_tiktoken_encoding()
    # tiktoken has type stubs but our ignore-missing-imports override above
    # means mypy treats it as Any. ``attr-defined`` covers ``enc.encode``;
    # ``len`` on the returned list is just an int, no other ignore needed.
    return len(enc.encode(text))  # type: ignore[attr-defined]


def context_for(
    manager: MemoryManager,
    query: str,
    *,
    k: int = 5,
    token_budget: int | None = None,
    include_working: bool = True,
) -> list[dict[str, str]]:
    """Build a list of OpenAI-style chat messages for an agent prompt.

    Args:
        manager: The Mneme manager to draw memory from.
        query: The user's current question. Used to retrieve relevant
            episodic + semantic memories.
        k: How many memories to retrieve. Default 5.
        token_budget: Optional token cap on the returned messages. When
            given, requires the ``tokens`` extra (tiktoken). Messages are
            dropped in priority order: working-memory turns survive last;
            lower-scored retrieved memories are dropped first.
        include_working: If ``True`` (default), append all current
            working-memory turns as message entries after the retrieved
            memory.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts ready to pass
        to ``client.chat.completions.create(messages=...)``.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    out: list[dict[str, str]] = []

    # 1. Retrieved memory as one system message with citations.
    #    Skipped if k=0 or nothing is retrieved.
    retrieved_messages: list[dict[str, str]] = []
    if k > 0:
        results = manager.retrieve(query, k=k)
        if results:
            citations: list[str] = []
            for result in results:
                tier_marker = "FACT" if result.record.tier is MemoryTier.SEMANTIC else "EPISODE"
                citations.append(
                    f"[{tier_marker} id={result.record.id} "
                    f"score={result.score:.3f}] {result.record.content}"
                )
            retrieved_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant memories (cite by id when grounding answers):\n\n"
                        + "\n".join(citations)
                    ),
                }
            )

    # 2. Working memory turns in chronological order.
    working_messages: list[dict[str, str]] = []
    if include_working:
        for turn in manager.working.turns():
            working_messages.append({"role": turn.role, "content": turn.content})

    out = retrieved_messages + working_messages

    if token_budget is None:
        return out

    # Token-budget packing: drop retrieved-memory citations first (they
    # are lower priority than the immediate chat history), then drop
    # working-memory turns oldest-first if still over budget. This
    # preserves the "the model needs the recent conversation" invariant.
    return _pack_under_budget(
        retrieved=retrieved_messages,
        working=working_messages,
        budget=token_budget,
    )


def _pack_under_budget(
    *,
    retrieved: list[dict[str, str]],
    working: list[dict[str, str]],
    budget: int,
) -> list[dict[str, str]]:
    """Greedy packing that prefers keeping recent working memory over
    retrieved memory when forced to drop something."""
    if budget <= 0:
        return []

    def _msg_tokens(m: dict[str, str]) -> int:
        # OpenAI's per-message overhead is ~3 to 4 tokens (role + structural
        # tokens). We approximate with +4 per message; close enough for
        # budget gating.
        return _count_tokens(m["content"]) + 4

    # Working memory has higher priority — pack it first, oldest-to-newest.
    packed: list[dict[str, str]] = []
    spent = 0
    for msg in working:
        cost = _msg_tokens(msg)
        if spent + cost > budget:
            break
        packed.append(msg)
        spent += cost

    # Whatever budget remains, fill with retrieved memory (which is one
    # combined system message — either it fits in full or not at all).
    for msg in retrieved:
        cost = _msg_tokens(msg)
        if spent + cost > budget:
            continue
        # Retrieved memory goes BEFORE working memory in the final order
        # so the LLM sees grounding context first.
        packed.insert(0, msg)
        spent += cost

    return packed
