"""``SummaryBufferStrategy`` — the LangChain-style rolling summary.

Keeps the last N turns verbatim. When the buffer overflows, an LLM
summarises the oldest turns into a single ``system`` message and the
detail is dropped. This is what most existing memory libraries
(LangChain's ``ConversationSummaryBufferMemory``, LlamaIndex's
``ChatSummaryMemoryBuffer``) actually do.

The default uses a simple non-LLM "summary" (a join of the dropped turn
contents) so the baseline works without an OpenAI key. Pass a real
``summarize_fn`` to use an LLM. The eval CLI wires this up when run
with ``--with-answers``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any


def _trivial_summary(messages: list[dict[str, str]]) -> str:
    """Trivial non-LLM summarisation: join the contents with ; separators.

    Real summary-buffer implementations call an LLM to produce a coherent
    paragraph. We use this trivial form so the baseline works deterministically
    without an API key — the comparison to Mneme is still meaningful since
    both strategies see the same raw input.
    """
    return "Earlier in the conversation: " + " | ".join(
        f"{m['role']}: {m['content']}" for m in messages
    )


SummaryFn = Callable[[list[dict[str, str]]], str]


class SummaryBufferStrategy:
    """Rolling buffer + a summary of the overflow."""

    name = "summary_buffer"

    def __init__(
        self,
        *,
        max_recent_turns: int = 6,
        summarize_fn: SummaryFn = _trivial_summary,
    ) -> None:
        self._max_recent = max_recent_turns
        self._summarize_fn = summarize_fn
        self._recent: deque[dict[str, str]] = deque()
        self._summary_so_far: str | None = None

    def add(self, role: str, content: str) -> None:
        self._recent.append({"role": role, "content": content})
        # Overflow: evict from the front, fold into the summary.
        while len(self._recent) > self._max_recent:
            evicted = self._recent.popleft()
            running = (
                [{"role": "system", "content": self._summary_so_far}]
                if self._summary_so_far is not None
                else []
            )
            self._summary_so_far = self._summarize_fn([*running, evicted])

    def context_for(self, query: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if self._summary_so_far is not None:
            out.append({"role": "system", "content": self._summary_so_far})
        out.extend(self._recent)
        return out

    def retrieve_records(self, query: str, *, k: int = 5) -> list[str]:
        # No semantic retrieval — best we can do is the recent buffer
        # contents plus the summary (if any). This is the "honest
        # interpretation" of what the strategy would put in front of
        # the model when asked about ``query``.
        out: list[str] = []
        if self._summary_so_far is not None:
            out.append(self._summary_so_far)
        out.extend(t["content"] for t in self._recent)
        return out[:k]

    def reset(self) -> None:
        self._recent.clear()
        self._summary_so_far = None

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_recent_turns": self._max_recent,
            "summarize_fn": self._summarize_fn.__name__,
        }
