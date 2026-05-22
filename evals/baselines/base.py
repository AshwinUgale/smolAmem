"""The :class:`MemoryStrategy` protocol — what the eval runner sees.

Any memory approach (no-memory, summary-buffer, full-history, Mneme) plugs
into the runner by implementing four methods. Same shape every time;
strategies are interchangeable inputs to the same eval flow.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryStrategy(Protocol):
    """Eval-facing memory abstraction.

    Implementations:

    * accept ``add(role, content)`` for every turn played through the corpus
    * expose ``context_for(query)`` that returns the OpenAI-style messages
      this strategy would put in front of an LLM at this point
    * expose ``retrieve_records(query, k)`` that returns the top-k records
      this strategy considers most relevant to ``query`` (used for recall@k
      measurement — for baselines without semantic retrieval, this is just
      the last-k turns of history or an empty list)
    * expose ``name`` for human-readable identification in result JSON
    """

    name: str

    def add(self, role: str, content: str) -> None:
        """Record a single turn the conversation just produced."""
        ...

    def context_for(self, query: str) -> list[dict[str, str]]:
        """Return the chat-messages-style context this strategy would
        prepend before answering ``query``. Used for end-to-end accuracy
        and token-cost metrics."""
        ...

    def retrieve_records(self, query: str, *, k: int = 5) -> list[str]:
        """Return up to ``k`` record-content strings most relevant to
        ``query``. Used for the cheap retrieval-only recall@k metric.

        For baselines that don't have semantic retrieval (no_memory,
        summary_buffer), returning the last-k turn-contents is the
        honest interpretation — it's "what would this strategy show
        the model in lieu of a real retrieval result." Mneme returns
        the actual top-k from :meth:`MemoryManager.retrieve`.
        """
        ...

    def reset(self) -> None:
        """Wipe state. Called before each conversation in the corpus."""
        ...

    # ---- optional metadata for result JSON ----

    def config_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict describing this strategy's
        config. Default: just the name. Override to surface model,
        backend, etc.
        """
        return {"name": self.name}
