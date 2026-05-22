"""``FullHistoryStrategy`` — the oracle upper bound.

Keeps every turn verbatim and feeds the entire conversation to the model
on every query. Wins on accuracy (no information is lost) but at
quadratic token cost. The eval compares Mneme to this on both axes —
Mneme should approach full-history accuracy at a fraction of the tokens.
"""

from __future__ import annotations

from typing import Any


class FullHistoryStrategy:
    """All prior turns, forever. The accuracy ceiling at unbounded cost."""

    name = "full_history"

    def __init__(self) -> None:
        self._turns: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})

    def context_for(self, query: str) -> list[dict[str, str]]:
        # The whole transcript, in chronological order. The query itself
        # is appended by the runner so we don't double-include it here.
        return list(self._turns)

    def retrieve_records(self, query: str, *, k: int = 5) -> list[str]:
        # Full-history has no semantic retrieval, but the model sees the
        # ENTIRE transcript regardless of how big it is. For recall@k
        # purposes that's equivalent to "every prior turn is retrieved" —
        # so we return all turn contents and ignore ``k``. The point of
        # this baseline is to be the oracle ceiling; capping its
        # retrieved set at k would understate that.
        return [t["content"] for t in self._turns]

    def reset(self) -> None:
        self._turns.clear()

    def config_summary(self) -> dict[str, Any]:
        return {"name": self.name}
