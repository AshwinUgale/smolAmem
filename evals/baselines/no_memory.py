"""``NoMemoryStrategy`` — the cheapest, dumbest baseline.

Every turn is fresh. The strategy never remembers anything; ``context_for``
returns an empty list. This is what an agent without any memory layer
looks like, and it's the floor we should beat at every test point.
"""

from __future__ import annotations

from typing import Any


class NoMemoryStrategy:
    """No memory at all. The lower bound."""

    name = "no_memory"

    def add(self, role: str, content: str) -> None:
        return None

    def context_for(self, query: str) -> list[dict[str, str]]:
        return []

    def retrieve_records(self, query: str, *, k: int = 5) -> list[str]:
        return []

    def reset(self) -> None:
        return None

    def config_summary(self) -> dict[str, Any]:
        return {"name": self.name}
