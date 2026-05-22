"""Working memory tier.

The last N turns of the current conversation, held in process memory.
FIFO eviction — when the buffer is full, the oldest turn falls out as a
new one is added.

**Working memory does not participate in semantic retrieval at v0.1.** It is
not embedded, not persisted, and not searched. The typical use is "give me
the last N turns" to dump into the LLM prompt verbatim. If you want past
turns to be searchable, persist them as episodic memories — that's what the
episodic tier is for.

This tier is the cheapest, the simplest, and intentionally so. Most "memory"
libraries stop here. Mneme treats it as one tier of three.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from mneme.types import MemoryTier, WorkingMemory


class WorkingMemoryTier:
    """A FIFO buffer of recent conversation turns for one agent.

    Args:
        agent_id: Namespaces this buffer to a single agent. Matches the
            ``agent_id`` on the persisted tiers' records.
        max_size: Maximum number of turns to retain. Adding a turn when the
            buffer is full evicts the oldest. Default 20 — enough for most
            multi-turn conversations without bloating prompts.
    """

    def __init__(self, *, agent_id: str, max_size: int = 20) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.agent_id = agent_id
        self.max_size = max_size
        self._turns: deque[WorkingMemory] = deque(maxlen=max_size)

    def add(
        self,
        *,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemory:
        """Append a turn. Evicts the oldest if the buffer is full.

        Returns the created :class:`WorkingMemory` so callers can hold on to
        it (typically only the manager does).
        """
        turn = WorkingMemory(
            agent_id=self.agent_id,
            content=content,
            role=role,
            metadata=metadata or {},
        )
        self._turns.append(turn)
        return turn

    def turns(self) -> list[WorkingMemory]:
        """Return the current turns in chronological order (oldest first)."""
        return list(self._turns)

    def clear(self) -> int:
        """Remove every turn. Returns the number removed."""
        count = len(self._turns)
        self._turns.clear()
        return count

    def __len__(self) -> int:
        return len(self._turns)

    # Tier identity, exposed for symmetry with the persisted tiers.
    tier = MemoryTier.WORKING
