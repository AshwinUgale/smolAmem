"""Core data types shared across every Mneme tier and backend.

These are plain data carriers — no behaviour beyond what the tiers and the
retrieval ranker genuinely need. The field shape here is load-bearing: the
``MnemeBackend`` protocol, the freshness-decay logic, and the authority-weighted
retrieval ranker are all defined against these records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

__all__ = [
    "EpisodicMemory",
    "MemoryRecord",
    "MemoryTier",
    "RetrievalResult",
    "SemanticFact",
    "WorkingMemory",
]


def _utcnow() -> datetime:
    """Timezone-aware UTC now. All timestamps in Mneme are aware and in UTC."""
    return datetime.now(UTC)


def _new_id() -> str:
    """A fresh opaque record id. Hex so it is safe in URLs and collection keys."""
    return uuid4().hex


class MemoryTier(StrEnum):
    """The three tiers a memory can belong to.

    A ``StrEnum`` so a tier serialises to its plain value (``"episodic"``) in
    JSON and storage backends without a custom encoder.
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass(kw_only=True)
class MemoryRecord:
    """Common shape for anything Mneme stores or ranks.

    ``kw_only`` means every field is passed by keyword, which frees subclasses
    to add required fields without fighting dataclass default-ordering rules.

    Attributes:
        agent_id: Namespaces a record to one agent. Memory is never shared
            across agents in v1.0.
        content: The natural-language payload that gets embedded and shown.
        tier: Which tier this record belongs to. Concrete subclasses default it.
        id: Opaque unique id, generated if not supplied.
        embedding: The content's vector, or ``None`` until it has been embedded.
        created_at: When the record was first written (UTC, aware).
        last_accessed: When it was last returned by retrieval, or ``None`` if
            never. Drives access-frequency decay.
        access_count: How many times retrieval has surfaced it. Also drives decay.
        expires_at: When the forgetting pass should delete this record. ``None``
            (default) means "never". Set this when you have a known TTL — the
            forgetting pass deletes anything with ``expires_at <= now``.
            :meth:`MemoryManager.retrieve` also defensively filters out expired
            records even between forgetting passes.
        metadata: Free-form caller-supplied tags (source, conversation id, etc.).
    """

    agent_id: str
    content: str
    tier: MemoryTier
    id: str = field(default_factory=_new_id)
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=_utcnow)
    last_accessed: datetime | None = None
    access_count: int = 0
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self, *, now: datetime | None = None) -> None:
        """Record that retrieval just surfaced this memory.

        Bumps ``access_count`` and stamps ``last_accessed``. The forgetting pass
        reads both to decide what has gone cold.
        """
        self.last_accessed = now if now is not None else _utcnow()
        self.access_count += 1


@dataclass(kw_only=True)
class WorkingMemory(MemoryRecord):
    """A single turn in the current conversation.

    Lives in process memory (FIFO), not the storage backend. Adds a ``role`` so
    a turn can be replayed as a chat message.
    """

    role: str
    tier: MemoryTier = MemoryTier.WORKING


@dataclass(kw_only=True)
class EpisodicMemory(MemoryRecord):
    """A specific past interaction, embedded and persisted to the backend.

    Retrieved by similarity to the current query, optionally filtered by recency
    or metadata.
    """

    tier: MemoryTier = MemoryTier.EPISODIC


@dataclass(kw_only=True)
class SemanticFact(MemoryRecord):
    """A consolidated fact extracted from episodic memories.

    Attributes:
        confidence: How sure consolidation is of this fact, in ``[0.0, 1.0]``.
            Weights the record during retrieval.
        provenance: Ids of the episodic memories that contributed to this fact,
            so a fact can be traced back to its evidence.
    """

    confidence: float = 1.0
    provenance: list[str] = field(default_factory=list)
    tier: MemoryTier = MemoryTier.SEMANTIC


@dataclass(kw_only=True)
class RetrievalResult:
    """One ranked hit returned by ``mneme.retrieve``.

    Wraps the underlying record with the final ``score`` plus the component
    scores that produced it, so callers can see *why* something ranked where it
    did instead of getting an opaque number.

    Attributes:
        record: The memory that matched.
        score: Final fused score used for ranking (higher is better).
        similarity: Raw query/record vector similarity, if computed.
        recency: Freshness-decay contribution, if computed.
        authority: Tier-authority contribution (semantic > episodic > working),
            if computed.
    """

    record: MemoryRecord
    score: float
    similarity: float | None = None
    recency: float | None = None
    authority: float | None = None
