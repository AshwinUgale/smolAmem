"""Episodic memory tier.

Specific past interactions, embedded once at write time and persisted to the
configured backend. Retrieved by semantic similarity to a query, with the
backend's tier filter scoping the search to episodic-only records.

This tier is the load-bearing one for "remember what was said." Every
turn worth keeping past the working-memory window flows through here.
"""

from __future__ import annotations

from typing import Any, cast

from mneme.backends import MnemeBackend
from mneme.embeddings import EmbeddingProvider
from mneme.types import EpisodicMemory, MemoryTier


class EpisodicMemoryTier:
    """Persistent, semantically-searchable conversation history for one agent.

    Args:
        agent_id: Namespaces this tier's records.
        backend: The :class:`MnemeBackend` used for persistence and search.
        embedder: The :class:`EmbeddingProvider` used to compute vectors at
            write time and at query time.
    """

    tier = MemoryTier.EPISODIC

    def __init__(
        self,
        *,
        agent_id: str,
        backend: MnemeBackend,
        embedder: EmbeddingProvider,
    ) -> None:
        self.agent_id = agent_id
        self.backend = backend
        self.embedder = embedder

    # -- writes --------------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodicMemory:
        """Embed ``content`` and persist it as an episodic memory."""
        [embedding] = self.embedder.embed([content])
        record = EpisodicMemory(
            agent_id=self.agent_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        self.backend.upsert(record)
        return record

    # -- reads ---------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[EpisodicMemory, float]]:
        """Return the top-``k`` episodes most similar to ``query``.

        Embeds the query, calls :meth:`MnemeBackend.search` scoped to the
        episodic tier, and returns ``(record, similarity)`` pairs sorted by
        similarity descending. Records are typed as :class:`EpisodicMemory`
        — the tier filter guarantees the backend returns only those.
        """
        [query_embedding] = self.embedder.embed([query])
        raw = self.backend.search(
            query_embedding=query_embedding,
            agent_id=self.agent_id,
            k=k,
            tiers=[MemoryTier.EPISODIC],
            metadata_filter=metadata_filter,
        )
        # Safe cast: tier filter restricts results to EpisodicMemory.
        return [(cast(EpisodicMemory, rec), score) for rec, score in raw]

    def get(self, record_id: str) -> EpisodicMemory | None:
        record = self.backend.get(record_id)
        if record is None or record.tier is not MemoryTier.EPISODIC:
            return None
        return cast(EpisodicMemory, record)

    def delete(self, record_id: str) -> bool:
        # Only delete if it's actually one of ours — protects against deleting
        # a semantic fact by id via the wrong tier handle.
        existing = self.get(record_id)
        if existing is None:
            return False
        return self.backend.delete(record_id)

    def count(self) -> int:
        return self.backend.count(agent_id=self.agent_id, tier=MemoryTier.EPISODIC)

    def clear(self) -> int:
        """Delete every episodic record for this agent. Returns count deleted."""
        return self.backend.clear(agent_id=self.agent_id, tier=MemoryTier.EPISODIC)
