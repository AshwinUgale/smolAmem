"""Semantic memory tier.

Distilled facts about an agent's user, project, or domain, produced by the
consolidation pass over episodic memories. Each fact carries a
``confidence`` score (how sure consolidation is) and a ``provenance`` list
(ids of the episodes that produced it).

In v0.1 this tier exposed a stubbed ``consolidate()``. In v0.2 the real
consolidation algorithm lives on :class:`mneme.MemoryManager` because it
needs access to the episodic tier and the LLM judge — both outside this
tier's scope. ``manager.consolidate(...)`` is the call site; this tier's
``add()`` is what consolidation calls to write the produced facts.
"""

from __future__ import annotations

from typing import Any, cast

from mneme.backends import MnemeBackend
from mneme.embeddings import EmbeddingProvider
from mneme.types import MemoryTier, SemanticFact


class SemanticMemoryTier:
    """Persistent, semantically-searchable consolidated facts for one agent.

    Shape is parallel to :class:`EpisodicMemoryTier` with one difference:

    * :meth:`add` accepts ``confidence`` and ``provenance`` so callers (and
      the consolidator) can record where a fact came from and how sure we
      are about it.
    """

    tier = MemoryTier.SEMANTIC

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
        confidence: float = 1.0,
        provenance: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticFact:
        """Embed ``content`` and persist it as a semantic fact.

        Args:
            content: The fact, in natural language. E.g., "user prefers
                TypeScript examples".
            confidence: How sure we are, in ``[0.0, 1.0]``. Retrieval ranking
                (Step 6) weights by this. Default 1.0 (set by hand → trust it).
            provenance: Optional list of episodic-memory ids that produced
                this fact. Populated by the consolidator; manual ``add``
                callers typically leave it empty.
            metadata: Free-form caller tags.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")
        [embedding] = self.embedder.embed([content])
        fact = SemanticFact(
            agent_id=self.agent_id,
            content=content,
            embedding=embedding,
            confidence=confidence,
            provenance=list(provenance) if provenance else [],
            metadata=metadata or {},
        )
        self.backend.upsert(fact)
        return fact

    # -- reads ---------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[SemanticFact, float]]:
        """Return the top-``k`` facts most similar to ``query``."""
        [query_embedding] = self.embedder.embed([query])
        raw = self.backend.search(
            query_embedding=query_embedding,
            agent_id=self.agent_id,
            k=k,
            tiers=[MemoryTier.SEMANTIC],
            metadata_filter=metadata_filter,
        )
        return [(cast(SemanticFact, rec), score) for rec, score in raw]

    def get(self, record_id: str) -> SemanticFact | None:
        record = self.backend.get(record_id)
        if record is None or record.tier is not MemoryTier.SEMANTIC:
            return None
        return cast(SemanticFact, record)

    def delete(self, record_id: str) -> bool:
        existing = self.get(record_id)
        if existing is None:
            return False
        return self.backend.delete(record_id)

    def count(self) -> int:
        return self.backend.count(agent_id=self.agent_id, tier=MemoryTier.SEMANTIC)

    def clear(self) -> int:
        return self.backend.clear(agent_id=self.agent_id, tier=MemoryTier.SEMANTIC)
