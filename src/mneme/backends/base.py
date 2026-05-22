"""The ``MnemeBackend`` protocol — Mneme's pluggable storage seam.

Backends own persistence and vector search for the episodic and semantic tiers.
Working memory lives in process and never touches a backend.

The protocol is intentionally narrow. A backend does seven things:

1. **Upsert** a record (insert or replace by id).
2. **Get** a record by id.
3. **Delete** a record by id.
4. **Search** by query vector, returning ``(record, similarity)`` pairs.
5. **List recent** records by tier in time order (added v0.2 for consolidation).
6. **Touch** a set of records to bump access tracking (added v0.3 for forgetting).
7. **Count / clear** for observability and tests.

Things a backend deliberately does *not* do:

* **Compute embeddings.** The tier layer embeds and passes the vector in. This
  keeps backends free of LLM-model dependencies and makes them swappable.
* **Apply authority weighting, freshness decay, or any fused scoring.** Those
  live in :func:`mneme.retrieve`. The backend returns raw similarity.
* **Generate ids or timestamps.** :class:`mneme.MemoryRecord` already does.

This narrowness is what makes a SQLite-vec, Qdrant, or pgvector backend trade
out cleanly behind the same call sites.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from mneme.types import MemoryRecord, MemoryTier


@runtime_checkable
class MnemeBackend(Protocol):
    """Pluggable storage + vector-search layer for Mneme's persisted tiers.

    Implementations are usually classes but can be any object that satisfies
    this structural type. ``@runtime_checkable`` enables ``isinstance`` checks
    in tests; method calls are still resolved by structural typing, not by
    inheritance.

    Method contract notes:

    * Every record passed to :meth:`upsert` must have ``embedding`` set —
      backends are not expected to embed text. Records without embeddings are
      not searchable and should be rejected with ``ValueError``.
    * :meth:`search` returns results sorted by similarity, descending. Ties
      should be broken deterministically (implementations may use ``created_at``
      descending as a tiebreaker so newer wins on equal score).
    * Backends are multi-tenant: every read/write is scoped by ``agent_id``.
      There is no cross-agent leakage.
    """

    def upsert(self, record: MemoryRecord) -> None:
        """Insert a new record or replace an existing one by ``record.id``.

        Raises:
            ValueError: if ``record.embedding`` is ``None``.
        """
        ...

    def get(self, record_id: str) -> MemoryRecord | None:
        """Return the record with the given id, or ``None`` if not found."""
        ...

    def delete(self, record_id: str) -> bool:
        """Delete a record by id. Returns ``True`` if the id existed, else ``False``."""
        ...

    def search(
        self,
        *,
        query_embedding: list[float],
        agent_id: str,
        k: int = 5,
        tiers: Iterable[MemoryTier] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Return up to ``k`` records most similar to ``query_embedding``.

        Args:
            query_embedding: The query vector. Must be the same dimension as
                stored records' embeddings; backends may raise if not.
            agent_id: Namespace filter — only records belonging to this agent
                are searched.
            k: Maximum number of results. Backends should return fewer if
                fewer matching records exist.
            tiers: If given, restrict search to records in these tiers. If
                ``None`` (default), all tiers are searched.
            metadata_filter: If given, only records whose ``metadata`` dict
                contains every key/value pair are returned. Equality only;
                no operators.

        Returns:
            ``[(record, similarity), ...]`` sorted by similarity descending.
            Similarity is in ``[-1.0, 1.0]`` for cosine; backends should
            document any other metric they use.
        """
        ...

    def list_recent(
        self,
        *,
        agent_id: str,
        tier: MemoryTier,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[MemoryRecord]:
        """Return records in a tier ordered by ``created_at`` descending.

        Added in v0.2 to support consolidation, which needs to enumerate
        episodes by time rather than similarity. Distinct from :meth:`search`
        in that there is no query vector and no ranking — just a windowed
        time-ordered listing.

        Args:
            agent_id: Namespace filter (mandatory, like every other method).
            tier: Which tier to list. Single tier only — consolidation
                doesn't have a use case for cross-tier listings, and a
                single tier per call keeps the contract simple.
            since: If given, only records with ``created_at >= since`` are
                returned. ``None`` means "all of them" subject to ``limit``.
            limit: Maximum number of records to return. Default 1000.

        Returns:
            ``[record, ...]`` newest first. Length is at most ``limit``.
        """
        ...

    def touch(
        self,
        record_ids: Iterable[str],
        *,
        now: datetime,
    ) -> int:
        """Mark records as just-accessed: bump ``access_count``, set ``last_accessed``.

        Added in v0.3. Called by :meth:`MemoryManager.retrieve` for the
        top-k records it actually returns so access-frequency decay (used
        by :meth:`MemoryManager.forget`) reflects real usage across
        restarts.

        Args:
            record_ids: Ids to touch. Unknown ids are silently skipped.
            now: Timestamp to write into ``last_accessed``.

        Returns:
            The number of records actually touched (i.e. found in storage).
            Useful for observability; safe to ignore.
        """
        ...

    def count(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        """Number of records for an agent, optionally scoped to one tier."""
        ...

    def clear(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        """Delete records for an agent, optionally tier-scoped. Returns count deleted.

        Useful in tests and for explicit "forget this agent" operations.
        """
        ...
