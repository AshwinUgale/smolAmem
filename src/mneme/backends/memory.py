"""In-process reference implementation of :class:`MnemeBackend`.

Stores everything in a plain dict, computes cosine similarity in Python.
No external dependencies. Useful for:

* the conformance test suite (every backend implementation runs these tests),
* notebooks and quickstart examples where infra setup would be friction,
* unit tests in downstream code that uses Mneme.

Not appropriate for production: O(n) search, no persistence across process
restarts, no thread safety. For that, use the SQLite or Qdrant backend.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from mneme.types import MemoryRecord, MemoryTier


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns 0.0 if either vector is the zero vector (rather than raising), so
    callers do not need to guard. ``strict=True`` on ``zip`` makes a dimension
    mismatch fail loudly instead of silently truncating.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _metadata_matches(record_meta: dict[str, Any], filt: dict[str, Any]) -> bool:
    """``True`` iff every key/value in ``filt`` is present and equal in ``record_meta``."""
    return all(record_meta.get(k) == v for k, v in filt.items())


class InMemoryBackend:
    """Process-local backend. Satisfies :class:`mneme.backends.MnemeBackend`.

    Holds every record in a single ``dict[str, MemoryRecord]`` keyed by id.
    Search scans the dict, filters, computes cosine similarity for each
    candidate, and returns the top-k. This is O(n) and fine for a few thousand
    records — past that, switch to a real vector index.
    """

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    # -- writes --------------------------------------------------------------

    def upsert(self, record: MemoryRecord) -> None:
        if record.embedding is None:
            raise ValueError(
                f"record {record.id!r} has no embedding; backends only store "
                "embedded records (compute the embedding before calling upsert)"
            )
        self._records[record.id] = record

    def delete(self, record_id: str) -> bool:
        return self._records.pop(record_id, None) is not None

    # -- reads ---------------------------------------------------------------

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def search(
        self,
        *,
        query_embedding: list[float],
        agent_id: str,
        k: int = 5,
        tiers: Iterable[MemoryTier] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        if k <= 0:
            return []

        tier_set = set(tiers) if tiers is not None else None

        candidates: list[tuple[MemoryRecord, float]] = []
        for record in self._records.values():
            if record.agent_id != agent_id:
                continue
            if tier_set is not None and record.tier not in tier_set:
                continue
            if metadata_filter and not _metadata_matches(record.metadata, metadata_filter):
                continue
            if record.embedding is None:
                # Should never happen because upsert rejects it, but be defensive.
                continue
            similarity = _cosine_similarity(query_embedding, record.embedding)
            candidates.append((record, similarity))

        # Sort by similarity desc, then created_at desc as a stable tiebreaker
        # (newer wins on equal score; matches the contract in MnemeBackend).
        candidates.sort(key=lambda pair: (pair[1], pair[0].created_at), reverse=True)
        return candidates[:k]

    def touch(
        self,
        record_ids: Iterable[str],
        *,
        now: datetime,
    ) -> int:
        touched = 0
        for rid in record_ids:
            record = self._records.get(rid)
            if record is None:
                continue
            record.last_accessed = now
            record.access_count += 1
            touched += 1
        return touched

    def list_recent(
        self,
        *,
        agent_id: str,
        tier: MemoryTier,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        candidates = [
            r
            for r in self._records.values()
            if r.agent_id == agent_id
            and r.tier == tier
            and (since is None or r.created_at >= since)
        ]
        candidates.sort(key=lambda r: r.created_at, reverse=True)
        return candidates[:limit]

    # -- observability / maintenance -----------------------------------------

    def count(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        return sum(
            1
            for r in self._records.values()
            if r.agent_id == agent_id and (tier is None or r.tier == tier)
        )

    def clear(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        to_delete = [
            rid
            for rid, r in self._records.items()
            if r.agent_id == agent_id and (tier is None or r.tier == tier)
        ]
        for rid in to_delete:
            del self._records[rid]
        return len(to_delete)
