"""Qdrant backend for Mneme.

Wraps ``qdrant-client``. One Qdrant collection per backend instance;
multi-tenancy via a ``payload`` filter on ``agent_id`` (same shape as the
SQLite backend).

All scalar fields (agent_id, tier, content, timestamps, expires_at,
confidence, provenance, role, metadata) ride in the point payload. The
embedding is the point's vector. Distance is cosine.

Not appropriate when:

* You need fully self-contained storage with no daemon — use SQLiteBackend.
* You need transactional cross-document writes — Qdrant is eventually
  consistent across replicas in a clustered setup, fine for single-node.

Appropriate when:

* You're running production agents with thousands+ of records and want
  HNSW-backed sub-millisecond search.
* You're already on Qdrant Cloud for other vector workloads and want
  Mneme to share infra.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mneme.types import (
    EpisodicMemory,
    MemoryRecord,
    MemoryTier,
    SemanticFact,
    WorkingMemory,
)

if TYPE_CHECKING:
    # Type-checking-only imports; runtime imports happen lazily in __init__
    # so the class loads without the [qdrant] extra installed.
    from qdrant_client import QdrantClient


__all__ = ["QdrantBackend"]


# Qdrant requires point IDs to be either unsigned integers or *unsigned UUIDs*.
# Mneme uses hex-string ids generated from ``uuid4().hex`` — we round-trip
# them through ``uuid.UUID`` to satisfy Qdrant's validator, and store the
# original hex string in payload for human-readable reads.
def _id_to_qdrant(record_id: str) -> str:
    """Render a Mneme hex id as a Qdrant-acceptable UUID string."""
    return str(uuid.UUID(hex=record_id))


def _payload_to_record(point_id: Any, payload: dict[str, Any], vector: Any) -> MemoryRecord:
    """Reconstruct the right ``MemoryRecord`` subclass from a Qdrant payload."""
    tier = MemoryTier(payload["tier"])
    common: dict[str, Any] = {
        "id": payload["id"],
        "agent_id": payload["agent_id"],
        "content": payload["content"],
        "embedding": list(vector) if vector is not None else None,
        "created_at": datetime.fromisoformat(payload["created_at"]),
        "last_accessed": (
            datetime.fromisoformat(payload["last_accessed"])
            if payload.get("last_accessed")
            else None
        ),
        "access_count": payload.get("access_count", 0),
        "expires_at": (
            datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None
        ),
        "metadata": payload.get("metadata", {}),
    }
    if tier is MemoryTier.EPISODIC:
        return EpisodicMemory(**common)
    if tier is MemoryTier.SEMANTIC:
        return SemanticFact(
            confidence=payload.get("confidence", 1.0),
            provenance=list(payload.get("provenance", [])),
            **common,
        )
    if tier is MemoryTier.WORKING:
        return WorkingMemory(role=payload.get("role") or "user", **common)
    raise ValueError(f"unknown tier value in qdrant payload: {payload['tier']!r}")


def _record_to_payload(record: MemoryRecord) -> dict[str, Any]:
    """Serialise a record to a Qdrant payload dict (vector handled separately)."""
    payload: dict[str, Any] = {
        "id": record.id,
        "agent_id": record.agent_id,
        "tier": record.tier.value,
        "content": record.content,
        "created_at": record.created_at.isoformat(),
        "last_accessed": (record.last_accessed.isoformat() if record.last_accessed else None),
        "access_count": record.access_count,
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        "metadata": dict(record.metadata),
    }
    if isinstance(record, SemanticFact):
        payload["confidence"] = record.confidence
        payload["provenance"] = list(record.provenance)
    if isinstance(record, WorkingMemory):
        payload["role"] = record.role
    return payload


class QdrantBackend:
    """Qdrant-backed implementation of :class:`mneme.MnemeBackend`.

    Args:
        client: A constructed ``qdrant_client.QdrantClient``. Pass the
            client the application already has if Qdrant is shared.
        collection: Collection name. Created on first construction if it
            doesn't exist already. Default ``"mneme"``.
        dimensions: Embedding dimensionality. Locked at collection
            creation; mismatches raise :class:`ValueError`.

    Not thread-safe at the Python layer; Qdrant's HTTP/gRPC client itself
    is generally safe for concurrent calls, but Mneme's own writers should
    coordinate.
    """

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection: str = "mneme",
        dimensions: int,
    ) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")

        # Lazy import qdrant_client so the class itself loads without the
        # [qdrant] extra installed.
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise ImportError(
                "QdrantBackend requires the 'qdrant' extra. Install with:\n"
                "    pip install 'mneme[qdrant]'\n"
                "    # or, in a uv-managed project:\n"
                "    uv add mneme --extra qdrant"
            ) from exc

        self._client = client
        self._collection = collection
        self._dimensions = dimensions
        self._models = models

        self._ensure_collection()

    # -- lifecycle -----------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._models.VectorParams(
                    size=self._dimensions,
                    distance=self._models.Distance.COSINE,
                ),
            )
        # Payload indexes. Qdrant requires a range index on any field used
        # in ``order_by`` (we use it on ``created_at`` in ``list_recent``).
        # KEYWORD indexes on ``agent_id`` and ``tier`` aren't strictly
        # required but they're hot filter paths on every read — indexing
        # them keeps multi-tenant query latency sane as the collection
        # grows. ``create_payload_index`` is idempotent: re-running it on
        # an existing index is a cheap no-op, but we still wrap in
        # ``contextlib.suppress`` defensively so a future Qdrant version
        # that errors on duplicate-index doesn't break construction.
        import contextlib

        for field, schema in (
            ("created_at", self._models.PayloadSchemaType.DATETIME),
            ("agent_id", self._models.PayloadSchemaType.KEYWORD),
            ("tier", self._models.PayloadSchemaType.KEYWORD),
        ):
            with contextlib.suppress(Exception):
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                )

    # -- writes --------------------------------------------------------------

    def upsert(self, record: MemoryRecord) -> None:
        if record.embedding is None:
            raise ValueError(
                f"record {record.id!r} has no embedding; backends only store "
                "embedded records (compute the embedding before calling upsert)"
            )
        if len(record.embedding) != self._dimensions:
            raise ValueError(
                f"record {record.id!r} embedding has {len(record.embedding)} dims, "
                f"backend was constructed with dimensions={self._dimensions}"
            )

        point = self._models.PointStruct(
            id=_id_to_qdrant(record.id),
            vector=list(record.embedding),
            payload=_record_to_payload(record),
        )
        self._client.upsert(collection_name=self._collection, points=[point])

    def delete(self, record_id: str) -> bool:
        try:
            qid = _id_to_qdrant(record_id)
        except ValueError:
            return False
        existing = self._client.retrieve(
            collection_name=self._collection,
            ids=[qid],
            with_payload=False,
            with_vectors=False,
        )
        if not existing:
            return False
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._models.PointIdsList(points=[qid]),
        )
        return True

    def touch(self, record_ids: Iterable[str], *, now: datetime) -> int:
        ids = list(record_ids)
        if not ids:
            return 0
        # Skip IDs that aren't valid hex UUIDs — same "silently skip
        # unknown id" contract as get() / delete(). A malformed id is
        # by definition not in the store, so the right behavior is to
        # treat it like a miss, not raise.
        qids: list[str] = []
        for rid in ids:
            try:
                qids.append(_id_to_qdrant(rid))
            except ValueError:
                continue
        if not qids:
            return 0
        # Read current access_counts, increment in Python, set_payload.
        # Qdrant lacks a server-side "increment" operator; this is two RTTs
        # but each round-trip is cheap.
        existing = self._client.retrieve(
            collection_name=self._collection,
            ids=qids,
            with_payload=["access_count"],
            with_vectors=False,
        )
        touched = 0
        now_iso = now.isoformat()
        for point in existing:
            current = (point.payload or {}).get("access_count", 0)
            self._client.set_payload(
                collection_name=self._collection,
                payload={
                    "access_count": current + 1,
                    "last_accessed": now_iso,
                },
                points=[point.id],
            )
            touched += 1
        return touched

    # -- reads ---------------------------------------------------------------

    def get(self, record_id: str) -> MemoryRecord | None:
        try:
            qid = _id_to_qdrant(record_id)
        except ValueError:
            return None
        result = self._client.retrieve(
            collection_name=self._collection,
            ids=[qid],
            with_payload=True,
            with_vectors=True,
        )
        if not result:
            return None
        point = result[0]
        return _payload_to_record(point.id, point.payload or {}, point.vector)

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
        if len(query_embedding) != self._dimensions:
            raise ValueError(
                f"query embedding has {len(query_embedding)} dims, "
                f"backend was constructed with dimensions={self._dimensions}"
            )

        tier_list = list(tiers) if tiers is not None else None
        if tier_list is not None and not tier_list:
            return []

        # ``must`` is typed as ``list[Any]`` because qdrant-client's
        # ``Filter.must`` accepts a union of condition types and Python's
        # invariant ``list`` won't accept ``list[FieldCondition]`` for it.
        must: list[Any] = [
            self._models.FieldCondition(
                key="agent_id",
                match=self._models.MatchValue(value=agent_id),
            )
        ]
        if tier_list is not None:
            must.append(
                self._models.FieldCondition(
                    key="tier",
                    match=self._models.MatchAny(any=[t.value for t in tier_list]),
                )
            )

        # Metadata equality filter pushed down where possible; same shape as
        # other backends — every key/value pair must match.
        if metadata_filter:
            for key, value in metadata_filter.items():
                must.append(
                    self._models.FieldCondition(
                        key=f"metadata.{key}",
                        match=self._models.MatchValue(value=value),
                    )
                )

        qdrant_filter = self._models.Filter(must=must)

        hits = self._client.query_points(
            collection_name=self._collection,
            query=list(query_embedding),
            query_filter=qdrant_filter,
            limit=k,
            with_payload=True,
            with_vectors=True,
        ).points

        out: list[tuple[MemoryRecord, float]] = []
        for hit in hits:
            record = _payload_to_record(hit.id, hit.payload or {}, hit.vector)
            # Qdrant returns cosine *similarity* (not distance) when the
            # collection is configured with COSINE — score is already in
            # [-1, 1] matching the protocol contract.
            out.append((record, float(hit.score)))
        return out

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

        must: list[Any] = [
            self._models.FieldCondition(
                key="agent_id", match=self._models.MatchValue(value=agent_id)
            ),
            self._models.FieldCondition(
                key="tier", match=self._models.MatchValue(value=tier.value)
            ),
        ]
        if since is not None:
            must.append(
                self._models.FieldCondition(
                    key="created_at",
                    range=self._models.DatetimeRange(gte=since),
                )
            )

        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=self._models.Filter(must=must),
            limit=limit,
            with_payload=True,
            with_vectors=True,
            order_by=self._models.OrderBy(key="created_at", direction=self._models.Direction.DESC),
        )
        return [_payload_to_record(p.id, p.payload or {}, p.vector) for p in points]

    # -- observability / maintenance -----------------------------------------

    def count(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        must: list[Any] = [
            self._models.FieldCondition(
                key="agent_id", match=self._models.MatchValue(value=agent_id)
            )
        ]
        if tier is not None:
            must.append(
                self._models.FieldCondition(
                    key="tier", match=self._models.MatchValue(value=tier.value)
                )
            )
        result = self._client.count(
            collection_name=self._collection,
            count_filter=self._models.Filter(must=must),
            exact=True,
        )
        return int(result.count)

    def clear(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        must: list[Any] = [
            self._models.FieldCondition(
                key="agent_id", match=self._models.MatchValue(value=agent_id)
            )
        ]
        if tier is not None:
            must.append(
                self._models.FieldCondition(
                    key="tier", match=self._models.MatchValue(value=tier.value)
                )
            )
        # count first so we can return the number deleted; Qdrant's
        # delete_by_filter doesn't return a count.
        n = self.count(agent_id=agent_id, tier=tier)
        if n == 0:
            return 0
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._models.FilterSelector(filter=self._models.Filter(must=must)),
        )
        return n
