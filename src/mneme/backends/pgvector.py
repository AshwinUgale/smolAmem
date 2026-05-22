"""pgvector backend for Mneme.

Wraps ``psycopg3`` against a Postgres instance with the ``vector`` extension
installed. Schema mirrors :mod:`mneme.backends.sqlite` — single table, one
column per scalar field, a ``vector(D)`` column for the embedding.

Appropriate when:

* Postgres is already in your stack and you don't want a separate vector
  service. Reuse roles, backups, replication, point-in-time recovery.
* You want SQL-native joins / filters on memory records alongside your
  application data.

Not appropriate when:

* You don't already run Postgres and the extra ops surface isn't worth it.
  Use SQLiteBackend (zero infra) or QdrantBackend (single-binary service).
* You need HNSW at hundreds of millions of vectors. pgvector's HNSW is
  fine to ~10M; past that Qdrant or a dedicated vector DB wins.

The pgvector extension must be installed in the target database
(``CREATE EXTENSION vector``). We attempt the CREATE on construction —
permission-denied is silently ignored on the assumption that someone with
DDL rights has already done it.
"""

from __future__ import annotations

import contextlib
import json
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
    # Type-checking only; runtime imports live inside __init__.
    import psycopg


__all__ = ["PgVectorBackend"]


def _row_to_record(row: dict[str, Any]) -> MemoryRecord:
    """Reconstruct the right :class:`MemoryRecord` subclass from a pg row."""
    tier = MemoryTier(row["tier"])
    embedding = list(row["embedding"]) if row["embedding"] is not None else None
    common: dict[str, Any] = {
        "id": row["id"],
        "agent_id": row["agent_id"],
        "content": row["content"],
        "embedding": embedding,
        "created_at": row["created_at"],
        "last_accessed": row["last_accessed"],
        "access_count": row["access_count"],
        "expires_at": row["expires_at"],
        "metadata": row["metadata"]
        if isinstance(row["metadata"], dict)
        else json.loads(row["metadata"]),
    }
    if tier is MemoryTier.EPISODIC:
        return EpisodicMemory(**common)
    if tier is MemoryTier.SEMANTIC:
        provenance = row.get("provenance") or []
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        return SemanticFact(
            confidence=row.get("confidence") or 1.0,
            provenance=list(provenance),
            **common,
        )
    if tier is MemoryTier.WORKING:
        return WorkingMemory(role=row.get("role") or "user", **common)
    raise ValueError(f"unknown tier value in row: {row['tier']!r}")


class PgVectorBackend:
    """Postgres + pgvector implementation of :class:`mneme.MnemeBackend`.

    Args:
        dsn: A psycopg connection string (``"postgresql://user:pw@host/db"``).
            One connection per backend instance; not a connection pool.
        dimensions: Embedding dimensionality. Locked at table creation;
            mismatches raise :class:`ValueError`.
        table: Table name. Default ``"mneme_records"``. Created on first
            construction if it doesn't exist already.

    Not thread-safe; psycopg3 sync connections aren't safe to share
    across threads. Open one backend per thread (or use a connection
    pool above this layer) if you need concurrency.
    """

    def __init__(
        self,
        *,
        dsn: str,
        dimensions: int,
        table: str = "mneme_records",
    ) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        if not table.replace("_", "").isalnum():
            # Defensive: the table name is interpolated into SQL (psycopg
            # can't parameterise identifiers). Reject anything weird.
            raise ValueError(f"invalid table name: {table!r}")

        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise ImportError(
                "PgVectorBackend requires the 'pgvector' extra. Install with:\n"
                "    pip install 'mneme[pgvector]'\n"
                "    # or, in a uv-managed project:\n"
                "    uv add mneme --extra pgvector"
            ) from exc

        self._dimensions = dimensions
        self._table = table
        self._conn: psycopg.Connection[Any] = psycopg.connect(dsn, autocommit=True)

        # Best-effort CREATE EXTENSION — silently OK if already installed
        # or the role lacks DDL rights (the user should have done it).
        try:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error:
            self._conn.rollback()

        register_vector(self._conn)
        self._init_schema()

    def close(self) -> None:
        """Close the underlying psycopg connection. Idempotent."""
        with contextlib.suppress(Exception):
            self._conn.close()

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id             TEXT        PRIMARY KEY,
                    agent_id       TEXT        NOT NULL,
                    tier           TEXT        NOT NULL,
                    content        TEXT        NOT NULL,
                    embedding      vector({self._dimensions}) NOT NULL,
                    created_at     TIMESTAMPTZ NOT NULL,
                    last_accessed  TIMESTAMPTZ,
                    access_count   INTEGER     NOT NULL DEFAULT 0,
                    expires_at     TIMESTAMPTZ,
                    metadata       JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
                    confidence     REAL,
                    provenance     JSONB,
                    role           TEXT
                );
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_agent_tier "
                f"ON {self._table} (agent_id, tier);"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_expires "
                f"ON {self._table} (expires_at) WHERE expires_at IS NOT NULL;"
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

        confidence: float | None = None
        provenance_json: str | None = None
        role: str | None = None
        if isinstance(record, SemanticFact):
            confidence = record.confidence
            provenance_json = json.dumps(record.provenance)
        if isinstance(record, WorkingMemory):
            role = record.role

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table} (
                    id, agent_id, tier, content, embedding, created_at,
                    last_accessed, access_count, expires_at, metadata,
                    confidence, provenance, role
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    agent_id      = EXCLUDED.agent_id,
                    tier          = EXCLUDED.tier,
                    content       = EXCLUDED.content,
                    embedding     = EXCLUDED.embedding,
                    created_at    = EXCLUDED.created_at,
                    last_accessed = EXCLUDED.last_accessed,
                    access_count  = EXCLUDED.access_count,
                    expires_at    = EXCLUDED.expires_at,
                    metadata      = EXCLUDED.metadata,
                    confidence    = EXCLUDED.confidence,
                    provenance    = EXCLUDED.provenance,
                    role          = EXCLUDED.role
                """,
                (
                    record.id,
                    record.agent_id,
                    record.tier.value,
                    record.content,
                    record.embedding,
                    record.created_at,
                    record.last_accessed,
                    record.access_count,
                    record.expires_at,
                    json.dumps(record.metadata),
                    confidence,
                    provenance_json,
                    role,
                ),
            )

    def delete(self, record_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE id = %s",
                (record_id,),
            )
            return cur.rowcount > 0

    def touch(self, record_ids: Iterable[str], *, now: datetime) -> int:
        ids = list(record_ids)
        if not ids:
            return 0
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {self._table}
                   SET last_accessed = %s,
                       access_count  = access_count + 1
                 WHERE id = ANY(%s)
                """,
                (now, ids),
            )
            return cur.rowcount

    # -- reads ---------------------------------------------------------------

    _SELECT_COLUMNS = (
        "id, agent_id, tier, content, embedding, created_at, last_accessed, "
        "access_count, expires_at, metadata, confidence, provenance, role"
    )

    def get(self, record_id: str) -> MemoryRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM {self._table} WHERE id = %s",
                (record_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            columns = [d.name for d in cur.description] if cur.description else []
            return _row_to_record(dict(zip(columns, row, strict=True)))

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

        where_parts = ["agent_id = %s"]
        params: list[Any] = [agent_id]
        if tier_list is not None:
            where_parts.append("tier = ANY(%s)")
            params.append([t.value for t in tier_list])

        fetch_k = k * 4 if metadata_filter else k
        where_clause = " AND ".join(where_parts)

        sql = (
            f"SELECT {self._SELECT_COLUMNS}, "
            "embedding <=> %s::vector AS distance "
            f"FROM {self._table} "
            f"WHERE {where_clause} "
            "ORDER BY distance ASC, created_at DESC "
            "LIMIT %s"
        )

        with self._conn.cursor() as cur:
            cur.execute(sql, (query_embedding, *params, fetch_k))
            columns = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchall()

        results: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            row_dict = dict(zip(columns, row, strict=True))
            if metadata_filter:
                meta = row_dict["metadata"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                if not all(meta.get(key) == value for key, value in metadata_filter.items()):
                    continue
            distance = float(row_dict.pop("distance"))
            record = _row_to_record(row_dict)
            similarity = 1.0 - distance
            results.append((record, similarity))
            if len(results) >= k:
                break
        return results

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
        params: list[Any] = [agent_id, tier.value]
        sql = f"SELECT {self._SELECT_COLUMNS} FROM {self._table} WHERE agent_id = %s AND tier = %s"
        if since is not None:
            sql += " AND created_at >= %s"
            params.append(since)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return [_row_to_record(dict(zip(columns, row, strict=True))) for row in rows]

    # -- observability / maintenance -----------------------------------------

    def count(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        if tier is None:
            sql = f"SELECT COUNT(*) FROM {self._table} WHERE agent_id = %s"
            params: tuple[Any, ...] = (agent_id,)
        else:
            sql = f"SELECT COUNT(*) FROM {self._table} WHERE agent_id = %s AND tier = %s"
            params = (agent_id, tier.value)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def clear(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        if tier is None:
            sql = f"DELETE FROM {self._table} WHERE agent_id = %s"
            params: tuple[Any, ...] = (agent_id,)
        else:
            sql = f"DELETE FROM {self._table} WHERE agent_id = %s AND tier = %s"
            params = (agent_id, tier.value)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount
