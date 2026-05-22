"""SQLite + sqlite-vec backend.

Zero-infra storage for Mneme. Everything lives in a single ``.db`` file (or
``:memory:`` for tests). Vector similarity uses the ``sqlite-vec`` extension's
``vec_distance_cosine`` function, executed in Python via ``sqlite3``.

Schema, in two tables joined by ``rowid``:

* ``records`` — every scalar field of a :class:`MemoryRecord`. ``metadata``
  is JSON-encoded text. ``confidence`` and ``provenance`` are nullable;
  only :class:`SemanticFact` rows populate them.
* ``vec_records`` — a ``vec0`` virtual table holding only the embedding at a
  fixed dimensionality declared at backend construction.

The two tables share a row by virtue of sharing a ``rowid`` (``records``'s
``INTEGER PRIMARY KEY`` column is an alias for ``ROWID``).

Not appropriate when you need real ANN performance at large scale or
multi-process write concurrency — switch to Qdrant or pgvector for that. Fine
for tens of thousands of records, single-process agents, local development,
and the eval harness.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from mneme.types import EpisodicMemory, MemoryRecord, MemoryTier, SemanticFact, WorkingMemory

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into a connection.

    Imported lazily so :class:`SQLiteBackend` can be imported without the
    ``sqlite`` extra installed. Failure here is the only place we raise the
    helpful "install the extra" message.
    """
    try:
        import sqlite_vec
    except ImportError as exc:
        raise ImportError(
            "SQLiteBackend requires the 'sqlite' extra. Install with:\n"
            "    pip install 'mneme[sqlite]'\n"
            "    # or, in a uv-managed project:\n"
            "    uv add mneme --extra sqlite"
        ) from exc

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _embedding_to_json(values: list[float]) -> str:
    """Serialise a vector as a JSON array for the sqlite-vec scalar functions.

    sqlite-vec's ``vec_distance_cosine`` accepts text in JSON form; we use
    that path because it sidesteps platform-specific float binary layouts.
    """
    return json.dumps(values)


def _row_to_record(row: sqlite3.Row, embedding: list[float] | None) -> MemoryRecord:
    """Reconstruct the right :class:`MemoryRecord` subclass from a row.

    ``row`` is from the ``records`` table; ``embedding`` is pre-decoded from
    the ``vec_records`` join (or ``None`` when the caller did not fetch it,
    which currently does not happen).
    """
    tier = MemoryTier(row["tier"])

    common: dict[str, Any] = {
        "id": row["id"],
        "agent_id": row["agent_id"],
        "content": row["content"],
        "embedding": embedding,
        "created_at": datetime.fromisoformat(row["created_at"]),
        "last_accessed": (
            datetime.fromisoformat(row["last_accessed"])
            if row["last_accessed"] is not None
            else None
        ),
        "access_count": row["access_count"],
        "expires_at": (
            datetime.fromisoformat(row["expires_at"]) if row["expires_at"] is not None else None
        ),
        "metadata": json.loads(row["metadata"]),
    }

    if tier is MemoryTier.EPISODIC:
        return EpisodicMemory(**common)
    if tier is MemoryTier.SEMANTIC:
        return SemanticFact(
            confidence=row["confidence"],
            provenance=json.loads(row["provenance"]) if row["provenance"] else [],
            **common,
        )
    if tier is MemoryTier.WORKING:
        # Working memory is not meant to be persisted, but if a user pushes one
        # through anyway we round-trip it faithfully rather than guessing.
        return WorkingMemory(role=row["role"] or "user", **common)

    raise ValueError(f"unknown tier value in records row: {row['tier']!r}")


def _metadata_matches(record_meta: dict[str, Any], filt: dict[str, Any]) -> bool:
    return all(record_meta.get(k) == v for k, v in filt.items())


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class SQLiteBackend:
    """SQLite + sqlite-vec implementation of :class:`mneme.MnemeBackend`.

    Args:
        path: File path for the database, or ``":memory:"`` for an in-process
            database (useful for tests). Accepts ``str`` or :class:`Path`.
        dimensions: Embedding dimensionality. Locked at construction time —
            ``vec0`` virtual tables require a fixed dimension. Records whose
            embedding length differs raise :class:`ValueError`.

    Not thread-safe in v0.1. Open one backend instance per process and route
    all calls through it. For multi-process write workloads, switch to a
    backend that supports concurrent writes (Qdrant, pgvector).
    """

    def __init__(self, *, path: str | Path, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self._dimensions = dimensions
        # ``check_same_thread=False`` so callers running an event loop or a
        # worker pool can share the backend; we document that we do not lock,
        # so concurrent writes are the caller's problem.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        _load_sqlite_vec(self._conn)
        self._init_schema()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying sqlite connection. Idempotent.

        ``sqlite3.Connection.close()`` raises ``ProgrammingError`` if called on
        an already-closed connection; we suppress that so callers can call
        ``close()`` defensively without guarding it.
        """
        with contextlib.suppress(sqlite3.ProgrammingError):
            self._conn.close()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS records (
                    rowid          INTEGER PRIMARY KEY,
                    id             TEXT    NOT NULL UNIQUE,
                    agent_id       TEXT    NOT NULL,
                    tier           TEXT    NOT NULL,
                    content        TEXT    NOT NULL,
                    created_at     TEXT    NOT NULL,
                    last_accessed  TEXT,
                    access_count   INTEGER NOT NULL DEFAULT 0,
                    expires_at     TEXT,
                    metadata       TEXT    NOT NULL,
                    confidence     REAL,
                    provenance     TEXT,
                    role           TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_records_agent_tier
                    ON records (agent_id, tier);
                CREATE INDEX IF NOT EXISTS idx_records_expires_at
                    ON records (expires_at) WHERE expires_at IS NOT NULL;

                CREATE VIRTUAL TABLE IF NOT EXISTS vec_records USING vec0(
                    embedding float[{self._dimensions}]
                );
                """
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
        provenance: str | None = None
        role: str | None = None
        if isinstance(record, SemanticFact):
            confidence = record.confidence
            provenance = json.dumps(record.provenance)
        if isinstance(record, WorkingMemory):
            role = record.role

        embedding_json = _embedding_to_json(record.embedding)

        with self._conn:
            cur = self._conn.execute(
                "SELECT rowid FROM records WHERE id = ?",
                (record.id,),
            )
            existing = cur.fetchone()

            row_values = (
                record.id,
                record.agent_id,
                record.tier.value,
                record.content,
                record.created_at.isoformat(),
                record.last_accessed.isoformat() if record.last_accessed else None,
                record.access_count,
                record.expires_at.isoformat() if record.expires_at else None,
                json.dumps(record.metadata),
                confidence,
                provenance,
                role,
            )

            if existing is None:
                cur = self._conn.execute(
                    """
                    INSERT INTO records (
                        id, agent_id, tier, content, created_at,
                        last_accessed, access_count, expires_at, metadata,
                        confidence, provenance, role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row_values,
                )
                rowid = cur.lastrowid
                self._conn.execute(
                    "INSERT INTO vec_records (rowid, embedding) VALUES (?, ?)",
                    (rowid, embedding_json),
                )
            else:
                rowid = existing["rowid"]
                self._conn.execute(
                    """
                    UPDATE records SET
                        agent_id      = ?,
                        tier          = ?,
                        content       = ?,
                        created_at    = ?,
                        last_accessed = ?,
                        access_count  = ?,
                        expires_at    = ?,
                        metadata      = ?,
                        confidence    = ?,
                        provenance    = ?,
                        role          = ?
                    WHERE id = ?
                    """,
                    (*row_values[1:], record.id),
                )
                self._conn.execute(
                    "UPDATE vec_records SET embedding = ? WHERE rowid = ?",
                    (embedding_json, rowid),
                )

    def delete(self, record_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "SELECT rowid FROM records WHERE id = ?",
                (record_id,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            rowid = row["rowid"]
            self._conn.execute("DELETE FROM records WHERE rowid = ?", (rowid,))
            self._conn.execute("DELETE FROM vec_records WHERE rowid = ?", (rowid,))
            return True

    # -- reads ---------------------------------------------------------------

    def get(self, record_id: str) -> MemoryRecord | None:
        cur = self._conn.execute(
            """
            SELECT r.*, vec_to_json(v.embedding) AS embedding_json
            FROM records r
            LEFT JOIN vec_records v ON v.rowid = r.rowid
            WHERE r.id = ?
            """,
            (record_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
        return _row_to_record(row, embedding)

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

        # Build the WHERE clause dynamically for tier filtering. agent_id is
        # mandatory; tiers and metadata are optional. Metadata filtering is
        # applied in Python after fetch — see ADR-008.
        tier_list = list(tiers) if tiers is not None else None
        where_parts = ["r.agent_id = ?"]
        params: list[Any] = [agent_id]
        if tier_list is not None:
            if not tier_list:
                # Caller asked for "search no tiers" — return nothing.
                return []
            placeholders = ",".join("?" for _ in tier_list)
            where_parts.append(f"r.tier IN ({placeholders})")
            params.extend(t.value for t in tier_list)
        where_clause = " AND ".join(where_parts)

        # Over-fetch if a metadata filter is active so we still have enough
        # candidates after the Python-side filter. Bounded by total rows.
        fetch_k = k * 4 if metadata_filter else k

        query_json = _embedding_to_json(query_embedding)

        sql = f"""
            SELECT r.*,
                   vec_to_json(v.embedding) AS embedding_json,
                   vec_distance_cosine(v.embedding, ?) AS distance
            FROM records r
            JOIN vec_records v ON v.rowid = r.rowid
            WHERE {where_clause}
            ORDER BY distance ASC, r.created_at DESC
            LIMIT ?
        """
        cur = self._conn.execute(sql, (query_json, *params, fetch_k))

        results: list[tuple[MemoryRecord, float]] = []
        for row in cur:
            metadata = json.loads(row["metadata"])
            if metadata_filter and not _metadata_matches(metadata, metadata_filter):
                continue
            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            record = _row_to_record(row, embedding)
            # vec_distance_cosine returns ``1 - cosine_similarity``; flip back
            # to match the protocol contract (higher is better).
            similarity = 1.0 - float(row["distance"])
            results.append((record, similarity))
            if len(results) >= k:
                break

        return results

    def touch(
        self,
        record_ids: Iterable[str],
        *,
        now: datetime,
    ) -> int:
        ids = list(record_ids)
        if not ids:
            return 0
        now_iso = now.isoformat()
        placeholders = ",".join("?" for _ in ids)
        with self._conn:
            cur = self._conn.execute(
                f"""
                UPDATE records
                   SET last_accessed = ?,
                       access_count  = access_count + 1
                 WHERE id IN ({placeholders})
                """,
                (now_iso, *ids),
            )
        return cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0

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

        if since is None:
            cur = self._conn.execute(
                """
                SELECT r.*, vec_to_json(v.embedding) AS embedding_json
                FROM records r
                LEFT JOIN vec_records v ON v.rowid = r.rowid
                WHERE r.agent_id = ? AND r.tier = ?
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (agent_id, tier.value, limit),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT r.*, vec_to_json(v.embedding) AS embedding_json
                FROM records r
                LEFT JOIN vec_records v ON v.rowid = r.rowid
                WHERE r.agent_id = ? AND r.tier = ? AND r.created_at >= ?
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (agent_id, tier.value, since.isoformat(), limit),
            )

        records: list[MemoryRecord] = []
        for row in cur:
            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            records.append(_row_to_record(row, embedding))
        return records

    # -- observability / maintenance -----------------------------------------

    def count(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        if tier is None:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM records WHERE agent_id = ?",
                (agent_id,),
            )
        else:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM records WHERE agent_id = ? AND tier = ?",
                (agent_id, tier.value),
            )
        return int(cur.fetchone()["n"])

    def clear(
        self,
        *,
        agent_id: str,
        tier: MemoryTier | None = None,
    ) -> int:
        with self._conn:
            if tier is None:
                cur = self._conn.execute(
                    "SELECT rowid FROM records WHERE agent_id = ?",
                    (agent_id,),
                )
            else:
                cur = self._conn.execute(
                    "SELECT rowid FROM records WHERE agent_id = ? AND tier = ?",
                    (agent_id, tier.value),
                )
            rowids = [row["rowid"] for row in cur]
            if not rowids:
                return 0
            placeholders = ",".join("?" for _ in rowids)
            self._conn.execute(
                f"DELETE FROM records WHERE rowid IN ({placeholders})",
                rowids,
            )
            self._conn.execute(
                f"DELETE FROM vec_records WHERE rowid IN ({placeholders})",
                rowids,
            )
            return len(rowids)
