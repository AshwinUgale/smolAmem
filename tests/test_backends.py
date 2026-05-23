"""Conformance suite for any :class:`mneme.MnemeBackend` implementation.

Every test here is parametrised over the ``backend`` fixture below. When we
add the SQLite, Qdrant, or pgvector backends, we extend the ``params`` list
and these tests automatically run against the new implementation. A backend
that passes this suite is considered conforming.

The suite covers:
* basic upsert / get / delete
* tenant isolation by ``agent_id``
* tier filtering on search
* metadata filtering on search
* k bounds (k=0, k larger than count)
* search ordering (similarity desc, tiebreak by recency)
* count / clear (whole agent + tier-scoped)
* upsert rejecting un-embedded records
"""

import contextlib
import os
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from mneme import (
    EpisodicMemory,
    InMemoryBackend,
    MemoryTier,
    MnemeBackend,
    PgVectorBackend,
    QdrantBackend,
    SemanticFact,
    SQLiteBackend,
)

# Embedding dimension used everywhere in this suite. Three is enough to make
# similarity ordering trivially verifiable (cardinal axes are orthogonal).
_DIM = 3

# Env vars opt into the Qdrant / pgvector integration paths. Without them
# the corresponding params are skipped — contributors without Docker can
# still run the full suite against memory + sqlite.
_QDRANT_URL = os.environ.get("MNEME_TEST_QDRANT_URL")
_PGVECTOR_DSN = os.environ.get("MNEME_TEST_PGVECTOR_DSN")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _qdrant_client():
    """Lazy import so the file loads without the [qdrant] extra installed.

    ``check_compatibility=False`` because the bundled docker-compose pins
    an older qdrant server while ``qdrant-client`` floats; the mismatch
    warning is noise for our purposes (we exercise only the stable subset
    of the API). Production callers can leave compatibility checking on.
    """
    from qdrant_client import QdrantClient

    return QdrantClient(url=_QDRANT_URL, check_compatibility=False)


@pytest.fixture(params=["memory", "sqlite", "qdrant", "pgvector"])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[MnemeBackend]:
    """Return a fresh empty backend for each test.

    Yields rather than returns so backends with resources can be closed
    between tests. Qdrant and pgvector tests skip cleanly if the
    ``MNEME_TEST_QDRANT_URL`` / ``MNEME_TEST_PGVECTOR_DSN`` env vars
    aren't set — keeping the suite green for contributors without Docker.
    """
    name = request.param
    if name == "memory":
        yield InMemoryBackend()
        return
    if name == "sqlite":
        b_sqlite = SQLiteBackend(path=tmp_path / "test.db", dimensions=_DIM)
        try:
            yield b_sqlite
        finally:
            b_sqlite.close()
        return
    if name == "qdrant":
        if not _QDRANT_URL:
            pytest.skip("MNEME_TEST_QDRANT_URL not set; skipping qdrant conformance")
        # Each test gets its own collection so cross-test state can't leak.
        collection = f"mneme_test_{uuid.uuid4().hex[:8]}"
        client = _qdrant_client()
        b_qdrant = QdrantBackend(client=client, collection=collection, dimensions=_DIM)
        try:
            yield b_qdrant
        finally:
            with contextlib.suppress(Exception):
                client.delete_collection(collection_name=collection)
        return
    if name == "pgvector":
        if not _PGVECTOR_DSN:
            pytest.skip("MNEME_TEST_PGVECTOR_DSN not set; skipping pgvector conformance")
        # Each test gets its own table so cross-test state can't leak.
        table = f"mneme_test_{uuid.uuid4().hex[:8]}"
        b_pg = PgVectorBackend(dsn=_PGVECTOR_DSN, dimensions=_DIM, table=table)
        try:
            yield b_pg
        finally:
            with (
                contextlib.suppress(Exception),
                b_pg._conn.cursor() as cur,  # type: ignore[attr-defined]
            ):
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            b_pg.close()
        return
    raise ValueError(f"unknown backend fixture param: {name!r}")


def _episode(
    *,
    agent_id: str = "agent-1",
    content: str = "hello",
    embedding: Iterable[float] = (1.0, 0.0, 0.0),
    metadata: dict[str, object] | None = None,
) -> EpisodicMemory:
    """Tiny constructor so tests stay readable."""
    return EpisodicMemory(
        agent_id=agent_id,
        content=content,
        embedding=list(embedding),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_backend_satisfies_protocol(backend):
    # @runtime_checkable on MnemeBackend enables this check.
    assert isinstance(backend, MnemeBackend)


# ---------------------------------------------------------------------------
# Upsert / get / delete
# ---------------------------------------------------------------------------


def test_upsert_then_get_round_trip(backend):
    rec = _episode(content="round trip")
    backend.upsert(rec)
    fetched = backend.get(rec.id)
    assert fetched is not None
    assert fetched.id == rec.id
    assert fetched.content == "round trip"


def test_get_returns_none_for_missing_id(backend):
    assert backend.get("nonexistent") is None


def test_upsert_replaces_existing_by_id(backend):
    rec = _episode(content="original")
    backend.upsert(rec)
    rec.content = "updated"
    backend.upsert(rec)
    fetched = backend.get(rec.id)
    assert fetched is not None
    assert fetched.content == "updated"


def test_delete_returns_true_when_present_false_when_absent(backend):
    rec = _episode()
    backend.upsert(rec)
    assert backend.delete(rec.id) is True
    assert backend.delete(rec.id) is False
    assert backend.get(rec.id) is None


def test_upsert_rejects_records_without_embedding(backend):
    rec = EpisodicMemory(agent_id="a", content="no embedding")
    assert rec.embedding is None
    with pytest.raises(ValueError):
        backend.upsert(rec)


# ---------------------------------------------------------------------------
# Search — basics, ordering, k bounds
# ---------------------------------------------------------------------------


def test_search_returns_most_similar_first(backend):
    near = _episode(content="near", embedding=(1.0, 0.0, 0.0))
    middle = _episode(content="middle", embedding=(0.7, 0.7, 0.0))
    far = _episode(content="far", embedding=(0.0, 0.0, 1.0))
    for r in (far, middle, near):  # insert in non-order
        backend.upsert(r)

    results = backend.search(
        query_embedding=[1.0, 0.0, 0.0],
        agent_id="agent-1",
        k=3,
    )
    contents = [r.content for r, _ in results]
    assert contents == ["near", "middle", "far"]
    # scores monotonically non-increasing
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_k_zero_returns_empty(backend):
    backend.upsert(_episode())
    assert backend.search(query_embedding=[1.0, 0.0, 0.0], agent_id="agent-1", k=0) == []


def test_search_k_larger_than_count_returns_all(backend):
    backend.upsert(_episode(content="only"))
    results = backend.search(query_embedding=[1.0, 0.0, 0.0], agent_id="agent-1", k=99)
    assert len(results) == 1


def test_search_empty_backend_returns_empty(backend):
    assert backend.search(query_embedding=[1.0, 0.0, 0.0], agent_id="agent-1", k=5) == []


# ---------------------------------------------------------------------------
# Search — tenant + tier + metadata filtering
# ---------------------------------------------------------------------------


def test_search_isolates_by_agent_id(backend):
    backend.upsert(_episode(agent_id="alice", content="alice memory"))
    backend.upsert(_episode(agent_id="bob", content="bob memory"))

    alice_results = backend.search(query_embedding=[1.0, 0.0, 0.0], agent_id="alice", k=5)
    assert [r.content for r, _ in alice_results] == ["alice memory"]


def test_search_filters_by_tier(backend):
    backend.upsert(_episode(content="an episode"))
    fact = SemanticFact(
        agent_id="agent-1",
        content="a fact",
        embedding=[1.0, 0.0, 0.0],
    )
    backend.upsert(fact)

    only_episodic = backend.search(
        query_embedding=[1.0, 0.0, 0.0],
        agent_id="agent-1",
        k=5,
        tiers=[MemoryTier.EPISODIC],
    )
    assert [r.content for r, _ in only_episodic] == ["an episode"]

    only_semantic = backend.search(
        query_embedding=[1.0, 0.0, 0.0],
        agent_id="agent-1",
        k=5,
        tiers=[MemoryTier.SEMANTIC],
    )
    assert [r.content for r, _ in only_semantic] == ["a fact"]


def test_search_tiers_none_returns_all_tiers(backend):
    backend.upsert(_episode(content="an episode"))
    backend.upsert(SemanticFact(agent_id="agent-1", content="a fact", embedding=[1.0, 0.0, 0.0]))
    results = backend.search(query_embedding=[1.0, 0.0, 0.0], agent_id="agent-1", k=5, tiers=None)
    assert len(results) == 2


def test_search_metadata_filter_equality(backend):
    backend.upsert(_episode(content="from chat", metadata={"source": "chat"}))
    backend.upsert(_episode(content="from docs", metadata={"source": "docs"}))

    chat_only = backend.search(
        query_embedding=[1.0, 0.0, 0.0],
        agent_id="agent-1",
        k=5,
        metadata_filter={"source": "chat"},
    )
    assert [r.content for r, _ in chat_only] == ["from chat"]


def test_search_metadata_filter_requires_all_keys(backend):
    backend.upsert(_episode(content="matches", metadata={"source": "chat", "lang": "en"}))
    backend.upsert(_episode(content="missing lang", metadata={"source": "chat"}))

    filtered = backend.search(
        query_embedding=[1.0, 0.0, 0.0],
        agent_id="agent-1",
        k=5,
        metadata_filter={"source": "chat", "lang": "en"},
    )
    assert [r.content for r, _ in filtered] == ["matches"]


# ---------------------------------------------------------------------------
# touch — added v0.3 for persisted access tracking
# ---------------------------------------------------------------------------


def test_touch_bumps_access_count_and_sets_last_accessed(backend):
    from datetime import UTC, datetime

    rec = _episode()
    backend.upsert(rec)
    assert rec.access_count == 0

    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    touched = backend.touch([rec.id], now=now)
    assert touched == 1

    fetched = backend.get(rec.id)
    assert fetched is not None
    assert fetched.access_count == 1
    assert fetched.last_accessed == now


def test_touch_multiple_ids(backend):
    from datetime import UTC, datetime

    a = _episode(content="a")
    b = _episode(content="b")
    backend.upsert(a)
    backend.upsert(b)

    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    touched = backend.touch([a.id, b.id], now=now)
    assert touched == 2


def test_touch_unknown_ids_are_silently_skipped(backend):
    from datetime import UTC, datetime

    rec = _episode()
    backend.upsert(rec)
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)

    touched = backend.touch([rec.id, "does-not-exist"], now=now)
    assert touched == 1


def test_touch_empty_input_is_a_noop(backend):
    from datetime import UTC, datetime

    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    assert backend.touch([], now=now) == 0


def test_touch_increments_repeatedly(backend):
    from datetime import UTC, datetime

    rec = _episode()
    backend.upsert(rec)
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    backend.touch([rec.id], now=now)
    backend.touch([rec.id], now=now)
    backend.touch([rec.id], now=now)
    fetched = backend.get(rec.id)
    assert fetched is not None
    assert fetched.access_count == 3


# ---------------------------------------------------------------------------
# expires_at — added v0.3 for TTL
# ---------------------------------------------------------------------------


def test_expires_at_round_trips_through_backend(backend):
    from datetime import UTC, datetime

    expiry = datetime(2026, 12, 31, tzinfo=UTC)
    rec = EpisodicMemory(
        agent_id="agent-1",
        content="ttl",
        embedding=[1.0, 0.0, 0.0],
        expires_at=expiry,
    )
    backend.upsert(rec)
    fetched = backend.get(rec.id)
    assert fetched is not None
    assert fetched.expires_at == expiry


def test_expires_at_defaults_to_none(backend):
    rec = _episode()
    backend.upsert(rec)
    fetched = backend.get(rec.id)
    assert fetched is not None
    assert fetched.expires_at is None


# ---------------------------------------------------------------------------
# list_recent — added v0.2 for consolidation
# ---------------------------------------------------------------------------


def test_list_recent_returns_newest_first(backend):
    from datetime import UTC, datetime

    older = EpisodicMemory(
        agent_id="agent-1",
        content="older",
        embedding=[1.0, 0.0, 0.0],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = EpisodicMemory(
        agent_id="agent-1",
        content="newer",
        embedding=[1.0, 0.0, 0.0],
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    backend.upsert(older)
    backend.upsert(newer)

    out = backend.list_recent(agent_id="agent-1", tier=MemoryTier.EPISODIC)
    assert [r.content for r in out] == ["newer", "older"]


def test_list_recent_filters_by_tier(backend):
    backend.upsert(_episode(content="ep"))
    backend.upsert(SemanticFact(agent_id="agent-1", content="fact", embedding=[1.0, 0.0, 0.0]))
    episodes = backend.list_recent(agent_id="agent-1", tier=MemoryTier.EPISODIC)
    facts = backend.list_recent(agent_id="agent-1", tier=MemoryTier.SEMANTIC)
    assert [r.content for r in episodes] == ["ep"]
    assert [r.content for r in facts] == ["fact"]


def test_list_recent_isolates_by_agent(backend):
    backend.upsert(_episode(agent_id="alice", content="alice memory"))
    backend.upsert(_episode(agent_id="bob", content="bob memory"))
    alice_out = backend.list_recent(agent_id="alice", tier=MemoryTier.EPISODIC)
    assert [r.content for r in alice_out] == ["alice memory"]


def test_list_recent_since_filter(backend):
    from datetime import UTC, datetime

    cutoff = datetime(2026, 3, 1, tzinfo=UTC)
    before = EpisodicMemory(
        agent_id="agent-1",
        content="before",
        embedding=[1.0, 0.0, 0.0],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    after = EpisodicMemory(
        agent_id="agent-1",
        content="after",
        embedding=[1.0, 0.0, 0.0],
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    backend.upsert(before)
    backend.upsert(after)

    recent = backend.list_recent(agent_id="agent-1", tier=MemoryTier.EPISODIC, since=cutoff)
    assert [r.content for r in recent] == ["after"]


def test_list_recent_honors_limit(backend):
    for i in range(5):
        backend.upsert(_episode(content=f"ep-{i}"))
    out = backend.list_recent(agent_id="agent-1", tier=MemoryTier.EPISODIC, limit=2)
    assert len(out) == 2


def test_list_recent_limit_zero_returns_empty(backend):
    backend.upsert(_episode())
    assert backend.list_recent(agent_id="agent-1", tier=MemoryTier.EPISODIC, limit=0) == []


def test_list_recent_empty_returns_empty(backend):
    assert backend.list_recent(agent_id="agent-1", tier=MemoryTier.EPISODIC) == []


# ---------------------------------------------------------------------------
# Count / clear
# ---------------------------------------------------------------------------


def test_count_all_and_by_tier(backend):
    backend.upsert(_episode(content="ep1"))
    backend.upsert(_episode(content="ep2"))
    backend.upsert(SemanticFact(agent_id="agent-1", content="fact1", embedding=[1.0, 0.0, 0.0]))

    assert backend.count(agent_id="agent-1") == 3
    assert backend.count(agent_id="agent-1", tier=MemoryTier.EPISODIC) == 2
    assert backend.count(agent_id="agent-1", tier=MemoryTier.SEMANTIC) == 1
    assert backend.count(agent_id="other") == 0


def test_clear_scoped_to_tier(backend):
    backend.upsert(_episode(content="ep1"))
    backend.upsert(_episode(content="ep2"))
    backend.upsert(SemanticFact(agent_id="agent-1", content="fact", embedding=[1.0, 0.0, 0.0]))

    deleted = backend.clear(agent_id="agent-1", tier=MemoryTier.EPISODIC)
    assert deleted == 2
    assert backend.count(agent_id="agent-1") == 1
    assert backend.count(agent_id="agent-1", tier=MemoryTier.SEMANTIC) == 1


def test_clear_all_for_agent(backend):
    backend.upsert(_episode(agent_id="alice"))
    backend.upsert(_episode(agent_id="alice"))
    backend.upsert(_episode(agent_id="bob"))

    deleted = backend.clear(agent_id="alice")
    assert deleted == 2
    assert backend.count(agent_id="alice") == 0
    assert backend.count(agent_id="bob") == 1


# ---------------------------------------------------------------------------
# SQLite-specific tests
#
# These do not use the parametrised ``backend`` fixture because they exercise
# behaviour that only the SQLite backend has (fixed embedding dimensionality,
# persistence across instances).
# ---------------------------------------------------------------------------


def test_sqlite_rejects_dimension_mismatch_on_upsert(tmp_path: Path):
    backend = SQLiteBackend(path=tmp_path / "dim.db", dimensions=_DIM)
    try:
        rec = EpisodicMemory(
            agent_id="a",
            content="wrong dim",
            embedding=[1.0, 0.0],  # 2 dims, not 3
        )
        with pytest.raises(ValueError, match="dims"):
            backend.upsert(rec)
    finally:
        backend.close()


def test_sqlite_rejects_dimension_mismatch_on_search(tmp_path: Path):
    backend = SQLiteBackend(path=tmp_path / "dim.db", dimensions=_DIM)
    try:
        with pytest.raises(ValueError, match="dims"):
            backend.search(query_embedding=[1.0, 0.0], agent_id="a", k=5)
    finally:
        backend.close()


def test_sqlite_rejects_non_positive_dimensions(tmp_path: Path):
    with pytest.raises(ValueError, match="positive"):
        SQLiteBackend(path=tmp_path / "z.db", dimensions=0)


def test_sqlite_persists_across_instances(tmp_path: Path):
    """A second backend opening the same file sees the first one's writes."""
    db = tmp_path / "persist.db"
    rec_id: str
    first = SQLiteBackend(path=db, dimensions=_DIM)
    try:
        rec = _episode(content="durable")
        first.upsert(rec)
        rec_id = rec.id
    finally:
        first.close()

    second = SQLiteBackend(path=db, dimensions=_DIM)
    try:
        fetched = second.get(rec_id)
        assert fetched is not None
        assert fetched.content == "durable"
        assert fetched.embedding == [1.0, 0.0, 0.0]
    finally:
        second.close()
