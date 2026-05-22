"""Tests for ``MemoryManager.forget`` and the v0.3 forgetting pass.

Two phases under test:

1. **TTL eviction** — records with ``expires_at <= now`` get deleted, full stop.
2. **Access-frequency decay** — records older than ``cold_age_days`` with
   ``access_count <= access_floor`` get deleted.

Plus the ``retrieve()`` integration tests: returned records get their access
tracking persisted via ``backend.touch``, and expired records are filtered
out defensively even if forget() hasn't run yet.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from mneme import (
    EpisodicMemory,
    HashEmbedder,
    InMemoryBackend,
    MemoryManager,
    MemoryTier,
    SemanticFact,
)

_DIM = 8
_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _make_manager() -> MemoryManager:
    return MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(dimensions=_DIM),
    )


def _ep(
    *,
    content: str = "ep",
    created_at: datetime = _NOW,
    expires_at: datetime | None = None,
    access_count: int = 0,
    embedding: list[float] | None = None,
) -> EpisodicMemory:
    return EpisodicMemory(
        agent_id="alice",
        content=content,
        embedding=embedding or [1.0] + [0.0] * (_DIM - 1),
        created_at=created_at,
        expires_at=expires_at,
        access_count=access_count,
    )


def _fact(
    *,
    content: str = "fact",
    created_at: datetime = _NOW,
    expires_at: datetime | None = None,
    access_count: int = 0,
) -> SemanticFact:
    return SemanticFact(
        agent_id="alice",
        content=content,
        embedding=[1.0] + [0.0] * (_DIM - 1),
        created_at=created_at,
        expires_at=expires_at,
        access_count=access_count,
    )


# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------


def test_forget_on_empty_backend_returns_zeros():
    m = _make_manager()
    assert m.forget(now=_NOW) == {"expired": 0, "cold": 0}


def test_forget_with_only_fresh_records_deletes_nothing():
    m = _make_manager()
    m.backend.upsert(_ep(content="fresh"))
    m.backend.upsert(_fact(content="fresh"))
    counts = m.forget(now=_NOW)
    assert counts == {"expired": 0, "cold": 0}
    assert m.episodic.count() == 1
    assert m.semantic.count() == 1


# ---------------------------------------------------------------------------
# Phase 1 — TTL eviction
# ---------------------------------------------------------------------------


def test_forget_deletes_records_past_expires_at():
    m = _make_manager()
    expired = _ep(content="expired", expires_at=_NOW - timedelta(hours=1))
    live = _ep(content="live", expires_at=_NOW + timedelta(days=7))
    m.backend.upsert(expired)
    m.backend.upsert(live)

    counts = m.forget(now=_NOW)
    assert counts["expired"] == 1
    assert m.backend.get(expired.id) is None
    assert m.backend.get(live.id) is not None


def test_forget_ttl_only_skips_access_decay():
    m = _make_manager()
    expired = _ep(content="expired", expires_at=_NOW - timedelta(hours=1))
    very_old_and_cold = _ep(
        content="ancient", created_at=_NOW - timedelta(days=365), access_count=0
    )
    m.backend.upsert(expired)
    m.backend.upsert(very_old_and_cold)

    counts = m.forget(now=_NOW, ttl_only=True)
    assert counts == {"expired": 1, "cold": 0}
    # TTL claimed the expired one; access-decay did NOT touch the ancient one.
    assert m.backend.get(expired.id) is None
    assert m.backend.get(very_old_and_cold.id) is not None


def test_forget_treats_expires_at_equal_to_now_as_expired():
    """``<= now`` boundary: a record expiring exactly at ``now`` is expired."""
    m = _make_manager()
    on_the_dot = _ep(content="now", expires_at=_NOW)
    m.backend.upsert(on_the_dot)
    counts = m.forget(now=_NOW)
    assert counts["expired"] == 1


def test_forget_ttl_applies_to_both_persisted_tiers():
    m = _make_manager()
    expired_ep = _ep(content="ep", expires_at=_NOW - timedelta(hours=1))
    expired_fact = _fact(content="fact", expires_at=_NOW - timedelta(hours=1))
    m.backend.upsert(expired_ep)
    m.backend.upsert(expired_fact)
    counts = m.forget(now=_NOW)
    assert counts["expired"] == 2


# ---------------------------------------------------------------------------
# Phase 2 — Access-frequency decay
# ---------------------------------------------------------------------------


def test_forget_deletes_cold_records():
    m = _make_manager()
    old_cold = _ep(
        content="never accessed in 60 days",
        created_at=_NOW - timedelta(days=60),
        access_count=0,
    )
    m.backend.upsert(old_cold)

    counts = m.forget(now=_NOW, cold_age_days=30)
    assert counts["cold"] == 1
    assert m.backend.get(old_cold.id) is None


def test_forget_keeps_old_record_with_accesses():
    """access_count above the floor saves a record from cold eviction."""
    m = _make_manager()
    old_but_used = _ep(
        content="ancient but loved",
        created_at=_NOW - timedelta(days=60),
        access_count=5,
    )
    m.backend.upsert(old_but_used)
    counts = m.forget(now=_NOW, cold_age_days=30, access_floor=0)
    assert counts["cold"] == 0
    assert m.backend.get(old_but_used.id) is not None


def test_forget_access_floor_controls_threshold():
    """With access_floor=10, anything with <= 10 accesses qualifies."""
    m = _make_manager()
    low = _ep(content="low", created_at=_NOW - timedelta(days=60), access_count=5)
    high = _ep(content="high", created_at=_NOW - timedelta(days=60), access_count=15)
    m.backend.upsert(low)
    m.backend.upsert(high)

    counts = m.forget(now=_NOW, cold_age_days=30, access_floor=10)
    assert counts["cold"] == 1
    assert m.backend.get(low.id) is None
    assert m.backend.get(high.id) is not None


def test_forget_keeps_recent_records_regardless_of_access_count():
    """A young record never qualifies for cold eviction even at 0 accesses."""
    m = _make_manager()
    young = _ep(
        content="young",
        created_at=_NOW - timedelta(days=5),
        access_count=0,
    )
    m.backend.upsert(young)
    counts = m.forget(now=_NOW, cold_age_days=30)
    assert counts["cold"] == 0
    assert m.backend.get(young.id) is not None


def test_forget_combined_phases_count_separately():
    m = _make_manager()
    m.backend.upsert(_ep(content="expired", expires_at=_NOW - timedelta(hours=1)))
    m.backend.upsert(_ep(content="another expired", expires_at=_NOW - timedelta(days=1)))
    m.backend.upsert(_ep(content="cold", created_at=_NOW - timedelta(days=60), access_count=0))
    m.backend.upsert(_ep(content="fresh"))

    counts = m.forget(now=_NOW, cold_age_days=30)
    assert counts == {"expired": 2, "cold": 1}
    assert m.episodic.count() == 1  # only "fresh" remains


# ---------------------------------------------------------------------------
# retrieve() integration: touch persists, expired records filtered
# ---------------------------------------------------------------------------


def test_retrieve_persists_touch_via_backend():
    """Top-k returned records get their access_count bumped in the backend."""
    m = _make_manager()
    rec = _ep(content="touchable")
    m.backend.upsert(rec)

    m.retrieve("query", k=5, now=_NOW)

    fetched = m.backend.get(rec.id)
    assert fetched is not None
    assert fetched.access_count == 1
    assert fetched.last_accessed == _NOW


def test_retrieve_does_not_touch_records_outside_top_k():
    """Over-fetched but not-returned records shouldn't get touched."""
    m = _make_manager()
    # Use very different embeddings so similarity ordering is stable.
    high_sim = _ep(content="match", embedding=[1.0] + [0.0] * (_DIM - 1))
    low_sim = _ep(content="miss", embedding=[0.0, 1.0] + [0.0] * (_DIM - 2))
    m.backend.upsert(high_sim)
    m.backend.upsert(low_sim)

    # Manually-built FixedEmbedder-style query: aligned with high_sim.
    m.retrieve("query", k=1, now=_NOW)

    high_after = m.backend.get(high_sim.id)
    low_after = m.backend.get(low_sim.id)
    assert high_after is not None
    assert low_after is not None
    # high_sim was the top-1, so it got touched. low_sim was over-fetched but
    # never returned, so it stays untouched.
    assert high_after.access_count == 1
    assert low_after.access_count == 0


def test_retrieve_filters_out_expired_records():
    """Even if forget() hasn't run, retrieve must not surface expired records."""
    m = _make_manager()
    expired = _ep(content="expired", expires_at=_NOW - timedelta(hours=1))
    live = _ep(content="live", expires_at=_NOW + timedelta(days=7))
    m.backend.upsert(expired)
    m.backend.upsert(live)

    results = m.retrieve("query", k=5, now=_NOW)
    contents = [r.record.content for r in results]
    assert "expired" not in contents
    assert "live" in contents


# ---------------------------------------------------------------------------
# Mixed: end-to-end sanity
# ---------------------------------------------------------------------------


def test_forget_end_to_end_through_manager():
    """One realistic scenario: TTL on episodes, semantic facts long-lived,
    a couple of stale never-retrieved episodes."""
    m = _make_manager()
    # 1. An old expired episode
    m.backend.upsert(_ep(content="A", expires_at=_NOW - timedelta(hours=2)))
    # 2. A 60-day-old episode no one ever retrieved
    m.backend.upsert(_ep(content="B", created_at=_NOW - timedelta(days=60)))
    # 3. A fresh episode, just added
    m.backend.upsert(_ep(content="C"))
    # 4. A semantic fact that's old but well-loved
    m.backend.upsert(
        _fact(
            content="D",
            created_at=_NOW - timedelta(days=200),
            access_count=42,
        )
    )

    counts = m.forget(now=_NOW, cold_age_days=30)
    assert counts == {"expired": 1, "cold": 1}
    survivors = {
        *(r.content for r in m.backend.list_recent(agent_id="alice", tier=MemoryTier.EPISODIC)),
        *(r.content for r in m.backend.list_recent(agent_id="alice", tier=MemoryTier.SEMANTIC)),
    }
    assert survivors == {"C", "D"}


def test_forget_max_per_tier_caps_scanning():
    """If a tier has more records than max_per_tier, we only scan that many."""
    m = _make_manager()
    # 5 expired episodes, but only scan 3.
    for i in range(5):
        m.backend.upsert(
            _ep(
                content=f"exp-{i}",
                created_at=_NOW - timedelta(minutes=i),
                expires_at=_NOW - timedelta(hours=1),
            )
        )
    counts = m.forget(now=_NOW, max_per_tier=3)
    # Up to 3 expired records deleted, not all 5.
    assert counts["expired"] <= 3


def test_forget_returns_zeros_when_called_with_no_argument(monkeypatch: Any):
    """``now=None`` defaults to datetime.now(UTC) — exercise that branch."""
    m = _make_manager()
    counts = m.forget()  # no args at all
    assert counts == {"expired": 0, "cold": 0}
