"""Tests for ``MemoryManager.retrieve``.

The interesting tests here are about *ranking*, not similarity. To remove
similarity noise from the picture we use a ``FixedEmbedder`` that returns
the same vector for every input — every record has identical similarity to
every query, so the rest of the score (authority * recency * confidence)
is what determines order.

Records are constructed directly and ``backend.upsert``ed so we control
``created_at`` precisely. The tier classes don't expose a ``created_at``
override (and shouldn't — that's a backdoor most callers don't need), but
the manager doesn't care how records got into the backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from mneme import (
    EpisodicMemory,
    HashEmbedder,
    InMemoryBackend,
    MemoryManager,
    MemoryTier,
    RetrievalResult,
    SemanticFact,
)

_DIM = 4

# Module-level constant so the FixedEmbedder below can be a simple class
# (no ClassVar gymnastics, no risk of accidentally mutating a class-level
# mutable). Anything in the test file that needs the canonical fixed vector
# imports it from here.
_FIXED_VEC: list[float] = [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FixedEmbedder:
    """Embedder that returns the same vector for every input.

    Makes ranking tests purely about authority + recency + confidence by
    holding similarity constant at 1.0 for every (record, query) pair.
    """

    dimensions: int = _DIM

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(_FIXED_VEC) for _ in texts]


_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _make_manager() -> MemoryManager:
    return MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=FixedEmbedder(),
    )


def _insert_episode(
    manager: MemoryManager,
    *,
    content: str,
    created_at: datetime,
    embedding: list[float] | None = None,
) -> EpisodicMemory:
    rec = EpisodicMemory(
        agent_id=manager.agent_id,
        content=content,
        embedding=embedding if embedding is not None else list(_FIXED_VEC),
        created_at=created_at,
    )
    manager.backend.upsert(rec)
    return rec


def _insert_fact(
    manager: MemoryManager,
    *,
    content: str,
    created_at: datetime,
    confidence: float = 1.0,
    embedding: list[float] | None = None,
) -> SemanticFact:
    fact = SemanticFact(
        agent_id=manager.agent_id,
        content=content,
        embedding=embedding if embedding is not None else list(_FIXED_VEC),
        created_at=created_at,
        confidence=confidence,
    )
    manager.backend.upsert(fact)
    return fact


# ---------------------------------------------------------------------------
# Defaults and shape
# ---------------------------------------------------------------------------


def test_k_zero_returns_empty():
    m = _make_manager()
    _insert_episode(m, content="x", created_at=_NOW)
    assert m.retrieve("query", k=0, now=_NOW) == []


def test_empty_backend_returns_empty():
    m = _make_manager()
    assert m.retrieve("anything", k=5, now=_NOW) == []


def test_returns_retrieval_result_with_all_components():
    m = _make_manager()
    _insert_episode(m, content="ep", created_at=_NOW)

    [result] = m.retrieve("query", k=5, now=_NOW)
    assert isinstance(result, RetrievalResult)
    assert result.similarity is not None
    assert result.recency is not None
    assert result.authority is not None
    assert result.score == pytest.approx(result.similarity * result.authority * result.recency)


def test_returns_at_most_k_results():
    m = _make_manager()
    for i in range(20):
        _insert_episode(m, content=f"ep-{i}", created_at=_NOW)
    results = m.retrieve("query", k=3, now=_NOW)
    assert len(results) == 3


def test_results_ordered_by_score_desc():
    m = _make_manager()
    _insert_episode(m, content="ep", created_at=_NOW - timedelta(days=14))
    _insert_episode(m, content="ep-new", created_at=_NOW)
    results = m.retrieve("query", k=2, now=_NOW)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tier filtering
# ---------------------------------------------------------------------------


def test_default_tiers_excludes_working():
    """``manager.working.add`` records would never be searched anyway, but the
    default tier list should be exactly [EPISODIC, SEMANTIC]."""
    m = _make_manager()
    _insert_episode(m, content="ep", created_at=_NOW)
    _insert_fact(m, content="fact", created_at=_NOW)

    results = m.retrieve("query", k=5, now=_NOW)
    found_tiers = {r.record.tier for r in results}
    assert found_tiers == {MemoryTier.EPISODIC, MemoryTier.SEMANTIC}


def test_passing_working_in_tiers_raises():
    m = _make_manager()
    with pytest.raises(ValueError, match="Working memory"):
        m.retrieve("query", k=5, tiers=[MemoryTier.WORKING], now=_NOW)
    with pytest.raises(ValueError, match="Working memory"):
        m.retrieve(
            "query",
            k=5,
            tiers=[MemoryTier.WORKING, MemoryTier.EPISODIC],
            now=_NOW,
        )


def test_can_restrict_to_one_tier():
    m = _make_manager()
    _insert_episode(m, content="ep", created_at=_NOW)
    _insert_fact(m, content="fact", created_at=_NOW)

    episodic_only = m.retrieve("query", k=5, tiers=[MemoryTier.EPISODIC], now=_NOW)
    assert all(r.record.tier is MemoryTier.EPISODIC for r in episodic_only)

    semantic_only = m.retrieve("query", k=5, tiers=[MemoryTier.SEMANTIC], now=_NOW)
    assert all(r.record.tier is MemoryTier.SEMANTIC for r in semantic_only)


def test_empty_tiers_list_returns_empty():
    m = _make_manager()
    _insert_episode(m, content="ep", created_at=_NOW)
    assert m.retrieve("query", k=5, tiers=[], now=_NOW) == []


# ---------------------------------------------------------------------------
# Authority weighting
# ---------------------------------------------------------------------------


def test_semantic_outranks_episodic_at_equal_similarity_and_age():
    m = _make_manager()
    _insert_episode(m, content="an episode", created_at=_NOW)
    _insert_fact(m, content="a fact", created_at=_NOW)

    results = m.retrieve("query", k=2, now=_NOW)
    assert results[0].record.tier is MemoryTier.SEMANTIC
    assert results[1].record.tier is MemoryTier.EPISODIC


def test_custom_authority_weights_change_order():
    """Boost EPISODIC above SEMANTIC; the order flips."""
    m = _make_manager()
    _insert_episode(m, content="ep", created_at=_NOW)
    _insert_fact(m, content="fact", created_at=_NOW)

    results = m.retrieve(
        "query",
        k=2,
        authority_weights={
            MemoryTier.EPISODIC: 1.0,
            MemoryTier.SEMANTIC: 0.5,
        },
        now=_NOW,
    )
    assert results[0].record.tier is MemoryTier.EPISODIC
    assert results[1].record.tier is MemoryTier.SEMANTIC


def test_missing_tier_in_weights_defaults_to_zero():
    """A tier absent from authority_weights gets weight 0 — effectively skipped."""
    m = _make_manager()
    _insert_episode(m, content="ep", created_at=_NOW)
    _insert_fact(m, content="fact", created_at=_NOW)

    results = m.retrieve(
        "query",
        k=5,
        authority_weights={MemoryTier.SEMANTIC: 1.0},  # no EPISODIC entry
        now=_NOW,
    )
    # Episodic ends up at score 0 and ranks last; semantic is on top.
    assert results[0].record.tier is MemoryTier.SEMANTIC


# ---------------------------------------------------------------------------
# Freshness decay
# ---------------------------------------------------------------------------


def test_newer_outranks_older_at_equal_similarity_and_tier():
    m = _make_manager()
    _insert_episode(m, content="old", created_at=_NOW - timedelta(days=30))
    _insert_episode(m, content="new", created_at=_NOW)

    [first, second] = m.retrieve("query", k=2, now=_NOW)
    assert first.record.content == "new"
    assert second.record.content == "old"


def test_recency_decays_to_half_at_one_half_life():
    """A 7-day-old record (default half-life) should score 0.5 on recency."""
    m = _make_manager()
    _insert_episode(m, content="seven-days-old", created_at=_NOW - timedelta(days=7))

    [result] = m.retrieve("query", k=1, now=_NOW)
    assert result.recency == pytest.approx(0.5, rel=1e-6)


def test_custom_half_life_makes_old_records_more_competitive():
    """With a 60-day half-life, a 30-day-old record still scores ~0.71 on recency."""
    m = _make_manager()
    _insert_episode(m, content="thirty-days-old", created_at=_NOW - timedelta(days=30))

    [result] = m.retrieve("query", k=1, half_life_days=60.0, now=_NOW)
    assert result.recency == pytest.approx(0.5 ** (30 / 60), rel=1e-6)


def test_future_created_at_returns_recency_of_one():
    """A future timestamp (clock skew, backdating) should not score > 1 on recency."""
    m = _make_manager()
    _insert_episode(m, content="from-the-future", created_at=_NOW + timedelta(days=5))

    [result] = m.retrieve("query", k=1, now=_NOW)
    assert result.recency == 1.0


# ---------------------------------------------------------------------------
# Confidence weighting (semantic facts only)
# ---------------------------------------------------------------------------


def test_higher_confidence_outranks_lower_at_equal_similarity_and_age():
    m = _make_manager()
    _insert_fact(m, content="weak", created_at=_NOW, confidence=0.3)
    _insert_fact(m, content="strong", created_at=_NOW, confidence=0.95)

    [first, second] = m.retrieve("query", k=2, tiers=[MemoryTier.SEMANTIC], now=_NOW)
    assert first.record.content == "strong"
    assert second.record.content == "weak"


def test_use_confidence_false_ignores_confidence():
    """With use_confidence=False, the weak fact and the strong fact tie on score."""
    m = _make_manager()
    _insert_fact(m, content="weak", created_at=_NOW, confidence=0.3)
    _insert_fact(m, content="strong", created_at=_NOW, confidence=0.95)

    results = m.retrieve(
        "query",
        k=2,
        tiers=[MemoryTier.SEMANTIC],
        use_confidence=False,
        now=_NOW,
    )
    # Both have the same score now — they tie. Order between them is
    # determined by the backend's tiebreaker (created_at desc), which is
    # equal here, so either ordering is acceptable; what matters is the
    # scores are equal.
    assert results[0].score == pytest.approx(results[1].score)


def test_confidence_does_not_affect_episodic_records():
    """Episodic records have no confidence; their score should equal sim * auth * rec."""
    m = _make_manager()
    _insert_episode(m, content="ep", created_at=_NOW)

    [result] = m.retrieve("query", k=1, tiers=[MemoryTier.EPISODIC], now=_NOW)
    expected = result.similarity * result.authority * result.recency
    assert result.score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# End-to-end with the real HashEmbedder (not FixedEmbedder) — sanity
# ---------------------------------------------------------------------------


def test_real_embedder_round_trip():
    """Smoke test against HashEmbedder + InMemoryBackend through the public API."""
    m = MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(dimensions=16),
    )
    m.episodic.add("user asked about React Suspense boundaries")
    m.semantic.add("user is building a Next.js app", confidence=0.9)

    results = m.retrieve("React Suspense", k=2)
    # HashEmbedder has no semantic structure, so similarity is whatever the
    # hash spits out — but both records should appear in the results.
    assert len(results) == 2
    tiers = {r.record.tier for r in results}
    assert tiers == {MemoryTier.EPISODIC, MemoryTier.SEMANTIC}
