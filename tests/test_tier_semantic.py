"""Tests for the semantic memory tier.

Semantic facts get the same surface as episodic (add/search/get/delete/etc)
plus ``confidence`` and ``provenance``, plus a stubbed ``consolidate()``.
"""

from __future__ import annotations

import pytest

from mneme import (
    EpisodicMemoryTier,
    HashEmbedder,
    InMemoryBackend,
    MemoryTier,
    SemanticMemoryTier,
)

_DIM = 16


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dimensions=_DIM)


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def tier(backend: InMemoryBackend, embedder: HashEmbedder) -> SemanticMemoryTier:
    return SemanticMemoryTier(agent_id="agent-1", backend=backend, embedder=embedder)


# ---------------------------------------------------------------------------
# add() — confidence + provenance + the basics
# ---------------------------------------------------------------------------


def test_add_default_confidence_and_empty_provenance(tier: SemanticMemoryTier):
    fact = tier.add("user prefers TypeScript")
    assert fact.confidence == 1.0
    assert fact.provenance == []
    assert fact.tier is MemoryTier.SEMANTIC


def test_add_with_explicit_confidence(tier: SemanticMemoryTier):
    fact = tier.add("user might prefer Rust", confidence=0.7)
    assert fact.confidence == 0.7


def test_add_with_provenance(tier: SemanticMemoryTier):
    fact = tier.add(
        "user works in TypeScript",
        provenance=["episode-1", "episode-7", "episode-12"],
    )
    assert fact.provenance == ["episode-1", "episode-7", "episode-12"]


def test_add_rejects_invalid_confidence(tier: SemanticMemoryTier):
    with pytest.raises(ValueError, match="confidence"):
        tier.add("nope", confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        tier.add("nope", confidence=-0.1)


def test_provenance_is_copied_not_aliased(tier: SemanticMemoryTier):
    """Caller mutating the input list after add() must not affect the fact."""
    ids = ["a", "b"]
    fact = tier.add("x", provenance=ids)
    ids.append("c")
    assert fact.provenance == ["a", "b"]


# ---------------------------------------------------------------------------
# search / get / delete / count / clear
# ---------------------------------------------------------------------------


def test_search_finds_added_fact(tier: SemanticMemoryTier):
    tier.add("user is on a Mac")
    [(fact, score)] = tier.search("user is on a Mac")
    assert fact.content == "user is on a Mac"
    assert score == pytest.approx(1.0)


def test_search_does_not_return_episodic_records(
    tier: SemanticMemoryTier,
    backend: InMemoryBackend,
    embedder: HashEmbedder,
):
    episodic = EpisodicMemoryTier(agent_id="agent-1", backend=backend, embedder=embedder)
    episodic.add("episode")
    tier.add("fact")
    hits = tier.search("anything", k=10)
    assert all(f.tier is MemoryTier.SEMANTIC for f, _ in hits)


def test_get_returns_semantic_fact(tier: SemanticMemoryTier):
    fact = tier.add("findable", confidence=0.5, provenance=["ep-1"])
    fetched = tier.get(fact.id)
    assert fetched is not None
    assert fetched.confidence == 0.5
    assert fetched.provenance == ["ep-1"]


def test_get_returns_none_for_wrong_tier(
    tier: SemanticMemoryTier,
    backend: InMemoryBackend,
    embedder: HashEmbedder,
):
    episodic = EpisodicMemoryTier(agent_id="agent-1", backend=backend, embedder=embedder)
    ep = episodic.add("episodic record")
    assert tier.get(ep.id) is None


def test_count_and_clear(tier: SemanticMemoryTier):
    for i in range(3):
        tier.add(f"fact-{i}")
    assert tier.count() == 3
    assert tier.clear() == 3
    assert tier.count() == 0


# consolidate() moved to MemoryManager in v0.2 — tested in tests/test_consolidate.py.
