"""Tests for the episodic memory tier.

Uses HashEmbedder + InMemoryBackend so the suite is fully deterministic and
hits no network. Real-embedder behaviour is covered by the embedding tests;
the tier tests are about *wiring* — does the tier embed on add, persist
through the backend, retrieve back via search, and isolate tier filters
correctly.
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
def tier(backend: InMemoryBackend, embedder: HashEmbedder) -> EpisodicMemoryTier:
    return EpisodicMemoryTier(agent_id="agent-1", backend=backend, embedder=embedder)


def test_add_embeds_and_persists(tier: EpisodicMemoryTier, backend: InMemoryBackend):
    rec = tier.add("user asked about Suspense")
    assert rec.tier is MemoryTier.EPISODIC
    assert rec.embedding is not None and len(rec.embedding) == _DIM
    # The backend should now hold exactly one record for this agent.
    assert backend.count(agent_id="agent-1", tier=MemoryTier.EPISODIC) == 1


def test_search_finds_added_record(tier: EpisodicMemoryTier):
    tier.add("user prefers TypeScript")
    results = tier.search("user prefers TypeScript", k=5)
    assert len(results) == 1
    record, score = results[0]
    assert record.content == "user prefers TypeScript"
    # HashEmbedder is deterministic and unit-norm, so identical inputs
    # produce identical vectors — cosine similarity is 1.0.
    assert score == pytest.approx(1.0)


def test_search_returns_episodic_typed_records(tier: EpisodicMemoryTier):
    """Type narrows from MemoryRecord to EpisodicMemory at the tier API edge."""
    tier.add("ep")
    [(rec, _)] = tier.search("ep")
    # Concrete subclass, not just the base.
    assert rec.tier is MemoryTier.EPISODIC


def test_search_does_not_return_semantic_facts(
    tier: EpisodicMemoryTier,
    backend: InMemoryBackend,
    embedder: HashEmbedder,
):
    """Tier filter scopes search to episodic only, even with both tiers stored."""
    semantic = SemanticMemoryTier(agent_id="agent-1", backend=backend, embedder=embedder)
    semantic.add("fact about user")
    tier.add("user said something")

    episodic_only = tier.search("user", k=10)
    assert all(r.tier is MemoryTier.EPISODIC for r, _ in episodic_only)


def test_search_isolates_by_agent(backend: InMemoryBackend, embedder: HashEmbedder):
    alice = EpisodicMemoryTier(agent_id="alice", backend=backend, embedder=embedder)
    bob = EpisodicMemoryTier(agent_id="bob", backend=backend, embedder=embedder)
    alice.add("alice memory")
    bob.add("bob memory")

    alice_hits = alice.search("memory", k=5)
    bob_hits = bob.search("memory", k=5)
    assert [r.content for r, _ in alice_hits] == ["alice memory"]
    assert [r.content for r, _ in bob_hits] == ["bob memory"]


def test_get_returns_record_by_id(tier: EpisodicMemoryTier):
    rec = tier.add("findable")
    fetched = tier.get(rec.id)
    assert fetched is not None
    assert fetched.id == rec.id
    assert fetched.content == "findable"


def test_get_returns_none_for_wrong_tier(
    tier: EpisodicMemoryTier,
    backend: InMemoryBackend,
    embedder: HashEmbedder,
):
    """Asking the episodic tier for a semantic id returns None, not the wrong type."""
    semantic = SemanticMemoryTier(agent_id="agent-1", backend=backend, embedder=embedder)
    fact = semantic.add("a fact")
    assert tier.get(fact.id) is None


def test_delete_removes_record(tier: EpisodicMemoryTier):
    rec = tier.add("doomed")
    assert tier.delete(rec.id) is True
    assert tier.get(rec.id) is None
    assert tier.delete(rec.id) is False  # idempotent


def test_delete_refuses_wrong_tier(
    tier: EpisodicMemoryTier,
    backend: InMemoryBackend,
    embedder: HashEmbedder,
):
    """Episodic.delete() will not delete a semantic fact even by exact id."""
    semantic = SemanticMemoryTier(agent_id="agent-1", backend=backend, embedder=embedder)
    fact = semantic.add("not yours to delete")
    assert tier.delete(fact.id) is False
    assert backend.get(fact.id) is not None  # still there


def test_count_and_clear(tier: EpisodicMemoryTier):
    for i in range(4):
        tier.add(f"ep-{i}")
    assert tier.count() == 4
    deleted = tier.clear()
    assert deleted == 4
    assert tier.count() == 0


def test_metadata_round_trips(tier: EpisodicMemoryTier):
    rec = tier.add("with meta", metadata={"source": "chat", "lang": "en"})
    fetched = tier.get(rec.id)
    assert fetched is not None
    assert fetched.metadata == {"source": "chat", "lang": "en"}


def test_metadata_filter_in_search(tier: EpisodicMemoryTier):
    tier.add("a", metadata={"source": "chat"})
    tier.add("b", metadata={"source": "docs"})
    chat_only = tier.search("a", k=10, metadata_filter={"source": "chat"})
    assert [r.content for r, _ in chat_only] == ["a"]
