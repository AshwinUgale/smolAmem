"""Tests for the MemoryManager facade.

Manager-level tests cover *wiring* — that the manager constructs all three
tiers correctly, that they share the backend + embedder + agent_id, and
that ``clear_all`` reports per-tier counts. Tier-level behaviour is covered
in the per-tier test files.
"""

from __future__ import annotations

import pytest

from mneme import (
    EpisodicMemoryTier,
    HashEmbedder,
    InMemoryBackend,
    MemoryManager,
    MemoryTier,
    SemanticMemoryTier,
    WorkingMemoryTier,
)

_DIM = 16


@pytest.fixture
def manager() -> MemoryManager:
    return MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(dimensions=_DIM),
    )


def test_exposes_three_tiers(manager: MemoryManager):
    assert isinstance(manager.working, WorkingMemoryTier)
    assert isinstance(manager.episodic, EpisodicMemoryTier)
    assert isinstance(manager.semantic, SemanticMemoryTier)


def test_all_tiers_share_agent_id(manager: MemoryManager):
    assert manager.working.agent_id == "alice"
    assert manager.episodic.agent_id == "alice"
    assert manager.semantic.agent_id == "alice"


def test_persistent_tiers_share_the_backend_and_embedder(manager: MemoryManager):
    assert manager.episodic.backend is manager.backend
    assert manager.semantic.backend is manager.backend
    assert manager.episodic.embedder is manager.embedder
    assert manager.semantic.embedder is manager.embedder


def test_working_size_defaults_to_20():
    m = MemoryManager(agent_id="x", backend=InMemoryBackend(), embedder=HashEmbedder())
    assert m.working.max_size == 20


def test_working_size_override():
    m = MemoryManager(
        agent_id="x",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(),
        working_size=3,
    )
    assert m.working.max_size == 3


def test_cross_tier_independence(manager: MemoryManager):
    """Adding to one tier should not change another's count."""
    manager.working.add(role="user", content="just a turn")
    manager.episodic.add("an episode")
    manager.semantic.add("a fact")

    assert len(manager.working) == 1
    assert manager.episodic.count() == 1
    assert manager.semantic.count() == 1


def test_clear_all_reports_per_tier(manager: MemoryManager):
    manager.working.add(role="user", content="w1")
    manager.working.add(role="user", content="w2")
    manager.episodic.add("e1")
    manager.semantic.add("s1")
    manager.semantic.add("s2")
    manager.semantic.add("s3")

    deleted = manager.clear_all()
    assert deleted == {"working": 2, "episodic": 1, "semantic": 3}
    assert len(manager.working) == 0
    assert manager.episodic.count() == 0
    assert manager.semantic.count() == 0


def test_two_managers_share_one_backend_without_leaking():
    """Multi-tenant pattern: many managers, one backend, agent_id isolates."""
    backend = InMemoryBackend()
    embedder = HashEmbedder(dimensions=_DIM)
    alice = MemoryManager(agent_id="alice", backend=backend, embedder=embedder)
    bob = MemoryManager(agent_id="bob", backend=backend, embedder=embedder)

    alice.episodic.add("alice memory")
    bob.episodic.add("bob memory")

    assert alice.episodic.count() == 1
    assert bob.episodic.count() == 1
    alice_hits = alice.episodic.search("memory")
    assert [r.content for r, _ in alice_hits] == ["alice memory"]


def test_end_to_end_through_manager(manager: MemoryManager):
    """One realistic flow: turns, an episode, a fact, then search."""
    manager.working.add(role="user", content="hey there")
    manager.working.add(role="assistant", content="hi, what's up?")
    manager.episodic.add("user asked about React Suspense boundaries")
    manager.semantic.add(
        "user is building a Next.js app",
        confidence=0.9,
        provenance=[],
    )

    # Working memory: dump as-is, no embedding.
    assert [t.role for t in manager.working.turns()] == ["user", "assistant"]

    # Episodic: search by similarity.
    hits = manager.episodic.search("React Suspense boundaries", k=1)
    assert hits[0][0].content == "user asked about React Suspense boundaries"
    assert hits[0][0].tier is MemoryTier.EPISODIC

    # Semantic: search by similarity.
    facts = manager.semantic.search("user is building a Next.js app", k=1)
    assert facts[0][0].confidence == 0.9
    assert facts[0][0].tier is MemoryTier.SEMANTIC
