"""Tests for the core data types."""

from datetime import UTC, datetime

import mneme
from mneme.types import (
    EpisodicMemory,
    MemoryRecord,
    MemoryTier,
    RetrievalResult,
    SemanticFact,
    WorkingMemory,
)


def test_public_exports():
    for name in (
        "MemoryTier",
        "MemoryRecord",
        "WorkingMemory",
        "EpisodicMemory",
        "SemanticFact",
        "RetrievalResult",
    ):
        assert hasattr(mneme, name)


def test_tier_serialises_to_plain_string():
    assert MemoryTier.EPISODIC == "episodic"
    assert MemoryTier.SEMANTIC.value == "semantic"


def test_record_generates_id_and_timestamp():
    rec = MemoryRecord(agent_id="a", content="hello", tier=MemoryTier.EPISODIC)
    assert rec.id  # non-empty generated hex id
    assert rec.created_at.tzinfo is not None  # timezone-aware
    assert rec.embedding is None
    assert rec.access_count == 0
    assert rec.last_accessed is None
    assert rec.metadata == {}


def test_ids_are_unique():
    a = MemoryRecord(agent_id="x", content="1", tier=MemoryTier.EPISODIC)
    b = MemoryRecord(agent_id="x", content="2", tier=MemoryTier.EPISODIC)
    assert a.id != b.id


def test_subclasses_default_their_tier():
    assert WorkingMemory(agent_id="a", content="hi", role="user").tier is MemoryTier.WORKING
    assert EpisodicMemory(agent_id="a", content="hi").tier is MemoryTier.EPISODIC
    assert SemanticFact(agent_id="a", content="hi").tier is MemoryTier.SEMANTIC


def test_working_memory_requires_role():
    turn = WorkingMemory(agent_id="a", content="hi", role="assistant")
    assert turn.role == "assistant"


def test_semantic_fact_defaults():
    fact = SemanticFact(agent_id="a", content="user prefers TypeScript")
    assert fact.confidence == 1.0
    assert fact.provenance == []


def test_touch_tracks_access():
    rec = EpisodicMemory(agent_id="a", content="hi")
    fixed = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    rec.touch(now=fixed)
    assert rec.access_count == 1
    assert rec.last_accessed == fixed
    rec.touch(now=fixed)
    assert rec.access_count == 2


def test_retrieval_result_wraps_record():
    rec = SemanticFact(agent_id="a", content="fact")
    result = RetrievalResult(record=rec, score=0.92, similarity=0.8, authority=1.0)
    assert result.record is rec
    assert result.score == 0.92
    assert result.recency is None
