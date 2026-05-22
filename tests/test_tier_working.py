"""Tests for the working memory tier.

Working memory is pure FIFO with no backend, no embedder, no search. The
tests reflect that — there's not much to verify beyond ordering and
eviction.
"""

from __future__ import annotations

import pytest

from mneme import MemoryTier, WorkingMemoryTier


def _tier(*, agent_id: str = "agent-1", max_size: int = 4) -> WorkingMemoryTier:
    return WorkingMemoryTier(agent_id=agent_id, max_size=max_size)


def test_starts_empty():
    t = _tier()
    assert len(t) == 0
    assert t.turns() == []


def test_add_returns_working_memory_with_role_and_content():
    t = _tier()
    turn = t.add(role="user", content="hello")
    assert turn.role == "user"
    assert turn.content == "hello"
    assert turn.tier is MemoryTier.WORKING
    assert turn.agent_id == "agent-1"


def test_turns_returned_in_insertion_order():
    t = _tier()
    t.add(role="user", content="one")
    t.add(role="assistant", content="two")
    t.add(role="user", content="three")
    assert [w.content for w in t.turns()] == ["one", "two", "three"]


def test_fifo_eviction_at_capacity():
    t = _tier(max_size=3)
    for i in range(5):
        t.add(role="user", content=f"turn-{i}")
    # Only the last 3 remain.
    assert [w.content for w in t.turns()] == ["turn-2", "turn-3", "turn-4"]
    assert len(t) == 3


def test_clear_returns_count_and_empties():
    t = _tier()
    t.add(role="user", content="a")
    t.add(role="user", content="b")
    assert t.clear() == 2
    assert len(t) == 0
    assert t.turns() == []


def test_metadata_passes_through():
    t = _tier()
    turn = t.add(role="user", content="hi", metadata={"convo": "thread-9"})
    assert turn.metadata == {"convo": "thread-9"}


def test_rejects_non_positive_max_size():
    with pytest.raises(ValueError, match="positive"):
        WorkingMemoryTier(agent_id="x", max_size=0)


def test_turns_returns_independent_copy():
    """Caller mutating the returned list must not affect the tier."""
    t = _tier()
    t.add(role="user", content="kept")
    snapshot = t.turns()
    snapshot.clear()
    assert len(t) == 1  # the tier still has it
