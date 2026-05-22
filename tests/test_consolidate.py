"""Tests for ``MemoryManager.consolidate``.

The consolidation algorithm has three independent moving parts:

1. **Plumbing** — episodes get fetched, batched, handed to the judge.
2. **Extraction** — the judge returns fact dicts; they get embedded and stored.
3. **Dedup** — high-similarity new facts merge into existing ones with
   provenance preservation.

We exercise each in isolation with ``MockLLMJudge`` plus ``HashEmbedder`` +
``InMemoryBackend`` so nothing is non-deterministic and no network is touched.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mneme import (
    EpisodicMemory,
    HashEmbedder,
    InMemoryBackend,
    MemoryManager,
    MemoryTier,
    MockLLMJudge,
    SemanticFact,
)

_DIM = 16
_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(*, judge: MockLLMJudge | None = None) -> MemoryManager:
    """Construct a manager with a HashEmbedder + InMemoryBackend."""
    return MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(dimensions=_DIM),
        llm_judge=judge,
    )


def _empty_handler(**_: Any) -> dict[str, Any]:
    """Handler that returns no facts — useful when we only care about plumbing."""
    return {"facts": []}


def _fixed_handler(facts: list[dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Handler that always returns the same fact list."""
    return lambda **_: {"facts": facts}


def _insert_episode_at(
    manager: MemoryManager,
    *,
    content: str,
    created_at: datetime,
    episode_id: str | None = None,
) -> EpisodicMemory:
    """Insert an episode at a specific timestamp via the backend.

    The tier's ``add()`` would stamp ``created_at`` for us, so we bypass
    when we need to control the timeline.
    """
    [embedding] = manager.embedder.embed([content])
    kwargs: dict[str, Any] = {
        "agent_id": manager.agent_id,
        "content": content,
        "embedding": embedding,
        "created_at": created_at,
    }
    if episode_id is not None:
        kwargs["id"] = episode_id
    rec = EpisodicMemory(**kwargs)
    manager.backend.upsert(rec)
    return rec


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_consolidate_without_judge_raises():
    m = _make_manager(judge=None)
    with pytest.raises(RuntimeError, match="llm_judge"):
        m.consolidate()


def test_consolidate_with_no_episodes_returns_empty():
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    assert m.consolidate() == []
    # No call to the LLM judge was made — nothing to consolidate.
    assert judge.calls == []


def test_consolidate_rejects_non_positive_batch_size():
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    with pytest.raises(ValueError, match="batch_size"):
        m.consolidate(batch_size=0)


def test_consolidate_max_episodes_zero_returns_empty():
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="anything", created_at=_NOW)
    assert m.consolidate(max_episodes=0) == []
    assert judge.calls == []


# ---------------------------------------------------------------------------
# Plumbing — episodes flow to the judge correctly
# ---------------------------------------------------------------------------


def test_consolidate_passes_episode_ids_and_content_in_prompt():
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    e1 = _insert_episode_at(m, content="user uses Mac", created_at=_NOW)
    e2 = _insert_episode_at(m, content="user uses VS Code", created_at=_NOW + timedelta(minutes=1))

    m.consolidate()

    assert len(judge.calls) == 1
    user_message = judge.calls[0]["messages"][1]["content"]
    assert e1.id in user_message
    assert e2.id in user_message
    assert "user uses Mac" in user_message
    assert "user uses VS Code" in user_message


def test_consolidate_sends_system_prompt_and_schema():
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="x", created_at=_NOW)

    m.consolidate()

    call = judge.calls[0]
    # System message present
    assert call["messages"][0]["role"] == "system"
    assert "memory consolidator" in call["messages"][0]["content"]
    # Schema requires facts list with the expected fields
    schema = call["response_schema"]
    assert schema["required"] == ["facts"]
    fact_props = schema["properties"]["facts"]["items"]["properties"]
    assert set(fact_props.keys()) == {"content", "confidence", "source_episode_ids"}
    # Deterministic — temperature 0
    assert call["temperature"] == 0.0


def test_consolidate_batches_episodes():
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    for i in range(25):
        _insert_episode_at(m, content=f"ep-{i}", created_at=_NOW + timedelta(minutes=i))

    m.consolidate(batch_size=10)

    # 25 episodes / batch 10 → 3 LLM calls.
    assert len(judge.calls) == 3


def test_consolidate_processes_episodes_oldest_first():
    """Within a batch, episodes should appear in the prompt oldest → newest."""
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="THIRD", created_at=_NOW + timedelta(minutes=2))
    _insert_episode_at(m, content="FIRST", created_at=_NOW)
    _insert_episode_at(m, content="SECOND", created_at=_NOW + timedelta(minutes=1))

    m.consolidate()
    user_message = judge.calls[0]["messages"][1]["content"]
    first_pos = user_message.index("FIRST")
    second_pos = user_message.index("SECOND")
    third_pos = user_message.index("THIRD")
    assert first_pos < second_pos < third_pos


def test_consolidate_since_filters_old_episodes():
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="OLD", created_at=_NOW - timedelta(days=30))
    _insert_episode_at(m, content="NEW", created_at=_NOW)

    m.consolidate(since=_NOW - timedelta(days=1))
    user_message = judge.calls[0]["messages"][1]["content"]
    assert "NEW" in user_message
    assert "OLD" not in user_message


def test_consolidate_max_episodes_limits_input():
    judge = MockLLMJudge(handler=_empty_handler)
    m = _make_manager(judge=judge)
    for i in range(50):
        _insert_episode_at(m, content=f"ep-{i}", created_at=_NOW + timedelta(minutes=i))

    m.consolidate(batch_size=10, max_episodes=20)
    # Hard cap at 20 → 2 batches.
    assert len(judge.calls) == 2


# ---------------------------------------------------------------------------
# Extraction — facts produced by the judge get stored
# ---------------------------------------------------------------------------


def test_consolidate_writes_extracted_facts_to_semantic_tier():
    facts = [
        {
            "content": "user uses TypeScript",
            "confidence": 0.9,
            "source_episode_ids": ["ep_1"],
        },
        {
            "content": "user is on a Mac",
            "confidence": 0.7,
            "source_episode_ids": ["ep_2"],
        },
    ]
    judge = MockLLMJudge(handler=_fixed_handler(facts))
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="trigger", created_at=_NOW)

    produced = m.consolidate()
    assert len(produced) == 2
    assert m.semantic.count() == 2

    contents = {f.content for f in produced}
    assert contents == {"user uses TypeScript", "user is on a Mac"}

    ts_fact = next(f for f in produced if f.content == "user uses TypeScript")
    assert ts_fact.confidence == 0.9
    assert ts_fact.provenance == ["ep_1"]
    assert ts_fact.tier is MemoryTier.SEMANTIC


def test_consolidate_handles_empty_facts_response():
    judge = MockLLMJudge(handler=_fixed_handler([]))
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="trigger", created_at=_NOW)

    produced = m.consolidate()
    assert produced == []
    assert m.semantic.count() == 0


def test_consolidate_raises_when_judge_returns_malformed_facts_field():
    """Defensive: a buggy judge shouldn't silently corrupt the semantic tier."""

    def bad_handler(**_: Any) -> dict[str, Any]:
        return {"facts": "not a list"}  # wrong shape

    judge = MockLLMJudge(handler=bad_handler)
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="x", created_at=_NOW)

    with pytest.raises(RuntimeError, match="non-list"):
        m.consolidate()


# ---------------------------------------------------------------------------
# Dedup — high-similarity facts merge into existing ones
# ---------------------------------------------------------------------------


def test_consolidate_dedups_identical_fact_into_existing():
    """Same content twice → one stored fact, provenance union, max confidence."""
    facts = [
        {
            "content": "user uses TypeScript",
            "confidence": 0.6,
            "source_episode_ids": ["ep_a"],
        }
    ]
    judge = MockLLMJudge(handler=_fixed_handler(facts))
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="first trigger", created_at=_NOW)

    first_run = m.consolidate()
    assert m.semantic.count() == 1
    first_id = first_run[0].id

    # Second consolidation run with a higher-confidence version + new provenance.
    facts[0]["confidence"] = 0.9
    facts[0]["source_episode_ids"] = ["ep_b"]
    _insert_episode_at(m, content="second trigger", created_at=_NOW + timedelta(minutes=5))

    second_run = m.consolidate()
    # Still one fact; the same record was merged into, not duplicated.
    assert m.semantic.count() == 1
    assert second_run[0].id == first_id

    fetched = m.semantic.get(first_id)
    assert fetched is not None
    assert fetched.confidence == 0.9  # max(0.6, 0.9)
    assert fetched.provenance == ["ep_a", "ep_b"]  # union, order preserved


def test_consolidate_skips_dedup_for_dissimilar_facts():
    """HashEmbedder has no semantic structure, so very different strings stay distinct."""
    facts_run_1 = [
        {
            "content": "user uses TypeScript",
            "confidence": 0.9,
            "source_episode_ids": [],
        }
    ]
    judge = MockLLMJudge(handler=_fixed_handler(facts_run_1))
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="trigger", created_at=_NOW)
    m.consolidate()

    # Replace the canned response and run again with an unrelated string.
    judge._handler = _fixed_handler(  # type: ignore[attr-defined]
        [
            {
                "content": "totally different domain unrelated content",
                "confidence": 0.7,
                "source_episode_ids": [],
            }
        ]
    )
    _insert_episode_at(m, content="trigger-2", created_at=_NOW + timedelta(minutes=1))
    m.consolidate()

    # HashEmbedder vectors of two unrelated strings are extremely unlikely to
    # exceed the 0.85 cosine-similarity threshold, so we get a second fact.
    assert m.semantic.count() == 2


def test_consolidate_returns_the_facts_it_touched():
    """The return value should include both newly-stored and merged-into facts."""
    facts = [
        {
            "content": "user uses TypeScript",
            "confidence": 0.8,
            "source_episode_ids": ["ep_x"],
        }
    ]
    judge = MockLLMJudge(handler=_fixed_handler(facts))
    m = _make_manager(judge=judge)
    _insert_episode_at(m, content="trigger", created_at=_NOW)
    produced = m.consolidate()
    assert len(produced) == 1
    assert isinstance(produced[0], SemanticFact)


def test_consolidate_pre_existing_unrelated_fact_is_untouched():
    """An existing fact unrelated to the extracted ones should not be modified."""
    pre_existing = SemanticFact(
        agent_id="alice",
        content="totally separate established fact xyz",
        embedding=[1.0, 0.0] + [0.0] * (_DIM - 2),
        confidence=0.5,
        provenance=["ep_existing"],
    )
    judge = MockLLMJudge(
        handler=_fixed_handler(
            [
                {
                    "content": "different new fact about user",
                    "confidence": 0.9,
                    "source_episode_ids": [],
                }
            ]
        )
    )
    m = _make_manager(judge=judge)
    m.backend.upsert(pre_existing)
    _insert_episode_at(m, content="trigger", created_at=_NOW)

    m.consolidate()

    after = m.semantic.get(pre_existing.id)
    assert after is not None
    assert after.content == "totally separate established fact xyz"
    assert after.confidence == 0.5
    assert after.provenance == ["ep_existing"]
