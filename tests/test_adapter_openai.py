"""Tests for the ``context_for`` raw-OpenAI helper.

These tests don't need an OpenAI account or key — ``context_for`` runs
fully against ``HashEmbedder`` + ``InMemoryBackend``. The only optional
dep is ``tiktoken``, which only matters when ``token_budget`` is passed;
that branch is gated via ``pytest.importorskip`` per-test.
"""

from __future__ import annotations

import pytest

from mneme import (
    EpisodicMemory,
    HashEmbedder,
    InMemoryBackend,
    MemoryManager,
    SemanticFact,
)
from mneme.adapters import context_for


def _make_manager() -> MemoryManager:
    return MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(dimensions=8),
    )


# ---------------------------------------------------------------------------
# Shape and ordering
# ---------------------------------------------------------------------------


def test_empty_manager_returns_empty_list():
    out = context_for(_make_manager(), "anything", k=5)
    assert out == []


def test_returns_list_of_role_content_dicts():
    m = _make_manager()
    m.episodic.add("user asked about React")
    out = context_for(m, "React", k=3)
    assert isinstance(out, list)
    assert all(set(msg.keys()) == {"role", "content"} for msg in out)


def test_working_memory_turns_appear_in_chronological_order():
    m = _make_manager()
    m.working.add(role="user", content="first")
    m.working.add(role="assistant", content="second")
    m.working.add(role="user", content="third")

    out = context_for(m, "anything", k=0)
    assert [msg["role"] for msg in out] == ["user", "assistant", "user"]
    assert [msg["content"] for msg in out] == ["first", "second", "third"]


def test_include_working_false_drops_working_memory():
    m = _make_manager()
    m.working.add(role="user", content="should be excluded")
    m.episodic.add("episodic record")

    out = context_for(m, "record", k=5, include_working=False)
    # No user message from working memory; just the system-citation block.
    roles = {msg["role"] for msg in out}
    assert "user" not in roles
    assert "system" in roles


def test_retrieved_block_appears_before_working_memory():
    m = _make_manager()
    m.episodic.add("retrieved fact")
    m.working.add(role="user", content="recent turn")

    out = context_for(m, "fact", k=3)
    # The system-citation block must come first so the model sees grounding
    # context before the conversation turns it has to respond to.
    assert out[0]["role"] == "system"
    assert "retrieved fact" in out[0]["content"]


# ---------------------------------------------------------------------------
# k bounds + tier labels
# ---------------------------------------------------------------------------


def test_k_zero_skips_retrieval():
    m = _make_manager()
    m.episodic.add("ignore me")
    m.working.add(role="user", content="kept")

    out = context_for(m, "anything", k=0)
    assert all(msg["role"] != "system" for msg in out)
    assert any(msg["content"] == "kept" for msg in out)


def test_negative_k_raises():
    m = _make_manager()
    with pytest.raises(ValueError, match="k must be non-negative"):
        context_for(m, "query", k=-1)


def test_citation_labels_fact_vs_episode():
    m = _make_manager()
    # Inject both an episode and a semantic fact directly so retrieve()
    # sees both tiers.
    embedding = [1.0] + [0.0] * 7
    m.backend.upsert(EpisodicMemory(agent_id="alice", content="ep one", embedding=embedding))
    m.backend.upsert(
        SemanticFact(
            agent_id="alice",
            content="fact one",
            embedding=embedding,
            confidence=0.9,
        )
    )

    out = context_for(m, "anything", k=5)
    system_msg = next(msg for msg in out if msg["role"] == "system")
    assert "[FACT" in system_msg["content"]
    assert "[EPISODE" in system_msg["content"]


# ---------------------------------------------------------------------------
# Token-budget packing (tiktoken-gated)
# ---------------------------------------------------------------------------


def test_token_budget_requires_tiktoken_if_set():
    """Passing token_budget without tiktoken installed raises an ImportError
    with a clear message. Skip when tiktoken IS installed (the negative
    case isn't checkable then)."""
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        m = _make_manager()
        m.episodic.add("anything")
        with pytest.raises(ImportError, match="tokens"):
            context_for(m, "query", k=1, token_budget=100)
    else:
        pytest.skip("tiktoken is installed; can't exercise the missing-dep path")


def test_token_budget_zero_returns_empty():
    pytest.importorskip("tiktoken")
    m = _make_manager()
    m.working.add(role="user", content="this would normally be returned")
    out = context_for(m, "anything", k=0, token_budget=0)
    assert out == []


def test_token_budget_drops_retrieved_first_keeps_working():
    """When budget is tight, working-memory turns survive over retrieved
    citations — the model needs the recent conversation to respond."""
    pytest.importorskip("tiktoken")
    m = _make_manager()
    # Long retrieved content so it dwarfs a small working turn
    m.episodic.add("a very long retrieved memory " * 50)
    m.working.add(role="user", content="short")

    # Budget tight enough to fit the short working turn but not the long
    # retrieved citation block.
    out = context_for(m, "anything", k=3, token_budget=20)
    contents = [msg["content"] for msg in out]
    assert "short" in contents
    assert all("very long retrieved" not in c for c in contents)


def test_token_budget_keeps_both_when_room_allows():
    pytest.importorskip("tiktoken")
    m = _make_manager()
    m.episodic.add("retrieved")
    m.working.add(role="user", content="recent")

    out = context_for(m, "anything", k=3, token_budget=5000)
    roles = [msg["role"] for msg in out]
    assert "system" in roles
    assert "user" in roles
