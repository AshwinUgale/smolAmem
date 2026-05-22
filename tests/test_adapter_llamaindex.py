"""Tests for the LlamaIndex adapter.

Skips entirely if llama-index-core isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.llms import (
    ChatMessage,
    MessageRole,
)

from mneme import HashEmbedder, InMemoryBackend, MemoryManager, MemoryTier
from mneme.adapters import MnemeLlamaIndexMemory


def _make_manager() -> MemoryManager:
    return MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(dimensions=8),
    )


# ---------------------------------------------------------------------------
# put — dual write semantics
# ---------------------------------------------------------------------------


def test_put_writes_to_both_tiers_by_default():
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory(manager)

    memory.put(ChatMessage(role=MessageRole.USER, content="hello"))

    assert len(manager.working) == 1
    assert manager.episodic.count() == 1


def test_also_persist_episodic_false_writes_working_only():
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory(manager, also_persist_episodic=False)

    memory.put(ChatMessage(role=MessageRole.USER, content="ephemeral"))

    assert len(manager.working) == 1
    assert manager.episodic.count() == 0


def test_role_mapping_user_assistant_system():
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory(manager)

    memory.put(ChatMessage(role=MessageRole.USER, content="u"))
    memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="a"))
    memory.put(ChatMessage(role=MessageRole.SYSTEM, content="s"))

    roles = [t.role for t in manager.working.turns()]
    assert roles == ["user", "assistant", "system"]


def test_episodic_metadata_records_source():
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory(manager)
    memory.put(ChatMessage(role=MessageRole.USER, content="tagged"))

    eps = manager.backend.list_recent(agent_id="alice", tier=MemoryTier.EPISODIC)
    assert eps[0].metadata["source"] == "llamaindex"


# ---------------------------------------------------------------------------
# get_all / get
# ---------------------------------------------------------------------------


def test_get_all_returns_chat_messages():
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory(manager)
    memory.put(ChatMessage(role=MessageRole.USER, content="hi"))
    memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="hello"))

    out = memory.get_all()
    assert len(out) == 2
    assert out[0].role == MessageRole.USER
    assert out[0].content == "hi"
    assert out[1].role == MessageRole.ASSISTANT


def test_get_returns_same_as_get_all_at_v0_5():
    """v0.5: get(input=...) ignores the input arg; both return recent chat."""
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory(manager)
    memory.put(ChatMessage(role=MessageRole.USER, content="anything"))

    all_ = memory.get_all()
    via_get = memory.get(input="totally different query")
    assert [m.content for m in via_get] == [m.content for m in all_]


# ---------------------------------------------------------------------------
# set / reset
# ---------------------------------------------------------------------------


def test_set_replaces_working_memory():
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory(manager)
    memory.put(ChatMessage(role=MessageRole.USER, content="old"))

    new_messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="fresh system"),
        ChatMessage(role=MessageRole.USER, content="fresh user"),
    ]
    memory.set(new_messages)

    assert [t.content for t in manager.working.turns()] == [
        "fresh system",
        "fresh user",
    ]


def test_reset_wipes_working_but_keeps_episodic():
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory(manager)
    memory.put(ChatMessage(role=MessageRole.USER, content="A"))

    memory.reset()

    assert len(manager.working) == 0
    assert manager.episodic.count() == 1


def test_from_defaults_factory():
    manager = _make_manager()
    memory = MnemeLlamaIndexMemory.from_defaults(manager=manager)
    assert isinstance(memory, MnemeLlamaIndexMemory)
