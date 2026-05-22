"""Tests for the LangChain adapter.

Skips entirely if langchain-core isn't installed.

The adapter has a small surface — the tests cover:

* ``add_message`` writes to both working and episodic
* ``also_persist_episodic=False`` writes to working only
* ``messages`` reflects current working memory in order
* ``clear()`` wipes working but NOT episodic (the deliberate semantic split)
* role mapping for Human / AI / System messages
* async variants delegate to sync correctly
"""

from __future__ import annotations

import asyncio

import pytest

# Skip the whole file if langchain-core isn't installed.
pytest.importorskip("langchain_core")

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from mneme import HashEmbedder, InMemoryBackend, MemoryManager, MemoryTier
from mneme.adapters import MnemeChatMessageHistory


def _make_manager() -> MemoryManager:
    return MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(dimensions=8),
    )


# ---------------------------------------------------------------------------
# add_message — dual write semantics
# ---------------------------------------------------------------------------


def test_add_message_writes_to_both_tiers_by_default():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager)

    history.add_message(HumanMessage(content="hello"))

    assert len(manager.working) == 1
    assert manager.episodic.count() == 1


def test_also_persist_episodic_false_writes_working_only():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager, also_persist_episodic=False)

    history.add_message(HumanMessage(content="ephemeral"))

    assert len(manager.working) == 1
    assert manager.episodic.count() == 0


def test_role_mapping_human_ai_system():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager)

    history.add_message(HumanMessage(content="from user"))
    history.add_message(AIMessage(content="from assistant"))
    history.add_message(SystemMessage(content="system prompt"))

    turns = manager.working.turns()
    assert [t.role for t in turns] == ["user", "assistant", "system"]


def test_episodic_records_carry_role_metadata():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager)

    history.add_message(HumanMessage(content="tagged"))

    episodes = manager.backend.list_recent(agent_id="alice", tier=MemoryTier.EPISODIC)
    assert len(episodes) == 1
    assert episodes[0].metadata["role"] == "user"
    assert episodes[0].metadata["source"] == "langchain"


# ---------------------------------------------------------------------------
# messages — read from working memory
# ---------------------------------------------------------------------------


def test_messages_reflects_working_memory_in_order():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager)
    history.add_message(HumanMessage(content="first"))
    history.add_message(AIMessage(content="second"))
    history.add_message(HumanMessage(content="third"))

    out = history.messages
    assert len(out) == 3
    assert isinstance(out[0], HumanMessage)
    assert isinstance(out[1], AIMessage)
    assert isinstance(out[2], HumanMessage)
    assert [m.content for m in out] == ["first", "second", "third"]


def test_messages_empty_when_history_empty():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager)
    assert history.messages == []


# ---------------------------------------------------------------------------
# clear — wipes working, NOT episodic
# ---------------------------------------------------------------------------


def test_clear_wipes_working_but_keeps_episodic():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager)

    history.add_message(HumanMessage(content="A"))
    history.add_message(AIMessage(content="B"))

    history.clear()

    assert len(manager.working) == 0
    # Episodic still holds the past — the whole point of dual-write semantics.
    assert manager.episodic.count() == 2


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


def test_async_add_and_get_messages():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager)

    async def _scenario() -> None:
        await history.aadd_message(HumanMessage(content="async hello"))
        msgs = await history.aget_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "async hello"

    asyncio.run(_scenario())


def test_async_clear():
    manager = _make_manager()
    history = MnemeChatMessageHistory(manager)
    history.add_message(HumanMessage(content="x"))

    asyncio.run(history.aclear())

    assert len(manager.working) == 0
