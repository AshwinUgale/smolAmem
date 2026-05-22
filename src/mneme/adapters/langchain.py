"""LangChain adapter — :class:`BaseChatMessageHistory` drop-in.

Wraps a :class:`mneme.MemoryManager` so any LangChain chain that takes a
``ChatMessageHistory`` gets Mneme's multi-tier memory underneath:

* ``add_message`` writes to both **working** memory (so subsequent
  ``.messages`` reads reflect the current chat) AND **episodic** memory
  (so ``manager.retrieve()`` and ``manager.consolidate()`` can see past
  conversations).
* ``messages`` returns the current working-memory turns as LangChain
  message objects (``HumanMessage`` / ``AIMessage`` / ``SystemMessage``).
* ``clear()`` wipes working memory only. Episodic stays — that's the
  point of having both tiers. To forget episodic too, call
  ``manager.episodic.clear()`` explicitly.

Async variants delegate to the sync implementations via ``asyncio.to_thread``
so LangChain LCEL chains that use ``ainvoke`` still work.

Importing this module requires the ``[langchain]`` extra
(``pip install 'mneme[langchain]'``). The parent package
``mneme.adapters`` defers this import via :pep:`562` ``__getattr__``,
so users who only want :func:`context_for` don't pay the langchain cost.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError as exc:  # pragma: no cover - exercised only when extra missing
    raise ImportError(
        "MnemeChatMessageHistory requires the 'langchain' extra. Install with:\n"
        "    pip install 'mneme[langchain]'\n"
        "    # or, in a uv-managed project:\n"
        "    uv add mneme --extra langchain"
    ) from exc

from mneme.manager import MemoryManager

__all__ = ["MnemeChatMessageHistory"]


class MnemeChatMessageHistory(BaseChatMessageHistory):
    """LangChain ``BaseChatMessageHistory`` backed by a Mneme ``MemoryManager``.

    Args:
        manager: A constructed :class:`mneme.MemoryManager`. The adapter
            owns nothing — pass the same manager into other adapters or
            use it directly alongside.
        also_persist_episodic: If ``True`` (default), every message also
            gets written to the episodic tier so it becomes searchable
            via ``manager.retrieve()`` and consolidatable. Disable if you
            want the LangChain history to be ephemeral (working-memory
            only, cleared on session end).
    """

    def __init__(
        self,
        manager: MemoryManager,
        *,
        also_persist_episodic: bool = True,
    ) -> None:
        self._manager = manager
        self._also_persist_episodic = also_persist_episodic

    # -- BaseChatMessageHistory contract -------------------------------------

    def add_message(self, message: BaseMessage) -> None:
        """Append a message to Mneme.

        Writes to working memory (for ``.messages`` reads) and to episodic
        memory (for retrieval/consolidation), unless ``also_persist_episodic``
        was set ``False`` at construction.
        """
        role = self._role_for(message)
        content = str(message.content)
        self._manager.working.add(role=role, content=content)
        if self._also_persist_episodic:
            self._manager.episodic.add(
                content,
                metadata={"role": role, "source": "langchain"},
            )

    # LangChain's BaseChatMessageHistory declares ``messages`` as a writeable
    # attribute (``messages: list[BaseMessage]``); ours is a derived view of
    # working memory and therefore read-only. The override is intentional —
    # mypy's ``[override]`` warning is a known false-positive here.
    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Return current working-memory turns as LangChain message objects."""
        return [
            self._message_for(turn.role, turn.content) for turn in self._manager.working.turns()
        ]

    def clear(self) -> None:
        """Wipe working memory. Episodic survives — call ``manager.episodic.clear()``
        explicitly if you want to forget past conversations too.
        """
        self._manager.working.clear()

    # -- Async variants (delegate to sync via asyncio.to_thread) -------------

    async def aget_messages(self) -> list[BaseMessage]:
        return await asyncio.to_thread(lambda: self.messages)

    async def aadd_message(self, message: BaseMessage) -> None:
        await asyncio.to_thread(self.add_message, message)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        for m in messages:
            await asyncio.to_thread(self.add_message, m)

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _role_for(message: BaseMessage) -> str:
        """Map a LangChain message type to a Mneme role string.

        LangChain has HumanMessage / AIMessage / SystemMessage / ToolMessage /
        FunctionMessage. We collapse to user/assistant/system; tool and
        function messages map to assistant by convention since they originate
        from the model's side of the conversation.
        """
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, SystemMessage):
            return "system"
        return "assistant"

    @staticmethod
    def _message_for(role: str, content: str) -> BaseMessage:
        """Inverse of :meth:`_role_for` — build a LangChain message for a turn."""
        if role == "user":
            return HumanMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        return AIMessage(content=content)
