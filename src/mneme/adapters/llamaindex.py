"""LlamaIndex adapter — :class:`BaseMemory` drop-in.

Wraps a :class:`mneme.MemoryManager` for LlamaIndex agents:

* ``put`` writes the message to both working and episodic memory.
* ``get_all`` returns working-memory turns as ``ChatMessage`` objects.
* ``get(input=...)`` returns the same in v0.5 — the ``input`` argument is
  accepted for protocol conformance but ignored. A future version may use
  it as a query into ``manager.retrieve`` so the returned context is
  search-aware; for v0.5 we keep the semantics "give me recent chat."
* ``set`` replaces working memory with the given message list.
* ``reset`` wipes working memory only. Episodic survives.

Importing this module requires the ``[llamaindex]`` extra
(``pip install 'mneme[llamaindex]'``). The parent package
``mneme.adapters`` defers this import via :pep:`562` ``__getattr__``.
"""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.llms import ChatMessage, MessageRole
    from llama_index.core.memory import BaseMemory
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - exercised only when extra missing
    raise ImportError(
        "MnemeLlamaIndexMemory requires the 'llamaindex' extra. Install with:\n"
        "    pip install 'mneme[llamaindex]'\n"
        "    # or, in a uv-managed project:\n"
        "    uv add mneme --extra llamaindex"
    ) from exc

from mneme.manager import MemoryManager
from mneme.types import WorkingMemory

__all__ = ["MnemeLlamaIndexMemory"]


# Pydantic-ish BaseMemory subclasses sometimes need ``arbitrary_types_allowed``
# to accept non-pydantic fields like our MemoryManager. We side-step that by
# storing the manager in a private slot and overriding __init__.


class MnemeLlamaIndexMemory(BaseMemory):
    """LlamaIndex ``BaseMemory`` backed by a Mneme ``MemoryManager``.

    Args:
        manager: A constructed :class:`mneme.MemoryManager`. The adapter
            owns nothing.
        also_persist_episodic: If ``True`` (default), every message also
            writes to episodic memory so it's searchable via
            ``manager.retrieve()`` and visible to consolidation.
    """

    # pydantic v2 model config: allow our MemoryManager (a non-pydantic
    # type) through. ``arbitrary_types_allowed`` is the field-level escape
    # hatch; we don't actually persist the manager as a pydantic field —
    # we store it in instance state below via object.__setattr__.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        manager: MemoryManager,
        *,
        also_persist_episodic: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Use ``object.__setattr__`` so pydantic doesn't try to treat
        # these as fields. They are pure instance state, not part of the
        # serialised model.
        object.__setattr__(self, "_manager", manager)
        object.__setattr__(self, "_also_persist_episodic", also_persist_episodic)

    # -- BaseMemory contract -------------------------------------------------

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> MnemeLlamaIndexMemory:
        """LlamaIndex's conventional factory. Requires ``manager`` kwarg."""
        return cls(**kwargs)

    def put(self, message: ChatMessage) -> None:
        """Append a chat message to Mneme."""
        manager: MemoryManager = self._manager  # type: ignore[attr-defined]
        role = self._role_for(message.role)
        content = self._content_for(message)
        manager.working.add(role=role, content=content)
        if self._also_persist_episodic:  # type: ignore[attr-defined]
            manager.episodic.add(
                content,
                metadata={"role": role, "source": "llamaindex"},
            )

    def get(self, input: str | None = None, **kwargs: Any) -> list[ChatMessage]:
        """Return recent chat history.

        ``input`` is accepted for protocol conformance but ignored at v0.5;
        :func:`get` returns the same as :func:`get_all`. A future version
        may use it as a retrieval query.
        """
        return self.get_all()

    def get_all(self) -> list[ChatMessage]:
        manager: MemoryManager = self._manager  # type: ignore[attr-defined]
        return [self._chatmessage_for(turn) for turn in manager.working.turns()]

    def set(self, messages: list[ChatMessage]) -> None:
        """Replace working memory with ``messages``."""
        manager: MemoryManager = self._manager  # type: ignore[attr-defined]
        manager.working.clear()
        for msg in messages:
            self.put(msg)

    def reset(self) -> None:
        """Wipe working memory. Episodic survives."""
        manager: MemoryManager = self._manager  # type: ignore[attr-defined]
        manager.working.clear()

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _role_for(role: MessageRole) -> str:
        """Map a LlamaIndex ``MessageRole`` to a Mneme role string."""
        if role == MessageRole.USER:
            return "user"
        if role == MessageRole.SYSTEM:
            return "system"
        # Assistant, function, tool, model — all collapse to assistant.
        return "assistant"

    @staticmethod
    def _content_for(message: ChatMessage) -> str:
        """Extract plain string content from a ChatMessage.

        LlamaIndex's ``ChatMessage.content`` is typed as ``str | None``
        in older versions and may be a ``MessageContent`` union in newer
        ones supporting multi-modal. We coerce to ``str`` either way.
        """
        if message.content is None:
            return ""
        return str(message.content)

    @staticmethod
    def _chatmessage_for(turn: WorkingMemory) -> ChatMessage:
        if turn.role == "user":
            return ChatMessage(role=MessageRole.USER, content=turn.content)
        if turn.role == "system":
            return ChatMessage(role=MessageRole.SYSTEM, content=turn.content)
        return ChatMessage(role=MessageRole.ASSISTANT, content=turn.content)
