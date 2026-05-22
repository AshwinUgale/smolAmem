"""Framework adapters for Mneme.

Drop-in replacements for the memory primitives of popular agent frameworks.
Each adapter wraps a :class:`mneme.MemoryManager` so callers get Mneme's
multi-tier memory + consolidation + forgetting without changing how they
write the rest of their agent.

Three adapters ship at v0.5:

* :class:`MnemeChatMessageHistory` — LangChain's ``BaseChatMessageHistory``.
  Requires the ``[langchain]`` extra (``pip install 'mneme[langchain]'``).
* :class:`MnemeLlamaIndexMemory` — LlamaIndex's ``BaseMemory``. Requires
  the ``[llamaindex]`` extra.
* :func:`context_for` — raw-OpenAI helper that returns a list of chat
  messages ready for ``client.chat.completions.create``. Optional
  ``token_budget`` uses ``tiktoken`` (``[tokens]`` extra).

The submodules are lazy-loaded via :pep:`562`'s ``__getattr__`` — importing
``mneme.adapters`` doesn't pull in langchain-core or llama-index-core; only
the first reference to a specific adapter triggers its module import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static type-checkers see the real symbols; runtime defers via __getattr__.
    from mneme.adapters.langchain import MnemeChatMessageHistory
    from mneme.adapters.llamaindex import MnemeLlamaIndexMemory
    from mneme.adapters.openai_helper import context_for


__all__ = [
    "MnemeChatMessageHistory",
    "MnemeLlamaIndexMemory",
    "context_for",
]


def __getattr__(name: str) -> Any:
    if name == "MnemeChatMessageHistory":
        from mneme.adapters.langchain import (
            MnemeChatMessageHistory as _MCMH,
        )

        return _MCMH
    if name == "MnemeLlamaIndexMemory":
        from mneme.adapters.llamaindex import (
            MnemeLlamaIndexMemory as _MLIM,
        )

        return _MLIM
    if name == "context_for":
        from mneme.adapters.openai_helper import context_for as _ctx

        return _ctx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
