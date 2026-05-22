"""Mneme adapter — exposes :class:`MemoryManager` as a :class:`MemoryStrategy`.

Constructs a fresh manager per conversation (the runner calls ``reset()``
between conversations). All Mneme features — multi-tier memory, consolidation,
forgetting — are available, but the cheap retrieval-only eval path doesn't
trigger consolidation (which would need an LLM judge).
"""

from __future__ import annotations

from typing import Any

from mneme import (
    EmbeddingProvider,
    HashEmbedder,
    InMemoryBackend,
    MemoryManager,
    MnemeBackend,
    SQLiteBackend,
)
from mneme.judge import LLMJudge


class MnemeStrategy:
    """Wrap a :class:`MemoryManager` as a :class:`MemoryStrategy`.

    Each ``add(role, content)`` writes to both working and episodic
    (the same dual-write pattern the framework adapters use), so
    ``retrieve_records`` can surface past turns via semantic search.
    """

    name = "mneme"

    def __init__(
        self,
        *,
        backend: MnemeBackend,
        embedder: EmbeddingProvider,
        llm_judge: LLMJudge | None = None,
        agent_id: str = "eval",
        config: dict[str, Any] | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._llm_judge = llm_judge
        self._agent_id = agent_id
        self._config = config or {}
        self._manager: MemoryManager | None = None
        self._build_manager()

    def _build_manager(self) -> None:
        self._manager = MemoryManager(
            agent_id=self._agent_id,
            backend=self._backend,
            embedder=self._embedder,
            llm_judge=self._llm_judge,
        )

    @property
    def manager(self) -> MemoryManager:
        assert self._manager is not None
        return self._manager

    def add(self, role: str, content: str) -> None:
        self.manager.working.add(role=role, content=content)
        self.manager.episodic.add(content, metadata={"role": role, "source": "eval"})

    def context_for(self, query: str) -> list[dict[str, str]]:
        # Use the same shape as the OpenAI context_for helper but inline
        # to avoid pulling in tiktoken here. Eval runs don't pack to a
        # token budget; that's a separate concern.
        from mneme.adapters import context_for as _ctx

        return _ctx(self.manager, query, k=5, token_budget=None)

    def retrieve_records(self, query: str, *, k: int = 5) -> list[str]:
        return [r.record.content for r in self.manager.retrieve(query, k=k)]

    def reset(self) -> None:
        # Clear out the existing manager's storage and rebuild. We do
        # NOT use clear_all() because some backends (SQLite) would just
        # delete records under the eval-specific agent_id; building a
        # fresh manager + fresh backend is cleaner per-conversation.
        if self._manager is not None:
            self._manager.clear_all()
        # The backend itself isn't reset between conversations — the
        # agent_id namespace gives us isolation. If two strategies were
        # to share a backend we'd want a stricter reset.

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": type(self._backend).__name__,
            "embedder": type(self._embedder).__name__,
            "llm_judge": (type(self._llm_judge).__name__ if self._llm_judge else None),
            **self._config,
        }


def build_mneme_strategy(
    *,
    backend_name: str = "memory",
    embedder_name: str = "hash",
    dimensions: int = 16,
) -> MnemeStrategy:
    """Convenience factory for the CLI.

    Args:
        backend_name: ``"memory"`` (default, fast) or ``"sqlite"`` (file-backed).
        embedder_name: ``"hash"`` (deterministic, no API key) or
            ``"openai"`` (real embeddings — requires ``OPENAI_API_KEY``).
        dimensions: Embedding dimensionality. Default 16 matches the
            HashEmbedder default; pass 1536 with ``embedder_name="openai"``.
    """
    if embedder_name == "hash":
        embedder: EmbeddingProvider = HashEmbedder(dimensions=dimensions)
    elif embedder_name == "openai":
        from mneme import OpenAIEmbeddings

        embedder = OpenAIEmbeddings()
        dimensions = embedder.dimensions
    else:
        raise ValueError(f"unknown embedder: {embedder_name}")

    backend: MnemeBackend
    if backend_name == "memory":
        backend = InMemoryBackend()
    elif backend_name == "sqlite":
        backend = SQLiteBackend(path=":memory:", dimensions=dimensions)
    else:
        raise ValueError(f"unknown backend: {backend_name}")

    return MnemeStrategy(
        backend=backend,
        embedder=embedder,
        config={"dimensions": dimensions},
    )
