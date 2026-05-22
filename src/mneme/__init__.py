"""Mneme — multi-tier long-term memory for LLM agents.

See https://github.com/ashwinugale/ashwinugale-mneme for usage.
"""

from mneme.backends import (
    InMemoryBackend,
    MnemeBackend,
    PgVectorBackend,
    QdrantBackend,
    SQLiteBackend,
)
from mneme.embeddings import EmbeddingProvider, HashEmbedder, OpenAIEmbeddings
from mneme.judge import LLMJudge, MockLLMJudge, OpenAILLMJudge
from mneme.manager import MemoryManager
from mneme.tiers import EpisodicMemoryTier, SemanticMemoryTier, WorkingMemoryTier
from mneme.types import (
    EpisodicMemory,
    MemoryRecord,
    MemoryTier,
    RetrievalResult,
    SemanticFact,
    WorkingMemory,
)

# Adapters are NOT re-exported at the package root because they each
# require an optional extra. Users import them explicitly:
#   from mneme.adapters import MnemeChatMessageHistory  # needs [langchain]
#   from mneme.adapters import MnemeLlamaIndexMemory    # needs [llamaindex]
#   from mneme.adapters import context_for              # needs [tokens] for budget

__version__ = "0.0.1"

__all__ = [
    "EmbeddingProvider",
    "EpisodicMemory",
    "EpisodicMemoryTier",
    "HashEmbedder",
    "InMemoryBackend",
    "LLMJudge",
    "MemoryManager",
    "MemoryRecord",
    "MemoryTier",
    "MnemeBackend",
    "MockLLMJudge",
    "OpenAIEmbeddings",
    "OpenAILLMJudge",
    "PgVectorBackend",
    "QdrantBackend",
    "RetrievalResult",
    "SQLiteBackend",
    "SemanticFact",
    "SemanticMemoryTier",
    "WorkingMemory",
    "WorkingMemoryTier",
    "__version__",
]
