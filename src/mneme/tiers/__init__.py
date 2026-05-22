"""Memory tier classes.

Each tier owns its dependencies (agent_id, optionally backend + embedder)
and exposes the operations that make sense for that tier. They are concrete
classes, not protocols — unlike :class:`MnemeBackend` and
:class:`EmbeddingProvider`, there is only ever one implementation per tier.

The :class:`MemoryManager` in ``mneme.manager`` constructs all three and
exposes them as attributes (``manager.working``, ``manager.episodic``,
``manager.semantic``).
"""

from mneme.tiers.episodic import EpisodicMemoryTier
from mneme.tiers.semantic import SemanticMemoryTier
from mneme.tiers.working import WorkingMemoryTier

__all__ = ["EpisodicMemoryTier", "SemanticMemoryTier", "WorkingMemoryTier"]
