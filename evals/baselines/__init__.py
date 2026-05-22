"""Baseline memory strategies for the eval harness.

Each baseline implements the :class:`MemoryStrategy` protocol — a tiny
interface that the runner uses to measure how a given approach to memory
performs on the corpus. Mneme itself is wrapped as a strategy by
``runners/mneme.py``; these baselines are what we compare against.
"""

from evals.baselines.base import MemoryStrategy
from evals.baselines.full_history import FullHistoryStrategy
from evals.baselines.no_memory import NoMemoryStrategy
from evals.baselines.summary_buffer import SummaryBufferStrategy

__all__ = [
    "FullHistoryStrategy",
    "MemoryStrategy",
    "NoMemoryStrategy",
    "SummaryBufferStrategy",
]
