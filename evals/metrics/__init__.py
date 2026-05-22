"""Metric computation over a :class:`RunResult`.

Each metric is a pure function taking a ``RunResult`` and returning a
JSON-serialisable dict. The CLI calls every metric in turn and merges
the results into the final output file.
"""

from evals.metrics.recall import recall_at_k
from evals.metrics.tokens import token_cost_summary

__all__ = ["recall_at_k", "token_cost_summary"]
