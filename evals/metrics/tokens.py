"""Token-cost summary over a run.

For the cheap retrieval-only path we don't actually call an LLM, so the
token cost we report is the *context size* the strategy WOULD prepend to
a real chat call. Counted via ``tiktoken`` if installed; falls back to
"~4 chars per token" approximation otherwise.

The interesting comparison is between strategies on the same corpus:
``full_history`` ballooning as the conversation grows vs Mneme staying
roughly flat thanks to retrieval.
"""

from __future__ import annotations

from typing import Any

from evals.runners.base import RunResult


def _approx_token_count(text: str) -> int:
    """Approximate token count, falling back when tiktoken isn't installed."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # mypy treats tiktoken as Any via our pyproject override.
        return len(enc.encode(text))  # type: ignore[attr-defined]
    except ImportError:
        # ~4 chars per token is the canonical rough approximation. Good
        # enough when tokenisation isn't available.
        return max(1, len(text) // 4)


def token_cost_summary(result: RunResult) -> dict[str, Any]:
    """For each conversation, sum the token cost of the *retrieved* context
    across every test point. Use that as the proxy for "how many tokens
    this strategy would consume at inference time."

    This is a strategy-vs-strategy comparison metric. Absolute values are
    meaningful for back-of-envelope cost estimates ($/1M tokens x this
    number = a real dollar figure), but the cross-strategy ratio is what
    drives decisions.
    """
    per_convo: dict[str, dict[str, Any]] = {}
    grand_total = 0
    grand_points = 0
    for convo in result.conversations:
        convo_total = 0
        for point in convo.test_points:
            for record in point.retrieved_records:
                convo_total += _approx_token_count(record)
        per_convo[convo.conversation_id] = {
            "context_tokens_sum": convo_total,
            "test_points": len(convo.test_points),
            "context_tokens_per_test_point": (
                convo_total / len(convo.test_points) if convo.test_points else 0
            ),
        }
        grand_total += convo_total
        grand_points += len(convo.test_points)
    return {
        "tokens": {
            "aggregate_context_tokens": grand_total,
            "aggregate_tokens_per_test_point": (grand_total / grand_points if grand_points else 0),
            "per_conversation": per_convo,
        }
    }
