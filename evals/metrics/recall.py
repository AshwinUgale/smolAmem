"""Recall@k — did the labelled fact rank in the top-k retrieved records?

A test point is a *hit* if any of its ``expected_keywords`` appears
(case-insensitively) inside any of the strategy's top-k retrieved
records. We use keyword overlap rather than exact string match because
the strategy's record might be the verbatim turn (``"I mostly work in
TypeScript..."``) while the expected_fact is a paraphrased canonical
(``"user prefers TypeScript"``) — the keywords bridge the gap.

A run's recall@k is the fraction of test points that were hits.
"""

from __future__ import annotations

from typing import Any

from evals.runners.base import RunResult, TestPointResult


def _point_hit(point: TestPointResult) -> bool:
    """True iff any expected_keyword appears in any retrieved record."""
    if not point.expected_keywords:
        # No keywords specified — fall back to expected_fact substring.
        # Case-insensitive whole-string-in-content.
        target = point.expected_fact.lower()
        return any(target in r.lower() for r in point.retrieved_records)
    haystack = "\n".join(point.retrieved_records).lower()
    return any(kw.lower() in haystack for kw in point.expected_keywords)


def recall_at_k(result: RunResult) -> dict[str, Any]:
    """Compute aggregate + per-conversation recall@k from a run."""
    all_points: list[TestPointResult] = []
    per_convo_hits: dict[str, dict[str, Any]] = {}

    for convo in result.conversations:
        hits = 0
        for point in convo.test_points:
            all_points.append(point)
            if _point_hit(point):
                hits += 1
        per_convo_hits[convo.conversation_id] = {
            "hits": hits,
            "total": len(convo.test_points),
            "recall": (hits / len(convo.test_points)) if convo.test_points else None,
        }

    total = len(all_points)
    total_hits = sum(1 for p in all_points if _point_hit(p))
    aggregate = (total_hits / total) if total else None

    return {
        "recall_at_k": {
            "aggregate": aggregate,
            "hits": total_hits,
            "total": total,
            "per_conversation": per_convo_hits,
        }
    }
