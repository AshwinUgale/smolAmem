"""``python -m evals`` — run a corpus through a strategy and dump JSON.

Glues the four pieces together::

    corpus (json files)
        v
    EvalRunner.run(strategy)        <- runners/base.py
        v
    RunResult                       <- per-test-point retrieved_records
        v
    recall_at_k + token_cost        <- metrics/*
        v
    final JSON written to disk

Examples::

    # cheap deterministic comparison, no API key needed
    python -m evals --runner mneme --output out/mneme.json
    python -m evals --runner no_memory --output out/no_memory.json
    python -m evals --runner summary_buffer --output out/summary.json
    python -m evals --runner full_history --output out/full.json

    # real embeddings (requires OPENAI_API_KEY)
    python -m evals --runner mneme --embedder openai --output out/mneme_oai.json

The ``--with-answers`` / ``--judge`` flags are reserved for a future
generation-pass that calls a real model on top of the retrieved context
and judges its answers. The cheap retrieval-only path is what this
script actually exercises today, and it's where recall@k and token-cost
already give a meaningful strategy-vs-strategy comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

# Optional .env loading so the CLI picks up OPENAI_API_KEY without the
# caller having to ``$env:OPENAI_API_KEY = ...`` in every shell. Encoding
# matches tests/conftest.py — Windows tooling tends to write UTF-8 with
# a BOM and vanilla load_dotenv() silently mishandles it.
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")

from evals.baselines import (
    FullHistoryStrategy,
    MemoryStrategy,
    NoMemoryStrategy,
    SummaryBufferStrategy,
)
from evals.metrics import recall_at_k, token_cost_summary
from evals.runners.base import EvalRunner, load_corpus
from evals.runners.mneme import build_mneme_strategy

_DEFAULT_CORPUS = Path(__file__).parent / "corpus"

_RUNNERS = ("mneme", "no_memory", "summary_buffer", "full_history")
_BACKENDS = ("memory", "sqlite")
_EMBEDDERS = ("hash", "openai")


def _build_strategy(args: argparse.Namespace) -> MemoryStrategy:
    """Construct the chosen strategy from CLI args."""
    if args.runner == "mneme":
        return build_mneme_strategy(
            backend_name=args.backend,
            embedder_name=args.embedder,
        )
    if args.runner == "no_memory":
        return NoMemoryStrategy()
    if args.runner == "summary_buffer":
        return SummaryBufferStrategy()
    if args.runner == "full_history":
        return FullHistoryStrategy()
    raise ValueError(f"unknown runner: {args.runner}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run Mneme's eval corpus through a memory strategy.",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=_DEFAULT_CORPUS,
        help=f"Directory of conversation *.json files (default: {_DEFAULT_CORPUS}).",
    )
    p.add_argument(
        "--runner",
        choices=_RUNNERS,
        default="mneme",
        help="Memory strategy to evaluate (default: mneme).",
    )
    p.add_argument(
        "--backend",
        choices=_BACKENDS,
        default="memory",
        help="Backend for the mneme runner (ignored otherwise; default: memory).",
    )
    p.add_argument(
        "--embedder",
        choices=_EMBEDDERS,
        default="hash",
        help=(
            "Embedder for the mneme runner (ignored otherwise; default: hash). "
            "'openai' requires OPENAI_API_KEY."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the JSON result file.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed recorded in the result file (no randomness in cheap path; default: 42).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k for retrieval-based recall measurement (default: 5).",
    )
    # Reserved for a future generation + LLM-judge pass. The cheap path
    # ignores both flags but we accept them so callers can pre-wire the
    # full command line.
    p.add_argument(
        "--with-answers",
        action="store_true",
        help="(reserved) generate an LLM answer per test point. Not used in cheap path.",
    )
    p.add_argument(
        "--judge",
        action="store_true",
        help="(reserved) run LLM-as-judge accuracy. Not used in cheap path.",
    )
    return p


def _run(args: argparse.Namespace) -> dict[str, Any]:
    strategy = _build_strategy(args)
    conversations = load_corpus(args.corpus)
    runner = EvalRunner(strategy=strategy, k=args.k, seed=args.seed)
    run_result = runner.run(conversations)

    # Compute metrics over the in-memory RunResult.
    metrics: dict[str, Any] = {}
    metrics.update(recall_at_k(run_result))
    metrics.update(token_cost_summary(run_result))

    # Final JSON shape: strategy config + per-conversation play-through
    # data + aggregate metrics. Everything is JSON-serialisable already.
    return {
        "strategy": run_result.strategy,
        "seed": run_result.seed,
        "k": args.k,
        "corpus_dir": str(args.corpus),
        "conversation_count": len(run_result.conversations),
        "metrics": metrics,
        "conversations": [
            {
                "conversation_id": c.conversation_id,
                "description": c.description,
                "turn_count": c.turn_count,
                "test_points": [
                    {
                        "turn_index": tp.turn_index,
                        "question": tp.question,
                        "expected_fact": tp.expected_fact,
                        "expected_keywords": tp.expected_keywords,
                        "retrieved_records": tp.retrieved_records,
                    }
                    for tp in c.test_points
                ],
            }
            for c in run_result.conversations
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.with_answers or args.judge:
        print(
            "note: --with-answers/--judge are reserved flags; the cheap "
            "retrieval-only path is what's actually run.",
            file=sys.stderr,
        )
    out = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # Tiny stdout summary so CI logs are useful at a glance.
    agg = out["metrics"].get("recall_at_k", {})
    tok = out["metrics"].get("tokens", {})
    print(
        f"strategy={out['strategy'].get('name')} "
        f"convs={out['conversation_count']} "
        f"recall@{args.k}={agg.get('aggregate')} "
        f"tokens/tp={tok.get('aggregate_tokens_per_test_point')} "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
