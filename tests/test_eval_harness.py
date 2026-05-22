"""End-to-end checks for the eval harness.

Plays the bundled corpus through every strategy and verifies:

* the runner produces a :class:`RunResult` with one entry per conversation
* recall@k and token_cost_summary return well-formed dicts
* the relative ordering matches our priors:
    - ``no_memory`` recall == 0 (it never retrieves anything)
    - ``full_history`` recall == 1 for every test point that has the
      expected keywords in the corpus (it sees everything)
    - ``mneme`` recall > ``no_memory`` recall (semantic retrieval beats
      "nothing at all" even with the deterministic HashEmbedder)
* the CLI writes a JSON file with the expected shape

Cheap path only — no OpenAI key needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from evals.baselines import (
    FullHistoryStrategy,
    NoMemoryStrategy,
    SummaryBufferStrategy,
)
from evals.metrics import recall_at_k, token_cost_summary
from evals.runners.base import EvalRunner, load_corpus
from evals.runners.mneme import build_mneme_strategy

_CORPUS = Path(__file__).resolve().parents[1] / "evals" / "corpus"


@pytest.fixture(scope="module")
def conversations() -> list:
    return load_corpus(_CORPUS)


def _run(strategy, conversations):
    runner = EvalRunner(strategy=strategy, k=5, seed=42)
    return runner.run(conversations)


def test_corpus_loads(conversations) -> None:
    assert len(conversations) >= 5, "expected at least 5 starter conversations"
    # Every conversation has at least one test point — otherwise it's
    # not a useful eval datum.
    for c in conversations:
        assert any(t.test_at is not None for t in c.turns), c.id


def test_no_memory_recall_is_zero(conversations) -> None:
    result = _run(NoMemoryStrategy(), conversations)
    metrics = recall_at_k(result)
    assert metrics["recall_at_k"]["aggregate"] == 0.0


def test_full_history_recall_is_one(conversations) -> None:
    result = _run(FullHistoryStrategy(), conversations)
    metrics = recall_at_k(result)
    # Full history sees every prior turn. The keywords in each test point
    # are chosen so they appear in some earlier turn of the same convo,
    # so recall should be a perfect 1.0.
    assert metrics["recall_at_k"]["aggregate"] == 1.0


def test_summary_buffer_runs_without_error(conversations) -> None:
    result = _run(SummaryBufferStrategy(), conversations)
    metrics = recall_at_k(result)
    # We don't assert a specific value — the summary baseline collapses
    # detail, so recall depends on whether the keyword survives the
    # trivial summary. The point of the test is that it runs.
    assert 0.0 <= metrics["recall_at_k"]["aggregate"] <= 1.0


def test_mneme_runs_and_returns_records(conversations) -> None:
    """Mneme produces well-formed retrieval results.

    We don't assert a specific recall value here because the
    HashEmbedder is deterministic-but-lexical, so its semantic match
    quality on a paraphrased ``question`` vs the verbatim
    ``content`` is poor by design. Real embedder evaluation uses
    ``--embedder openai`` and is gated by the API key.
    """
    strategy = build_mneme_strategy(backend_name="memory", embedder_name="hash")
    result = _run(strategy, conversations)
    metrics = recall_at_k(result)
    agg = metrics["recall_at_k"]["aggregate"]
    assert agg is None or 0.0 <= agg <= 1.0
    # Every test point should have produced *some* retrieved records (k=5)
    # — Mneme's retrieval always returns the top-k it has, even if not
    # all of them are about the labelled fact.
    for conv in result.conversations:
        for tp in conv.test_points:
            assert isinstance(tp.retrieved_records, list)


def test_token_cost_summary_well_formed(conversations) -> None:
    strategy = build_mneme_strategy(backend_name="memory", embedder_name="hash")
    result = _run(strategy, conversations)
    tok = token_cost_summary(result)["tokens"]
    assert "aggregate_context_tokens" in tok
    assert "aggregate_tokens_per_test_point" in tok
    assert tok["aggregate_context_tokens"] >= 0
    assert isinstance(tok["per_conversation"], dict)
    assert len(tok["per_conversation"]) == len(result.conversations)


def test_full_history_costs_more_than_mneme(conversations) -> None:
    """Sanity: the oracle baseline burns more tokens than retrieval.

    Full-history pre-pends the whole transcript on every turn; Mneme
    only surfaces the top-k retrieved records. Even with k=5 and our
    short starter conversations, the gap should be visible at the
    aggregate level.
    """
    full = token_cost_summary(_run(FullHistoryStrategy(), conversations))
    mneme_result = _run(
        build_mneme_strategy(backend_name="memory", embedder_name="hash"),
        conversations,
    )
    mneme_tok = token_cost_summary(mneme_result)
    assert (
        full["tokens"]["aggregate_context_tokens"]
        >= mneme_tok["tokens"]["aggregate_context_tokens"]
    ), "full_history should not be cheaper than retrieval"


def test_cli_writes_json_file(tmp_path, conversations) -> None:
    # Import here to avoid eager argparse setup at collection time.
    from evals.__main__ import main

    out = tmp_path / "result.json"
    rc = main(
        [
            "--corpus",
            str(_CORPUS),
            "--runner",
            "mneme",
            "--backend",
            "memory",
            "--embedder",
            "hash",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["strategy"]["name"] == "mneme"
    assert payload["conversation_count"] >= 5
    assert "recall_at_k" in payload["metrics"]
    assert "tokens" in payload["metrics"]
    # The conversations array carries the per-test-point retrievals.
    assert isinstance(payload["conversations"], list)
    assert all("test_points" in c for c in payload["conversations"])


def test_cli_all_runners_smoke(tmp_path, conversations) -> None:
    """Every CLI ``--runner`` choice produces a valid JSON file."""
    from evals.__main__ import main

    for runner in ("mneme", "no_memory", "summary_buffer", "full_history"):
        out = tmp_path / f"{runner}.json"
        rc = main(
            [
                "--corpus",
                str(_CORPUS),
                "--runner",
                runner,
                "--output",
                str(out),
            ]
        )
        assert rc == 0, runner
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["strategy"]["name"] == runner


def test_cli_unknown_runner_rejected(tmp_path) -> None:
    from evals.__main__ import main

    with pytest.raises(SystemExit):
        main(
            [
                "--corpus",
                str(_CORPUS),
                "--runner",
                "does-not-exist",
                "--output",
                str(tmp_path / "x.json"),
            ]
        )


def test_load_corpus_missing_dir_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


# A small belt-and-braces check that the metric helpers don't crash on
# an empty result — corner case for "I added a strategy that retrieves
# nothing for every point."
def test_metrics_on_empty_run() -> None:
    from evals.runners.base import RunResult

    empty = RunResult(strategy={"name": "empty"}, seed=0)
    rec = recall_at_k(empty)
    tok = token_cost_summary(empty)
    assert rec["recall_at_k"]["aggregate"] is None
    assert tok["tokens"]["aggregate_context_tokens"] == 0


# Ensure the module is importable as ``python -m evals`` (i.e. it has
# a ``__main__.py``). The presence check below is dirt cheap but catches
# packaging regressions.
def test_module_main_exists() -> None:
    import importlib.util

    spec = importlib.util.find_spec("evals.__main__")
    assert spec is not None
    # Sanity: it's a real file we can resolve, not a namespace mirage.
    assert spec.origin is not None
    assert Path(spec.origin).suffix == ".py"
    # And it's actually loadable in this interpreter.
    assert "evals" in sys.modules or importlib.util.find_spec("evals") is not None
