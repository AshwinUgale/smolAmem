"""Mneme's eval harness.

NOT shipped in the installed wheel. This directory holds the corpus, the
baseline memory strategies, the runner, the metrics, and the CLI used to
reproduce Mneme's published benchmark results.

Layout::

    evals/
    +-- corpus/                 Labelled conversations (JSON)
    +-- baselines/              Reference memory strategies for comparison
    +-- runners/                Code that plays a corpus through a strategy
    +-- metrics/                Recall@k, accuracy, token cost
    +-- schema.py               Pydantic models for corpus shape
    +-- __main__.py             CLI: ``python -m evals ...``
    +-- README.md               How to run + how to add conversations

Run with ``python -m evals --help`` from the repo root.
"""
