# Mneme Eval Harness

> Reproducible benchmark for Mneme's memory layer. Plays a labelled corpus of conversations through one of four memory strategies and reports recall@k + token cost.

Not shipped in the installed wheel — this is repo-shaped tooling. Run it from a clone of the repo.

---

## Quickstart

```bash
# From the repo root, after uv sync:
uv run python -m evals --runner mneme        --output out/mneme.json
uv run python -m evals --runner no_memory    --output out/no_memory.json
uv run python -m evals --runner summary_buffer --output out/summary.json
uv run python -m evals --runner full_history --output out/full.json
```

Each invocation produces a JSON file with per-conversation retrievals + aggregate metrics. No OpenAI key required for the default run (HashEmbedder + deterministic recall@k).

To use real embeddings instead of the hash baseline:

```bash
uv run python -m evals --runner mneme --embedder openai --output out/mneme_oai.json
```

Requires `OPENAI_API_KEY` in your environment.

---

## What gets measured

**recall@k** — for each test point, does any of `expected_keywords` (case-insensitively) appear in any of the strategy's top-k retrieved records? Aggregate is `hits / total_test_points`.

**token cost** — sum of token counts across every retrieved record, per conversation and as a grand total. Uses `tiktoken` (cl100k_base) when installed; falls back to "~4 chars per token". The cross-strategy ratio is what matters for cost comparisons.

Both metrics live in `evals/metrics/`. End-to-end accuracy (LLM answer generation + LLM-as-judge) is reserved behind `--with-answers --judge` and not wired in v0.6.

---

## The strategies

| Strategy | What it does | Expected behaviour |
|---|---|---|
| `no_memory` | Stores nothing, retrieves nothing | recall@k = 0.0 — the floor |
| `summary_buffer` | Last N turns verbatim + rolling summary of overflow (LangChain-style) | Depends on whether keywords survive the summary |
| `full_history` | Every turn, forever, all of it in context | recall@k = 1.0 — the ceiling; high token cost |
| `mneme` | `MemoryManager` with dual-write to working + episodic, semantic retrieval | The thing we're measuring |

All four implement the `MemoryStrategy` protocol in `baselines/base.py`. Add a new one by writing five methods.

---

## CLI reference

```
python -m evals \
  --corpus DIR              # default: evals/corpus/
  --runner {mneme,no_memory,summary_buffer,full_history}
  --backend {memory,sqlite} # mneme only; default memory
  --embedder {hash,openai}  # mneme only; default hash
  --output PATH             # required
  --seed INT                # default 42
  --k INT                   # default 5
  --with-answers            # reserved (LLM generation pass)
  --judge                   # reserved (LLM-as-judge accuracy)
```

The cheap path (no `--with-answers`, no `--judge`) runs with no API calls and is deterministic.

---

## Output format

```jsonc
{
  "strategy": {"name": "mneme", "backend": "InMemoryBackend", "embedder": "HashEmbedder", "dimensions": 16},
  "seed": 42,
  "k": 5,
  "corpus_dir": "evals/corpus",
  "conversation_count": 5,
  "metrics": {
    "recall_at_k": {
      "aggregate": 0.5,
      "hits": 3,
      "total": 6,
      "per_conversation": {"001_typescript_preference": {"hits": 0, "total": 1, "recall": 0.0}, ...}
    },
    "tokens": {
      "aggregate_context_tokens": 245,
      "aggregate_tokens_per_test_point": 40.83,
      "per_conversation": {...}
    }
  },
  "conversations": [
    {
      "conversation_id": "001_typescript_preference",
      "description": "...",
      "turn_count": 13,
      "test_points": [
        {"turn_index": 12, "question": "...", "expected_fact": "...", "expected_keywords": ["TypeScript"], "retrieved_records": ["...", "..."]}
      ]
    }
  ]
}
```

The `conversations` array is the raw play-through data — useful for debugging "why did recall miss here?" without rerunning.

---

## Adding a conversation

Drop a JSON in `corpus/` matching `schema.py`. Minimal shape:

```jsonc
{
  "schema_version": "1",
  "id": "006_my_new_case",
  "description": "One sentence: what is this conversation testing?",
  "turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {
      "role": "user",
      "content": "...",
      "test_at": {
        "question": "...",
        "expected_fact": "...",
        "expected_keywords": ["..."]
      }
    }
  ]
}
```

A turn becomes a test point when it has a `test_at` block. Multiple test points per conversation are fine. Test points run BEFORE the turn is added to memory.

Keywords are matched case-insensitively as substrings. Choose short, distinctive tokens — `"TypeScript"` not `"the user prefers TypeScript"`.

---

## Reproducibility notes

- The cheap path (HashEmbedder, no judge) is deterministic. Same input bytes → same output JSON.
- OpenAI embeddings have tiny precision drift between calls (~1e-5). The harness tolerates this; absolute embedding values aren't compared, only retrieval order.
- LLM-as-judge accuracy (when enabled) is reproducible only within a model release; record the model name + run date in any reported numbers.

---

## Comparing across strategies

The CLI runs one strategy per invocation. To produce a comparison, run all four and post-process the JSON files:

```bash
for r in mneme no_memory summary_buffer full_history; do
  uv run python -m evals --runner $r --output out/$r.json
done
jq '.metrics.recall_at_k.aggregate, .metrics.tokens.aggregate_tokens_per_test_point' out/*.json
```

The headline plot is recall@k on one axis, tokens-per-test-point on the other — Mneme should land in the top-left (high recall, low cost).
