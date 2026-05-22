# Eval harness

Mneme ships a reproducible benchmark in `evals/`. Plays a labelled corpus of conversations through one of four memory strategies and reports two strategy-vs-strategy metrics: `recall@k` and token cost.

Not installed by `pip install mneme` — `evals/` is a sibling package at the repo root, used by people who clone the repo to reproduce numbers.

---

## Headline numbers

From a fresh clone, with no API key, on the 5-conversation starter corpus:

| Strategy | recall@5 | tokens / test point |
|---|---|---|
| `no_memory` | 0.000 | 0.0 |
| **`mneme`** | **0.833** | **68.0** |
| `full_history` | 1.000 | 141.0 |
| `summary_buffer` | 1.000 | 165.3 |

Read this as: Mneme hits 5/6 labelled facts at less than half the token cost of full-history, *with the deterministic HashEmbedder*. Real OpenAI embeddings push recall higher; cost stays bounded by `k`.

Note that `summary_buffer` here is *worse* than full-history on tokens — the trivial built-in summariser concatenates rather than compressing. Plug in a real LLM summary callback (see the strategy module) to make it a fair comparison.

---

## Running it

From a clone of the repo, with `uv sync` done:

```bash
uv run python -m evals --runner mneme        --output out/mneme.json
uv run python -m evals --runner no_memory    --output out/no_memory.json
uv run python -m evals --runner summary_buffer --output out/summary.json
uv run python -m evals --runner full_history --output out/full.json
```

Each invocation produces one JSON file. To compare:

```bash
jq '{strategy: .strategy.name,
     recall: .metrics.recall_at_k.aggregate,
     tokens_per_tp: .metrics.tokens.aggregate_tokens_per_test_point}' out/*.json
```

For real embeddings:

```bash
export OPENAI_API_KEY="sk-..."
uv run python -m evals --runner mneme --embedder openai --output out/mneme_oai.json
```

---

## What gets measured

### `recall@k`

For each test point: is *any* of the test point's `expected_keywords` (case-insensitively) present in *any* of the strategy's top-k retrieved records? Aggregate is `hits / total_test_points`.

Keyword overlap rather than exact-string match. The retrieved record is the verbatim turn (`"I mostly work in TypeScript these days"`); the canonical fact is paraphrased (`"user prefers TypeScript"`). Keywords like `["TypeScript"]` bridge the two without either side having to match the other word-for-word.

### Token cost

Sum of token counts across every retrieved record, per conversation and aggregated. Uses `tiktoken` with `cl100k_base` when available; falls back to "~4 chars per token". The cross-strategy ratio is what matters for cost comparisons; absolutes feed back-of-envelope `$/1M tokens * this number = dollar figure` estimates.

### What's *not* measured (yet)

End-to-end accuracy via LLM-generated answers + LLM-as-judge scoring is reserved behind `--with-answers --judge`. Flags are accepted today, no-op'd. The v0.6 path is the cheap retrieval-only comparison; the expensive path will land in a later milestone when there's a corpus large enough for the spend to make sense.

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

## Output shape

```json
{
  "strategy": {
    "name": "mneme",
    "backend": "InMemoryBackend",
    "embedder": "HashEmbedder",
    "dimensions": 16
  },
  "seed": 42,
  "k": 5,
  "corpus_dir": "evals/corpus",
  "conversation_count": 5,
  "metrics": {
    "recall_at_k": {
      "aggregate": 0.8333,
      "hits": 5,
      "total": 6,
      "per_conversation": {
        "001_typescript_preference": {"hits": 0, "total": 1, "recall": 0.0}
      }
    },
    "tokens": {
      "aggregate_context_tokens": 408,
      "aggregate_tokens_per_test_point": 68.0,
      "per_conversation": {
        "001_typescript_preference": {
          "context_tokens_sum": 105,
          "test_points": 1,
          "context_tokens_per_test_point": 105.0
        }
      }
    }
  },
  "conversations": [
    {
      "conversation_id": "001_typescript_preference",
      "description": "...",
      "turn_count": 13,
      "test_points": [
        {
          "turn_index": 12,
          "question": "what programming language does the user prefer?",
          "expected_fact": "user prefers TypeScript",
          "expected_keywords": ["TypeScript"],
          "retrieved_records": ["...", "..."]
        }
      ]
    }
  ]
}
```

The `conversations` array carries the raw play-through — useful for debugging "why did recall miss here?" without rerunning the strategy.

---

## The corpus

JSON files in `evals/corpus/`. One conversation per file. Pydantic-validated via `evals/schema.py`. `schema_version: "1"` on every conversation.

Each conversation is a list of chat turns; at least one turn is annotated with a `test_at` block that names the question, the expected fact, and a keyword shortlist used for fuzzy matching:

```json
{
  "schema_version": "1",
  "id": "006_my_case",
  "description": "One-sentence: what is this conversation testing?",
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

Adding a case is a one-file edit. Drop a JSON in the directory; the runner picks it up.

Test-point retrieval runs BEFORE the turn carrying `test_at` is added to memory. Otherwise we'd be measuring whether the strategy can find a fact inside the question itself, not whether it remembered.

---

## The strategies

| Strategy | What it does | Expected behaviour |
|---|---|---|
| `no_memory` | Stores nothing, retrieves nothing | recall@k = 0.0 — the floor |
| `summary_buffer` | Last N turns verbatim + rolling summary of overflow (LangChain-style) | Depends on whether keywords survive the summary |
| `full_history` | Every turn, forever, all of it in context | recall@k = 1.0 — the ceiling at unbounded cost |
| `mneme` | `MemoryManager` with dual-write to working + episodic, semantic retrieval | What we're measuring |

All four implement the same `MemoryStrategy` protocol (`evals/baselines/base.py`). Add a new one with five methods — no inheritance required.

---

## Reproducibility

Two sources of non-determinism in a memory eval:

- **The embedder.** With `HashEmbedder`, everything is deterministic — same bytes, same vector, same retrieval order. With OpenAI, the API itself is deterministic in practice but tiny precision drift between calls is real (we hit ~1e-5 element-wise drift in v0.5; tolerance loosened to 1e-3 in the embedding tests).
- **The LLM judge / answer generation.** When those are on (`--with-answers`, `--judge`), every run varies. Temperature=0 + a fixed seed reduces the variance but doesn't eliminate it.

The cheap path is fully reproducible. The expensive path is reproducible *within a model release version + within OpenAI's nondeterminism budget*. Always record model name + run date when publishing numbers.

---

## Where to go next

- The full corpus authoring conventions: `evals/corpus/README.md` in the repo.
- The strategies in detail: `evals/baselines/` (one file per strategy).
- The metric implementations: `evals/metrics/recall.py` and `evals/metrics/tokens.py`.
