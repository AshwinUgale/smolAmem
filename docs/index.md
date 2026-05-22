# Mneme

> Multi-tier long-term memory for LLM agents. Working / episodic / semantic tiers, pluggable storage backends, framework-agnostic adapters, and explicit forgetting + consolidation.

!!! warning "Pre-alpha"
    APIs will change before 1.0. Pin to a specific version if you build on it. The shape of the public surface is now stable enough to depend on for prototypes.

---

## What it is

Most agent frameworks ship with one of two "memory" implementations: a sliding window of the last N messages, or an LLM-summarised rolling buffer. Both are fine for a 5-turn demo and useless once a conversation crosses a session boundary.

Mneme is the thing you wished those abstractions were. Three tiers of memory that mirror the cognitive-science distinction:

| Tier | What it holds | Lifetime |
|---|---|---|
| **Working** | The last N turns of the current conversation, in-process | Wiped on `clear()` |
| **Episodic** | Every interaction the agent ever had, stored verbatim with timestamps | Persisted; subject to TTL / decay |
| **Semantic** | Durable facts about the user / project, extracted from episodes by an LLM | Persisted; merged across runs |

Plus the things you'd want on top: semantic + lexical retrieval with authority weighting and freshness decay, multiplicative score fusion across signals, scheduled background consolidation, explicit forgetting with TTL + access-frequency decay, and adapters for LangChain / LlamaIndex / raw OpenAI.

---

## Install

```bash
pip install mneme                   # core only
pip install "mneme[sqlite,openai]"  # SQLite backend + OpenAI embeddings
pip install "mneme[qdrant,openai]"  # production-grade vector backend
```

Other extras: `pgvector`, `scheduler`, `langchain`, `llamaindex`, `tokens`, `docs`. See [Backends](backends.md) and [Embedders](embedders.md) for the full matrix.

---

## 60-second example

```python
from mneme import MemoryManager, InMemoryBackend, HashEmbedder

m = MemoryManager(
    agent_id="demo",
    backend=InMemoryBackend(),
    embedder=HashEmbedder(),
)

# Drop a fact into episodic memory.
m.episodic.add("user mostly works in TypeScript", metadata={"role": "user"})

# Ten turns later, retrieve.
for r in m.retrieve("what language does the user prefer?", k=3):
    print(r.score, r.record.content)
```

For a more realistic walkthrough — including `consolidate()`, `forget()`, and the framework adapters — see the [Quickstart](quickstart.md).

---

## Benchmark snapshot

Numbers from the v0.6 [eval harness](eval.md) on the 5-conversation starter corpus, k=5, with the deterministic HashEmbedder (no API key):

| Strategy | recall@5 | tokens / test point |
|---|---|---|
| `no_memory` | 0.000 | 0.0 |
| **`mneme`** | **0.833** | **68.0** |
| `full_history` | 1.000 | 141.0 |
| `summary_buffer` | 1.000 | 165.3 |

Mneme hits 5/6 labelled facts at less than half the token cost of the full-history oracle — *with the hash embedder*. Real OpenAI embeddings push recall higher; cost stays bounded by k. Reproduce locally:

```bash
uv run python -m evals --runner mneme --output out/mneme.json
```

---

## Where to go next

- New here? Start with the [Quickstart](quickstart.md).
- Want to understand the model? Read [Concepts](concepts.md).
- Picking storage? See [Backends](backends.md).
- Plugging into an existing agent? See [Adapters](adapters.md).
- Reproducing the numbers above? See [Eval harness](eval.md).
- Reading the API surface? See [API reference](api.md).

---

## Status

| Milestone | Status |
|---|---|
| v0.1 — walking skeleton | ✅ |
| v0.2 — LLM-driven consolidation | ✅ |
| v0.3 — forgetting + scheduler | ✅ |
| v0.4 — Qdrant + pgvector backends | ✅ |
| v0.5 — LangChain / LlamaIndex / OpenAI adapters | ✅ |
| v0.6 — eval harness | ✅ |
| v0.7 — docs site | ✅ (you're reading it) |
| v1.0 — PyPI release | ⏳ |

[GitHub](https://github.com/ashwinugale/ashwinugale-mneme){ .md-button } [Issues](https://github.com/ashwinugale/ashwinugale-mneme/issues){ .md-button }
