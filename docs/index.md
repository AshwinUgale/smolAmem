# Mneme

> Multi-tier long-term memory for LLM agents. Working / episodic / semantic tiers, pluggable storage backends, framework-agnostic adapters, and explicit forgetting + consolidation.

!!! success "1.0 released"
    Public APIs are stable. Semver from here — minor releases add, patch releases fix, major releases break (rarely). Pin to a major version (`smolAmem>=1,<2`) and you're safe.

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

!!! info "Mneme installs as `smolAmem`, imports as `mneme`"
    The PyPI slot `mneme` was already taken by an unrelated package, so Mneme ships under `smolAmem` on PyPI. The Python import name stays `mneme` — same dual-name pattern as `Pillow` / `PIL` or `pyyaml` / `yaml`. **Install with the PyPI name; write code with the import name.** That's the only naming quirk; the rest is exactly what you'd expect.

```bash
pip install smolAmem                   # core only
pip install "smolAmem[sqlite,openai]"  # SQLite backend + OpenAI embeddings
pip install "smolAmem[qdrant,openai]"  # production-grade vector backend
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

Numbers from the v0.6 [eval harness](eval.md) on the 5-conversation starter corpus, k=5:

| Strategy | recall@5 | tokens / test point |
|---|---|---|
| `no_memory` | 0.000 | 0.0 |
| `mneme` (hash, deterministic) | 0.833 | 68.0 |
| **`mneme` (OpenAI embeddings)** | **1.000** | **67.7** |
| `full_history` | 1.000 | 141.0 |
| `summary_buffer` | 1.000 | 165.3 |

With real embeddings, Mneme matches the full-history oracle's accuracy (1.000) at less than half its token cost (67.7 vs 141.0). The HashEmbedder row is the cheap deterministic baseline that runs without an API key — useful in CI and for reviewers reproducing numbers without spending tokens.

Reproduce locally:

```bash
# With OpenAI embeddings (set OPENAI_API_KEY first):
uv run python -m evals --runner mneme --embedder openai --output out/mneme.json

# Or the no-key deterministic baseline:
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

[GitHub](https://github.com/ashwinugale/smolAmem){ .md-button } [Issues](https://github.com/ashwinugale/smolAmem/issues){ .md-button } [Changelog](https://github.com/ashwinugale/smolAmem/blob/main/CHANGELOG.md){ .md-button }
