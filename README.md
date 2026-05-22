# Mneme

> Multi-tier long-term memory for LLM agents. Working / episodic / semantic tiers, pluggable storage backends, framework-agnostic adapters, and explicit forgetting + consolidation.

[![docs](https://img.shields.io/badge/docs-ashwinugale.github.io%2Fashwinugale--mneme-blue)](https://ashwinugale.github.io/ashwinugale-mneme/)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**Status:** pre-alpha. APIs will change before 1.0. Pin to a specific version if you build on it.

## Why

Every agent framework ships with toy memory: last-N messages, or an LLM-summarised buffer. Real agents need to remember user preferences across weeks, recall specific past interactions, and store semantic facts extracted from many interactions — and they need to forget things that stop being relevant.

Mneme is the library you wish existed.

## Install

```bash
pip install mneme                    # core only
pip install "mneme[sqlite,openai]"   # SQLite + OpenAI embeddings
pip install "mneme[qdrant,openai]"   # production vector backend
```

## Quickstart

```python
from mneme import MemoryManager, SQLiteBackend, OpenAIEmbeddings

m = MemoryManager(
    agent_id="alice",
    backend=SQLiteBackend(path="mneme.db", dimensions=1536),
    embedder=OpenAIEmbeddings(),
)

m.episodic.add("user mostly works in TypeScript", metadata={"role": "user"})

for r in m.retrieve("what language does the user prefer?", k=3):
    print(r.score, r.record.content)
```

Full walkthrough: [docs/quickstart](https://ashwinugale.github.io/ashwinugale-mneme/quickstart/).

## Documentation

The full site lives at **[ashwinugale.github.io/ashwinugale-mneme](https://ashwinugale.github.io/ashwinugale-mneme/)**.

- [Concepts](https://ashwinugale.github.io/ashwinugale-mneme/concepts/) — the three-tier model, retrieval, consolidation, forgetting.
- [Backends](https://ashwinugale.github.io/ashwinugale-mneme/backends/) — InMemory, SQLite, Qdrant, pgvector.
- [Adapters](https://ashwinugale.github.io/ashwinugale-mneme/adapters/) — LangChain, LlamaIndex, raw OpenAI.
- [Eval harness](https://ashwinugale.github.io/ashwinugale-mneme/eval/) — reproducible recall@k + token-cost benchmark.
- [API reference](https://ashwinugale.github.io/ashwinugale-mneme/api/) — autogen from docstrings.

## Benchmark snapshot

From the v0.6 [eval harness](https://ashwinugale.github.io/ashwinugale-mneme/eval/) on the 5-conversation starter corpus, k=5, deterministic HashEmbedder (no API key required):

| Strategy | recall@5 | tokens / test point |
|---|---|---|
| `no_memory` | 0.000 | 0.0 |
| **`mneme`** | **0.833** | **68.0** |
| `full_history` | 1.000 | 141.0 |
| `summary_buffer` | 1.000 | 165.3 |

Reproduce:

```bash
uv run python -m evals --runner mneme --output out/mneme.json
```

## License

MIT. See [LICENSE](./LICENSE).
