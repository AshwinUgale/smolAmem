# Mneme

> Multi-tier long-term memory for LLM agents. Working / episodic / semantic tiers, pluggable storage backends, framework-agnostic adapters, and explicit forgetting + consolidation.

[![docs](https://img.shields.io/badge/docs-ashwinugale.github.io%2Fashwinugale--mneme-blue)](https://ashwinugale.github.io/ashwinugale-mneme/)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**Status:** 1.0 released. APIs are stable; semver from here. Bug reports welcome.

## Why

Every agent framework ships with toy memory: last-N messages, or an LLM-summarised buffer. Real agents need to remember user preferences across weeks, recall specific past interactions, and store semantic facts extracted from many interactions — and they need to forget things that stop being relevant.

Mneme is the library you wish existed.

## Install

```bash
pip install smolAmem                    # core only
pip install "smolAmem[sqlite,openai]"   # SQLite + OpenAI embeddings
pip install "smolAmem[qdrant,openai]"   # production vector backend
```

> **Heads-up on naming:** PyPI install name is `smolAmem` (the `mneme` slot was already taken). Python import name stays `mneme` — same dual-name pattern as `Pillow` / `PIL` or `pyyaml` / `yaml`. So you `pip install smolAmem` but write `import mneme` everywhere in your code.

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

From the v0.6 [eval harness](https://ashwinugale.github.io/ashwinugale-mneme/eval/) on the 5-conversation starter corpus, k=5:

| Strategy | recall@5 | tokens / test point |
|---|---|---|
| `no_memory` | 0.000 | 0.0 |
| `mneme` (hash, deterministic) | 0.833 | 68.0 |
| **`mneme` (OpenAI embeddings)** | **1.000** | **67.7** |
| `full_history` | 1.000 | 141.0 |
| `summary_buffer` | 1.000 | 165.3 |

Mneme matches the full-history oracle for accuracy at **less than half** the token cost. The HashEmbedder row is the cheap deterministic baseline that runs without an API key — useful in CI and for reviewers reproducing numbers.

Reproduce:

```bash
uv run python -m evals --runner mneme --embedder openai --output out/mneme.json
```

## License

MIT. See [LICENSE](./LICENSE).
