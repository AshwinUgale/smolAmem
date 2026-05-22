# Embedders

An embedder turns text into a fixed-dimensional vector. Mneme stores the vector alongside the record so retrieval can rank by cosine similarity. The embedder is an injectable dependency — swap one for another and the rest of the library doesn't notice.

| Embedder | Best for | Install |
|---|---|---|
| `OpenAIEmbeddings` | Real semantic similarity in production | `pip install "mneme[openai]"` |
| `HashEmbedder` | Tests, deterministic CI, no-key paths | core install — no extras |

---

## `OpenAIEmbeddings`

Calls OpenAI's embeddings API. Defaults to `text-embedding-3-small` (1536-d, ~$0.02 per 1M tokens).

```python
from mneme import OpenAIEmbeddings

embedder = OpenAIEmbeddings()                           # text-embedding-3-small
embedder = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072-d
embedder = OpenAIEmbeddings(api_key="sk-...")           # explicit; defaults to OPENAI_API_KEY env
```

Notes:

- **Auto-batches at 2048.** The OpenAI API accepts up to 2048 inputs per call; the embedder splits longer lists transparently.
- **Vectors are unit-normalised** by the API. Cosine similarity and dot product give the same ranking.
- **`dimensions` is an attribute**, not a method. `embedder.dimensions` returns the size you'll need to pass to backends that take it (`SQLiteBackend`, `QdrantBackend`, `PgVectorBackend`).
- **Tiny non-determinism between calls.** Embedding the same string twice can produce vectors that differ by ~1e-5 element-wise — well below any retrieval-affecting threshold but enough that exact equality tests need a `~1e-3` tolerance.

The `[openai]` extra also pulls in everything `OpenAILLMJudge` needs (same SDK).

---

## `HashEmbedder`

Deterministic, no API, no key. Hashes tokens and produces a fingerprint vector. *Not* a semantic embedder — semantically equivalent paraphrases will not cluster.

```python
from mneme import HashEmbedder

embedder = HashEmbedder(dimensions=16)          # default
embedder = HashEmbedder(dimensions=1536)        # match an OpenAI-sized backend for parity tests
```

When to use it:

- **CI.** Your eval pipeline doesn't need to spend tokens on the easy parts of the test surface.
- **Tests.** Mneme's own test suite uses HashEmbedder almost everywhere — same vector every time, no API costs, no network flakiness.
- **Reproducible benchmarks.** The eval harness's cheap path uses HashEmbedder so reviewers can reproduce headline recall@k numbers without spending money.

When *not* to use it:

- **In production.** It's a deterministic stand-in, not a real semantic model. Retrieval quality will be poor on anything more than keyword overlap.

---

## The protocol

```python
class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Two surface methods. `dimensions` is checked against the backend at construction time so a mismatch can't sneak past. `embed` takes a batch and returns one vector per input — implementations should batch internally if their underlying API benefits from it.

To plug in a different provider (Cohere, Voyage, a local model) implement the protocol:

```python
class CohereEmbeddings:
    @property
    def dimensions(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        # call out to Cohere, return list[list[float]]
        ...
```

No inheritance required — `@runtime_checkable` Protocols match by structure.

---

## Picking a model

A quick decision table:

| If... | Pick |
|---|---|
| You want one model, cheap, no fuss | `OpenAIEmbeddings()` (text-embedding-3-small) |
| You're doing serious retrieval work and the small model's recall is hurting | `OpenAIEmbeddings(model="text-embedding-3-large")` |
| You can't or won't call out to OpenAI | Implement the protocol against your model of choice |
| You're running CI / writing tests | `HashEmbedder()` |

Switching is a one-line constructor change at the manager level. The backend doesn't care which embedder produced its vectors, as long as the dimensions line up.
