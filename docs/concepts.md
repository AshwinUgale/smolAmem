# Concepts

Mneme's model in one page. If you're shopping the library, read this first — it tells you what's in the box and where the seams are.

---

## The three tiers

The split mirrors a long-standing cognitive-science distinction (Tulving 1972, refined since). It's a useful frame, but the analogy isn't load-bearing — what matters is that the three tiers solve different problems and shouldn't be collapsed into one.

### Working memory

The last N turns of the current conversation, in process, no persistence.

- **Backed by:** a fixed-size FIFO buffer in `WorkingMemoryTier`.
- **Wiped by:** `manager.working.clear()` or `manager.clear_all()`.
- **Excluded from:** `manager.retrieve()` — it's not a search surface, it's the running context.
- **Why separate from episodic?** Because the access patterns are different. Working memory is a hot read on every turn; episodic is a search corpus. Mixing them collapses caching, retrieval, and pruning into one tangle.

### Episodic memory

Specific past interactions, stored verbatim with timestamp + role + metadata.

- **Backed by:** any `MnemeBackend` — `InMemoryBackend`, `SQLiteBackend`, `QdrantBackend`, `PgVectorBackend`.
- **Searchable:** by semantic similarity. `retrieve()` ranks results by `similarity * authority * recency` (no confidence multiplier — episodes are first-class observations).
- **Lifetime:** unbounded by default. Subject to TTL via `expires_at` and to access-frequency decay via `forget()`.
- **Why verbatim?** Because the LLM consolidator (semantic tier, below) needs the raw signal to extract facts. Compress at the wrong stage and you lose information you can't recover.

### Semantic memory

Durable facts about the user / project / preferences, extracted from episodes by an LLM.

- **Backed by:** the same `MnemeBackend` as episodic; tier is a field on the record.
- **Searchable:** the same way; the score formula adds a `* confidence` factor.
- **Lifetime:** also subject to forgetting, but typically tuned to live much longer than episodes.
- **Authority weight:** higher than episodic (default 1.0 vs 0.7). When the same fact surfaces from both tiers, the semantic record wins.
- **Why use this instead of just searching episodic?** Compression. A semantic fact is one sentence; the equivalent episodic content might be twenty interleaved turns. Putting one sentence in the prompt is cheaper, scans faster, and gives the model less to ignore.

[`02-architecture/01-tier-design`](https://github.com/ashwinugale/ashwinugale-mneme/tree/main/src/mneme/tiers) in the source has the tier classes.

---

## Retrieval

`manager.retrieve(query, *, k, tiers, authority_weights, half_life_days, use_confidence, now)`.

The pipeline:

```text
query -> embed -> backend.search(top 4*k by similarity)
                     |
                     v
                 for each candidate:
                   score = clamp(sim, 0)
                         * authority[tier]
                         * recency(half_life_days)
                         * (confidence if semantic and use_confidence else 1)
                     |
                     v
                 sort desc, return top k
```

A few design notes:

- **Multiplicative fusion**, not weighted sum. The components are independent signals that should each be capable of zeroing the score — a stale fact with high similarity should still rank low.
- **Authority weighting** lets you boost semantic facts over episodic noise. Default `{SEMANTIC: 1.0, EPISODIC: 0.7}`; override per call.
- **Exponential freshness decay** with a configurable half-life (default 7 days). `recency = 2 ** -(age_days / half_life_days)`. Future-dated records get 1.0.
- **`use_confidence`** lets you disable the confidence multiplier for semantic facts — useful when you're inspecting why a result ranked the way it did.
- **`now`** is injectable so tests can pin the clock. It defaults to `datetime.now(timezone.utc)`.

Every `RetrievalResult` carries `similarity`, `authority`, `recency`, `confidence` separately as well as the fused `score` — open one up if a ranking surprises you.

---

## Consolidation

The episodic → semantic pipeline. `manager.consolidate(*, since, batch_size, dedup_threshold, max_episodes)` does this:

```text
1. backend.list_recent(tier=EPISODIC, since=..., limit=max_episodes)
2. chunk into batches of batch_size
3. for each batch:
     facts = judge.complete(EXTRACTION_PROMPT, episodes_text, schema=FACT_SCHEMA)
4. for each extracted fact:
     embed the fact's content
     existing = semantic.search(content, k=1)
     if existing[0].score >= dedup_threshold:
         merge into existing (newer-wins, provenance union, max confidence)
     else:
         semantic.add(content, confidence=..., provenance=src_episode_ids)
5. return the list of facts written
```

Key points:

- **Constrained decoding.** The judge gets a JSON schema and is required to emit valid output. No `json.loads` retry loops.
- **Newer wins on conflict.** If consolidation extracts "user prefers Vue now" and an existing semantic fact says "user uses React", they're similar enough on cosine to merge — the newer content wins, the provenance keeps the older episode IDs around for audit.
- **Idempotent in practice.** Run with no new episodes since the last cutoff; nothing gets extracted; nothing gets written.
- **Manual or scheduled.** Call `consolidate()` from app code, or `start_scheduler(consolidate_every=...)` and walk away.

The extraction prompt lives at module scope in `mneme.manager._CONSOLIDATION_SYSTEM_PROMPT` — read it if you want to know exactly what the model is being asked to do.

---

## Forgetting

Two failure modes Mneme guards against, both via `manager.forget()`:

- **TTL eviction.** `MemoryRecord.expires_at` is a UTC datetime. Records whose `expires_at <= now` are deleted on `forget()`. `retrieve()` also filters them defensively so you never see an expired record between scheduler runs.
- **Access-frequency decay.** Records older than `cold_age_days` and accessed fewer than `access_floor` times are deleted. The access count is incremented by `backend.touch()` every time `retrieve()` returns the record, so retrieval keeps useful memories alive.

The signature:

```python
manager.forget(
    now=None,                # injectable clock; defaults to UTC now
    ttl_only=False,          # skip the access-decay phase if True
    access_floor=1,
    cold_age_days=30,
    max_per_tier=None,       # cap eviction count per tier per call
)
# -> {"expired": N, "cold": M}
```

Working memory is never touched. Eviction there happens at insert time when the FIFO overflows.

The `_FORGETTABLE_TIERS` constant on `MemoryManager` is `(EPISODIC, SEMANTIC)` — explicit, not derived, so a future tier can't accidentally inherit the policy.

---

## Background workers

Scheduling consolidation + forgetting is opt-in via APScheduler:

```python
m.start_scheduler(
    consolidate_every=30 * 60,    # seconds
    forget_every=24 * 3600,
    consolidate_kwargs={"batch_size": 10},
    forget_kwargs={"access_floor": 1},
)
# ...
m.stop_scheduler()
```

A few caveats worth knowing:

- **Daemon thread**, not asyncio. `BackgroundScheduler(daemon=True)` so the scheduler doesn't keep the process alive after main exits. There's a best-effort `__del__` cleanup but you should still call `stop_scheduler()` explicitly when you can.
- **`max_instances=1, coalesce=True`** on both jobs. If a previous run is still in flight, the next tick is dropped — not queued — so we never stack runs on top of each other.
- **`pip install "mneme[scheduler]"`** to get the dep. Without it, `start_scheduler` raises with a helpful hint.

---

## Multi-tenancy

Every `MnemeBackend` method takes an `agent_id` and filters by it. Mneme never permits cross-agent leakage. Practically:

- One backend instance can serve many agents.
- A query for `agent_id="alice"` will never see `agent_id="bob"`'s data, regardless of the backend.
- The `MemoryManager` constructor takes a single `agent_id` and reuses it everywhere; one manager per agent is the typical pattern.

The conformance suite has an explicit cross-agent isolation test for every backend.

---

## Dimensions are locked at backend construction

`SQLiteBackend`, `QdrantBackend`, and `PgVectorBackend` all take a `dimensions` parameter (or infer it on collection / table creation) and refuse mismatched embeddings on insert:

```python
backend = SQLiteBackend(path="mneme.db", dimensions=1536)
embedder = OpenAIEmbeddings()      # text-embedding-3-small, 1536-d
# OK — dimensions match.

embedder = HashEmbedder(dimensions=16)
m = MemoryManager(backend=backend, embedder=embedder, ...)
# Will raise ValueError on first add() — embedding doesn't fit the schema.
```

This is intentional. Letting embeddings of mixed dimensionality coexist in one collection would silently produce nonsense rankings. Better to crash early.

---

## What's deliberately out of scope

- **Cross-process consolidation locking.** If two processes both call `consolidate()` against the same backend at the same moment, you'll get two extractions and possibly two near-duplicate facts (the dedup pass catches most of these, but it's not a database-level lock). For now, schedule from one process or accept the slight duplication risk.
- **Async retrieve / consolidate.** All call paths are sync; the framework adapters' async variants delegate to threads via `asyncio.to_thread`. Native async is additive when there's demand and a use case the sync path can't meet.
- **A built-in LangGraph adapter.** Its checkpoint API is a separate, larger surface than message history. Likely lands as `mneme.adapters.langgraph` when there's a real use case asking for it.
- **A hybrid lexical + semantic retriever.** RRF + BM25 + embeddings is on the v1.x roadmap. Until then, retrieval is pure semantic similarity.
- **Cross-agent shared facts.** Some applications want a shared "global" agent_id; Mneme doesn't model this directly. Closest pattern: instantiate a second manager with a shared agent_id and read from it explicitly.

---

## Where to go next

- [Backends](backends.md) — pick the storage layer for your environment.
- [Embedders](embedders.md) — OpenAI vs the deterministic hash baseline.
- [Adapters](adapters.md) — drop-in shims for LangChain, LlamaIndex, raw OpenAI.
- [Eval harness](eval.md) — measure how much memory is actually helping.
- [API reference](api.md) — autogen from docstrings.
