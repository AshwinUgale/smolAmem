# Quickstart

A 60-second walk-through. By the end you'll have a `MemoryManager` that stores conversation turns, retrieves them by semantic similarity, consolidates them into durable facts, and forgets the ones that age out.

---

## 1. Install

```bash
pip install "smolAmem[sqlite,openai]"
```

The core install (`pip install smolAmem`) has zero dependencies. Every storage backend, embedder, judge, and framework adapter is an opt-in extra so you only pay for what you use.

!!! info "Install vs import"
    The PyPI name is `smolAmem` (the `mneme` slot was already taken) but the Python import name stays `mneme`. So `pip install smolAmem` then `import mneme` — same pattern as `Pillow` / `PIL`.

For this walkthrough you need:

- `sqlite` — file-backed storage so the example persists across runs.
- `openai` — for real semantic embeddings (otherwise the hash embedder works fine for testing).

Set your key:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 2. Construct a manager

```python
from mneme import MemoryManager, SQLiteBackend, OpenAIEmbeddings, OpenAILLMJudge

backend = SQLiteBackend(path="mneme.db", dimensions=1536)
embedder = OpenAIEmbeddings()           # text-embedding-3-small, 1536-d
judge = OpenAILLMJudge()                # gpt-4o-mini by default

m = MemoryManager(
    agent_id="alice",                   # multi-tenancy key
    backend=backend,
    embedder=embedder,
    llm_judge=judge,
)
```

`agent_id` is the multi-tenancy boundary: every read and write is scoped to it, and no query ever crosses agents. Pass a stable string per user / session / app — it's just a label, not a structured key.

---

## 3. Feed it some turns

The three tiers are exposed directly on the manager:

```python
m.working.add(role="user", content="I'm working on a Next.js app called dashboard-v2.")
m.working.add(role="assistant", content="What's the stack underneath?")
m.working.add(role="user", content="Postgres + Drizzle for the ORM. TypeScript everywhere.")

# Also drop the same content into episodic for long-term retrieval.
m.episodic.add(
    "user is building dashboard-v2, a Next.js app with Postgres + Drizzle",
    metadata={"role": "user", "source": "session-2026-05-22"},
)
```

Most production code uses one of the [adapters](adapters.md) (LangChain / LlamaIndex / `context_for`) that dual-write to working + episodic for you. Doing it manually is fine when you want explicit control.

---

## 4. Retrieve

```python
results = m.retrieve("what database is the user using?", k=3)
for r in results:
    print(f"{r.score:.3f}  [{r.record.tier}]  {r.record.content}")
```

`retrieve()` runs over episodic + semantic by default (working memory is excluded — it's not a retrieval surface). Each result carries the underlying record plus the components of the fused score: similarity, authority, recency, confidence. Inspect them when you're tuning.

By default the multiplicative fusion is:

```
score = clamp(similarity, 0) * authority * recency * (confidence if semantic else 1)
```

with `authority = {SEMANTIC: 1.0, EPISODIC: 0.7}` and a 7-day half-life on recency. All four knobs are kwargs on `retrieve()`.

---

## 5. Consolidate

Episodic memory is voluminous. The interesting facts are buried in 40 verbatim turns. `consolidate()` reads recent episodes, calls the LLM to extract durable facts, and writes them through the semantic tier with newer-wins dedup against existing facts.

```python
facts = m.consolidate()
for f in facts:
    print(f.confidence, f.content)
```

Run it manually for one-shot use, or schedule it (see [Concepts](concepts.md#consolidation)). Either way it's idempotent — running twice in a row with no new episodes does nothing.

---

## 6. Forget

TTL eviction and access-frequency decay:

```python
removed = m.forget(
    access_floor=1,            # records accessed < this count
    cold_age_days=30,          # ... and older than this
    ttl_only=False,            # also do TTL pass
)
print(removed)  # {"expired": N, "cold": M}
```

Working memory is never touched by `forget()` (it's a short-window FIFO; eviction happens at insert time, not on a schedule). See [Concepts → Forgetting](concepts.md#forgetting) for the policy in full.

---

## 7. Schedule it

If you want consolidation + forgetting to run automatically:

```python
m.start_scheduler(
    consolidate_every=30 * 60,    # seconds
    forget_every=24 * 3600,
)

# ... later, before process exit:
m.stop_scheduler()
```

This uses APScheduler's `BackgroundScheduler` under the hood. `pip install "smolAmem[scheduler]"` to get the dep. See [Concepts → Background workers](concepts.md#background-workers).

---

## 8. Plug into a framework

For a one-line drop-in:

=== "LangChain"

    ```python
    from mneme.adapters import MnemeChatMessageHistory

    history = MnemeChatMessageHistory(manager=m, session_id="alice")
    # Pass `history` to any LangChain runnable that wants a BaseChatMessageHistory.
    ```

=== "LlamaIndex"

    ```python
    from mneme.adapters import MnemeLlamaIndexMemory

    memory = MnemeLlamaIndexMemory(manager=m)
    # Pass to an agent / chat engine that wants a BaseMemory.
    ```

=== "Raw OpenAI"

    ```python
    from mneme.adapters import context_for

    messages = context_for(m, query="what database is the user using?", k=5)
    # Pass `messages` straight to client.chat.completions.create(messages=messages, ...)
    ```

All three dual-write to working + episodic by default, so retrieval and consolidation work without extra plumbing. See [Adapters](adapters.md) for the full surface and the trade-offs.

---

## Where to go next

- Want to understand *why* the tiers are split this way? Read [Concepts](concepts.md).
- Picking a backend? See [Backends](backends.md).
- Measuring whether memory is actually helping? See [Eval harness](eval.md).
