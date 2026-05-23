# Adapters

Three drop-in shims so Mneme works with the agent stack you already have:

| Adapter | What it satisfies | Install |
|---|---|---|
| `MnemeChatMessageHistory` | LangChain's `BaseChatMessageHistory` | `pip install "smolAmem[langchain]"` |
| `MnemeLlamaIndexMemory` | LlamaIndex's `BaseMemory` | `pip install "smolAmem[llamaindex]"` |
| `context_for` | A function that returns OpenAI-style `messages=[...]` | core install; `[tokens]` for budget packing |

All three follow the same design principles:

- **Dual-write by default.** Every message goes into BOTH working (so framework-shaped reads return chat history) AND episodic (so `retrieve()` and `consolidate()` can do their work). Toggleable via `also_persist_episodic=False`.
- **Sync first.** Async variants delegate to `asyncio.to_thread`. Native async is additive when there's a use case the sync path can't meet.
- **`clear()` wipes working only.** "Start a new session" doesn't mean "delete the agent's memory." Use `manager.clear_all()` for that.

---

## LangChain

```python
from mneme import MemoryManager, SQLiteBackend, OpenAIEmbeddings
from mneme.adapters import MnemeChatMessageHistory

m = MemoryManager(
    agent_id="alice",
    backend=SQLiteBackend(path="mneme.db", dimensions=1536),
    embedder=OpenAIEmbeddings(),
)

history = MnemeChatMessageHistory(
    manager=m,
    session_id="alice",
    also_persist_episodic=True,   # default
)

# Now plug into any LangChain runnable that takes a BaseChatMessageHistory.
from langchain_core.messages import HumanMessage, AIMessage

history.add_message(HumanMessage(content="I work mostly in TypeScript."))
history.add_message(AIMessage(content="Got it."))

for msg in history.messages:
    print(type(msg).__name__, msg.content)
```

Notes:

- **Role mapping is lossy.** LangChain's `ToolMessage` / `FunctionMessage` are stored as `"assistant"` — Mneme's tier model doesn't care about that distinction. Round-tripping doesn't preserve the original message type. Documented as a known limitation; file an issue if it hurts.
- **Async methods.** `aadd_messages`, `aclear` exist; they call the sync counterparts via `asyncio.to_thread`. The dual-write call sites are sync at the manager layer, so native async would require a deeper change.

---

## LlamaIndex

```python
from mneme.adapters import MnemeLlamaIndexMemory

memory = MnemeLlamaIndexMemory(
    manager=m,
    also_persist_episodic=True,
)

# Pass `memory` to any LlamaIndex agent / chat engine that takes a BaseMemory.
from llama_index.core.llms import ChatMessage

memory.put(ChatMessage(role="user", content="What's the stack?"))
memory.put(ChatMessage(role="assistant", content="Postgres + Drizzle."))

for msg in memory.get_all():
    print(msg.role, msg.content)
```

Notes:

- **`get(input=...)` ignores the input argument at v0.5.** Using it as a retrieval query would conflate "give me chat history" with "give me search results" — two distinct concerns the manager keeps separate. If you want retrieval-driven context, build it explicitly:

    ```python
    history = memory.get_all()
    retrieved = m.retrieve(input, k=5)
    # ... assemble whatever shape your agent expects
    ```

- **Pydantic v2 model.** `arbitrary_types_allowed=True` so the manager passes through without becoming a field. Don't try to serialise the adapter directly; round-trip the manager instead.

---

## Raw OpenAI: `context_for`

The "I don't use a framework" path. Returns OpenAI-style chat messages so you can plug them straight into `client.chat.completions.create(messages=...)`.

```python
from mneme.adapters import context_for

messages = context_for(
    m,
    query="what database is the user using?",
    k=5,
    token_budget=4000,   # optional — requires the [tokens] extra
)

# messages looks like:
# [
#   {"role": "system", "content": "[FACT id=...] user is on Postgres. [EPISODE id=...] user said ..."},
#   {"role": "user", "content": "..."},
#   {"role": "assistant", "content": "..."},
#   ...
# ]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages + [{"role": "user", "content": query}],
)
```

Notes:

- **One `system` block carries the retrieved memory.** Items get `[FACT id=...]` / `[EPISODE id=...]` citation markers so the model can weight by tier authority if you prompt it to (or so you can inspect why an answer cited what it did).
- **Working memory follows the system block** in chronological order. Roles are passed through verbatim.
- **`token_budget` packs to a tiktoken count.** Working memory wins over retrieved memory when forced to drop something — the model needs the recent conversation to respond at all; retrieved context is additive. Requires `pip install "smolAmem[tokens]"`; without it, raise with a helpful hint.
- **`cl100k_base`** is the encoding used for budget calculation. That covers every OpenAI chat model from gpt-3.5-turbo through gpt-4o. For other providers, the count is approximate.

---

## Why dual-write?

Single-write to working would mean `retrieve()` can't see the conversation — the search corpus is empty and `consolidate()` has nothing to read. Single-write to episodic would mean `messages` / `get_all()` (the framework-shaped reads) miss the recent turns or have to derive them from episodic, conflating "chat history" with "search result."

Both writes are cheap (working is a Python list; episodic is one embedder call + one backend insert). The cost difference is small; the semantic clarity is worth a lot.

Toggle it off if you genuinely want ephemeral chats:

```python
history = MnemeChatMessageHistory(
    manager=m,
    session_id="alice",
    also_persist_episodic=False,    # working only; no long-term storage
)
```

---

## Picking an adapter

If your stack uses an agent framework, use its adapter — that's the path of least resistance and you get the framework's other affordances (callbacks, streaming, evaluation hooks) for free.

If you're calling OpenAI / Anthropic / etc. directly, use `context_for`. It's just a function; no inheritance, no Pydantic field machinery, no async surface to navigate.

If you're using LangGraph, there's no adapter yet — `MnemeChatMessageHistory` is usable from inside a LangGraph node, but the checkpoint API is a separate, larger surface. Track [issue tracking on the repo](https://github.com/ashwinugale/ashwinugale-mneme/issues) or file a request with a concrete use case.
