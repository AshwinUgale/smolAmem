# Backends

Mneme's storage is pluggable. Every backend implements the same `MnemeBackend` protocol; swap one for another and the rest of the library doesn't notice. The conformance suite (`tests/test_backends.py`) parametrises every backend over the same set of behavioural tests, so they all behave identically on the bits that matter.

Pick by environment:

| Backend | Best for | Install |
|---|---|---|
| `InMemoryBackend` | Tests, ephemeral demos, fastest path | core install — no extras |
| `SQLiteBackend` | Single-process apps, prototypes, CLI tools | `pip install "smolAmem[sqlite]"` |
| `QdrantBackend` | Production agents, scale, multi-process | `pip install "smolAmem[qdrant]"` |
| `PgVectorBackend` | Postgres shops, transactional integration | `pip install "smolAmem[pgvector]"` |

---

## `InMemoryBackend`

Reference implementation. Holds everything in a Python dict; no persistence; no thread safety beyond what the GIL gives you.

```python
from mneme import InMemoryBackend, MemoryManager, HashEmbedder

backend = InMemoryBackend()             # dimensions inferred on first upsert
m = MemoryManager(
    agent_id="demo",
    backend=backend,
    embedder=HashEmbedder(),
)
```

- **Use when:** tests, demos, anything that doesn't need to survive a process restart.
- **Don't use when:** anything else.

---

## `SQLiteBackend`

File-backed, zero-infra. Uses [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search via a `vec0` virtual table joined by rowid to a normal `records` table.

```python
from mneme import SQLiteBackend, MemoryManager, OpenAIEmbeddings

backend = SQLiteBackend(path="mneme.db", dimensions=1536)
m = MemoryManager(
    agent_id="alice",
    backend=backend,
    embedder=OpenAIEmbeddings(),
)
```

Notes:

- **Dimensions are fixed at construction.** `dimensions=1536` matches OpenAI `text-embedding-3-small`; pass whatever your embedder reports. Mismatched embeddings raise `ValueError` on insert.
- **No KNN MATCH syntax.** We use `vec_distance_cosine` as a scalar in normal `WHERE / ORDER BY / LIMIT` so agent + tier + metadata filters compose without losing top-k.
- **4x over-fetch** for metadata filtering at the Python layer. Tunable if you need it.
- **Single connection per backend.** Not thread-safe at the connection layer; wrap with your own pool if you need concurrency.
- **`pip install "smolAmem[sqlite]"`** brings in `sqlite-vec`. Without it, `SQLiteBackend()` raises with a helpful import hint.

`:memory:` works too if you want a SQLite-flavoured ephemeral store for tests.

---

## `QdrantBackend`

Production-grade ANN search. Uses Qdrant's HNSW index with cosine distance. One collection per backend instance; multi-tenancy is enforced via an `agent_id` payload filter on every operation.

```python
from qdrant_client import QdrantClient
from mneme import QdrantBackend, MemoryManager, OpenAIEmbeddings

client = QdrantClient(url="http://localhost:6333")
backend = QdrantBackend(
    client=client,
    collection_name="mneme",
    dimensions=1536,
)
m = MemoryManager(
    agent_id="alice",
    backend=backend,
    embedder=OpenAIEmbeddings(),
)
```

Notes:

- **Constructor takes a pre-built `QdrantClient`.** Share the client with the rest of your app — auth, retries, timeouts, all live on the client.
- **Collection is created on construction if missing.** Default name `"mneme"`. Override for multi-tenant deployments that want one collection per app, not one per agent.
- **Qdrant point IDs are UUIDs.** Mneme's `uuid4().hex` round-trips through `uuid.UUID(hex=...)` at the boundary. The original hex string also lives in the payload for human readability.
- **Distance metric: COSINE.** Qdrant returns *similarity* for cosine-configured collections, matching our protocol contract.
- **`touch()` is two RTTs.** Qdrant has no server-side increment operator, so we retrieve the current `access_count` and `set_payload` with the new value. Acceptable for retrieval-touch volume; if your agent is read-hot, consider batching touches.

Running locally:

```bash
docker run -p 6333:6333 -v "$(pwd)/qdrant_data:/qdrant/storage" qdrant/qdrant:v1.12.0
```

Or use the bundled `docker-compose.yml` (`docker compose up -d qdrant`).

---

## `PgVectorBackend`

For Postgres-native deployments. Uses [pgvector](https://github.com/pgvector/pgvector) for the vector column + `<=>` cosine-distance operator. Schema mirrors `SQLiteBackend` so backend swaps don't require a schema migration.

```python
from mneme import PgVectorBackend, MemoryManager, OpenAIEmbeddings

backend = PgVectorBackend(
    dsn="postgresql://mneme:mneme@localhost:5433/mneme",
    table="mneme_records",
    dimensions=1536,
)
m = MemoryManager(
    agent_id="alice",
    backend=backend,
    embedder=OpenAIEmbeddings(),
)
```

Notes:

- **psycopg3 sync.** Async API later when retrieval becomes async too.
- **One connection per backend.** Not thread-safe at the connection layer. Open one backend per thread, or wrap with your own pool above.
- **`CREATE EXTENSION vector` is best-effort.** If your DB user lacks the privilege, the call is silently OK — your DBA should have done it once. The backend errors loudly if the extension isn't actually present.
- **Table name interpolated into SQL** (psycopg can't parameterise identifiers). Constructor validates `table.replace("_", "").isalnum()` to reject anything sketchy. No injection surface from the user-supplied name.
- **`register_vector` adapter** is set up at construction time so `list[float]` ↔ `vector` round-trips transparently.

Running locally:

```bash
docker run -p 5433:5432 -e POSTGRES_PASSWORD=mneme -e POSTGRES_USER=mneme -e POSTGRES_DB=mneme pgvector/pgvector:pg17
```

Or `docker compose up -d pgvector`. The exposed port is 5433 to avoid colliding with a local Postgres on 5432.

---

## The protocol

Every backend implements:

```python
class MnemeBackend(Protocol):
    def upsert(self, records: list[MemoryRecord], embeddings: list[list[float]], *, agent_id: str) -> None: ...
    def get(self, ids: list[str], *, agent_id: str) -> list[MemoryRecord]: ...
    def delete(self, ids: list[str], *, agent_id: str) -> None: ...
    def search(self, embedding: list[float], *, agent_id: str, tiers: list[MemoryTier], k: int, metadata_filter: dict | None) -> list[tuple[MemoryRecord, float]]: ...
    def touch(self, ids: list[str], *, agent_id: str, now: datetime) -> None: ...
    def list_recent(self, *, agent_id: str, tier: MemoryTier, since: datetime | None, limit: int) -> list[MemoryRecord]: ...
    def count(self, *, agent_id: str, tier: MemoryTier | None = None) -> int: ...
    def clear(self, *, agent_id: str, tier: MemoryTier | None = None) -> None: ...
```

To plug in your own backend, satisfy this protocol. There's no inheritance required — `@runtime_checkable` Protocols match by structure. Run the conformance suite against your implementation to verify behavioural parity:

```python
# tests/test_my_backend.py
import pytest
from mneme import MnemeBackend
from my_pkg import MyBackend

@pytest.fixture
def backend() -> MnemeBackend:
    return MyBackend(...)

# Then the conformance suite from tests/test_backends.py runs against your fixture.
```

---

## Which one should I use?

| If... | Pick |
|---|---|
| You're writing tests or doing a one-off script | `InMemoryBackend` |
| You're prototyping a single-process app or a CLI | `SQLiteBackend` |
| You're going to production with a real load profile | `QdrantBackend` (if your infra is mostly containers) or `PgVectorBackend` (if you already run Postgres) |
| You need transactional consistency with your application data | `PgVectorBackend` — same DB, same transaction |

Switching is a one-line constructor change. Don't over-think the early call; pick what's cheapest now and migrate when you've measured.
