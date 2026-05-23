"""End-to-end smoke test against a real backend + real OpenAI.

Wires :class:`mneme.MemoryManager` with real ``OpenAIEmbeddings`` + real
``OpenAILLMJudge`` against one of the persisted backends and runs a short
synthetic conversation through the full lifecycle:

    write episodes -> retrieve -> consolidate -> retrieve again -> forget

Outputs each step so you can eyeball whether the library does what it
says on the tin.

Usage::

    uv run python scripts/smoke.py --backend sqlite
    uv run python scripts/smoke.py --backend qdrant
    uv run python scripts/smoke.py --backend pgvector

Requirements:
    * ``OPENAI_API_KEY`` in the environment (or in .env at repo root).
    * For qdrant: ``MNEME_TEST_QDRANT_URL`` (default
      ``http://localhost:6333``) — start with ``docker compose up -d``.
    * For pgvector: ``MNEME_TEST_PGVECTOR_DSN`` (default
      ``postgresql://mneme:mneme@localhost:5433/mneme``) — same compose.

Costs about 1-3 cents per run (one embedding batch + one consolidation
LLM call).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import uuid

# Load .env if python-dotenv happens to be installed. Optional so the
# script also works in plain `pip install smolAmem` environments. We pass
# ``encoding="utf-8-sig"`` so a .env saved by Windows tooling (Notepad,
# VS Code) with a UTF-8 BOM still loads — the vanilla ``load_dotenv()``
# call mishandles the BOM and silently produces an empty result.
with contextlib.suppress(ImportError):
    from pathlib import Path

    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path, encoding="utf-8-sig")

from mneme import (
    MemoryManager,
    OpenAIEmbeddings,
    OpenAILLMJudge,
    PgVectorBackend,
    QdrantBackend,
    SQLiteBackend,
)
from mneme.backends.base import MnemeBackend

# Defaults match docker-compose.yml.
_DEFAULT_QDRANT_URL = "http://localhost:6333"
_DEFAULT_PGVECTOR_DSN = "postgresql://mneme:mneme@localhost:5433/mneme"


def build_backend(name: str, dim: int) -> tuple[MnemeBackend, object]:
    """Return ``(backend, cleanup_handle)``.

    ``cleanup_handle`` is None for backends that don't need teardown,
    otherwise an opaque object the caller hands back to ``cleanup()``.
    """
    if name == "sqlite":
        return SQLiteBackend(path=":memory:", dimensions=dim), None
    if name == "qdrant":
        from qdrant_client import QdrantClient

        url = os.environ.get("MNEME_TEST_QDRANT_URL", _DEFAULT_QDRANT_URL)
        client = QdrantClient(url=url, check_compatibility=False)
        collection = f"mneme_smoke_{uuid.uuid4().hex[:8]}"
        backend = QdrantBackend(client=client, collection=collection, dimensions=dim)
        return backend, (client, collection)
    if name == "pgvector":
        dsn = os.environ.get("MNEME_TEST_PGVECTOR_DSN", _DEFAULT_PGVECTOR_DSN)
        table = f"mneme_smoke_{uuid.uuid4().hex[:8]}"
        backend = PgVectorBackend(dsn=dsn, table=table, dimensions=dim)
        return backend, (backend, table)
    raise ValueError(f"unknown backend: {name!r}")


def cleanup(name: str, handle: object) -> None:
    """Drop the per-run table/collection so smoke runs don't leak state."""
    if handle is None:
        return
    if name == "qdrant":
        client, collection = handle  # type: ignore[misc]
        with contextlib.suppress(Exception):
            client.delete_collection(collection_name=collection)
    elif name == "pgvector":
        backend, table = handle  # type: ignore[misc]
        # PgVectorBackend exposes a private connection; use it for the
        # drop. We close the backend right after so this is fine.
        with (
            contextlib.suppress(Exception),
            backend._conn.cursor() as cur,  # type: ignore[attr-defined]
        ):
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        with contextlib.suppress(Exception):
            backend.close()


# Synthetic conversation. Mirrors the kind of dialog Mneme is built for:
# the user establishes facts about their stack and preferences early,
# and we expect retrieval/consolidation to surface them later.
_EPISODES: list[tuple[str, str]] = [
    ("user", "hi there"),
    ("assistant", "Hello! What are you working on today?"),
    (
        "user",
        "I'm building a Next.js 14 app called dashboard-v2. "
        "TypeScript everywhere, Postgres with Drizzle ORM under it.",
    ),
    ("assistant", "Nice stack. Anything I can help with?"),
    ("user", "I prefer a functional style and use Vitest for tests."),
    ("assistant", "Got it."),
    (
        "user",
        "Also — I usually deploy to Vercel, and I keep auth on Clerk.",
    ),
    ("assistant", "Sounds like a solid baseline."),
]

_QUERY = "what is the user's tech stack?"


def banner(label: str) -> None:
    print()
    print(f"=== {label} ===")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        choices=["sqlite", "qdrant", "pgvector"],
        default="sqlite",
        help="Which backend to exercise (default: sqlite).",
    )
    p.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Skip the consolidation LLM call (saves ~1c).",
    )
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: OPENAI_API_KEY not set. Put it in .env or export it.",
            file=sys.stderr,
        )
        return 1

    print(f"backend={args.backend}  skip_consolidate={args.skip_consolidate}")

    embedder = OpenAIEmbeddings()
    judge = OpenAILLMJudge()
    backend, handle = build_backend(args.backend, embedder.dimensions)

    try:
        m = MemoryManager(
            agent_id="smoke",
            backend=backend,
            embedder=embedder,
            llm_judge=judge,
        )

        # --- 1. Write the conversation ---
        banner("1. write episodes")
        t0 = time.time()
        for role, content in _EPISODES:
            m.working.add(role=role, content=content)
            m.episodic.add(content, metadata={"role": role})
        print(f"  wrote {len(_EPISODES)} episodes in {time.time() - t0:.2f}s")
        print(f"  episodic count: {m.episodic.count()}")

        # --- 2. Retrieve before consolidation ---
        banner(f"2. retrieve before consolidate: '{_QUERY}'")
        t0 = time.time()
        for r in m.retrieve(_QUERY, k=3):
            tier = r.record.tier.value
            content = r.record.content[:80].replace("\n", " ")
            print(f"  {r.score:.3f}  [{tier:<8}]  {content}")
        print(f"  ({time.time() - t0:.2f}s)")

        # --- 3. Consolidate (LLM extracts semantic facts) ---
        if not args.skip_consolidate:
            banner("3. consolidate (LLM)")
            t0 = time.time()
            facts = m.consolidate()
            print(f"  {len(facts)} facts in {time.time() - t0:.2f}s")
            for f in facts:
                print(f"    [{f.confidence:.2f}]  {f.content}")
            print(f"  semantic count: {m.semantic.count()}")

            # --- 4. Retrieve after consolidation ---
            banner(f"4. retrieve after consolidate: '{_QUERY}'")
            t0 = time.time()
            for r in m.retrieve(_QUERY, k=3):
                tier = r.record.tier.value
                content = r.record.content[:80].replace("\n", " ")
                print(f"  {r.score:.3f}  [{tier:<8}]  {content}")
            print(f"  ({time.time() - t0:.2f}s)")
        else:
            print()
            print("  (skipping consolidate + post-consolidate retrieve)")

        # --- 5. Forget (TTL only; nothing should be evicted from a fresh run) ---
        banner("5. forget(ttl_only=True)")
        removed = m.forget(ttl_only=True)
        print(f"  {removed}")

        banner("done")
        print(f"  episodic={m.episodic.count()}  semantic={m.semantic.count()}")
        return 0
    finally:
        cleanup(args.backend, handle)


if __name__ == "__main__":
    raise SystemExit(main())
