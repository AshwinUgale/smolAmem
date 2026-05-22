"""Storage + vector-search backends for Mneme.

A backend implements the :class:`MnemeBackend` protocol and is responsible
for persisting :class:`mneme.MemoryRecord` instances and serving similarity
search over them. Embedding computation happens *outside* the backend (in the
tier layer); backends only store and search.

Backends shipped:

* :class:`InMemoryBackend` — process-local, zero-deps, the reference impl.
* :class:`SQLiteBackend` — single-file storage via ``sqlite-vec``. Requires
  the ``sqlite`` extra: ``pip install 'mneme[sqlite]'``.
* :class:`QdrantBackend` — Qdrant client (v0.4). Requires the ``qdrant`` extra.
* :class:`PgVectorBackend` — Postgres + pgvector (v0.4). Requires the
  ``pgvector`` extra.

Adding a backend means satisfying :class:`MnemeBackend` and passing the
conformance suite in ``tests/test_backends.py``.
"""

from mneme.backends.base import MnemeBackend
from mneme.backends.memory import InMemoryBackend
from mneme.backends.pgvector import PgVectorBackend
from mneme.backends.qdrant import QdrantBackend
from mneme.backends.sqlite import SQLiteBackend

__all__ = [
    "InMemoryBackend",
    "MnemeBackend",
    "PgVectorBackend",
    "QdrantBackend",
    "SQLiteBackend",
]
