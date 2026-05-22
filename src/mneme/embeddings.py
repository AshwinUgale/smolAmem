"""Embedding providers for Mneme.

An :class:`EmbeddingProvider` turns a list of strings into a list of vectors.
The tier layer uses one to embed records before persisting them to a backend
and to embed queries before calling :meth:`MnemeBackend.search`.

Two providers ship:

* :class:`OpenAIEmbeddings` — production default; wraps the OpenAI SDK,
  auto-batches at the per-request limit. Requires the ``openai`` extra and
  an ``OPENAI_API_KEY``.
* :class:`HashEmbedder` — deterministic test double. No API calls, no
  semantic structure (similar strings do NOT produce similar vectors).
  Useful when downstream code wants to exercise the wiring without paying
  for or depending on a real model.

Adding a provider (Cohere, Voyage, Sentence-Transformers, etc.) is the same
shape as adding a backend: implement the protocol, ship it, done.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Only imported for typing; the actual import in OpenAIEmbeddings is lazy
    # so the module loads without the ``openai`` extra installed.
    from openai import OpenAI


__all__ = ["EmbeddingProvider", "HashEmbedder", "OpenAIEmbeddings"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    Implementations declare a fixed :attr:`dimensions` and expose a single
    :meth:`embed` call that takes a list of strings and returns one vector
    per string in the same order.

    Single-text use is just ``provider.embed([text])[0]``. We deliberately do
    not add a convenience method for that case — every implementation would
    then have to provide both, and the one-line caller is just as clear.
    """

    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` and return one vector per input in order.

        Implementations should accept any ``Sequence[str]`` and return a
        plain ``list[list[float]]``. Vectors should have length
        :attr:`dimensions`; callers may assume this and pass them straight
        to a backend constructed with matching dimensions.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


# Per-request input limit for OpenAI's embeddings endpoint. We auto-batch
# above this size so callers can ``embed(big_list)`` without thinking.
_OPENAI_BATCH = 2048

# Known model → dimension mapping. Used to default ``dimensions`` when the
# caller does not pass it. Add new model strings here as they ship.
_OPENAI_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings:
    """Embedding provider backed by OpenAI's ``/v1/embeddings`` endpoint.

    Args:
        model: Embedding model name. Defaults to ``text-embedding-3-small``
            (1536 dims, cheap, the right call for almost everything).
        dimensions: Optional override. Most callers leave this ``None`` and
            let the model's default dimension apply. If you pass a custom
            value, OpenAI returns a truncated/projected vector of that size
            (only supported on ``-3-small`` and ``-3-large``).
        api_key: Optional explicit key. If ``None``, the OpenAI SDK falls
            back to ``OPENAI_API_KEY`` from the environment.

    Reads ``OPENAI_API_KEY`` from the environment when ``api_key`` is not
    provided. Mneme itself never auto-loads ``.env`` files; do that in your
    application code (or via the test conftest in this repo).
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None = None,
    ) -> None:
        # Lazy import so importing ``mneme.embeddings`` works without the
        # ``openai`` extra installed.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbeddings requires the 'openai' extra. Install with:\n"
                "    pip install 'mneme[openai]'\n"
                "    # or, in a uv-managed project:\n"
                "    uv add mneme --extra openai"
            ) from exc

        self.model = model
        if dimensions is None:
            try:
                dimensions = _OPENAI_MODEL_DIMS[model]
            except KeyError as exc:
                raise ValueError(
                    f"unknown OpenAI embedding model {model!r}; pass dimensions= "
                    "explicitly or use one of "
                    f"{sorted(_OPENAI_MODEL_DIMS)}"
                ) from exc
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.dimensions: int = dimensions

        # Pass api_key=None to OpenAI() — the SDK reads OPENAI_API_KEY itself.
        self._client: OpenAI = OpenAI(api_key=api_key) if api_key else OpenAI()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        # Only pass ``dimensions`` to the API when the caller wanted a
        # non-default vector size — passing it on -ada-002 raises. We branch
        # the call rather than packing kwargs into a dict because mypy can't
        # type-check ``**dict[str, object]`` against the SDK's strict signature.
        send_dimensions = self.dimensions != _OPENAI_MODEL_DIMS.get(self.model)

        out: list[list[float]] = []
        for start in range(0, len(texts), _OPENAI_BATCH):
            chunk = list(texts[start : start + _OPENAI_BATCH])
            if send_dimensions:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=chunk,
                    dimensions=self.dimensions,
                )
            else:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=chunk,
                )
            # response.data is ordered to match the input batch.
            out.extend(item.embedding for item in response.data)
        return out


# ---------------------------------------------------------------------------
# Deterministic test double
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic embedder for tests. No API calls, no model.

    Given the same input, returns the same vector. Different inputs return
    different vectors. **There is no semantic structure** — "cat" and
    "kitten" are no more similar in this space than "cat" and "quantum
    chromodynamics". Use this to test plumbing (does an embedding flow from
    caller to backend correctly?) — never to test retrieval quality.

    Vectors are length-1 (unit-norm) so cosine similarity behaves sanely.

    Args:
        dimensions: Length of each output vector. Default 16 — small enough
            to keep tests fast, large enough that hash collisions in any
            single coordinate don't dominate similarity.
        seed: Hashed in with each input so two HashEmbedders with different
            seeds produce different vector spaces for the same strings.
    """

    def __init__(self, *, dimensions: int = 16, seed: int = 0) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.dimensions: int = dimensions
        self._seed = seed

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(f"{self._seed}::{text}".encode()).digest()
            # Seed a PRNG with the hash so the vector is fully determined by
            # (seed, text). hashlib.sha256().digest() is bytes; Random accepts
            # any hashable, so we pass it directly.
            rng = random.Random(digest)
            vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            out.append(vec)
        return out
