"""Tests for the embedding adapters.

The protocol-satisfaction tests are parametrised over every shipped
embedder (currently HashEmbedder; OpenAIEmbeddings is added when a key
is present so we never fail just because the user hasn't set one).

Provider-specific behaviour gets its own dedicated test below.
"""

from __future__ import annotations

import math
import os

import pytest

from mneme import EmbeddingProvider, HashEmbedder, OpenAIEmbeddings


def _has_real_openai_key() -> bool:
    """True only when the env var looks like an actual OpenAI key, not a placeholder.

    Real OpenAI keys start with ``sk-`` and are at least 40 characters long.
    Common placeholders (``sk-...``, ``sk-your-real-key-here``, the literal
    value from ``.env.example``) get filtered out so the integration tests
    skip cleanly instead of producing a confusing 401.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key.startswith("sk-") or len(key) < 40:
        return False
    # Filter common placeholder values (".env.example" literals, copy-paste
    # remnants like sk-...).
    return "your" not in key.lower() and "..." not in key


_HAS_OPENAI_KEY = _has_real_openai_key()


# ---------------------------------------------------------------------------
# Protocol satisfaction (shared across all embedders)
# ---------------------------------------------------------------------------


def _every_embedder() -> list[EmbeddingProvider]:
    """Return one instance of each embedder we want to conformance-test."""
    embedders: list[EmbeddingProvider] = [HashEmbedder(dimensions=8)]
    if _HAS_OPENAI_KEY:
        embedders.append(OpenAIEmbeddings())
    return embedders


@pytest.fixture(params=_every_embedder(), ids=lambda e: type(e).__name__)
def embedder(request: pytest.FixtureRequest) -> EmbeddingProvider:
    return request.param  # type: ignore[no-any-return]


def test_embedder_satisfies_protocol(embedder: EmbeddingProvider):
    # @runtime_checkable on EmbeddingProvider enables this check.
    assert isinstance(embedder, EmbeddingProvider)
    assert embedder.dimensions > 0


def test_embed_returns_one_vector_per_input(embedder: EmbeddingProvider):
    out = embedder.embed(["foo", "bar", "baz"])
    assert len(out) == 3
    assert all(len(v) == embedder.dimensions for v in out)
    assert all(isinstance(x, float) for v in out for x in v)


def test_embed_empty_input_returns_empty(embedder: EmbeddingProvider):
    assert embedder.embed([]) == []


def test_embed_preserves_order(embedder: EmbeddingProvider):
    # Embedding "x" then "y" must produce vectors in the same order.
    # OpenAI returns vectors that differ by ~1e-5 between single-input
    # and batched calls (internal kernel selection / quantisation), so
    # the tolerance has to accommodate that. HashEmbedder is exact.
    one = embedder.embed(["alpha"])[0]
    pair = embedder.embed(["alpha", "beta"])
    assert pair[0] == one or _allclose(pair[0], one, tol=1e-3)


# ---------------------------------------------------------------------------
# HashEmbedder specifics
# ---------------------------------------------------------------------------


def test_hash_embedder_is_deterministic():
    e = HashEmbedder(dimensions=8, seed=42)
    assert e.embed(["hello"]) == e.embed(["hello"])


def test_hash_embedder_different_texts_give_different_vectors():
    e = HashEmbedder(dimensions=8)
    a, b = e.embed(["hello", "world"])
    assert a != b


def test_hash_embedder_different_seeds_give_different_spaces():
    a = HashEmbedder(dimensions=8, seed=1)
    b = HashEmbedder(dimensions=8, seed=2)
    assert a.embed(["hello"]) != b.embed(["hello"])


def test_hash_embedder_dimensions_respected():
    e = HashEmbedder(dimensions=64)
    [vec] = e.embed(["x"])
    assert len(vec) == 64


def test_hash_embedder_vectors_are_unit_norm():
    e = HashEmbedder(dimensions=16)
    [vec] = e.embed(["unit"])
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-9)


def test_hash_embedder_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="positive"):
        HashEmbedder(dimensions=0)


# ---------------------------------------------------------------------------
# OpenAIEmbeddings specifics (real-API; skipped without a key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_default_model_dimensions():
    e = OpenAIEmbeddings()
    assert e.model == "text-embedding-3-small"
    assert e.dimensions == 1536


@pytest.mark.skipif(not _HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_embeds_short_batch():
    e = OpenAIEmbeddings()
    out = e.embed(["hello", "world"])
    assert len(out) == 2
    assert len(out[0]) == 1536
    # OpenAI returns normalised vectors — a useful sanity check.
    norm0 = math.sqrt(sum(x * x for x in out[0]))
    assert math.isclose(norm0, 1.0, abs_tol=1e-3)


def test_openai_rejects_unknown_model_without_explicit_dimensions():
    # No API call needed — the dimension lookup raises before we instantiate
    # the OpenAI client. Pass a placeholder api_key so we don't depend on env.
    with pytest.raises(ValueError, match="unknown OpenAI embedding model"):
        OpenAIEmbeddings(model="not-a-real-model", api_key="sk-test")


def test_openai_accepts_unknown_model_with_explicit_dimensions():
    # Proves the constructor branches correctly. No API call is made; the
    # OpenAI SDK client is built but never used.
    e = OpenAIEmbeddings(
        model="hypothetical-future-model",
        dimensions=512,
        api_key="sk-test",
    )
    assert e.dimensions == 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allclose(a: list[float], b: list[float], tol: float = 1e-6) -> bool:
    return len(a) == len(b) and all(
        math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b, strict=True)
    )
