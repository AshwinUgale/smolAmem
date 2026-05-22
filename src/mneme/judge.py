"""LLM-as-judge adapters for Mneme.

Consolidation (and any future LLM-driven task) doesn't talk to the OpenAI SDK
directly — it talks through an :class:`LLMJudge`. The protocol is a pure
*transport*: it knows how to ask a language model for a structured response
and hand back a parsed dict. Prompts live in the caller (the consolidator),
not in the judge.

Two implementations ship:

* :class:`OpenAILLMJudge` — production default. Wraps
  ``openai.OpenAI().chat.completions.create`` with the strict structured-output
  ``response_format``. Requires the ``openai`` extra and ``OPENAI_API_KEY``.
* :class:`MockLLMJudge` — deterministic test double. Takes a handler callable
  that receives the messages + schema and returns whatever dict the test wants.

Adding a third judge (Anthropic, Cohere, a local model) is the same shape as
adding a backend or embedder: implement the protocol, ship it, done.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = ["LLMJudge", "MockLLMJudge", "OpenAILLMJudge"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMJudge(Protocol):
    """A model that returns structured JSON for a given chat prompt.

    Implementations declare:

    * :attr:`model` — informational name used in logs (and exposed so callers
      can record which model produced a fact).
    * :meth:`complete` — single method. Takes OpenAI-style chat messages and a
      JSON schema; returns the parsed JSON as a dict.

    The judge does not own the prompt. Callers compose the messages, hand
    them in, and parse the response from the dict shape they requested.
    """

    model: str

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_schema: Mapping[str, Any],
        temperature: float = 0.0,
        schema_name: str = "Response",
    ) -> dict[str, Any]:
        """Run a chat completion with structured output.

        Args:
            messages: OpenAI-style messages — each ``{"role": "...", "content": "..."}``.
                Mneme uses ``"system"`` and ``"user"`` roles for v0.2; future
                tasks may add ``"assistant"`` for few-shot prompts.
            response_schema: A JSON schema (the inner schema object, not the
                OpenAI ``response_format`` wrapper) describing the expected
                response shape. Implementations should enforce it strictly —
                use ``additionalProperties: false`` and explicit ``required``.
            temperature: Sampling temperature. Default 0.0 — fact extraction
                wants deterministic, low-variance output.
            schema_name: Human-readable name for the schema. Surfaced in
                OpenAI's API and useful for logs / debugging.

        Returns:
            The parsed JSON response as a dict. Validation against the schema
            is the implementation's responsibility (most APIs that accept JSON
            schemas enforce it server-side; the dict you get back conforms).
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAILLMJudge:
    """LLM judge backed by OpenAI's chat completions with JSON schema response.

    Args:
        model: Chat model name. Defaults to ``gpt-4o-mini`` — cheap, fast, good
            enough for consolidation. Switch to ``gpt-4o`` when the quality
            differential matters for your eval.
        api_key: Optional explicit key. If ``None``, the OpenAI SDK falls back
            to ``OPENAI_API_KEY`` from the environment.

    Uses ``response_format={"type": "json_schema", ...}`` with ``strict: True``
    so the model is constrained at decode time to produce schema-valid output.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        # Lazy import so importing mneme.judge works without the openai extra.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAILLMJudge requires the 'openai' extra. Install with:\n"
                "    pip install 'mneme[openai]'\n"
                "    # or, in a uv-managed project:\n"
                "    uv add mneme --extra openai"
            ) from exc

        self.model = model
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_schema: Mapping[str, Any],
        temperature: float = 0.0,
        schema_name: str = "Response",
    ) -> dict[str, Any]:
        # The SDK's typed ChatCompletionMessageParam + ResponseFormatJSONSchema
        # would tighten this up, but our public protocol intentionally exposes
        # plain dicts so non-OpenAI judges can satisfy it without importing
        # openai. We pay the cost with one call-overload ignore at the boundary.
        response = self._client.chat.completions.create(  # type: ignore[call-overload]
            model=self.model,
            messages=[dict(m) for m in messages],
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": dict(response_schema),
                    "strict": True,
                },
            },
        )

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                f"OpenAI returned no content for {schema_name!r}; "
                "finish_reason="
                f"{response.choices[0].finish_reason!r}"
            )
        parsed: dict[str, Any] = json.loads(content)
        return parsed


# ---------------------------------------------------------------------------
# Mock test double
# ---------------------------------------------------------------------------


# Type alias for the handler signature MockLLMJudge accepts.
MockJudgeHandler = Callable[..., dict[str, Any]]


class MockLLMJudge:
    """Deterministic LLM judge for tests. No API calls.

    Construct with a ``handler`` callable that receives the same kwargs
    :meth:`complete` was called with and returns the dict the judge should
    return. This lets tests inspect prompts and inject whatever structured
    response the test needs without any real model in the loop.

    Args:
        handler: Callable taking ``(*, messages, response_schema, temperature,
            schema_name)`` and returning a dict.
        model: Informational model name. Defaults to ``"mock"``.
    """

    def __init__(
        self,
        *,
        handler: MockJudgeHandler,
        model: str = "mock",
    ) -> None:
        self._handler = handler
        self.model = model

        # Call log so tests can assert on what the judge saw.
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_schema: Mapping[str, Any],
        temperature: float = 0.0,
        schema_name: str = "Response",
    ) -> dict[str, Any]:
        call = {
            "messages": [dict(m) for m in messages],
            "response_schema": dict(response_schema),
            "temperature": temperature,
            "schema_name": schema_name,
        }
        self.calls.append(call)
        return self._handler(**call)
