"""Tests for the LLM judge module.

Protocol-satisfaction is parametrised over every shipped judge (MockLLMJudge
always; OpenAILLMJudge only when ``OPENAI_API_KEY`` is set). Judge-specific
behaviour gets dedicated tests below.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from mneme import LLMJudge, MockLLMJudge, OpenAILLMJudge


def _has_real_openai_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key.startswith("sk-") or len(key) < 40:
        return False
    return "your" not in key.lower() and "..." not in key


_HAS_OPENAI_KEY = _has_real_openai_key()


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def _every_judge() -> list[LLMJudge]:
    def _stub_handler(**_: Any) -> dict[str, Any]:
        return {"ok": True}

    judges: list[LLMJudge] = [MockLLMJudge(handler=_stub_handler)]
    if _HAS_OPENAI_KEY:
        judges.append(OpenAILLMJudge())
    return judges


@pytest.fixture(params=_every_judge(), ids=lambda j: type(j).__name__)
def judge(request: pytest.FixtureRequest) -> LLMJudge:
    return request.param  # type: ignore[no-any-return]


def test_judge_satisfies_protocol(judge: LLMJudge):
    assert isinstance(judge, LLMJudge)
    assert isinstance(judge.model, str)
    assert judge.model != ""


# ---------------------------------------------------------------------------
# MockLLMJudge — handler routing + call log
# ---------------------------------------------------------------------------


def test_mock_judge_calls_handler_with_request_kwargs():
    captured: dict[str, Any] = {}

    def handler(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"echo": True}

    judge = MockLLMJudge(handler=handler)
    out = judge.complete(
        messages=[{"role": "user", "content": "hi"}],
        response_schema={"type": "object"},
        temperature=0.3,
        schema_name="MySchema",
    )
    assert out == {"echo": True}
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["response_schema"] == {"type": "object"}
    assert captured["temperature"] == 0.3
    assert captured["schema_name"] == "MySchema"


def test_mock_judge_records_calls():
    judge = MockLLMJudge(handler=lambda **_: {})
    for i in range(3):
        judge.complete(
            messages=[{"role": "user", "content": f"call-{i}"}],
            response_schema={},
        )
    assert len(judge.calls) == 3
    assert judge.calls[1]["messages"][0]["content"] == "call-1"


def test_mock_judge_default_model_is_mock():
    assert MockLLMJudge(handler=lambda **_: {}).model == "mock"


def test_mock_judge_custom_model_name():
    judge = MockLLMJudge(handler=lambda **_: {}, model="gpt-fake-9999")
    assert judge.model == "gpt-fake-9999"


# ---------------------------------------------------------------------------
# OpenAILLMJudge — real-API tests (skipped without a key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_judge_returns_structured_dict():
    """A real chat completion with structured output should round-trip cleanly."""
    judge = OpenAILLMJudge()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "greeting": {"type": "string"},
            "language": {"type": "string"},
        },
        "required": ["greeting", "language"],
    }
    response = judge.complete(
        messages=[
            {"role": "system", "content": "Reply with a greeting in JSON."},
            {"role": "user", "content": "Hello"},
        ],
        response_schema=schema,
        schema_name="Greeting",
    )
    assert isinstance(response, dict)
    assert isinstance(response["greeting"], str)
    assert isinstance(response["language"], str)


def test_openai_judge_default_model():
    """Constructor doesn't hit the API — safe with a placeholder key."""
    judge = OpenAILLMJudge(api_key="sk-test")
    assert judge.model == "gpt-4o-mini"


def test_openai_judge_custom_model():
    judge = OpenAILLMJudge(model="gpt-4o", api_key="sk-test")
    assert judge.model == "gpt-4o"


# ---------------------------------------------------------------------------
# Schema-name passthrough (mock end of the test)
# ---------------------------------------------------------------------------


def test_mock_judge_can_return_json_serialisable_complex_shapes():
    """Sanity check that MockLLMJudge handles realistic consolidation outputs."""
    payload = {
        "facts": [
            {
                "content": "user uses TypeScript",
                "confidence": 0.9,
                "source_episode_ids": ["ep_1", "ep_2"],
            }
        ]
    }
    judge = MockLLMJudge(handler=lambda **_: payload)
    result = judge.complete(
        messages=[{"role": "system", "content": "..."}],
        response_schema={},
    )
    # Round-tripping through JSON proves it's serialisable.
    assert json.loads(json.dumps(result)) == payload
