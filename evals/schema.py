"""Corpus schema.

A *corpus* is a directory of JSON conversations. Each conversation has an
``id``, a ``description`` (one sentence explaining what's being tested),
and a list of ``turns``. Turns are normal chat turns OR test points;
test points are turns where we expect the memory layer to have something
specific to surface.

Schema versioning: ``schema_version`` is checked-in to every conversation.
v0.6 is ``"1"``. Future incompatible changes bump it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TestPoint(BaseModel):
    """A turn where the eval measures whether the memory layer remembered.

    Attached to a single ``Turn``. At eval time, the runner pauses just
    before this turn is added to memory and runs::

        results = manager.retrieve(test_point.question, k=k)

    Then evaluates:
    * **recall@k**: does ``expected_fact`` (or any keyword in
      ``expected_keywords``) appear in the top-k retrieved records?
    * **accuracy** (opt-in, requires real LLM): generates an answer to
      ``question`` using the memory layer's context and checks whether
      ``expected_keywords`` appear in the answer.
    """

    question: str = Field(
        description=(
            "The query to evaluate retrieval against. Phrased the way a "
            "user/agent would ask, not the way the fact is stored."
        )
    )
    expected_fact: str = Field(
        description=(
            "The canonical fact the memory layer SHOULD surface. Used "
            "for fuzzy keyword matching against retrieved records."
        )
    )
    expected_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Substring tokens that should appear in the retrieved "
            "record's content OR in the agent's answer (for accuracy "
            "metrics). Case-insensitive."
        ),
    )


class Turn(BaseModel):
    """One chat turn in a conversation."""

    role: str = Field(description="user / assistant / system")
    content: str
    test_at: TestPoint | None = Field(
        default=None,
        description=(
            "If set, this turn is a test point — we run retrieval at "
            "this position BEFORE adding the turn to memory."
        ),
    )


class Conversation(BaseModel):
    """One labelled conversation in the corpus."""

    schema_version: str = Field(
        default="1",
        description="Bump when the on-disk format changes incompatibly.",
    )
    id: str = Field(
        description=(
            "Short slug, used as the filename stem and as a JSON key in aggregate results."
        )
    )
    description: str = Field(
        description=(
            "One sentence: what is this conversation testing? "
            "E.g. 'User establishes TypeScript preference in turn 3; "
            "tested at turn 30 after distractor turns.'"
        )
    )
    turns: list[Turn]

    def test_points(self) -> list[tuple[int, TestPoint]]:
        """Return ``[(turn_index, TestPoint), ...]`` for every test turn."""
        return [(i, t.test_at) for i, t in enumerate(self.turns) if t.test_at is not None]
