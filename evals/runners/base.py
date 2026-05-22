"""``EvalRunner`` — plays a corpus through a :class:`MemoryStrategy`.

For each conversation in the corpus:

1. ``strategy.reset()`` — start fresh.
2. Walk the turns in order. Each turn:

   a. If the turn has a ``test_at`` test point, BEFORE adding it,
      run ``strategy.retrieve_records(test_point.question)`` and
      record the result for recall@k measurement.
   b. ``strategy.add(turn.role, turn.content)`` — feed the turn in.

3. After the conversation, record per-conversation stats.

The runner is intentionally simple — it does the orchestration and
record-keeping; metric computation is a separate module so different
metrics can be added without touching the runner.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evals.baselines.base import MemoryStrategy
from evals.schema import Conversation


@dataclass
class TestPointResult:
    """Recorded data from one test point during the run."""

    conversation_id: str
    turn_index: int
    question: str
    expected_fact: str
    expected_keywords: list[str]
    retrieved_records: list[str]


@dataclass
class ConversationResult:
    """Per-conversation summary."""

    conversation_id: str
    description: str
    turn_count: int
    test_points: list[TestPointResult] = field(default_factory=list)


@dataclass
class RunResult:
    """Top-level eval result — one file's worth of output."""

    strategy: dict[str, Any]
    seed: int
    conversations: list[ConversationResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def load_corpus(corpus_dir: Path) -> list[Conversation]:
    """Read every ``*.json`` in ``corpus_dir`` and parse as Conversation."""
    files = sorted(p for p in corpus_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no *.json files found in {corpus_dir}")
    out: list[Conversation] = []
    for path in files:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        out.append(Conversation(**data))
    return out


class EvalRunner:
    """Orchestrates the play-through of a corpus."""

    def __init__(
        self,
        *,
        strategy: MemoryStrategy,
        k: int = 5,
        seed: int = 42,
    ) -> None:
        self._strategy = strategy
        self._k = k
        self._seed = seed

    def run(self, conversations: Iterable[Conversation]) -> RunResult:
        run_result = RunResult(
            strategy=self._strategy.config_summary(),
            seed=self._seed,
        )
        for convo in conversations:
            self._strategy.reset()
            convo_result = ConversationResult(
                conversation_id=convo.id,
                description=convo.description,
                turn_count=len(convo.turns),
            )
            for turn_idx, turn in enumerate(convo.turns):
                # Test point happens BEFORE the turn is added — we measure
                # what the memory layer remembers from the conversation
                # *up to* this point, not including the current question.
                if turn.test_at is not None:
                    retrieved = self._strategy.retrieve_records(turn.test_at.question, k=self._k)
                    convo_result.test_points.append(
                        TestPointResult(
                            conversation_id=convo.id,
                            turn_index=turn_idx,
                            question=turn.test_at.question,
                            expected_fact=turn.test_at.expected_fact,
                            expected_keywords=list(turn.test_at.expected_keywords),
                            retrieved_records=list(retrieved),
                        )
                    )
                self._strategy.add(turn.role, turn.content)
            run_result.conversations.append(convo_result)
        return run_result
