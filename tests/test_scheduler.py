"""Tests for ``MemoryManager.start_scheduler`` / ``stop_scheduler``.

These tests focus on **wiring**: that start/stop are idempotent, that
preconditions raise, that the right jobs get registered with the right
intervals. We don't try to assert that jobs *actually fire on schedule* —
that's APScheduler's job and testing it well would require time.sleep
loops that bloat the suite for no real value. If the wiring is right
and APScheduler does what it says on the tin, the scheduled cadence
works in production.

If the ``[scheduler]`` extra isn't installed, the whole file skips
cleanly so a contributor without that dep can still run ``uv run pytest``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from mneme import (
    HashEmbedder,
    InMemoryBackend,
    MemoryManager,
    MockLLMJudge,
)

# Skip the entire module if APScheduler isn't available.
apscheduler = pytest.importorskip("apscheduler.schedulers.background")
_BackgroundScheduler = apscheduler.BackgroundScheduler


def _no_facts(**_: Any) -> dict[str, Any]:
    return {"facts": []}


def _make_manager(*, with_judge: bool = True) -> MemoryManager:
    judge = MockLLMJudge(handler=_no_facts) if with_judge else None
    return MemoryManager(
        agent_id="alice",
        backend=InMemoryBackend(),
        embedder=HashEmbedder(dimensions=8),
        llm_judge=judge,
    )


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_consolidation_without_judge_raises():
    m = _make_manager(with_judge=False)
    try:
        with pytest.raises(RuntimeError, match="llm_judge"):
            m.start_scheduler(consolidate_every=timedelta(minutes=30))
    finally:
        m.stop_scheduler()


def test_can_skip_consolidation_with_no_judge():
    """A manager without a judge can still run a forget-only scheduler."""
    m = _make_manager(with_judge=False)
    try:
        m.start_scheduler(
            consolidate_every=None,
            forget_every=timedelta(days=1),
        )
    finally:
        m.stop_scheduler()


def test_starting_twice_raises():
    m = _make_manager()
    try:
        m.start_scheduler()
        with pytest.raises(RuntimeError, match="already running"):
            m.start_scheduler()
    finally:
        m.stop_scheduler()


def test_stop_without_start_is_noop():
    m = _make_manager()
    m.stop_scheduler()  # should not raise


def test_stop_twice_is_noop():
    m = _make_manager()
    m.start_scheduler()
    m.stop_scheduler()
    m.stop_scheduler()  # second call must be safe


# ---------------------------------------------------------------------------
# Wiring — right jobs at right intervals
# ---------------------------------------------------------------------------


def test_both_jobs_registered_by_default():
    m = _make_manager()
    try:
        m.start_scheduler()
        jobs = m._scheduler.get_jobs()
        job_ids = {j.id for j in jobs}
        assert "mneme_consolidate" in job_ids
        assert "mneme_forget" in job_ids
    finally:
        m.stop_scheduler()


def test_consolidate_every_none_skips_that_job():
    m = _make_manager()
    try:
        m.start_scheduler(consolidate_every=None)
        job_ids = {j.id for j in m._scheduler.get_jobs()}
        assert "mneme_consolidate" not in job_ids
        assert "mneme_forget" in job_ids
    finally:
        m.stop_scheduler()


def test_forget_every_none_skips_that_job():
    m = _make_manager()
    try:
        m.start_scheduler(forget_every=None)
        job_ids = {j.id for j in m._scheduler.get_jobs()}
        assert "mneme_consolidate" in job_ids
        assert "mneme_forget" not in job_ids
    finally:
        m.stop_scheduler()


def test_intervals_match_caller_values():
    m = _make_manager()
    try:
        m.start_scheduler(
            consolidate_every=timedelta(minutes=15),
            forget_every=timedelta(hours=6),
        )
        jobs_by_id = {j.id: j for j in m._scheduler.get_jobs()}

        # APScheduler IntervalTrigger exposes ``interval`` as a timedelta.
        consolidate = jobs_by_id["mneme_consolidate"]
        forget = jobs_by_id["mneme_forget"]
        assert consolidate.trigger.interval == timedelta(minutes=15)
        assert forget.trigger.interval == timedelta(hours=6)
    finally:
        m.stop_scheduler()


def test_scheduler_is_running_after_start():
    m = _make_manager()
    try:
        m.start_scheduler()
        assert m._scheduler is not None
        assert m._scheduler.running is True
    finally:
        m.stop_scheduler()


def test_scheduler_attribute_resets_to_none_after_stop():
    m = _make_manager()
    m.start_scheduler()
    m.stop_scheduler()
    assert m._scheduler is None


# ---------------------------------------------------------------------------
# Default cadence
# ---------------------------------------------------------------------------


def test_default_cadence_30m_consolidate_1d_forget():
    m = _make_manager()
    try:
        m.start_scheduler()
        jobs_by_id = {j.id: j for j in m._scheduler.get_jobs()}
        assert jobs_by_id["mneme_consolidate"].trigger.interval == timedelta(minutes=30)
        assert jobs_by_id["mneme_forget"].trigger.interval == timedelta(days=1)
    finally:
        m.stop_scheduler()
