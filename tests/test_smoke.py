"""Smoke test — proves the package imports and the test harness works.

This is the dumbest possible test on purpose. The point at this stage is to
verify that:
    1. `uv sync` installed the package in editable mode correctly,
    2. pytest can discover and run tests against installed code,
    3. our src/ layout works without sys.path hacks.

Real tests start arriving once we have actual functions to test.
"""

import mneme


def test_version_is_a_string() -> None:
    assert isinstance(mneme.__version__, str)
    assert mneme.__version__ != ""
