"""Pytest configuration.

Loads ``.env`` at the repo root so tests that need ``OPENAI_API_KEY`` (or
any other secret) can pick it up locally without exporting variables in the
shell first. Library code never touches dotenv — that's a per-application
choice and would be a surprising side-effect on import.

If ``.env`` is missing, malformed, or written in the wrong encoding (a
common PowerShell ``>`` redirection mistake on Windows), we warn rather
than crash the whole test session. Tests that need the key are
``@pytest.mark.skipif``-guarded and will just skip.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from dotenv import load_dotenv

# Repo root is two levels up from this file (tests/conftest.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _REPO_ROOT / ".env"

try:
    # ``utf-8-sig`` handles a UTF-8 file with a leading BOM (notepad's
    # default save). UTF-16 files (PowerShell ``>`` redirection on
    # Windows) still fail — we catch that below.
    load_dotenv(_ENV_PATH, override=False, encoding="utf-8-sig")
except UnicodeDecodeError:
    warnings.warn(
        f"{_ENV_PATH} exists but is not UTF-8 (likely UTF-16 from "
        "PowerShell `>` redirection on Windows). Re-save it as UTF-8 — "
        "for example:\n"
        "    Set-Content -Encoding utf8 -Path .env -Value "
        "'OPENAI_API_KEY=sk-...'\n"
        "Continuing without loading .env; tests that need keys will skip.",
        stacklevel=1,
    )
