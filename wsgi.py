"""Gunicorn entry point for the review dashboard."""

from __future__ import annotations

import os

from review_dashboard import create_app


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


_output_dir = _env("PROCTORING_OUTPUT_DIR", "/var/lib/proctoring/live")

app = create_app(
    results=_env("PROCTORING_RESULTS_DIR", _output_dir),
    snapshots=_env("PROCTORING_SNAPSHOTS_DIR", _output_dir),
    live_db=_env("PROCTORING_LIVE_DB", f"{_output_dir}/live_results.sqlite3") or None,
    review_db=_env("PROCTORING_REVIEW_DB") or None,
)
