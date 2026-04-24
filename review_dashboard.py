"""
review_dashboard.py
===================
Local web dashboard for reviewing exam proctoring results.

Usage:
    python review_dashboard.py --results ./real_results --snapshots ./real_test_data
    python review_dashboard.py --results ./results --snapshots ./test_data

Then open:  http://127.0.0.1:5000

Reads:
  <results>/student_summary.csv
  <results>/image_level_results.csv

Shows:
  /                        Overview table: all students, risk level, stats
  /student/<student_id>    Per-student timeline, risk chart, frame evidence cards
  /snapshot/<path>         Serves a snapshot image (for cards)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
from pathlib import Path
import shutil
from typing import Any

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for  # type: ignore

from data_io.live_result_store import LiveResultStore
from data_io.review_label_store import ReviewLabelStore

app = Flask(__name__)
log = logging.getLogger(__name__)

# Populated by main() before app.run() – always absolute paths
RESULTS_DIR: Path = Path("results").resolve()
SNAPSHOTS_DIR: Path = Path("test_data").resolve()
LIVE_STORE: LiveResultStore | None = None
REVIEW_STORE: ReviewLabelStore | None = None
LIVE_MODE: bool = False

REVIEW_LABEL_OPTIONS = [
    {"id": "LOOK_AWAY", "label": "Looking Away", "badge": "secondary"},
    {"id": "EXTRA_PERSON", "label": "Extra Person", "badge": "danger"},
    {"id": "PHONE", "label": "Phone", "badge": "warning"},
    {"id": "MULTI_FACE", "label": "Multiple Faces", "badge": "danger"},
    {"id": "FACE_HIDDEN", "label": "Face Hidden", "badge": "danger"},
    {"id": "NO_FACE", "label": "No Face", "badge": "dark"},
    {"id": "TALKING", "label": "Talking", "badge": "warning"},
    {"id": "BOOK_NOTES", "label": "Book or Notes", "badge": "warning"},
    {"id": "LOW_QUALITY", "label": "Low Quality", "badge": "info"},
    {"id": "FALSE_POSITIVE", "label": "False Positive", "badge": "success"},
]
REVIEW_LABEL_IDS = {item["id"] for item in REVIEW_LABEL_OPTIONS}
API_FLAG_OPTIONS = [
    "LOOK_AWAY",
    "TALKING",
    "PHONE",
    "EXTRA_PERSON",
    "MULTI_FACE",
    "NO_FACE",
    "FACE_HIDDEN",
    "BOOK_NOTES",
    "LOW_QUALITY",
    "FACE_OBSTRUCTED",
    "IDENTITY_MISMATCH",
]
API_DEFAULT_SUSPICIOUS_THRESHOLD = 30.0
SWAGGER_UI_CSS_URL = "https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"
SWAGGER_UI_BUNDLE_URL = "https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_student_summary() -> list[dict]:
    if LIVE_STORE is not None:
        return LIVE_STORE.fetch_summaries()

    path = RESULTS_DIR / "student_summary.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Parse JSON-in-CSV columns
    for r in rows:
        for key in ("top_reasons", "flagged_timeline", "incidents", "question_overview"):
            if key in r and r[key]:
                try:
                    r[key] = json.loads(r[key])
                except Exception:
                    pass
        for key in ("total_frames", "valid_frames", "suspicious_frames",
                    "incident_count"):
            if key in r:
                try:
                    r[key] = int(r[key])
                except Exception:
                    pass
        for key in ("percentage_suspicious", "max_risk_score", "mean_risk_score",
                    "identity_stability_score"):
            if key in r:
                try:
                    r[key] = float(r[key])
                except Exception:
                    pass
    return rows


def load_image_results() -> list[dict]:
    if LIVE_STORE is not None:
        return LIVE_STORE.fetch_frame_dicts()

    path = RESULTS_DIR / "image_level_results.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("reasons",):
            if key in r and r[key]:
                try:
                    r[key] = json.loads(r[key])
                except Exception:
                    r[key] = []
        for key in ("risk_score", "yaw", "pitch", "roll", "blur_score",
                    "brightness_score", "glare_score", "identity_similarity",
                    "talking_confidence", "mouth_open_ratio", "mouth_open_delta"):
            if key in r:
                val = r[key]
                if val is None or str(val).strip() == "":
                    r[key] = None
                else:
                    try:
                        r[key] = float(val)
                    except (ValueError, TypeError):
                        r[key] = None
        for key in ("face_count", "source_log_id"):
            if key in r and r[key] != "":
                try:
                    r[key] = int(r[key])
                except Exception:
                    pass
        for key in ("look_away_flag", "talking_flag", "identity_mismatch", "phone_detected",
                    "extra_person_detected", "book_detected",
                    "face_obstructed", "person_detected", "low_quality"):
            if key in r:
                r[key] = r[key].strip().lower() == "true"
    return rows


def risk_badge_class(level: str) -> str:
    return {"high": "danger", "medium": "warning", "low": "success"}.get(
        level.lower(), "secondary"
    )


def reason_badge_class(reason: str) -> str:
    mapping = {
        "NO_FACE": "dark",
        "FACE_HIDDEN": "danger",
        "MULTI_FACE": "danger",
        "IDENTITY_MISMATCH": "danger",
        "PHONE": "warning",
        "EXTRA_PERSON": "danger",
        "BOOK_NOTES": "warning",
        "FACE_OBSTRUCTED": "warning",
        "TALKING_POSSIBLE": "warning",
        "TALKING_LIKELY": "danger",
        "LOOK_AWAY_SEVERE": "warning",
        "LOOK_AWAY_MODERATE": "secondary",
        "LOW_QUALITY": "info",
    }
    return mapping.get(reason.upper(), "secondary")


def review_badge_class(label: str) -> str:
    for option in REVIEW_LABEL_OPTIONS:
        if option["id"] == label:
            return option["badge"]
    return "secondary"


def frame_review_key(frame: dict) -> str:
    source_log_id = frame.get("source_log_id")
    if source_log_id not in (None, "", 0):
        try:
            return ReviewLabelStore.build_frame_key(
                student_id=str(frame.get("student_id", "")),
                attempt_id=str(frame.get("attempt_id", "")),
                image_path=str(frame.get("image_path", "")),
                source_log_id=int(source_log_id),
            )
        except Exception:
            pass
    return ReviewLabelStore.build_frame_key(
        student_id=str(frame.get("student_id", "")),
        attempt_id=str(frame.get("attempt_id", "")),
        image_path=str(frame.get("image_path", "")),
    )


def default_review_db_path(results_dir: Path) -> Path:
    return Path("review_data") / results_dir.name / "review_labels.sqlite3"


def configure_dashboard(
    *,
    results: str | Path = "results",
    snapshots: str | Path = "test_data",
    live_db: str | Path | None = None,
    review_db: str | Path | None = None,
) -> None:
    """Configure global dashboard paths for CLI and WSGI entry points."""
    global RESULTS_DIR, SNAPSHOTS_DIR, LIVE_STORE, REVIEW_STORE, LIVE_MODE

    RESULTS_DIR = Path(results).resolve()
    SNAPSHOTS_DIR = Path(snapshots).resolve()
    LIVE_MODE = bool(live_db)
    LIVE_STORE = LiveResultStore(live_db) if live_db else None
    if LIVE_STORE is not None:
        RESULTS_DIR = Path(live_db).resolve().parent

    if review_db:
        review_db_path = Path(review_db).resolve()
    else:
        review_db_path = default_review_db_path(RESULTS_DIR).resolve()
        legacy_review_db = RESULTS_DIR / "review_labels.sqlite3"
        if not review_db_path.exists() and legacy_review_db.exists():
            review_db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_review_db, review_db_path)
    REVIEW_STORE = ReviewLabelStore(review_db_path)

    if LIVE_STORE is None and not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")


def create_app(
    *,
    results: str | Path = "results",
    snapshots: str | Path = "test_data",
    live_db: str | Path | None = None,
    review_db: str | Path | None = None,
) -> Flask:
    """WSGI factory used by Gunicorn/systemd deployments."""
    configure_dashboard(
        results=results,
        snapshots=snapshots,
        live_db=live_db,
        review_db=review_db,
    )
    return app


def _openapi_parameter(
    name: str,
    description: str,
    *,
    schema: dict[str, Any] | None = None,
    style: str | None = None,
    explode: bool | None = None,
) -> dict[str, Any]:
    parameter: dict[str, Any] = {
        "name": name,
        "in": "query",
        "required": False,
        "description": description,
        "schema": schema or {"type": "string"},
    }
    if style is not None:
        parameter["style"] = style
    if explode is not None:
        parameter["explode"] = explode
    return parameter


def build_openapi_spec() -> dict[str, Any]:
    results_query_parameters = [
        _openapi_parameter("course_id", "Filter by one or more course IDs.", schema={"type": "array", "items": {"type": "string"}}, style="form", explode=True),
        _openapi_parameter("quiz_id", "Filter by one or more quiz IDs.", schema={"type": "array", "items": {"type": "string"}}, style="form", explode=True),
        _openapi_parameter("student_id", "Filter by one or more student IDs.", schema={"type": "array", "items": {"type": "string"}}, style="form", explode=True),
        _openapi_parameter("attempt_id", "Filter by one or more attempt IDs.", schema={"type": "array", "items": {"type": "string"}}, style="form", explode=True),
        _openapi_parameter("question_id", "Filter by Moodle question ID.", schema={"type": "array", "items": {"type": "string"}}, style="form", explode=True),
        _openapi_parameter("question_slot", "Filter by Moodle question slot.", schema={"type": "array", "items": {"type": "string"}}, style="form", explode=True),
        _openapi_parameter("question_label", "Filter by question label text.", schema={"type": "array", "items": {"type": "string"}}, style="form", explode=True),
        _openapi_parameter("quiz_page", "Filter by quiz page number.", schema={"type": "array", "items": {"type": "string"}}, style="form", explode=True),
        _openapi_parameter("risk_level", "Filter attempts by overall risk level.", schema={"type": "array", "items": {"type": "string", "enum": ["low", "medium", "high"]}}, style="form", explode=True),
        _openapi_parameter("min_risk", "Minimum frame risk score.", schema={"type": "number"}),
        _openapi_parameter("max_risk", "Maximum frame risk score.", schema={"type": "number"}),
        _openapi_parameter("flag", "Filter by detected analysis flags.", schema={"type": "array", "items": {"type": "string", "enum": API_FLAG_OPTIONS}}, style="form", explode=True),
        _openapi_parameter("review_label", "Filter by human review labels.", schema={"type": "array", "items": {"type": "string", "enum": sorted(REVIEW_LABEL_IDS)}}, style="form", explode=True),
        _openapi_parameter("flag_mode", "Use 'any' or 'all' matching for flag/review_label filters.", schema={"type": "string", "enum": ["any", "all"], "default": "any"}),
        _openapi_parameter("search", "Free-text search across student/exam/question/reasons/review notes."),
        _openapi_parameter("start", "Include only frames with timestamp >= this value.", schema={"type": "string", "example": "2026-04-21T10:00:00"}),
        _openapi_parameter("end", "Include only frames with timestamp <= this value.", schema={"type": "string", "example": "2026-04-21T10:30:00"}),
        _openapi_parameter("only_suspicious", "When true, only return frames with risk >= suspicious threshold.", schema={"type": "boolean"}),
        _openapi_parameter("has_review_label", "When true/false, require the frame to have or not have review labels/notes.", schema={"type": "boolean"}),
        _openapi_parameter("has_error", "When true/false, require the frame to have or not have an analysis error.", schema={"type": "boolean"}),
        _openapi_parameter("include_frames", "Include frame payloads in the response.", schema={"type": "boolean", "default": True}),
        _openapi_parameter("include_attempts", "Include attempt summary payloads in the response.", schema={"type": "boolean", "default": True}),
        _openapi_parameter("frame_limit", "Maximum number of frames returned. Use 0 for no limit.", schema={"type": "integer", "default": 250, "minimum": 0}),
        _openapi_parameter("frame_offset", "Frame pagination offset.", schema={"type": "integer", "default": 0, "minimum": 0}),
        _openapi_parameter("attempt_limit", "Maximum number of attempts returned. Use 0 for no limit.", schema={"type": "integer", "default": 100, "minimum": 0}),
        _openapi_parameter("attempt_offset", "Attempt pagination offset.", schema={"type": "integer", "default": 0, "minimum": 0}),
        _openapi_parameter("sort_by", "Frame sort field.", schema={"type": "string", "enum": ["timestamp", "risk_score"], "default": "timestamp"}),
        _openapi_parameter("sort_order", "Frame sort direction.", schema={"type": "string", "enum": ["asc", "desc"], "default": "desc"}),
    ]

    results_filter_properties = {
        "course_id": {"type": "array", "items": {"type": "string"}},
        "quiz_id": {"type": "array", "items": {"type": "string"}},
        "student_id": {"type": "array", "items": {"type": "string"}},
        "attempt_id": {"type": "array", "items": {"type": "string"}},
        "question_id": {"type": "array", "items": {"type": "string"}},
        "question_slot": {"type": "array", "items": {"type": "string"}},
        "question_label": {"type": "array", "items": {"type": "string"}},
        "quiz_page": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "array", "items": {"type": "string", "enum": ["low", "medium", "high"]}},
        "min_risk": {"type": "number"},
        "max_risk": {"type": "number"},
        "flag": {"type": "array", "items": {"type": "string", "enum": API_FLAG_OPTIONS}},
        "review_label": {"type": "array", "items": {"type": "string", "enum": sorted(REVIEW_LABEL_IDS)}},
        "flag_mode": {"type": "string", "enum": ["any", "all"], "default": "any"},
        "search": {"type": "string"},
        "start": {"type": "string"},
        "end": {"type": "string"},
        "only_suspicious": {"type": "boolean"},
        "has_review_label": {"type": "boolean"},
        "has_error": {"type": "boolean"},
        "include_frames": {"type": "boolean", "default": True},
        "include_attempts": {"type": "boolean", "default": True},
        "frame_limit": {"type": "integer", "default": 250, "minimum": 0},
        "frame_offset": {"type": "integer", "default": 0, "minimum": 0},
        "attempt_limit": {"type": "integer", "default": 100, "minimum": 0},
        "attempt_offset": {"type": "integer", "default": 0, "minimum": 0},
        "sort_by": {"type": "string", "enum": ["timestamp", "risk_score"], "default": "timestamp"},
        "sort_order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
    }
    example_results_request = {
        "exam": {"course_id": "16", "quiz_id": "51"},
        "student": {"student_id": "ps2124-11487"},
        "flag": ["LOOK_AWAY", "PHONE"],
        "flag_mode": "all",
        "min_risk": 40,
        "frame_limit": 50,
        "sort_by": "risk_score",
        "sort_order": "desc",
    }
    example_attempt_results_response = {
        "filters": {
            "course_id": [],
            "quiz_id": [],
            "student_id": ["ps2124-11487"],
            "attempt_id": ["quiz51_attempt18365"],
            "question_id": [],
            "question_slot": [],
            "question_label": [],
            "quiz_page": [],
            "risk_level": [],
            "min_risk": None,
            "max_risk": None,
            "flag": [],
            "review_label": [],
            "flag_mode": "any",
            "search": "",
            "only_suspicious": False,
            "start": "",
            "end": "",
        },
        "meta": {
            "live_mode": True,
            "results_dir": "/var/lib/proctoring/live",
            "snapshots_dir": "/var/lib/proctoring/live",
            "available_flags": API_FLAG_OPTIONS,
            "available_review_labels": sorted(REVIEW_LABEL_IDS),
            "total_matching_attempts": 1,
            "returned_attempts": 1,
            "attempt_limit": 100,
            "attempt_offset": 0,
            "total_matching_frames": 1,
            "returned_frames": 1,
            "frame_limit": 250,
            "frame_offset": 0,
            "sort_by": "timestamp",
            "sort_order": "desc",
        },
        "attempts": [
            {
                "student_id": "ps2124-11487",
                "attempt_id": "quiz51_attempt18365",
                "course_id": "16",
                "quiz_id": "51",
                "quiz_name": "YN_IK",
                "overall_risk_level": "high",
                "total_frames": 1,
                "valid_frames": 1,
                "suspicious_frames": 1,
                "percentage_suspicious": 100.0,
                "max_risk_score": 100.0,
                "mean_risk_score": 100.0,
                "top_reasons": ["FACE_HIDDEN", "EXTRA_PERSON"],
                "incident_count": 1,
                "question_overview": ["Q:"],
                "matched_frame_count": 1,
                "reviewed_frame_count": 0,
                "updated_at": "2026-04-20T13:07:08Z",
            }
        ],
        "frames": [
            {
                "frame_key": "log:18365",
                "timestamp": "2026-04-20T13:06:58",
                "snapshot_path": "course_16/quiz_51/ps2124-11487/quiz51_attempt18365/18365.jpg",
                "snapshot_url": "/snapshot/course_16/quiz_51/ps2124-11487/quiz51_attempt18365/18365.jpg",
                "risk_score": 100.0,
                "reasons": ["FACE_HIDDEN", "EXTRA_PERSON"],
                "analysis_flags": {
                    "look_away": False,
                    "talking": False,
                    "phone": False,
                    "extra_person": True,
                    "multi_face": False,
                    "no_face": False,
                    "face_hidden": True,
                    "book_notes": False,
                    "low_quality": False,
                    "face_obstructed": False,
                    "identity_mismatch": False,
                },
                "analysis_flag_names": ["EXTRA_PERSON", "FACE_HIDDEN"],
                "face_count": 1,
                "look_away_severity": "none",
                "talking_severity": "none",
                "talking_confidence": 0.0,
                "identity_similarity": 0.0,
                "yaw": None,
                "pitch": None,
                "roll": None,
                "gaze_direction": None,
                "pose_method": "unknown",
                "error": None,
                "source_log_id": 18365,
                "review": {
                    "labels": [],
                    "notes": "",
                    "updated_at": "",
                    "reviewed": False,
                },
                "student": {
                    "student_id": "ps2124-11487",
                    "attempt_id": "quiz51_attempt18365",
                },
                "exam": {
                    "course_id": "16",
                    "quiz_id": "51",
                    "quiz_name": "YN_IK",
                    "quiz_page": "1",
                    "question_id": "",
                    "question_slot": "",
                    "question_name": "",
                    "question_label": "Q:",
                },
            }
        ],
    }
    results_response_content = {
        "application/json": {
            "schema": {"$ref": "#/components/schemas/ResultsResponse"},
            "examples": {
                "specificStudentAttempt": {
                    "summary": "Specific student attempt with one flagged frame",
                    "value": example_attempt_results_response,
                }
            },
        }
    }
    results_post_request_content = {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "exam": {
                        "type": "object",
                        "properties": {
                            "course_id": {"type": "string"},
                            "quiz_id": {"type": "string"},
                            "question_id": {"type": "string"},
                            "question_slot": {"type": "string"},
                        },
                    },
                    "student": {
                        "type": "object",
                        "properties": {
                            "student_id": {"type": "string"},
                            "attempt_id": {"type": "string"},
                        },
                    },
                    **results_filter_properties,
                },
            },
            "examples": {
                "flaggedStudentFrames": {
                    "summary": "Phone and look-away frames for one student",
                    "value": example_results_request,
                }
            },
        }
    }
    review_label_request_example = {
        "student_id": "ps2124-11487",
        "attempt_id": "quiz51_attempt18365",
        "image_path": "course_16/quiz_51/ps2124-11487/quiz51_attempt18365/18365.jpg",
        "source_log_id": 18365,
        "labels": ["EXTRA_PERSON", "FACE_HIDDEN"],
        "notes": "Reviewer confirmed this frame should be flagged.",
    }
    review_label_response_example = {
        "ok": True,
        "frame_key": "log:18365",
        "labels": ["EXTRA_PERSON", "FACE_HIDDEN"],
        "notes": "Reviewer confirmed this frame should be flagged.",
    }

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Exam Proctoring Review API",
            "version": "1.0.0",
            "description": (
                "HTTP API for exam proctoring analysis results, human review labels, "
                "and live pipeline state."
            ),
        },
        "servers": [
            {"url": request.host_url.rstrip("/")}
        ],
        "tags": [
            {"name": "Results", "description": "Read filtered proctoring results and evidence frames."},
            {"name": "Review", "description": "Store or export human review labels."},
            {"name": "Live", "description": "Inspect realtime worker state."},
        ],
        "paths": {
            "/api/live/state": {
                "get": {
                    "tags": ["Live"],
                    "summary": "Get live pipeline state",
                    "responses": {
                        "200": {
                            "description": "Realtime worker counters and last processed log ID.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/LiveStateResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/results": {
                "get": {
                    "tags": ["Results"],
                    "summary": "List filtered result attempts and frames",
                    "parameters": results_query_parameters,
                    "responses": {
                        "200": {
                            "description": "Filtered attempt summaries and frame-level evidence.",
                            "content": results_response_content,
                        },
                        "400": {
                            "description": "Invalid filter parameter.",
                        },
                    },
                },
                "post": {
                    "tags": ["Results"],
                    "summary": "List filtered results using a JSON request body",
                    "requestBody": {
                        "required": False,
                        "content": results_post_request_content,
                    },
                    "responses": {
                        "200": {
                            "description": "Filtered attempt summaries and frame-level evidence.",
                            "content": results_response_content,
                        },
                        "400": {
                            "description": "Invalid filter body.",
                        },
                    },
                },
            },
            "/api/results/{student_id}/{attempt_id}": {
                "get": {
                    "tags": ["Results"],
                    "summary": "Get results for a single student attempt",
                    "parameters": [
                        {
                            "name": "student_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "attempt_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        *results_query_parameters,
                    ],
                    "responses": {
                        "200": {
                            "description": "Attempt-scoped result payload.",
                            "content": results_response_content,
                        }
                    },
                },
                "post": {
                    "tags": ["Results"],
                    "summary": "Get results for a single student attempt with a JSON body",
                    "parameters": [
                        {
                            "name": "student_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "attempt_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": results_filter_properties,
                                },
                                "examples": {
                                    "attemptScopedFilters": {
                                        "summary": "Attempt-scoped high risk frames",
                                        "value": {
                                            "min_risk": 40,
                                            "frame_limit": 50,
                                            "sort_by": "risk_score",
                                            "sort_order": "desc",
                                        },
                                    }
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Attempt-scoped result payload.",
                            "content": results_response_content,
                        }
                    },
                },
            },
            "/api/review-labels": {
                "post": {
                    "tags": ["Review"],
                    "summary": "Create or update human review labels for one frame",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ReviewLabelRequest"},
                                "examples": {
                                    "confirmFlags": {
                                        "summary": "Confirm frame labels after human review",
                                        "value": review_label_request_example,
                                    }
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Saved review labels.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ReviewLabelResponse"},
                                    "examples": {
                                        "saved": {
                                            "summary": "Saved review labels",
                                            "value": review_label_response_example,
                                        }
                                    },
                                }
                            },
                        },
                        "400": {"description": "Invalid review payload."},
                    },
                }
            },
            "/review-labels.csv": {
                "get": {
                    "tags": ["Review"],
                    "summary": "Export all stored review labels as CSV",
                    "responses": {
                        "200": {
                            "description": "CSV download of review labels."
                        }
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "ResultsResponse": {
                    "type": "object",
                    "properties": {
                        "filters": {"type": "object"},
                        "meta": {"$ref": "#/components/schemas/ResultsMeta"},
                        "attempts": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ResultAttempt"},
                        },
                        "frames": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ResultFrame"},
                        },
                    },
                },
                "ResultsMeta": {
                    "type": "object",
                    "properties": {
                        "live_mode": {"type": "boolean"},
                        "results_dir": {"type": "string"},
                        "snapshots_dir": {"type": "string"},
                        "available_flags": {"type": "array", "items": {"type": "string"}},
                        "available_review_labels": {"type": "array", "items": {"type": "string"}},
                        "total_matching_attempts": {"type": "integer"},
                        "returned_attempts": {"type": "integer"},
                        "attempt_limit": {"type": "integer"},
                        "attempt_offset": {"type": "integer"},
                        "total_matching_frames": {"type": "integer"},
                        "returned_frames": {"type": "integer"},
                        "frame_limit": {"type": "integer"},
                        "frame_offset": {"type": "integer"},
                        "sort_by": {"type": "string"},
                        "sort_order": {"type": "string"},
                    },
                },
                "ResultAttempt": {
                    "type": "object",
                    "properties": {
                        "student_id": {"type": "string"},
                        "attempt_id": {"type": "string"},
                        "course_id": {"type": "string"},
                        "quiz_id": {"type": "string"},
                        "quiz_name": {"type": "string"},
                        "overall_risk_level": {"type": "string"},
                        "total_frames": {"type": "integer"},
                        "valid_frames": {"type": "integer"},
                        "suspicious_frames": {"type": "integer"},
                        "percentage_suspicious": {"type": "number"},
                        "max_risk_score": {"type": "number"},
                        "mean_risk_score": {"type": "number"},
                        "top_reasons": {"type": "array", "items": {"type": "string"}},
                        "incident_count": {"type": "integer"},
                        "question_overview": {"type": "array", "items": {"type": "string"}},
                        "matched_frame_count": {"type": "integer"},
                        "reviewed_frame_count": {"type": "integer"},
                        "updated_at": {"type": "string"},
                    },
                },
                "ResultFrame": {
                    "type": "object",
                    "properties": {
                        "frame_key": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "snapshot_path": {"type": "string"},
                        "snapshot_url": {"type": "string"},
                        "risk_score": {"type": "number"},
                        "reasons": {"type": "array", "items": {"type": "string"}},
                        "analysis_flags": {
                            "type": "object",
                            "additionalProperties": {"type": "boolean"},
                        },
                        "analysis_flag_names": {"type": "array", "items": {"type": "string"}},
                        "face_count": {"type": "integer"},
                        "look_away_severity": {"type": "string"},
                        "talking_severity": {"type": "string"},
                        "talking_confidence": {"type": "number", "nullable": True},
                        "identity_similarity": {"type": "number", "nullable": True},
                        "yaw": {"type": "number", "nullable": True},
                        "pitch": {"type": "number", "nullable": True},
                        "roll": {"type": "number", "nullable": True},
                        "gaze_direction": {"type": "string", "nullable": True},
                        "pose_method": {"type": "string", "nullable": True},
                        "error": {"type": "string", "nullable": True},
                        "source_log_id": {"type": "integer", "nullable": True},
                        "review": {"$ref": "#/components/schemas/FrameReview"},
                        "student": {"$ref": "#/components/schemas/FrameStudent"},
                        "exam": {"$ref": "#/components/schemas/FrameExam"},
                    },
                },
                "FrameReview": {
                    "type": "object",
                    "properties": {
                        "labels": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"},
                        "updated_at": {"type": "string"},
                        "reviewed": {"type": "boolean"},
                    },
                },
                "FrameStudent": {
                    "type": "object",
                    "properties": {
                        "student_id": {"type": "string"},
                        "attempt_id": {"type": "string"},
                    },
                },
                "FrameExam": {
                    "type": "object",
                    "properties": {
                        "course_id": {"type": "string"},
                        "quiz_id": {"type": "string"},
                        "quiz_name": {"type": "string"},
                        "quiz_page": {"type": "string"},
                        "question_id": {"type": "string"},
                        "question_slot": {"type": "string"},
                        "question_name": {"type": "string"},
                        "question_label": {"type": "string"},
                    },
                },
                "ReviewLabelRequest": {
                    "type": "object",
                    "required": ["student_id", "attempt_id", "image_path"],
                    "properties": {
                        "student_id": {"type": "string"},
                        "attempt_id": {"type": "string"},
                        "image_path": {"type": "string"},
                        "source_log_id": {"type": "integer", "nullable": True},
                        "labels": {"type": "array", "items": {"type": "string", "enum": sorted(REVIEW_LABEL_IDS)}},
                        "notes": {"type": "string"},
                    },
                },
                "ReviewLabelResponse": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "frame_key": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"},
                    },
                },
                "LiveStateResponse": {
                    "type": "object",
                    "properties": {
                        "live_mode": {"type": "boolean"},
                        "last_source_log_id": {"type": "integer"},
                        "last_updated_at": {"type": "string"},
                        "frame_count": {"type": "integer"},
                        "summary_count": {"type": "integer"},
                    },
                },
            }
        },
    }


def build_swagger_ui_html() -> str:
    openapi_url = url_for("openapi_spec", _external=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Exam Proctoring API Docs</title>
  <link rel="stylesheet" href="{SWAGGER_UI_CSS_URL}">
  <style>
    body {{
      margin: 0;
      background: #f6f8fb;
    }}
    .topbar {{
      display: none;
    }}
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="{SWAGGER_UI_BUNDLE_URL}"></script>
  <script>
    window.onload = function () {{
      window.ui = SwaggerUIBundle({{
        url: "{openapi_url}",
        dom_id: '#swagger-ui',
        deepLinking: true,
        displayRequestDuration: true,
        persistAuthorization: true,
        docExpansion: 'list',
        filter: true
      }});
    }};
  </script>
</body>
</html>"""


def load_review_label_map() -> dict[str, dict[str, Any]]:
    if REVIEW_STORE is None:
        return {}
    results: dict[str, dict[str, Any]] = {}
    for row in REVIEW_STORE.export_rows():
        frame_key = ReviewLabelStore.build_frame_key(
            student_id=str(row.get("student_id", "")),
            attempt_id=str(row.get("attempt_id", "")),
            image_path=str(row.get("image_path", "")),
            source_log_id=row.get("source_log_id"),
        )
        results[frame_key] = {
            "frame_key": frame_key,
            "labels": list(row.get("labels") or []),
            "notes": str(row.get("notes") or ""),
            "updated_at": str(row.get("updated_at") or ""),
            "source_log_id": row.get("source_log_id"),
        }
    return results


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_list_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    values: list[str] = []
    for item in items:
        if item is None:
            continue
        for part in str(item).split(","):
            text = part.strip()
            if text:
                values.append(text)
    return values


def _payload_nested_value(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _request_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _request_scalar(name: str, payload: dict[str, Any], default: Any = None) -> Any:
    if name in payload:
        return payload.get(name)
    nested_map = {
        "course_id": ("exam", "course_id"),
        "quiz_id": ("exam", "quiz_id"),
        "student_id": ("student", "student_id"),
        "attempt_id": ("student", "attempt_id"),
        "question_id": ("exam", "question_id"),
        "question_slot": ("exam", "question_slot"),
    }
    if name in nested_map:
        nested = _payload_nested_value(payload, *nested_map[name])
        if nested not in (None, ""):
            return nested
    return request.args.get(name, default)


def _request_list(name: str, payload: dict[str, Any]) -> list[str]:
    if name in payload:
        return _normalize_list_values(payload.get(name))
    values: list[str] = []
    for raw in request.args.getlist(name):
        values.extend(_normalize_list_values(raw))
    return values


def _request_float(name: str, payload: dict[str, Any]) -> float | None:
    raw = _request_scalar(name, payload)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        abort(400, description=f"{name} must be a number.")


def _request_int(name: str, payload: dict[str, Any], default: int) -> int:
    raw = _request_scalar(name, payload, default)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        abort(400, description=f"{name} must be an integer.")
    if value < 0:
        abort(400, description=f"{name} must be >= 0.")
    return value


def _frame_analysis_flags(frame: dict[str, Any]) -> dict[str, bool]:
    reasons = {str(reason).upper() for reason in (frame.get("reasons") or [])}
    face_count = int(frame.get("face_count") or 0)
    return {
        "look_away": bool(frame.get("look_away_flag")),
        "talking": bool(frame.get("talking_flag")),
        "phone": bool(frame.get("phone_detected")),
        "extra_person": bool(frame.get("extra_person_detected")),
        "multi_face": face_count > 1 or "MULTI_FACE" in reasons,
        "no_face": face_count == 0 or "NO_FACE" in reasons,
        "face_hidden": "FACE_HIDDEN" in reasons,
        "book_notes": bool(frame.get("book_detected")),
        "low_quality": bool(frame.get("low_quality")),
        "face_obstructed": bool(frame.get("face_obstructed")),
        "identity_mismatch": bool(frame.get("identity_mismatch")),
    }


def _frame_flag_names(frame: dict[str, Any]) -> list[str]:
    flags = _frame_analysis_flags(frame)
    names = []
    name_map = {
        "look_away": "LOOK_AWAY",
        "talking": "TALKING",
        "phone": "PHONE",
        "extra_person": "EXTRA_PERSON",
        "multi_face": "MULTI_FACE",
        "no_face": "NO_FACE",
        "face_hidden": "FACE_HIDDEN",
        "book_notes": "BOOK_NOTES",
        "low_quality": "LOW_QUALITY",
        "face_obstructed": "FACE_OBSTRUCTED",
        "identity_mismatch": "IDENTITY_MISMATCH",
    }
    for key, enabled in flags.items():
        if enabled:
            names.append(name_map[key])
    return names


def _value_matches(actual: Any, expected_values: list[str]) -> bool:
    if not expected_values:
        return True
    actual_text = _normalize_text(actual).lower()
    return actual_text in {value.lower() for value in expected_values}


def _text_contains(haystack: list[Any], needle: str) -> bool:
    if not needle:
        return True
    search = needle.lower()
    return search in " ".join(_normalize_text(item).lower() for item in haystack)


def _matches_requested_flags(
    frame: dict[str, Any],
    requested_flags: list[str],
    *,
    mode: str,
) -> bool:
    if not requested_flags:
        return True
    available = set(_frame_flag_names(frame))
    requested = {flag.upper() for flag in requested_flags}
    if mode == "all":
        return requested.issubset(available)
    return bool(requested & available)


def _matches_requested_review_labels(
    review: dict[str, Any] | None,
    requested_labels: list[str],
    *,
    mode: str,
) -> bool:
    if not requested_labels:
        return True
    available = {str(label).upper() for label in (review or {}).get("labels", [])}
    requested = {label.upper() for label in requested_labels}
    if mode == "all":
        return requested.issubset(available)
    return bool(requested & available)


def _sort_frames(frames: list[dict[str, Any]], *, sort_by: str, sort_order: str) -> list[dict[str, Any]]:
    reverse = sort_order != "asc"
    if sort_by == "risk_score":
        return sorted(frames, key=lambda frame: float(frame.get("risk_score") or 0.0), reverse=reverse)
    return sorted(frames, key=lambda frame: _normalize_text(frame.get("timestamp")), reverse=reverse)


def _sort_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        attempts,
        key=lambda item: (
            order.get(_normalize_text(item.get("overall_risk_level")).lower(), 9),
            -float(item.get("mean_risk_score") or 0.0),
            _normalize_text(item.get("student_id")),
            _normalize_text(item.get("attempt_id")),
        ),
    )


def _paginate(items: list[dict[str, Any]], *, limit: int, offset: int) -> list[dict[str, Any]]:
    if limit == 0:
        return items[offset:]
    return items[offset:offset + limit]


def _build_frame_api_payload(frame: dict[str, Any], review: dict[str, Any] | None = None) -> dict[str, Any]:
    review = review or {}
    return {
        "frame_key": frame_review_key(frame),
        "timestamp": frame.get("timestamp", ""),
        "snapshot_path": frame.get("image_path", ""),
        "snapshot_url": url_for("serve_snapshot", image_path=frame.get("image_path", ""), _external=False),
        "risk_score": float(frame.get("risk_score") or 0.0),
        "reasons": list(frame.get("reasons") or []),
        "analysis_flags": _frame_analysis_flags(frame),
        "analysis_flag_names": _frame_flag_names(frame),
        "face_count": int(frame.get("face_count") or 0),
        "look_away_severity": frame.get("severity") or "none",
        "talking_severity": frame.get("talking_severity") or "none",
        "talking_confidence": frame.get("talking_confidence"),
        "identity_similarity": frame.get("identity_similarity"),
        "yaw": frame.get("yaw"),
        "pitch": frame.get("pitch"),
        "roll": frame.get("roll"),
        "gaze_direction": frame.get("gaze_direction"),
        "pose_method": frame.get("pose_method"),
        "error": frame.get("error"),
        "source_log_id": frame.get("source_log_id"),
        "review": {
            "labels": list(review.get("labels") or []),
            "notes": review.get("notes") or "",
            "updated_at": review.get("updated_at") or "",
            "reviewed": bool((review.get("labels") or []) or _normalize_text(review.get("notes"))),
        },
        "student": {
            "student_id": frame.get("student_id", ""),
            "attempt_id": frame.get("attempt_id", ""),
        },
        "exam": {
            "course_id": frame.get("course_id", ""),
            "quiz_id": frame.get("quiz_id", ""),
            "quiz_name": frame.get("quiz_name", ""),
            "quiz_page": frame.get("quiz_page", ""),
            "question_id": frame.get("question_id", ""),
            "question_slot": frame.get("question_slot", ""),
            "question_name": frame.get("question_name", ""),
            "question_label": frame.get("question_label", ""),
        },
    }


def _build_attempt_api_payload(
    summary: dict[str, Any],
    *,
    matched_frame_count: int,
    reviewed_frame_count: int,
) -> dict[str, Any]:
    return {
        "student_id": summary.get("student_id", ""),
        "attempt_id": summary.get("attempt_id", ""),
        "course_id": summary.get("course_id", ""),
        "quiz_id": summary.get("quiz_id", ""),
        "quiz_name": summary.get("quiz_name", ""),
        "overall_risk_level": summary.get("overall_risk_level", "low"),
        "total_frames": int(summary.get("total_frames") or 0),
        "valid_frames": int(summary.get("valid_frames") or 0),
        "suspicious_frames": int(summary.get("suspicious_frames") or 0),
        "percentage_suspicious": float(summary.get("percentage_suspicious") or 0.0),
        "max_risk_score": float(summary.get("max_risk_score") or 0.0),
        "mean_risk_score": float(summary.get("mean_risk_score") or 0.0),
        "top_reasons": list(summary.get("top_reasons") or []),
        "incident_count": int(summary.get("incident_count") or 0),
        "question_overview": list(summary.get("question_overview") or []),
        "matched_frame_count": matched_frame_count,
        "reviewed_frame_count": reviewed_frame_count,
        "updated_at": summary.get("updated_at", ""),
    }


def _build_results_payload(
    *,
    forced_student_id: str | None = None,
    forced_attempt_id: str | None = None,
) -> dict[str, Any]:
    payload = _request_payload()
    requested_flags = [flag.upper() for flag in _request_list("flag", payload)]
    invalid_flags = [flag for flag in requested_flags if flag not in API_FLAG_OPTIONS]
    if invalid_flags:
        abort(400, description=f"Unsupported flags: {', '.join(invalid_flags)}")

    requested_review_labels = [label.upper() for label in _request_list("review_label", payload)]
    invalid_review_labels = [label for label in requested_review_labels if label not in REVIEW_LABEL_IDS]
    if invalid_review_labels:
        abort(400, description=f"Unsupported review labels: {', '.join(invalid_review_labels)}")

    flag_mode = _normalize_text(_request_scalar("flag_mode", payload, "any")).lower() or "any"
    if flag_mode not in {"any", "all"}:
        abort(400, description="flag_mode must be 'any' or 'all'.")

    sort_by = _normalize_text(_request_scalar("sort_by", payload, "timestamp")).lower() or "timestamp"
    if sort_by not in {"timestamp", "risk_score"}:
        abort(400, description="sort_by must be 'timestamp' or 'risk_score'.")
    sort_order = _normalize_text(_request_scalar("sort_order", payload, "desc")).lower() or "desc"
    if sort_order not in {"asc", "desc"}:
        abort(400, description="sort_order must be 'asc' or 'desc'.")

    frame_limit = _request_int("frame_limit", payload, 250)
    frame_offset = _request_int("frame_offset", payload, 0)
    attempt_limit = _request_int("attempt_limit", payload, 100)
    attempt_offset = _request_int("attempt_offset", payload, 0)

    course_filters = _request_list("course_id", payload)
    quiz_filters = _request_list("quiz_id", payload)
    student_filters = [forced_student_id] if forced_student_id else _request_list("student_id", payload)
    attempt_filters = [forced_attempt_id] if forced_attempt_id else _request_list("attempt_id", payload)
    question_id_filters = _request_list("question_id", payload)
    question_slot_filters = _request_list("question_slot", payload)
    question_label_filters = _request_list("question_label", payload)
    quiz_page_filters = _request_list("quiz_page", payload)
    risk_level_filters = [level.lower() for level in _request_list("risk_level", payload)]
    min_risk = _request_float("min_risk", payload)
    max_risk = _request_float("max_risk", payload)
    start_ts = _normalize_text(_request_scalar("start", payload, ""))
    end_ts = _normalize_text(_request_scalar("end", payload, ""))
    search = _normalize_text(_request_scalar("search", payload, ""))
    include_frames = _truthy(_request_scalar("include_frames", payload, True), True)
    include_attempts = _truthy(_request_scalar("include_attempts", payload, True), True)
    only_suspicious = _truthy(_request_scalar("only_suspicious", payload, False), False)
    has_review_label = _request_scalar("has_review_label", payload, None)
    has_error = _request_scalar("has_error", payload, None)

    summaries = load_student_summary()
    summary_by_key = {
        (str(summary.get("student_id", "")), str(summary.get("attempt_id", ""))): summary
        for summary in summaries
    }
    review_label_map = load_review_label_map()
    frames = load_image_results()

    def summary_matches(summary: dict[str, Any]) -> bool:
        return (
            _value_matches(summary.get("course_id"), course_filters)
            and _value_matches(summary.get("quiz_id"), quiz_filters)
            and _value_matches(summary.get("student_id"), student_filters)
            and _value_matches(summary.get("attempt_id"), attempt_filters)
            and (
                not risk_level_filters
                or _normalize_text(summary.get("overall_risk_level")).lower() in set(risk_level_filters)
            )
            and _text_contains(
                [
                    summary.get("student_id"),
                    summary.get("attempt_id"),
                    summary.get("course_id"),
                    summary.get("quiz_id"),
                    summary.get("quiz_name"),
                    *(summary.get("question_overview") or []),
                ],
                search,
            )
        )

    matching_summary_keys = {
        key for key, summary in summary_by_key.items()
        if summary_matches(summary)
    }
    if not matching_summary_keys and summaries:
        matching_summary_keys = set()

    filtered_frames: list[dict[str, Any]] = []
    reviewed_counts: dict[tuple[str, str], int] = {}
    matched_frame_counts: dict[tuple[str, str], int] = {}

    for frame in frames:
        frame_key = (str(frame.get("student_id", "")), str(frame.get("attempt_id", "")))
        summary = summary_by_key.get(frame_key)
        if summaries and summary is not None and frame_key not in matching_summary_keys:
            continue

        review = review_label_map.get(frame_review_key(frame), {})
        if not _value_matches(frame.get("course_id"), course_filters):
            continue
        if not _value_matches(frame.get("quiz_id"), quiz_filters):
            continue
        if not _value_matches(frame.get("student_id"), student_filters):
            continue
        if not _value_matches(frame.get("attempt_id"), attempt_filters):
            continue
        if not _value_matches(frame.get("question_id"), question_id_filters):
            continue
        if not _value_matches(frame.get("question_slot"), question_slot_filters):
            continue
        if not _value_matches(frame.get("question_label"), question_label_filters):
            continue
        if not _value_matches(frame.get("quiz_page"), quiz_page_filters):
            continue
        if min_risk is not None and float(frame.get("risk_score") or 0.0) < min_risk:
            continue
        if max_risk is not None and float(frame.get("risk_score") or 0.0) > max_risk:
            continue
        if only_suspicious and float(frame.get("risk_score") or 0.0) < API_DEFAULT_SUSPICIOUS_THRESHOLD:
            continue
        if start_ts and _normalize_text(frame.get("timestamp")) < start_ts:
            continue
        if end_ts and _normalize_text(frame.get("timestamp")) > end_ts:
            continue
        if has_review_label is not None:
            reviewed = bool((review.get("labels") or []) or _normalize_text(review.get("notes")))
            if reviewed != _truthy(has_review_label):
                continue
        if has_error is not None and bool(frame.get("error")) != _truthy(has_error):
            continue
        if not _matches_requested_flags(frame, requested_flags, mode=flag_mode):
            continue
        if not _matches_requested_review_labels(review, requested_review_labels, mode=flag_mode):
            continue
        if not _text_contains(
            [
                frame.get("student_id"),
                frame.get("attempt_id"),
                frame.get("course_id"),
                frame.get("quiz_id"),
                frame.get("quiz_name"),
                frame.get("question_label"),
                frame.get("question_name"),
                frame.get("quiz_page"),
                *(frame.get("reasons") or []),
                *(review.get("labels") or []),
                review.get("notes"),
            ],
            search,
        ):
            continue

        filtered_frames.append(frame)
        matched_frame_counts[frame_key] = matched_frame_counts.get(frame_key, 0) + 1
        if (review.get("labels") or []) or _normalize_text(review.get("notes")):
            reviewed_counts[frame_key] = reviewed_counts.get(frame_key, 0) + 1

    sorted_frames = _sort_frames(filtered_frames, sort_by=sort_by, sort_order=sort_order)
    paged_frames = _paginate(sorted_frames, limit=frame_limit, offset=frame_offset)

    matched_attempt_keys = set(matched_frame_counts)
    attempts: list[dict[str, Any]] = []
    for key in matched_attempt_keys:
        summary = summary_by_key.get(key)
        if summary is None:
            continue
        attempts.append(
            _build_attempt_api_payload(
                summary,
                matched_frame_count=matched_frame_counts.get(key, 0),
                reviewed_frame_count=reviewed_counts.get(key, 0),
            )
        )
    attempts = _sort_attempts(attempts)
    paged_attempts = _paginate(attempts, limit=attempt_limit, offset=attempt_offset)

    frame_payload = [
        _build_frame_api_payload(frame, review_label_map.get(frame_review_key(frame)))
        for frame in paged_frames
    ] if include_frames else []

    return {
        "filters": {
            "course_id": course_filters,
            "quiz_id": quiz_filters,
            "student_id": student_filters,
            "attempt_id": attempt_filters,
            "question_id": question_id_filters,
            "question_slot": question_slot_filters,
            "question_label": question_label_filters,
            "quiz_page": quiz_page_filters,
            "risk_level": risk_level_filters,
            "min_risk": min_risk,
            "max_risk": max_risk,
            "flag": requested_flags,
            "review_label": requested_review_labels,
            "flag_mode": flag_mode,
            "search": search,
            "only_suspicious": only_suspicious,
            "start": start_ts,
            "end": end_ts,
        },
        "meta": {
            "live_mode": LIVE_MODE,
            "results_dir": str(RESULTS_DIR),
            "snapshots_dir": str(SNAPSHOTS_DIR),
            "available_flags": API_FLAG_OPTIONS,
            "available_review_labels": sorted(REVIEW_LABEL_IDS),
            "total_matching_attempts": len(attempts),
            "returned_attempts": len(paged_attempts) if include_attempts else 0,
            "attempt_limit": attempt_limit,
            "attempt_offset": attempt_offset,
            "total_matching_frames": len(sorted_frames),
            "returned_frames": len(frame_payload),
            "frame_limit": frame_limit,
            "frame_offset": frame_offset,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
        "attempts": paged_attempts if include_attempts else [],
        "frames": frame_payload,
    }


app.jinja_env.globals["risk_badge_class"] = risk_badge_class
app.jinja_env.globals["reason_badge_class"] = reason_badge_class
app.jinja_env.globals["review_badge_class"] = review_badge_class


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def overview():
    students = load_student_summary()
    review_counts = REVIEW_STORE.fetch_attempt_review_counts() if REVIEW_STORE is not None else {}
    for student in students:
        student["_reviewed_count"] = review_counts.get(
            (str(student.get("student_id", "")), str(student.get("attempt_id", ""))),
            0,
        )
    # Sort: high first, then by mean risk desc
    order = {"high": 0, "medium": 1, "low": 2}
    students.sort(
        key=lambda s: (order.get(s.get("overall_risk_level", "low"), 9),
                       -float(s.get("mean_risk_score", 0)))
    )
    # Summary cards
    total = len(students)
    high = sum(1 for s in students if s.get("overall_risk_level") == "high")
    medium = sum(1 for s in students if s.get("overall_risk_level") == "medium")
    low_ = sum(1 for s in students if s.get("overall_risk_level") == "low")
    courses = sorted({s.get("course_id", "") for s in students})
    return render_template(
        "overview.html",
        students=students,
        total=total, high=high, medium=medium, low=low_,
        courses=courses,
        results_dir=str(RESULTS_DIR),
        live_mode=LIVE_MODE,
        live_state=LIVE_STORE.fetch_state() if LIVE_STORE is not None else None,
        has_review_labels=bool(review_counts),
    )


@app.route("/student/<student_id>")
def student_detail(student_id: str):
    all_students = load_student_summary()
    summary = next(
        (s for s in all_students if s["student_id"] == student_id), None
    )
    if summary is None:
        abort(404)
    return redirect(f"/student/{student_id}/{summary['attempt_id']}")


@app.route("/student/<student_id>/<attempt_id>")
def student_attempt_detail(student_id: str, attempt_id: str):
    all_students = load_student_summary()
    summary = next(
        (
            s for s in all_students
            if s["student_id"] == student_id and s["attempt_id"] == attempt_id
        ),
        None,
    )
    if summary is None:
        abort(404)

    all_frames = load_image_results()
    frames = [
        f for f in all_frames
        if f["student_id"] == student_id and f["attempt_id"] == attempt_id
    ]
    frames.sort(key=lambda f: f.get("timestamp", ""))
    review_labels = REVIEW_STORE.fetch_attempt_labels(student_id, attempt_id) if REVIEW_STORE is not None else {}

    # Chart data
    chart_labels = [f.get("timestamp", "")[-8:] for f in frames]  # HH:MM:SS
    chart_scores = [f.get("risk_score", 0) for f in frames]
    chart_colors = [
        "rgba(220,53,69,0.8)" if s >= 40 else
        "rgba(255,193,7,0.8)" if s >= 20 else
        "rgba(25,135,84,0.8)"
        for s in chart_scores
    ]

    # Annotate frames with image exists flag
    for f in frames:
        candidate = SNAPSHOTS_DIR / f["image_path"]
        f["_img_url"] = f"/snapshot/{f['image_path']}"
        f["_img_exists"] = candidate.exists()
        f["_review_key"] = frame_review_key(f)
        saved = review_labels.get(f["_review_key"], {})
        f["_review_labels"] = saved.get("labels", [])
        f["_review_notes"] = saved.get("notes", "")
        f["_reviewed"] = bool(f["_review_labels"] or f["_review_notes"])

    return render_template(
        "student.html",
        summary=summary,
        frames=frames,
        chart_labels=json.dumps(chart_labels),
        chart_scores=json.dumps(chart_scores),
        chart_colors=json.dumps(chart_colors),
        threshold=30.0,
        live_mode=LIVE_MODE,
        live_state=LIVE_STORE.fetch_state() if LIVE_STORE is not None else None,
        review_label_options=REVIEW_LABEL_OPTIONS,
    )


@app.route("/snapshot/<path:image_path>")
def serve_snapshot(image_path: str):
    full = SNAPSHOTS_DIR / image_path
    if not full.exists():
        log.warning("Snapshot not found: %s (snapshots_dir=%s)", full, SNAPSHOTS_DIR)
        abort(404)
    return send_from_directory(str(SNAPSHOTS_DIR), image_path)


@app.route("/openapi.json")
def openapi_spec():
    return jsonify(build_openapi_spec())


@app.route("/docs")
def swagger_docs():
    return Response(build_swagger_ui_html(), mimetype="text/html")


@app.route("/api/live/state")
def live_state():
    if LIVE_STORE is None:
        return jsonify({"live_mode": False})
    return jsonify({"live_mode": True, **LIVE_STORE.fetch_state()})


@app.route("/api/results", methods=["GET", "POST"])
def api_results():
    return jsonify(_build_results_payload())


@app.route("/api/results/<student_id>/<attempt_id>", methods=["GET", "POST"])
def api_results_for_attempt(student_id: str, attempt_id: str):
    return jsonify(
        _build_results_payload(
            forced_student_id=student_id,
            forced_attempt_id=attempt_id,
        )
    )


@app.route("/api/review-labels", methods=["POST"])
def save_review_labels():
    if REVIEW_STORE is None:
        abort(500, description="Review label store is not configured.")

    payload = request.get_json(silent=True) or {}
    student_id = str(payload.get("student_id", "")).strip()
    attempt_id = str(payload.get("attempt_id", "")).strip()
    image_path = str(payload.get("image_path", "")).strip()
    notes = str(payload.get("notes", "") or "")
    source_log_id_raw = payload.get("source_log_id")

    raw_labels = payload.get("labels") or []
    if not isinstance(raw_labels, list):
        abort(400, description="labels must be a list.")
    labels = sorted({str(label).strip() for label in raw_labels if str(label).strip()})

    if not student_id or not attempt_id or not image_path:
        abort(400, description="student_id, attempt_id, and image_path are required.")
    unknown = [label for label in labels if label not in REVIEW_LABEL_IDS]
    if unknown:
        abort(400, description=f"Unsupported labels: {', '.join(unknown)}")

    source_log_id = None
    if source_log_id_raw not in (None, ""):
        try:
            source_log_id = int(source_log_id_raw)
        except Exception:
            abort(400, description="source_log_id must be an integer.")

    frame_key = REVIEW_STORE.save_label(
        student_id=student_id,
        attempt_id=attempt_id,
        image_path=image_path,
        labels=labels,
        notes=notes,
        source_log_id=source_log_id,
    )
    return jsonify(
        {
            "ok": True,
            "frame_key": frame_key,
            "labels": labels,
            "notes": notes.strip(),
        }
    )


@app.route("/review-labels.csv")
def export_review_labels_csv():
    if REVIEW_STORE is None:
        abort(404)

    rows = REVIEW_STORE.export_rows()
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "student_id",
            "attempt_id",
            "image_path",
            "source_log_id",
            "labels_pipe",
            "notes",
            "updated_at",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "student_id": row["student_id"],
                "attempt_id": row["attempt_id"],
                "image_path": row["image_path"],
                "source_log_id": row["source_log_id"],
                "labels_pipe": row["labels_pipe"],
                "notes": row["notes"],
                "updated_at": row["updated_at"],
            }
        )

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=review_labels.csv"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exam proctoring review dashboard")
    parser.add_argument("--results", default="results",
                        help="Directory containing student_summary.csv etc.")
    parser.add_argument("--snapshots", default="test_data",
                        help="Directory containing the snapshots/ sub-folder")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--live-db",
        default=None,
        help="SQLite store created by live_moodle_pipeline.py. Enables live dashboard mode.",
    )
    parser.add_argument(
        "--review-db",
        default=None,
        help="SQLite file used to store human review labels. Defaults to <results>/review_labels.sqlite3.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        configure_dashboard(
            results=args.results,
            snapshots=args.snapshots,
            live_db=args.live_db,
            review_db=args.review_db,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1)

    print(f"\n  Exam Proctoring Review Dashboard")
    print(f"  Results : {RESULTS_DIR.resolve()}")
    print(f"  Snapshots: {SNAPSHOTS_DIR.resolve()}")
    if LIVE_STORE is not None:
        print(f"  Live DB : {Path(args.live_db).resolve()}")
        print("  Mode    : realtime SQLite")
    if REVIEW_STORE is not None:
        print(f"  Review DB: {REVIEW_STORE.db_path.resolve()}")
    print(f"\n  Open: http://{args.host}:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
