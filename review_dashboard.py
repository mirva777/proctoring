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

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory  # type: ignore

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


@app.route("/api/live/state")
def live_state():
    if LIVE_STORE is None:
        return jsonify({"live_mode": False})
    return jsonify({"live_mode": True, **LIVE_STORE.fetch_state()})


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
