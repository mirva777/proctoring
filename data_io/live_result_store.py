"""
SQLite-backed result store for realtime proctoring analysis.

The live worker writes one row per processed frame and one row per
student-attempt summary.  The dashboard reads the same store and can refresh
without waiting for CSV exports.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scoring.aggregator import FrameRecord, StudentSummary, aggregate_student


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frame_key(record: FrameRecord) -> str:
    if record.source_log_id is not None:
        return f"log:{record.source_log_id}"
    return "|".join([
        "frame",
        record.student_id,
        record.attempt_id,
        record.timestamp,
        record.image_path,
    ])


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(raw_value: Any, fallback: Any) -> Any:
    if raw_value is None or raw_value == "":
        return fallback
    try:
        return json.loads(raw_value)
    except Exception:
        return fallback


class LiveResultStore:
    """Persistence layer for realtime frame and summary rows."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS frame_results (
                    frame_key TEXT PRIMARY KEY,
                    source_log_id INTEGER,
                    image_path TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    course_id TEXT NOT NULL,
                    quiz_id TEXT,
                    quiz_name TEXT,
                    quiz_page TEXT,
                    question_id TEXT,
                    question_slot TEXT,
                    question_name TEXT,
                    question_label TEXT,
                    face_count INTEGER,
                    look_away_flag INTEGER,
                    severity TEXT,
                    identity_mismatch INTEGER,
                    identity_similarity REAL,
                    phone_detected INTEGER,
                    extra_person_detected INTEGER,
                    book_detected INTEGER,
                    face_obstructed INTEGER,
                    talking_flag INTEGER,
                    talking_severity TEXT,
                    talking_confidence REAL,
                    mouth_open_ratio REAL,
                    mouth_open_delta REAL,
                    person_detected INTEGER,
                    low_quality INTEGER,
                    blur_score REAL,
                    brightness_score REAL,
                    glare_score REAL,
                    risk_score REAL,
                    reasons_json TEXT,
                    yaw REAL,
                    pitch REAL,
                    roll REAL,
                    gaze_direction TEXT,
                    pose_method TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_frame_attempt
                    ON frame_results(student_id, attempt_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_frame_log_id
                    ON frame_results(source_log_id);

                CREATE TABLE IF NOT EXISTS attempt_summaries (
                    student_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    course_id TEXT NOT NULL,
                    quiz_id TEXT,
                    quiz_name TEXT,
                    total_frames INTEGER,
                    valid_frames INTEGER,
                    suspicious_frames INTEGER,
                    percentage_suspicious REAL,
                    max_risk_score REAL,
                    mean_risk_score REAL,
                    top_reasons_json TEXT,
                    incident_count INTEGER,
                    incidents_json TEXT,
                    flagged_timeline_json TEXT,
                    question_overview_json TEXT,
                    identity_stability_score REAL,
                    overall_risk_level TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (student_id, attempt_id)
                );

                CREATE TABLE IF NOT EXISTS pipeline_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_source_log_id INTEGER NOT NULL,
                    last_updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO pipeline_state
                    (id, last_source_log_id, last_updated_at)
                VALUES (1, 0, ?)
                """,
                (_utc_now_iso(),),
            )

    def upsert_frame(self, record: FrameRecord) -> None:
        row = self._record_to_db_row(record)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO frame_results (
                    frame_key, source_log_id, image_path, student_id, attempt_id,
                    timestamp, course_id, quiz_id, quiz_name, quiz_page,
                    question_id, question_slot, question_name, question_label,
                    face_count, look_away_flag, severity, identity_mismatch,
                    identity_similarity, phone_detected, extra_person_detected,
                    book_detected, face_obstructed, talking_flag, talking_severity,
                    talking_confidence, mouth_open_ratio, mouth_open_delta,
                    person_detected, low_quality, blur_score, brightness_score,
                    glare_score, risk_score, reasons_json, yaw, pitch, roll,
                    gaze_direction, pose_method, error, updated_at
                ) VALUES (
                    :frame_key, :source_log_id, :image_path, :student_id, :attempt_id,
                    :timestamp, :course_id, :quiz_id, :quiz_name, :quiz_page,
                    :question_id, :question_slot, :question_name, :question_label,
                    :face_count, :look_away_flag, :severity, :identity_mismatch,
                    :identity_similarity, :phone_detected, :extra_person_detected,
                    :book_detected, :face_obstructed, :talking_flag, :talking_severity,
                    :talking_confidence, :mouth_open_ratio, :mouth_open_delta,
                    :person_detected, :low_quality, :blur_score, :brightness_score,
                    :glare_score, :risk_score, :reasons_json, :yaw, :pitch, :roll,
                    :gaze_direction, :pose_method, :error, :updated_at
                )
                ON CONFLICT(frame_key) DO UPDATE SET
                    source_log_id=excluded.source_log_id,
                    image_path=excluded.image_path,
                    student_id=excluded.student_id,
                    attempt_id=excluded.attempt_id,
                    timestamp=excluded.timestamp,
                    course_id=excluded.course_id,
                    quiz_id=excluded.quiz_id,
                    quiz_name=excluded.quiz_name,
                    quiz_page=excluded.quiz_page,
                    question_id=excluded.question_id,
                    question_slot=excluded.question_slot,
                    question_name=excluded.question_name,
                    question_label=excluded.question_label,
                    face_count=excluded.face_count,
                    look_away_flag=excluded.look_away_flag,
                    severity=excluded.severity,
                    identity_mismatch=excluded.identity_mismatch,
                    identity_similarity=excluded.identity_similarity,
                    phone_detected=excluded.phone_detected,
                    extra_person_detected=excluded.extra_person_detected,
                    book_detected=excluded.book_detected,
                    face_obstructed=excluded.face_obstructed,
                    talking_flag=excluded.talking_flag,
                    talking_severity=excluded.talking_severity,
                    talking_confidence=excluded.talking_confidence,
                    mouth_open_ratio=excluded.mouth_open_ratio,
                    mouth_open_delta=excluded.mouth_open_delta,
                    person_detected=excluded.person_detected,
                    low_quality=excluded.low_quality,
                    blur_score=excluded.blur_score,
                    brightness_score=excluded.brightness_score,
                    glare_score=excluded.glare_score,
                    risk_score=excluded.risk_score,
                    reasons_json=excluded.reasons_json,
                    yaw=excluded.yaw,
                    pitch=excluded.pitch,
                    roll=excluded.roll,
                    gaze_direction=excluded.gaze_direction,
                    pose_method=excluded.pose_method,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                row,
            )

    def update_attempt_summary(self, student_id: str, attempt_id: str) -> StudentSummary | None:
        records = self.fetch_frames(student_id=student_id, attempt_id=attempt_id)
        if not records:
            return None
        summary = aggregate_student(records)
        self.upsert_summary(summary)
        return summary

    def upsert_summary(self, summary: StudentSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO attempt_summaries (
                    student_id, attempt_id, course_id, quiz_id, quiz_name,
                    total_frames, valid_frames, suspicious_frames,
                    percentage_suspicious, max_risk_score, mean_risk_score,
                    top_reasons_json, incident_count, incidents_json,
                    flagged_timeline_json, question_overview_json,
                    identity_stability_score, overall_risk_level, updated_at
                ) VALUES (
                    :student_id, :attempt_id, :course_id, :quiz_id, :quiz_name,
                    :total_frames, :valid_frames, :suspicious_frames,
                    :percentage_suspicious, :max_risk_score, :mean_risk_score,
                    :top_reasons_json, :incident_count, :incidents_json,
                    :flagged_timeline_json, :question_overview_json,
                    :identity_stability_score, :overall_risk_level, :updated_at
                )
                ON CONFLICT(student_id, attempt_id) DO UPDATE SET
                    course_id=excluded.course_id,
                    quiz_id=excluded.quiz_id,
                    quiz_name=excluded.quiz_name,
                    total_frames=excluded.total_frames,
                    valid_frames=excluded.valid_frames,
                    suspicious_frames=excluded.suspicious_frames,
                    percentage_suspicious=excluded.percentage_suspicious,
                    max_risk_score=excluded.max_risk_score,
                    mean_risk_score=excluded.mean_risk_score,
                    top_reasons_json=excluded.top_reasons_json,
                    incident_count=excluded.incident_count,
                    incidents_json=excluded.incidents_json,
                    flagged_timeline_json=excluded.flagged_timeline_json,
                    question_overview_json=excluded.question_overview_json,
                    identity_stability_score=excluded.identity_stability_score,
                    overall_risk_level=excluded.overall_risk_level,
                    updated_at=excluded.updated_at
                """,
                {
                    "student_id": summary.student_id,
                    "attempt_id": summary.attempt_id,
                    "course_id": summary.course_id,
                    "quiz_id": summary.quiz_id,
                    "quiz_name": summary.quiz_name,
                    "total_frames": summary.total_frames,
                    "valid_frames": summary.valid_frames,
                    "suspicious_frames": summary.suspicious_frames,
                    "percentage_suspicious": summary.percentage_suspicious,
                    "max_risk_score": summary.max_risk_score,
                    "mean_risk_score": summary.mean_risk_score,
                    "top_reasons_json": _json_dumps(summary.top_reasons),
                    "incident_count": summary.incident_count,
                    "incidents_json": _json_dumps(summary.incidents),
                    "flagged_timeline_json": _json_dumps(summary.flagged_timeline),
                    "question_overview_json": _json_dumps(summary.question_overview),
                    "identity_stability_score": summary.identity_stability_score,
                    "overall_risk_level": summary.overall_risk_level,
                    "updated_at": _utc_now_iso(),
                },
            )

    def fetch_summaries(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM attempt_summaries
                ORDER BY
                    CASE overall_risk_level
                        WHEN 'high' THEN 0
                        WHEN 'medium' THEN 1
                        ELSE 2
                    END,
                    mean_risk_score DESC,
                    student_id ASC,
                    attempt_id ASC
                """
            ).fetchall()
        return [self._summary_row_to_dict(row) for row in rows]

    def fetch_summary(self, student_id: str, attempt_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM attempt_summaries
                WHERE student_id = ? AND attempt_id = ?
                """,
                (student_id, attempt_id),
            ).fetchone()
        return self._summary_row_to_dict(row) if row else None

    def fetch_frames(
        self,
        student_id: str | None = None,
        attempt_id: str | None = None,
    ) -> list[FrameRecord]:
        sql = "SELECT * FROM frame_results"
        params: list[Any] = []
        clauses = []
        if student_id is not None:
            clauses.append("student_id = ?")
            params.append(student_id)
        if attempt_id is not None:
            clauses.append("attempt_id = ?")
            params.append(attempt_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp ASC, source_log_id ASC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._frame_row_to_record(row) for row in rows]

    def fetch_frame_dicts(
        self,
        student_id: str | None = None,
        attempt_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [self._record_to_public_dict(rec) for rec in self.fetch_frames(student_id, attempt_id)]

    def get_last_source_log_id(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_source_log_id FROM pipeline_state WHERE id = 1"
            ).fetchone()
        return int(row["last_source_log_id"]) if row else 0

    def set_last_source_log_id(self, log_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE pipeline_state
                SET last_source_log_id = ?, last_updated_at = ?
                WHERE id = 1
                """,
                (int(log_id), _utc_now_iso()),
            )

    def fetch_state(self) -> dict[str, Any]:
        with self._connect() as conn:
            state_row = conn.execute(
                "SELECT last_source_log_id, last_updated_at FROM pipeline_state WHERE id = 1"
            ).fetchone()
            frame_count = conn.execute("SELECT COUNT(*) AS n FROM frame_results").fetchone()["n"]
            summary_count = conn.execute("SELECT COUNT(*) AS n FROM attempt_summaries").fetchone()["n"]
        return {
            "last_source_log_id": int(state_row["last_source_log_id"]) if state_row else 0,
            "last_updated_at": state_row["last_updated_at"] if state_row else "",
            "frame_count": int(frame_count),
            "summary_count": int(summary_count),
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @staticmethod
    def _record_to_db_row(record: FrameRecord) -> dict[str, Any]:
        return {
            "frame_key": _frame_key(record),
            "source_log_id": record.source_log_id,
            "image_path": record.image_path,
            "student_id": record.student_id,
            "attempt_id": record.attempt_id,
            "timestamp": record.timestamp,
            "course_id": record.course_id,
            "quiz_id": record.quiz_id,
            "quiz_name": record.quiz_name,
            "quiz_page": record.quiz_page,
            "question_id": record.question_id,
            "question_slot": record.question_slot,
            "question_name": record.question_name,
            "question_label": record.question_label,
            "face_count": record.face_count,
            "look_away_flag": int(bool(record.look_away_flag)),
            "severity": record.severity,
            "identity_mismatch": int(bool(record.identity_mismatch)),
            "identity_similarity": record.identity_similarity,
            "phone_detected": int(bool(record.phone_detected)),
            "extra_person_detected": int(bool(record.extra_person_detected)),
            "book_detected": int(bool(record.book_detected)),
            "face_obstructed": int(bool(record.face_obstructed)),
            "talking_flag": int(bool(record.talking_flag)),
            "talking_severity": record.talking_severity,
            "talking_confidence": record.talking_confidence,
            "mouth_open_ratio": record.mouth_open_ratio,
            "mouth_open_delta": record.mouth_open_delta,
            "person_detected": int(bool(record.person_detected)),
            "low_quality": int(bool(record.low_quality)),
            "blur_score": record.blur_score,
            "brightness_score": record.brightness_score,
            "glare_score": record.glare_score,
            "risk_score": record.risk_score,
            "reasons_json": _json_dumps(record.reasons),
            "yaw": record.yaw,
            "pitch": record.pitch,
            "roll": record.roll,
            "gaze_direction": record.gaze_direction,
            "pose_method": record.pose_method,
            "error": record.error,
            "updated_at": _utc_now_iso(),
        }

    @staticmethod
    def _record_to_public_dict(record: FrameRecord) -> dict[str, Any]:
        data = asdict(record)
        data["reasons"] = list(record.reasons)
        return data

    @staticmethod
    def _frame_row_to_record(row: sqlite3.Row) -> FrameRecord:
        return FrameRecord(
            image_path=row["image_path"],
            student_id=row["student_id"],
            attempt_id=row["attempt_id"],
            timestamp=row["timestamp"],
            course_id=row["course_id"],
            face_count=int(row["face_count"] or 0),
            look_away_flag=bool(row["look_away_flag"]),
            severity=row["severity"] or "none",
            identity_mismatch=bool(row["identity_mismatch"]),
            identity_similarity=float(row["identity_similarity"] or 0.0),
            phone_detected=bool(row["phone_detected"]),
            extra_person_detected=bool(row["extra_person_detected"]),
            book_detected=bool(row["book_detected"]),
            face_obstructed=bool(row["face_obstructed"]),
            talking_flag=bool(row["talking_flag"]),
            talking_severity=row["talking_severity"] or "none",
            talking_confidence=float(row["talking_confidence"] or 0.0),
            mouth_open_ratio=row["mouth_open_ratio"],
            mouth_open_delta=row["mouth_open_delta"],
            low_quality=bool(row["low_quality"]),
            blur_score=float(row["blur_score"] or 0.0),
            brightness_score=float(row["brightness_score"] or 0.0),
            glare_score=float(row["glare_score"] or 0.0),
            risk_score=float(row["risk_score"] or 0.0),
            reasons=_json_loads(row["reasons_json"], []),
            person_detected=bool(row["person_detected"]),
            yaw=row["yaw"],
            pitch=row["pitch"],
            roll=row["roll"],
            gaze_direction=row["gaze_direction"],
            pose_method=row["pose_method"] or "unknown",
            quiz_id=row["quiz_id"] or "",
            quiz_name=row["quiz_name"] or "",
            quiz_page=row["quiz_page"] or "",
            question_id=row["question_id"] or "",
            question_slot=row["question_slot"] or "",
            question_name=row["question_name"] or "",
            question_label=row["question_label"] or "",
            source_log_id=row["source_log_id"],
            error=row["error"],
        )

    @staticmethod
    def _summary_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "student_id": row["student_id"],
            "attempt_id": row["attempt_id"],
            "course_id": row["course_id"],
            "quiz_id": row["quiz_id"] or "",
            "quiz_name": row["quiz_name"] or "",
            "total_frames": int(row["total_frames"] or 0),
            "valid_frames": int(row["valid_frames"] or 0),
            "suspicious_frames": int(row["suspicious_frames"] or 0),
            "percentage_suspicious": float(row["percentage_suspicious"] or 0.0),
            "max_risk_score": float(row["max_risk_score"] or 0.0),
            "mean_risk_score": float(row["mean_risk_score"] or 0.0),
            "top_reasons": _json_loads(row["top_reasons_json"], []),
            "incident_count": int(row["incident_count"] or 0),
            "incidents": _json_loads(row["incidents_json"], []),
            "flagged_timeline": _json_loads(row["flagged_timeline_json"], []),
            "question_overview": _json_loads(row["question_overview_json"], []),
            "identity_stability_score": float(row["identity_stability_score"] or 1.0),
            "overall_risk_level": row["overall_risk_level"] or "low",
            "updated_at": row["updated_at"],
        }
