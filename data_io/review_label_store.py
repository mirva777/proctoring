from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class ReviewLabelStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @staticmethod
    def build_frame_key(
        student_id: str,
        attempt_id: str,
        image_path: str,
        source_log_id: int | None = None,
    ) -> str:
        if source_log_id:
            return f"log:{int(source_log_id)}"
        return f"{student_id}::{attempt_id}::{image_path}"

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS frame_review_labels (
                    frame_key TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    source_log_id INTEGER,
                    labels_json TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_frame_review_attempt
                    ON frame_review_labels(student_id, attempt_id);

                CREATE INDEX IF NOT EXISTS idx_frame_review_source_log
                    ON frame_review_labels(source_log_id);
                """
            )
            self._conn.commit()

    def save_label(
        self,
        *,
        student_id: str,
        attempt_id: str,
        image_path: str,
        labels: list[str],
        notes: str = "",
        source_log_id: int | None = None,
    ) -> str:
        frame_key = self.build_frame_key(
            student_id=student_id,
            attempt_id=attempt_id,
            image_path=image_path,
            source_log_id=source_log_id,
        )
        normalized_labels = sorted({str(label).strip() for label in labels if str(label).strip()})
        normalized_notes = (notes or "").strip()

        with self._lock:
            if not normalized_labels and not normalized_notes:
                self._conn.execute(
                    "DELETE FROM frame_review_labels WHERE frame_key = ?",
                    (frame_key,),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO frame_review_labels (
                        frame_key, student_id, attempt_id, image_path,
                        source_log_id, labels_json, notes, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(frame_key) DO UPDATE SET
                        student_id = excluded.student_id,
                        attempt_id = excluded.attempt_id,
                        image_path = excluded.image_path,
                        source_log_id = excluded.source_log_id,
                        labels_json = excluded.labels_json,
                        notes = excluded.notes,
                        updated_at = excluded.updated_at
                    """,
                    (
                        frame_key,
                        student_id,
                        attempt_id,
                        image_path,
                        source_log_id,
                        json.dumps(normalized_labels),
                        normalized_notes,
                        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ),
                )
            self._conn.commit()

        return frame_key

    def fetch_attempt_labels(self, student_id: str, attempt_id: str) -> dict[str, dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT frame_key, image_path, source_log_id, labels_json, notes, updated_at
                FROM frame_review_labels
                WHERE student_id = ? AND attempt_id = ?
                ORDER BY updated_at DESC
                """,
                (student_id, attempt_id),
            ).fetchall()

        results: dict[str, dict] = {}
        for row in rows:
            labels = []
            if row["labels_json"]:
                try:
                    labels = json.loads(row["labels_json"])
                except Exception:
                    labels = []
            results[str(row["frame_key"])] = {
                "frame_key": str(row["frame_key"]),
                "image_path": str(row["image_path"]),
                "source_log_id": row["source_log_id"],
                "labels": labels,
                "notes": str(row["notes"] or ""),
                "updated_at": str(row["updated_at"] or ""),
            }
        return results

    def fetch_attempt_review_counts(self) -> dict[tuple[str, str], int]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT student_id, attempt_id, COUNT(*) AS reviewed_count
                FROM frame_review_labels
                GROUP BY student_id, attempt_id
                """
            ).fetchall()
        return {
            (str(row["student_id"]), str(row["attempt_id"])): int(row["reviewed_count"])
            for row in rows
        }

    def export_rows(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT student_id, attempt_id, image_path, source_log_id,
                       labels_json, notes, updated_at
                FROM frame_review_labels
                ORDER BY updated_at DESC, student_id ASC, attempt_id ASC, image_path ASC
                """
            ).fetchall()

        exported: list[dict] = []
        for row in rows:
            labels = []
            if row["labels_json"]:
                try:
                    labels = json.loads(row["labels_json"])
                except Exception:
                    labels = []
            exported.append(
                {
                    "student_id": str(row["student_id"]),
                    "attempt_id": str(row["attempt_id"]),
                    "image_path": str(row["image_path"]),
                    "source_log_id": row["source_log_id"],
                    "labels": labels,
                    "labels_pipe": "|".join(labels),
                    "notes": str(row["notes"] or ""),
                    "updated_at": str(row["updated_at"] or ""),
                }
            )
        return exported

    def close(self) -> None:
        with self._lock:
            self._conn.close()
