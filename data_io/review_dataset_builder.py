from __future__ import annotations

import ast
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from data_io.review_label_store import ReviewLabelStore


def frame_key_for_row(row: dict) -> str:
    source_log_id = row.get("source_log_id")
    if source_log_id not in (None, "", 0):
        try:
            return ReviewLabelStore.build_frame_key(
                student_id=str(row.get("student_id", "")),
                attempt_id=str(row.get("attempt_id", "")),
                image_path=str(row.get("image_path", "")),
                source_log_id=int(source_log_id),
            )
        except Exception:
            pass
    return ReviewLabelStore.build_frame_key(
        student_id=str(row.get("student_id", "")),
        attempt_id=str(row.get("attempt_id", "")),
        image_path=str(row.get("image_path", "")),
    )


def load_image_results_csv(results_dir: str | Path) -> list[dict]:
    path = Path(results_dir) / "image_level_results.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        raw_reasons = (row.get("reasons") or "").strip()
        try:
            row["reasons_list"] = ast.literal_eval(raw_reasons) if raw_reasons else []
        except Exception:
            row["reasons_list"] = []
    return rows


def load_review_rows_from_csv(csv_path: str | Path) -> list[dict]:
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    normalized: list[dict] = []
    for row in rows:
        labels_pipe = str(row.get("labels_pipe") or "").strip()
        labels = [label.strip() for label in labels_pipe.split("|") if label.strip()]
        normalized.append(
            {
                "student_id": str(row.get("student_id") or ""),
                "attempt_id": str(row.get("attempt_id") or ""),
                "image_path": str(row.get("image_path") or ""),
                "source_log_id": row.get("source_log_id"),
                "labels": labels,
                "labels_pipe": labels_pipe,
                "notes": str(row.get("notes") or ""),
                "updated_at": str(row.get("updated_at") or ""),
            }
        )
    return normalized


def build_review_dataset_rows(
    image_rows: Iterable[dict],
    review_rows: Iterable[dict],
    *,
    snapshots_dir: str | Path | None = None,
    include_unreviewed: bool = False,
) -> list[dict]:
    review_lookup = {frame_key_for_row(row): row for row in review_rows}
    snapshots_root = Path(snapshots_dir).resolve() if snapshots_dir else None

    dataset_rows: list[dict] = []
    for image_row in image_rows:
        review = review_lookup.get(frame_key_for_row(image_row))
        if review is None and not include_unreviewed:
            continue

        labels = list((review or {}).get("labels") or [])
        dataset_rows.append(
            {
                "student_id": str(image_row.get("student_id") or ""),
                "attempt_id": str(image_row.get("attempt_id") or ""),
                "image_path": str(image_row.get("image_path") or ""),
                "absolute_image_path": (
                    str((snapshots_root / str(image_row.get("image_path") or "")).resolve())
                    if snapshots_root is not None
                    else ""
                ),
                "timestamp": str(image_row.get("timestamp") or ""),
                "course_id": str(image_row.get("course_id") or ""),
                "quiz_id": str(image_row.get("quiz_id") or ""),
                "quiz_name": str(image_row.get("quiz_name") or ""),
                "quiz_page": str(image_row.get("quiz_page") or ""),
                "question_label": str(image_row.get("question_label") or ""),
                "source_log_id": str(image_row.get("source_log_id") or ""),
                "risk_score": str(image_row.get("risk_score") or ""),
                "model_reasons_pipe": "|".join(str(x) for x in image_row.get("reasons_list", [])),
                "face_count": str(image_row.get("face_count") or ""),
                "gaze_direction": str(image_row.get("gaze_direction") or ""),
                "severity": str(image_row.get("severity") or ""),
                "talking_flag": str(image_row.get("talking_flag") or ""),
                "phone_detected": str(image_row.get("phone_detected") or ""),
                "extra_person_detected": str(image_row.get("extra_person_detected") or ""),
                "book_detected": str(image_row.get("book_detected") or ""),
                "face_obstructed": str(image_row.get("face_obstructed") or ""),
                "person_detected": str(image_row.get("person_detected") or ""),
                "low_quality": str(image_row.get("low_quality") or ""),
                "human_labels_pipe": "|".join(labels),
                "human_notes": str((review or {}).get("notes") or ""),
                "review_updated_at": str((review or {}).get("updated_at") or ""),
                "reviewed": "true" if review is not None and (labels or (review or {}).get("notes")) else "false",
            }
        )
    return dataset_rows


def summarize_review_dataset(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    label_counts: Counter[str] = Counter()
    for row in rows:
        for label in [item.strip() for item in str(row.get("human_labels_pipe") or "").split("|") if item.strip()]:
            label_counts[label] += 1
    return {
        "reviewed_frames": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
    }
