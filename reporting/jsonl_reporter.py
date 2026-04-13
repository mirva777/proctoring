"""
JSONL reporter – one JSON object per line, one line per image frame.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from scoring.aggregator import FrameRecord

logger = logging.getLogger(__name__)


def _record_to_dict(rec: FrameRecord) -> dict:
    """Convert a FrameRecord to a JSON-serialisable dict."""
    return {
        "image_path": rec.image_path,
        "student_id": rec.student_id,
        "attempt_id": rec.attempt_id,
        "timestamp": rec.timestamp,
        "course_id": rec.course_id,
        "quiz": {
            "id": rec.quiz_id,
            "name": rec.quiz_name,
            "page": rec.quiz_page,
            "question_id": rec.question_id,
            "question_slot": rec.question_slot,
            "question_name": rec.question_name,
            "question_label": rec.question_label,
            "source_log_id": rec.source_log_id,
        },
        "face": {
            "count": rec.face_count,
        },
        "attention": {
            "look_away_flag": rec.look_away_flag,
            "severity": rec.severity,
            "yaw": rec.yaw,
            "pitch": rec.pitch,
            "roll": rec.roll,
            "gaze_direction": rec.gaze_direction,
            "pose_method": rec.pose_method,
            "talking_flag": rec.talking_flag,
            "talking_severity": rec.talking_severity,
            "talking_confidence": rec.talking_confidence,
            "mouth_open_ratio": rec.mouth_open_ratio,
            "mouth_open_delta": rec.mouth_open_delta,
        },
        "identity": {
            "mismatch_flag": rec.identity_mismatch,
            "similarity_score": rec.identity_similarity,
        },
        "objects": {
            "phone_detected": rec.phone_detected,
            "extra_person_detected": rec.extra_person_detected,
            "book_detected": rec.book_detected,
            "face_obstructed": rec.face_obstructed,
            "person_detected": rec.person_detected,
        },
        "quality": {
            "blur_score": rec.blur_score,
            "brightness_score": rec.brightness_score,
            "glare_score": rec.glare_score,
            "low_quality_flag": rec.low_quality,
        },
        "risk": {
            "score": rec.risk_score,
            "reasons": rec.reasons,
        },
        "error": rec.error,
    }


def write_image_results_jsonl(
    records: Sequence[FrameRecord],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(_record_to_dict(rec), ensure_ascii=False) + "\n")
    logger.info("Wrote %d JSONL records to %s", len(records), path)
