"""
CSV reporters for image-level results and student summary.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Sequence

from scoring.aggregator import FrameRecord, StudentSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image-level CSV
# ---------------------------------------------------------------------------
IMAGE_LEVEL_FIELDS = [
    "image_path", "student_id", "attempt_id", "timestamp", "course_id",
    "quiz_id", "quiz_name", "quiz_page",
    "question_id", "question_slot", "question_name", "question_label",
    "source_log_id", "source_webcampicture", "source_filename",
    "source_contenthash", "source_moodledata_path",
    "face_count", "look_away_flag", "severity",
    "yaw", "pitch", "roll", "gaze_direction", "pose_method",
    "talking_flag", "talking_severity", "talking_confidence",
    "mouth_open_ratio", "mouth_open_delta",
    "identity_mismatch", "identity_similarity",
    "phone_detected", "extra_person_detected", "book_detected", "face_obstructed",
    "person_detected",
    "low_quality", "blur_score", "brightness_score", "glare_score",
    "risk_score", "reasons", "error",
]

# ---------------------------------------------------------------------------
# Student-summary CSV
# ---------------------------------------------------------------------------
STUDENT_SUMMARY_FIELDS = [
    "student_id", "attempt_id", "course_id", "quiz_id", "quiz_name",
    "total_frames", "valid_frames", "suspicious_frames", "percentage_suspicious",
    "max_risk_score", "mean_risk_score",
    "top_reasons", "incident_count",
    "identity_stability_score", "overall_risk_level",
    "question_overview", "flagged_timeline",
]


def write_image_results_csv(
    records: Sequence[FrameRecord],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=IMAGE_LEVEL_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "image_path": rec.image_path,
                "student_id": rec.student_id,
                "attempt_id": rec.attempt_id,
                "timestamp": rec.timestamp,
                "course_id": rec.course_id,
                "quiz_id": rec.quiz_id,
                "quiz_name": rec.quiz_name,
                "quiz_page": rec.quiz_page,
                "question_id": rec.question_id,
                "question_slot": rec.question_slot,
                "question_name": rec.question_name,
                "question_label": rec.question_label,
                "source_log_id": rec.source_log_id if rec.source_log_id is not None else "",
                "source_webcampicture": rec.source_webcampicture,
                "source_filename": rec.source_filename,
                "source_contenthash": rec.source_contenthash,
                "source_moodledata_path": rec.source_moodledata_path,
                "face_count": rec.face_count,
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
                "identity_mismatch": rec.identity_mismatch,
                "identity_similarity": rec.identity_similarity,
                "phone_detected": rec.phone_detected,
                "extra_person_detected": rec.extra_person_detected,
                "book_detected": rec.book_detected,
                "face_obstructed": rec.face_obstructed,
                "person_detected": rec.person_detected,
                "low_quality": rec.low_quality,
                "blur_score": rec.blur_score,
                "brightness_score": rec.brightness_score,
                "glare_score": rec.glare_score,
                "risk_score": rec.risk_score,
                "reasons": json.dumps(rec.reasons),
                "error": rec.error or "",
            })
    logger.info("Wrote %d image-level rows to %s", len(records), path)


def write_student_summary_csv(
    summaries: Sequence[StudentSummary],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STUDENT_SUMMARY_FIELDS)
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "student_id": s.student_id,
                "attempt_id": s.attempt_id,
                "course_id": s.course_id,
                "quiz_id": s.quiz_id,
                "quiz_name": s.quiz_name,
                "total_frames": s.total_frames,
                "valid_frames": s.valid_frames,
                "suspicious_frames": s.suspicious_frames,
                "percentage_suspicious": s.percentage_suspicious,
                "max_risk_score": s.max_risk_score,
                "mean_risk_score": s.mean_risk_score,
                "top_reasons": json.dumps(s.top_reasons),
                "incident_count": s.incident_count,
                "identity_stability_score": s.identity_stability_score,
                "overall_risk_level": s.overall_risk_level,
                "question_overview": json.dumps(s.question_overview),
                "flagged_timeline": json.dumps(s.flagged_timeline),
            })
    logger.info("Wrote %d student summaries to %s", len(summaries), path)
