"""
Student-level aggregator.

Groups image-level results by (student_id, attempt_id) and produces a summary
with temporal incident logic.

All logic is implemented as pure functions for easy unit testing.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from scoring.risk_scorer import RiskScore


@dataclass
class FrameRecord:
    """Everything we know about one analysed frame."""

    image_path: str
    student_id: str
    attempt_id: str
    timestamp: str
    course_id: str
    face_count: int
    look_away_flag: bool
    severity: str
    identity_mismatch: bool
    identity_similarity: float
    phone_detected: bool
    extra_person_detected: bool
    book_detected: bool
    face_obstructed: bool
    talking_flag: bool
    talking_severity: str
    talking_confidence: float
    mouth_open_ratio: float | None
    mouth_open_delta: float | None
    low_quality: bool
    blur_score: float
    brightness_score: float
    glare_score: float
    risk_score: float
    reasons: list[str]
    person_detected: bool = False   # human body visible even without a face
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    gaze_direction: str | None = None
    pose_method: str = "unknown"
    quiz_id: str = ""
    quiz_name: str = ""
    quiz_page: str = ""
    question_id: str = ""
    question_slot: str = ""
    question_name: str = ""
    question_label: str = ""
    source_log_id: int | None = None
    source_webcampicture: str = ""
    source_filename: str = ""
    source_contenthash: str = ""
    source_moodledata_path: str = ""
    error: str | None = None


@dataclass
class Incident:
    """A temporal cluster of suspicious consecutive frames."""

    start_timestamp: str
    end_timestamp: str
    frame_count: int
    reasons: list[str]
    max_risk: float


@dataclass
class StudentSummary:
    student_id: str
    attempt_id: str
    course_id: str
    quiz_id: str = ""
    quiz_name: str = ""
    total_frames: int = 0
    valid_frames: int = 0
    suspicious_frames: int = 0
    percentage_suspicious: float = 0.0
    max_risk_score: float = 0.0
    mean_risk_score: float = 0.0
    top_reasons: list[str] = field(default_factory=list)
    incident_count: int = 0
    incidents: list[dict] = field(default_factory=list)
    flagged_timeline: list[dict] = field(default_factory=list)
    question_overview: list[str] = field(default_factory=list)
    identity_stability_score: float = 1.0
    overall_risk_level: str = "low"  # low | medium | high


# ---- Pure helper functions ------------------------------------------------


def _parse_ts(ts: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime(2000, 1, 1)


def build_incidents(
    records: list[FrameRecord],
    suspicious_threshold: float = 30.0,
    window_seconds: int = 30,
    min_frames: int = 2,
    single_frame_high_risk_threshold: float = 50.0,
) -> list[Incident]:
    """
    Group suspicious frames into temporal incidents.

    A new incident starts when a gap of > window_seconds seconds occurs or
    when the reasons change substantially.  Only incidents with >= min_frames
    are returned to reduce false positives.
    """
    if not records:
        return []

    suspicious = sorted(
        [r for r in records if r.risk_score >= suspicious_threshold],
        key=lambda r: _parse_ts(r.timestamp),
    )
    if not suspicious:
        return []

    incidents: list[Incident] = []
    window = timedelta(seconds=window_seconds)

    def _should_emit(frames: list[FrameRecord]) -> bool:
        if len(frames) >= min_frames:
            return True
        return max((f.risk_score for f in frames), default=0.0) >= single_frame_high_risk_threshold

    cur_start = suspicious[0].timestamp
    cur_end = suspicious[0].timestamp
    cur_frames = [suspicious[0]]

    for rec in suspicious[1:]:
        prev_ts = _parse_ts(cur_end)
        cur_ts = _parse_ts(rec.timestamp)
        if cur_ts - prev_ts <= window:
            cur_frames.append(rec)
            cur_end = rec.timestamp
        else:
            if _should_emit(cur_frames):
                all_reasons: list[str] = []
                for f in cur_frames:
                    all_reasons.extend(f.reasons)
                incidents.append(Incident(
                    start_timestamp=cur_start,
                    end_timestamp=cur_end,
                    frame_count=len(cur_frames),
                    reasons=list(dict.fromkeys(all_reasons)),
                    max_risk=max(f.risk_score for f in cur_frames),
                ))
            cur_start = rec.timestamp
            cur_end = rec.timestamp
            cur_frames = [rec]

    if _should_emit(cur_frames):
        all_reasons = []
        for f in cur_frames:
            all_reasons.extend(f.reasons)
        incidents.append(Incident(
            start_timestamp=cur_start,
            end_timestamp=cur_end,
            frame_count=len(cur_frames),
            reasons=list(dict.fromkeys(all_reasons)),
            max_risk=max(f.risk_score for f in cur_frames),
        ))

    return incidents


def aggregate_student(
    records: list[FrameRecord],
    suspicious_threshold: float = 30.0,
    window_seconds: int = 30,
    min_frames_per_incident: int = 2,
    single_frame_high_risk_threshold: float = 50.0,
) -> StudentSummary:
    """Pure function: compute student summary from a list of FrameRecords."""
    if not records:
        return StudentSummary(student_id="", attempt_id="", course_id="")

    first = records[0]
    summary = StudentSummary(
        student_id=first.student_id,
        attempt_id=first.attempt_id,
        course_id=first.course_id,
        quiz_id=first.quiz_id,
        quiz_name=first.quiz_name,
    )

    summary.total_frames = len(records)
    valid = [r for r in records if r.error is None]
    summary.valid_frames = len(valid)

    suspicious = [r for r in valid if r.risk_score >= suspicious_threshold]
    summary.suspicious_frames = len(suspicious)
    summary.percentage_suspicious = (
        round(100.0 * len(suspicious) / len(valid), 2) if valid else 0.0
    )

    scores = [r.risk_score for r in valid]
    summary.max_risk_score = max(scores) if scores else 0.0
    summary.mean_risk_score = round(sum(scores) / len(scores), 2) if scores else 0.0

    # Top reasons (by frequency)
    reason_counter: Counter = Counter()
    for r in suspicious:
        reason_counter.update(r.reasons)
    summary.top_reasons = [r for r, _ in reason_counter.most_common(10)]

    # Incidents
    incidents = build_incidents(
        records=valid,
        suspicious_threshold=suspicious_threshold,
        window_seconds=window_seconds,
        min_frames=min_frames_per_incident,
        single_frame_high_risk_threshold=single_frame_high_risk_threshold,
    )
    summary.incident_count = len(incidents)
    summary.incidents = [
        {
            "start": i.start_timestamp,
            "end": i.end_timestamp,
            "frames": i.frame_count,
            "reasons": i.reasons,
            "max_risk": i.max_risk,
        }
        for i in incidents
    ]

    # Flagged timeline
    summary.flagged_timeline = [
        {
            "timestamp": r.timestamp,
            "risk_score": r.risk_score,
            "reasons": r.reasons,
            "quiz_page": r.quiz_page,
            "question_label": r.question_label,
        }
        for r in suspicious
    ]

    question_labels: list[str] = []
    for r in sorted(valid, key=lambda item: _parse_ts(item.timestamp)):
        label = r.question_label.strip() if r.question_label else ""
        if label and label not in question_labels:
            question_labels.append(label)
    summary.question_overview = question_labels[:50]

    # Identity stability: mean similarity over frames with reference
    identity_scores = [
        r.identity_similarity
        for r in valid
        if r.identity_similarity is not None and not r.error
    ]
    summary.identity_stability_score = (
        round(sum(identity_scores) / len(identity_scores), 4)
        if identity_scores
        else 1.0
    )

    # Overall risk level (for human reviewers)
    if summary.max_risk_score >= 70 or summary.percentage_suspicious >= 30:
        summary.overall_risk_level = "high"
    elif summary.max_risk_score >= 40 or summary.percentage_suspicious >= 10:
        summary.overall_risk_level = "medium"
    else:
        summary.overall_risk_level = "low"

    return summary


class StudentAggregator:
    """Stateful accumulator that groups FrameRecords by (student_id, attempt_id)."""

    def __init__(
        self,
        suspicious_threshold: float = 30.0,
        window_seconds: int = 30,
        min_frames_per_incident: int = 2,
        single_frame_high_risk_threshold: float = 50.0,
    ) -> None:
        self.suspicious_threshold = suspicious_threshold
        self.window_seconds = window_seconds
        self.min_frames = min_frames_per_incident
        self.single_frame_high_risk_threshold = single_frame_high_risk_threshold
        self._buckets: dict[tuple[str, str], list[FrameRecord]] = defaultdict(list)

    def add(self, record: FrameRecord) -> None:
        self._buckets[(record.student_id, record.attempt_id)].append(record)

    def summaries(self) -> list[StudentSummary]:
        result = []
        for records in self._buckets.values():
            result.append(
                aggregate_student(
                    records,
                    suspicious_threshold=self.suspicious_threshold,
                    window_seconds=self.window_seconds,
                    min_frames_per_incident=self.min_frames,
                    single_frame_high_risk_threshold=self.single_frame_high_risk_threshold,
                )
            )
        return result
