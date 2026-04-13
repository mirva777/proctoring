"""
Unit tests for risk scorer and student aggregator (pure functions).
Run with: pytest tests/ -v
"""

from __future__ import annotations

import pytest

from detectors.base import (
    FaceDetectionResult,
    IdentityResult,
    ObjectDetectionResult,
    PoseGazeResult,
    QualityResult,
)
from scoring.risk_scorer import RiskScorer, DEFAULT_WEIGHTS
from scoring.aggregator import (
    FrameRecord,
    StudentAggregator,
    aggregate_student,
    build_incidents,
)


# ---------------------------------------------------------------------------
# RiskScorer tests
# ---------------------------------------------------------------------------

def _make_scorer() -> RiskScorer:
    return RiskScorer()


def _ok_face() -> FaceDetectionResult:
    return FaceDetectionResult(face_count=1, bboxes=[], detector_name="test")


def _ok_pose() -> PoseGazeResult:
    return PoseGazeResult(look_away_flag=False, severity="none", method="test", confidence=0.9)


def _ok_identity() -> IdentityResult:
    return IdentityResult(similarity_score=0.95, mismatch_flag=False, reference_available=True)


def _ok_objects() -> ObjectDetectionResult:
    return ObjectDetectionResult(detector_name="test")


def _ok_quality() -> QualityResult:
    return QualityResult(blur_score=100.0, brightness_score=120.0, low_quality_flag=False)


class TestRiskScorer:
    def test_clean_frame_zero_score(self):
        scorer = _make_scorer()
        result = scorer.score(_ok_face(), _ok_pose(), _ok_identity(), _ok_objects(), _ok_quality())
        assert result.score == 0.0
        assert result.reasons == []

    def test_no_face(self):
        scorer = _make_scorer()
        face = FaceDetectionResult(face_count=0, detector_name="test")
        result = scorer.score(face, _ok_pose(), _ok_identity(), _ok_objects(), _ok_quality())
        assert "NO_FACE" in result.reasons
        assert result.score == DEFAULT_WEIGHTS["no_face"]

    def test_multi_face(self):
        scorer = _make_scorer()
        face = FaceDetectionResult(face_count=3, detector_name="test")
        result = scorer.score(face, _ok_pose(), _ok_identity(), _ok_objects(), _ok_quality())
        assert "MULTI_FACE" in result.reasons
        assert result.score == DEFAULT_WEIGHTS["multi_face"]

    def test_phone_detection(self):
        scorer = _make_scorer()
        obj = ObjectDetectionResult(phone_detected=True, phone_confidence=0.75, detector_name="test")
        result = scorer.score(_ok_face(), _ok_pose(), _ok_identity(), obj, _ok_quality())
        assert "PHONE" in result.reasons
        assert result.score == DEFAULT_WEIGHTS["phone"]

    def test_book_detection(self):
        scorer = _make_scorer()
        obj = ObjectDetectionResult(book_detected=True, book_confidence=0.75, detector_name="test")
        result = scorer.score(_ok_face(), _ok_pose(), _ok_identity(), obj, _ok_quality())
        assert "BOOK_NOTES" in result.reasons
        assert result.score == DEFAULT_WEIGHTS["book_notes"]

    def test_face_obstructed(self):
        scorer = _make_scorer()
        obj = ObjectDetectionResult(face_obstructed=True, obstruction_confidence=0.75, detector_name="test")
        result = scorer.score(_ok_face(), _ok_pose(), _ok_identity(), obj, _ok_quality())
        assert "FACE_OBSTRUCTED" in result.reasons
        assert result.score == DEFAULT_WEIGHTS["face_obstructed"]

    def test_score_capped_at_100(self):
        scorer = _make_scorer()
        face = FaceDetectionResult(face_count=0, detector_name="test")
        pose = PoseGazeResult(look_away_flag=True, severity="severe", method="test")
        obj = ObjectDetectionResult(
            phone_detected=True, phone_confidence=0.9,
            extra_person_detected=True, extra_person_confidence=0.8,
            detector_name="test",
        )
        identity = IdentityResult(similarity_score=0.1, mismatch_flag=True, reference_available=True)
        quality = QualityResult(low_quality_flag=True)
        result = scorer.score(face, pose, identity, obj, quality)
        assert result.score == 100.0

    def test_look_away_moderate(self):
        scorer = _make_scorer()
        pose = PoseGazeResult(look_away_flag=True, severity="moderate", method="test")
        result = scorer.score(_ok_face(), pose, _ok_identity(), _ok_objects(), _ok_quality())
        assert "LOOK_AWAY_MODERATE" in result.reasons
        assert result.score == DEFAULT_WEIGHTS["look_away_moderate"]

    def test_look_away_severe(self):
        scorer = _make_scorer()
        pose = PoseGazeResult(look_away_flag=True, severity="severe", method="test")
        result = scorer.score(_ok_face(), pose, _ok_identity(), _ok_objects(), _ok_quality())
        assert "LOOK_AWAY_SEVERE" in result.reasons
        assert result.score == DEFAULT_WEIGHTS["look_away_severe"]

    def test_identity_mismatch(self):
        scorer = _make_scorer()
        identity = IdentityResult(similarity_score=0.3, mismatch_flag=True, reference_available=True)
        result = scorer.score(_ok_face(), _ok_pose(), identity, _ok_objects(), _ok_quality())
        assert "IDENTITY_MISMATCH" in result.reasons
        assert result.score == DEFAULT_WEIGHTS["identity_mismatch"]

    def test_no_identity_without_reference(self):
        scorer = _make_scorer()
        identity = IdentityResult(similarity_score=0.0, mismatch_flag=True, reference_available=False)
        result = scorer.score(_ok_face(), _ok_pose(), identity, _ok_objects(), _ok_quality())
        assert "IDENTITY_MISMATCH" not in result.reasons

    def test_custom_weights(self):
        scorer = RiskScorer(weights={"no_face": 99.0})
        face = FaceDetectionResult(face_count=0, detector_name="test")
        result = scorer.score(face, _ok_pose(), _ok_identity(), _ok_objects(), _ok_quality())
        assert result.score == 99.0

    def test_low_quality_flag(self):
        scorer = _make_scorer()
        quality = QualityResult(low_quality_flag=True, blur_score=10.0)
        result = scorer.score(_ok_face(), _ok_pose(), _ok_identity(), _ok_objects(), quality)
        assert "LOW_QUALITY" in result.reasons


# ---------------------------------------------------------------------------
# Aggregator / incident tests
# ---------------------------------------------------------------------------

def _make_record(
    student_id: str = "s1",
    attempt_id: str = "a1",
    course_id: str = "c1",
    timestamp: str = "2024-01-01T09:00:00",
    risk_score: float = 0.0,
    reasons: list | None = None,
    error: str | None = None,
) -> FrameRecord:
    return FrameRecord(
        image_path="img.jpg",
        student_id=student_id,
        attempt_id=attempt_id,
        timestamp=timestamp,
        course_id=course_id,
        face_count=1,
        look_away_flag=False,
        severity="none",
        identity_mismatch=False,
        identity_similarity=1.0,
        phone_detected=False,
        extra_person_detected=False,
        book_detected=False,
        face_obstructed=False,
        talking_flag=False,
        talking_severity="none",
        talking_confidence=0.0,
        mouth_open_ratio=None,
        mouth_open_delta=None,
        low_quality=False,
        blur_score=100.0,
        brightness_score=120.0,
        glare_score=0.0,
        risk_score=risk_score,
        reasons=reasons or [],
        error=error,
    )


class TestAggregator:
    def test_clean_frames_zero_suspicious(self):
        records = [_make_record(risk_score=5.0) for _ in range(10)]
        summary = aggregate_student(records)
        assert summary.suspicious_frames == 0
        assert summary.overall_risk_level == "low"

    def test_suspicious_percentage(self):
        clean = [_make_record(risk_score=10.0) for _ in range(8)]
        susp = [
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:00:00", reasons=["PHONE"]),
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:00:05", reasons=["PHONE"]),
        ]
        summary = aggregate_student(clean + susp)
        assert summary.suspicious_frames == 2
        assert summary.percentage_suspicious == 20.0

    def test_incident_detection(self):
        susp = [
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:00:00", reasons=["PHONE"]),
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:00:05", reasons=["PHONE"]),
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:00:10", reasons=["PHONE"]),
        ]
        incidents = build_incidents(susp, suspicious_threshold=30.0, window_seconds=30, min_frames=2)
        assert len(incidents) == 1
        assert incidents[0].frame_count == 3

    def test_no_incident_single_frame(self):
        susp = [_make_record(risk_score=40.0, timestamp="2024-01-01T09:00:00", reasons=["LOOK_AWAY_SEVERE"])]
        incidents = build_incidents(susp, suspicious_threshold=30.0, window_seconds=30, min_frames=2)
        assert len(incidents) == 0

    def test_strong_single_frame_incident(self):
        susp = [_make_record(risk_score=80.0, timestamp="2024-01-01T09:00:00", reasons=["PHONE"])]
        incidents = build_incidents(susp, suspicious_threshold=30.0, window_seconds=30, min_frames=2)
        assert len(incidents) == 1
        assert incidents[0].frame_count == 1

    def test_two_separate_incidents(self):
        susp = [
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:00:00"),
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:00:05"),
            # 5-minute gap
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:05:00"),
            _make_record(risk_score=50.0, timestamp="2024-01-01T09:05:05"),
        ]
        incidents = build_incidents(susp, suspicious_threshold=30.0, window_seconds=30, min_frames=2)
        assert len(incidents) == 2

    def test_aggregator_groups_by_student_attempt(self):
        agg = StudentAggregator()
        for _ in range(5):
            agg.add(_make_record(student_id="s1", attempt_id="a1"))
        for _ in range(3):
            agg.add(_make_record(student_id="s2", attempt_id="a2"))
        summaries = agg.summaries()
        assert len(summaries) == 2
        totals = {(s.student_id, s.attempt_id): s.total_frames for s in summaries}
        assert totals[("s1", "a1")] == 5
        assert totals[("s2", "a2")] == 3

    def test_overall_risk_high(self):
        susp = [
            _make_record(risk_score=80.0, timestamp=f"2024-01-01T09:00:{i:02d}", reasons=["PHONE"])
            for i in range(5)
        ]
        clean = [_make_record(risk_score=5.0) for _ in range(10)]
        summary = aggregate_student(susp + clean)
        assert summary.overall_risk_level == "high"

    def test_empty_records(self):
        summary = aggregate_student([])
        assert summary.total_frames == 0

    def test_error_records_not_counted_as_valid(self):
        records = [
            _make_record(risk_score=0.0, error="image_load_failed") for _ in range(5)
        ]
        summary = aggregate_student(records)
        assert summary.valid_frames == 0
