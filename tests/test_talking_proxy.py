from __future__ import annotations

import numpy as np

from detectors.base import (
    FaceDetectionResult,
    IdentityResult,
    LandmarkResult,
    ObjectDetectionResult,
    PoseGazeResult,
    QualityResult,
)
from detectors.pose_gaze_estimator import HeadPoseEstimator
from pipeline.calibration import BaselineCalibrator
from scoring.risk_scorer import DEFAULT_WEIGHTS, RiskScorer


def _make_landmarks(gap: float) -> LandmarkResult:
    pts = np.full((468, 3), 0.5, dtype=np.float64)
    pts[78, :2] = [0.4, 0.5]
    pts[308, :2] = [0.6, 0.5]

    pairs = [
        (13, 14),
        (81, 178),
        (82, 87),
        (311, 402),
        (312, 317),
    ]
    for upper, lower in pairs:
        pts[upper, :2] = [0.5, 0.5 - gap / 2]
        pts[lower, :2] = [0.5, 0.5 + gap / 2]

    return LandmarkResult(raw_landmarks=[pts], detector_name="test")


def test_measure_mouth_open_ratio_is_small_for_closed_mouth():
    lms = _make_landmarks(gap=0.008)
    ratio = HeadPoseEstimator.measure_mouth_open_ratio(lms, (200, 200, 3))
    assert ratio is not None
    assert ratio < 0.05


def test_measure_mouth_open_ratio_is_large_for_open_mouth():
    lms = _make_landmarks(gap=0.04)
    ratio = HeadPoseEstimator.measure_mouth_open_ratio(lms, (200, 200, 3))
    assert ratio is not None
    assert ratio > 0.15


def test_classify_talking_uses_personal_baseline_delta():
    estimator = HeadPoseEstimator()
    flag, severity, confidence, delta = estimator.classify_talking(
        mouth_open_ratio=0.125,
        baseline_mouth_open_ratio=0.075,
        yaw=0.0,
        pitch=0.0,
    )
    assert flag is True
    assert severity == "likely"
    assert confidence > 0.5
    assert delta is not None and round(delta, 3) == 0.05


def test_classify_talking_suppressed_for_extreme_pose():
    estimator = HeadPoseEstimator()
    flag, severity, confidence, delta = estimator.classify_talking(
        mouth_open_ratio=0.13,
        baseline_mouth_open_ratio=0.07,
        yaw=50.0,
        pitch=0.0,
    )
    assert flag is False
    assert severity == "none"
    assert delta is not None
    assert confidence < 0.5


def test_talking_possible_adds_risk_reason():
    scorer = RiskScorer()
    pose = PoseGazeResult(
        look_away_flag=False,
        severity="none",
        mouth_open_ratio=0.1,
        mouth_open_delta=0.025,
        talking_flag=True,
        talking_severity="possible",
        talking_confidence=0.8,
        method="test",
    )
    result = scorer.score(
        FaceDetectionResult(face_count=1, detector_name="test"),
        pose,
        IdentityResult(reference_available=True, mismatch_flag=False, similarity_score=1.0),
        ObjectDetectionResult(detector_name="test"),
        QualityResult(low_quality_flag=False),
    )
    assert "TALKING_POSSIBLE" in result.reasons
    assert result.score == DEFAULT_WEIGHTS["talking_possible"]


def test_calibrator_stores_mouth_baseline():
    calibrator = BaselineCalibrator(n_frames=2)
    calibrator.update("s1", "a1", yaw=0.0, pitch=1.0, mouth_open_ratio=0.05)
    calibrator.update("s1", "a1", yaw=2.0, pitch=3.0, mouth_open_ratio=0.07)

    yaw, pitch, mouth = calibrator.get_baseline("s1", "a1")

    assert round(yaw or 0.0, 2) == 1.0
    assert round(pitch or 0.0, 2) == 2.0
    assert round(mouth or 0.0, 3) == 0.06
