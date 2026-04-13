"""
Risk scorer – computes a 0-100 risk score per image frame.

All weights are configurable via config.yaml.  Pure function design ensures
easy unit testing – no I/O or model calls happen here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from detectors.base import (
    FaceDetectionResult,
    IdentityResult,
    ObjectDetectionResult,
    PoseGazeResult,
    QualityResult,
)


@dataclass
class RiskScore:
    """Output of a single-frame risk assessment."""

    score: float = 0.0                   # capped 0–100
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default weight table (keep in sync with config.yaml)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "no_face": 40.0,
    "face_hidden": 55.0,   # person body visible but face not detected (deliberate hiding)
    "multi_face": 60.0,
    "phone": 50.0,
    "book_notes": 25.0,
    "extra_person": 70.0,
    "face_obstructed": 30.0,
    "talking_possible": 20.0,
    "talking_likely": 35.0,
    "look_away_moderate": 20.0,
    "look_away_severe": 35.0,
    "identity_mismatch": 80.0,
    "low_quality": 10.0,
}


class RiskScorer:
    """
    Stateless risk scorer.  Accepts per-detection results and returns a
    RiskScore dataclass.  All weights come from config.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def score(
        self,
        face_result: FaceDetectionResult,
        pose_result: PoseGazeResult,
        identity_result: IdentityResult,
        object_result: ObjectDetectionResult,
        quality_result: QualityResult,
    ) -> RiskScore:
        total = 0.0
        reasons: list[str] = []
        details: dict = {}

        # --- Face count ---
        fc = face_result.face_count
        details["face_count"] = fc
        if face_result.error is None:
            if fc == 0:
                # Distinguish: body present but face hidden vs. nobody in frame
                body_visible = (
                    object_result.error is None
                    and object_result.person_count > 0
                )
                if body_visible:
                    total += self.weights["face_hidden"]
                    reasons.append("FACE_HIDDEN")
                    details["person_count"] = object_result.person_count
                else:
                    total += self.weights["no_face"]
                    reasons.append("NO_FACE")
            elif fc > 1:
                total += self.weights["multi_face"]
                reasons.append("MULTI_FACE")

        # --- Gaze / attention ---
        if pose_result.error is None or pose_result.look_away_flag:
            details["look_away_flag"] = pose_result.look_away_flag
            details["severity"] = pose_result.severity
            if pose_result.severity == "severe":
                total += self.weights["look_away_severe"]
                reasons.append("LOOK_AWAY_SEVERE")
            elif pose_result.severity == "moderate":
                total += self.weights["look_away_moderate"]
                reasons.append("LOOK_AWAY_MODERATE")

            details["mouth_open_ratio"] = pose_result.mouth_open_ratio
            details["mouth_open_delta"] = pose_result.mouth_open_delta
            if pose_result.talking_severity == "likely":
                total += self.weights["talking_likely"]
                reasons.append("TALKING_LIKELY")
                details["talking_confidence"] = pose_result.talking_confidence
            elif pose_result.talking_severity == "possible":
                total += self.weights["talking_possible"]
                reasons.append("TALKING_POSSIBLE")
                details["talking_confidence"] = pose_result.talking_confidence

        # --- Identity ---
        if identity_result.reference_available and identity_result.mismatch_flag:
            total += self.weights["identity_mismatch"]
            reasons.append("IDENTITY_MISMATCH")
            details["identity_similarity"] = identity_result.similarity_score

        # --- Objects ---
        if object_result.error is None:
            if object_result.phone_detected:
                total += self.weights["phone"]
                reasons.append("PHONE")
                details["phone_confidence"] = object_result.phone_confidence
            if object_result.extra_person_detected:
                total += self.weights["extra_person"]
                reasons.append("EXTRA_PERSON")
                details["extra_person_confidence"] = object_result.extra_person_confidence
            if object_result.book_detected:
                total += self.weights["book_notes"]
                reasons.append("BOOK_NOTES")
                details["book_confidence"] = object_result.book_confidence
            if object_result.face_obstructed:
                total += self.weights["face_obstructed"]
                reasons.append("FACE_OBSTRUCTED")

        # --- Quality ---
        if quality_result.low_quality_flag:
            total += self.weights["low_quality"]
            reasons.append("LOW_QUALITY")
        details["blur_score"] = quality_result.blur_score
        details["brightness"] = quality_result.brightness_score

        return RiskScore(
            score=min(100.0, round(total, 2)),
            reasons=reasons,
            details=details,
        )
