"""
Abstract base classes and result dataclasses for all detector components.

Every concrete detector must inherit from the corresponding base class and
implement the ``process(image)`` method.  This makes the pipeline fully
pluggable – swap any backend without touching the pipeline or scoring code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FaceDetectionResult:
    """Output of a face-detection pass on one frame."""

    face_count: int = 0
    # Each bbox: [x1, y1, x2, y2] in pixel coords, confidence
    bboxes: list[dict[str, Any]] = field(default_factory=list)
    detector_name: str = "unknown"
    error: Optional[str] = None


@dataclass
class LandmarkResult:
    """68-point (or similar) facial landmark output."""

    landmarks: list[dict[str, Any]] = field(default_factory=list)
    # One entry per detected face: list of (x, y) normalised [0,1] coords
    raw_landmarks: list[np.ndarray] = field(default_factory=list)
    # Optional 4x4 rigid transforms from MediaPipe FaceLandmarker.
    face_transforms: list[np.ndarray] = field(default_factory=list)
    detector_name: str = "unknown"
    error: Optional[str] = None


@dataclass
class PoseGazeResult:
    """Head pose / gaze estimation result."""

    yaw: Optional[float] = None        # degrees, + = turn right
    pitch: Optional[float] = None      # degrees, + = look up
    roll: Optional[float] = None       # degrees
    gaze_vector: Optional[tuple[float, float, float]] = None
    gaze_direction: Optional[str] = None   # "center"|"left"|"right"|"up"|"down"
    look_away_flag: bool = False
    severity: str = "none"             # "none" | "moderate" | "severe"
    mouth_open_ratio: Optional[float] = None
    mouth_open_delta: Optional[float] = None
    talking_flag: bool = False
    talking_severity: str = "none"     # "none" | "possible" | "likely"
    talking_confidence: float = 0.0
    confidence: float = 0.0
    method: str = "unknown"            # "direct_gaze" | "head_pose_proxy" | "none"
    error: Optional[str] = None


@dataclass
class IdentityResult:
    """Face-verification result comparing current frame to reference."""

    similarity_score: float = 1.0      # 1.0 = perfect match
    mismatch_flag: bool = False
    reference_available: bool = False
    verifier_name: str = "unknown"
    error: Optional[str] = None


@dataclass
class ObjectDetectionResult:
    """Scene / suspicious-object detection output."""

    phone_detected: bool = False
    phone_confidence: float = 0.0
    book_detected: bool = False
    book_confidence: float = 0.0
    extra_person_detected: bool = False
    extra_person_confidence: float = 0.0
    # Total bodies detected by YOLO (including the student themselves).
    # > 0 means at least one human body is visible even if MediaPipe found no face.
    person_count: int = 0
    person_confidence: float = 0.0
    face_obstructed: bool = False
    obstruction_confidence: float = 0.0
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    detector_name: str = "unknown"
    error: Optional[str] = None


@dataclass
class QualityResult:
    """Image quality / tampering signals."""

    blur_score: float = 0.0            # Laplacian variance – higher = sharper
    brightness_score: float = 0.0     # Mean pixel value [0–255]
    glare_score: float = 0.0          # Fraction of saturated pixels
    low_quality_flag: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract base classes (the plug-in contracts)
# ---------------------------------------------------------------------------


class BaseFaceDetector(ABC):
    """Detect faces and return bounding boxes + count."""

    @abstractmethod
    def process(self, image: np.ndarray) -> FaceDetectionResult:
        ...

    def warmup(self) -> None:
        """Optional: pre-load model weights."""


class BaseLandmarkDetector(ABC):
    """Detect facial landmarks (used for pose proxy & obstruction check)."""

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        face_bboxes: Optional[list[dict[str, Any]]] = None,
    ) -> LandmarkResult:
        ...

    def warmup(self) -> None:
        """Optional: pre-load model weights."""


class BasePoseGazeEstimator(ABC):
    """Estimate head pose or gaze direction from landmarks / image."""

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        landmark_result: Optional[LandmarkResult] = None,
    ) -> PoseGazeResult:
        ...

    def warmup(self) -> None:
        """Optional: pre-load model weights."""


class BaseIdentityVerifier(ABC):
    """Compare current-frame face to a per-student reference embedding."""

    @abstractmethod
    def build_reference(
        self,
        reference_images: list[np.ndarray],
        student_id: str,
    ) -> None:
        """Compute and store reference embedding for a student."""
        ...

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        student_id: str,
    ) -> IdentityResult:
        ...

    def warmup(self) -> None:
        """Optional: pre-load model weights."""

    def close(self) -> None:
        """Optional: release any model resources."""


class BaseObjectDetector(ABC):
    """Detect suspicious objects / scene events in the frame."""

    @abstractmethod
    def process(self, image: np.ndarray) -> ObjectDetectionResult:
        ...

    def warmup(self) -> None:
        """Optional: pre-load model weights."""


class BaseQualityAnalyzer(ABC):
    """Compute image quality / tampering signals."""

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        face_bboxes: Optional[list[dict[str, Any]]] = None,
    ) -> QualityResult:
        ...
