"""Detector package – exposes all detector interfaces and default implementations."""

from .base import (
    FaceDetectionResult,
    LandmarkResult,
    PoseGazeResult,
    IdentityResult,
    ObjectDetectionResult,
    QualityResult,
    BaseFaceDetector,
    BaseLandmarkDetector,
    BasePoseGazeEstimator,
    BaseIdentityVerifier,
    BaseObjectDetector,
    BaseQualityAnalyzer,
)

__all__ = [
    "FaceDetectionResult",
    "LandmarkResult",
    "PoseGazeResult",
    "IdentityResult",
    "ObjectDetectionResult",
    "QualityResult",
    "BaseFaceDetector",
    "BaseLandmarkDetector",
    "BasePoseGazeEstimator",
    "BaseIdentityVerifier",
    "BaseObjectDetector",
    "BaseQualityAnalyzer",
]
