"""
Head-pose proxy and optional direct gaze estimator.

Default: MediaPipe FaceLandmarker rigid transform matrices when available,
with OpenCV solvePnP as a fallback using 6 canonical 3-D face model points
derived from MediaPipe landmarks (468-point mesh). This gives yaw/pitch/roll
without any additional model download.

Optional (if l2csgaze or other gaze model is available): direct gaze estimation
replaces the proxy and outputs a normalised gaze vector + direction.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import cv2  # type: ignore
import numpy as np

from .base import BasePoseGazeEstimator, LandmarkResult, PoseGazeResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical 3-D face model (millimetres) for solvePnP
# These correspond to MediaPipe mesh indices for the 6 canonical points:
# nose tip, chin, left eye corner, right eye corner, left mouth, right mouth.
# ---------------------------------------------------------------------------
_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip        – idx 4
        [0.0, -330.0, -65.0],     # Chin            – idx 152
        [-225.0, 170.0, -135.0],  # Left eye corner – idx 263
        [225.0, 170.0, -135.0],   # Right eye corner– idx 33
        [-150.0, -150.0, -125.0], # Left mouth      – idx 287
        [150.0, -150.0, -125.0],  # Right mouth     – idx 57
    ],
    dtype=np.float64,
)

# Corresponding MediaPipe landmark indices
_MP_IDX = [4, 152, 263, 33, 287, 57]
_MTCNN_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip
        [-225.0, 170.0, -135.0],  # Left eye center
        [225.0, 170.0, -135.0],   # Right eye center
        [-150.0, -150.0, -125.0], # Left mouth corner
        [150.0, -150.0, -125.0],  # Right mouth corner
    ],
    dtype=np.float64,
)
_MTCNN_IDX = [0, 1, 2, 3, 4]
_MOUTH_WIDTH_IDX = (78, 308)
_MOUTH_GAP_PAIRS = [
    (13, 14),
    (81, 178),
    (82, 87),
    (311, 402),
    (312, 317),
]


def _rotation_matrix_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3×3 rotation matrix to yaw/pitch/roll in degrees."""
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(rmat[2, 1], rmat[2, 2])
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        roll = math.atan2(-rmat[1, 2], rmat[1, 1])
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = 0.0
    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


def _normalize_signed_angle(angle: float) -> float:
    """Wrap an angle into [-180, 180)."""
    return ((angle + 180.0) % 360.0) - 180.0


def _fold_face_angle(angle: float) -> float:
    """
    Fold ambiguous head-pose angles into the frontal-face range.

    solvePnP + Euler decomposition can return near-180 degree yaw/pitch for a
    face that is actually close to frontal. Because a visible webcam face should
    stay within roughly +/-90 degrees, we keep the closest equivalent angle.
    """
    angle = _normalize_signed_angle(angle)
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


class HeadPoseEstimator(BasePoseGazeEstimator):
    """
    Head-pose proxy using MediaPipe FaceLandmarker transforms with solvePnP fallback.

    Accuracy:  Moderate.  ±5–10° error is typical.
    Speed:     < 1 ms CPU math on top of the landmark detector output.
    License:   OpenCV BSD + MediaPipe Apache 2.0.
    """

    def __init__(
        self,
        moderate_yaw_deg: float = 20.0,
        severe_yaw_deg: float = 35.0,
        pitch_warning_deg: float = 20.0,
        talking_ratio_threshold: float = 0.08,
        talking_likely_ratio_threshold: float = 0.12,
        talking_delta_threshold: float = 0.018,
        talking_likely_delta_threshold: float = 0.035,
        talking_max_abs_yaw_deg: float = 35.0,
        talking_max_abs_pitch_deg: float = 35.0,
        baseline_yaw: Optional[float] = None,
        baseline_pitch: Optional[float] = None,
    ) -> None:
        self.moderate_yaw = moderate_yaw_deg
        self.severe_yaw = severe_yaw_deg
        self.pitch_warning = pitch_warning_deg
        self.talking_ratio_threshold = talking_ratio_threshold
        self.talking_likely_ratio_threshold = talking_likely_ratio_threshold
        self.talking_delta_threshold = talking_delta_threshold
        self.talking_likely_delta_threshold = talking_likely_delta_threshold
        self.talking_max_abs_yaw = talking_max_abs_yaw_deg
        self.talking_max_abs_pitch = talking_max_abs_pitch_deg
        self.baseline_yaw = baseline_yaw
        self.baseline_pitch = baseline_pitch

    @staticmethod
    def normalize_pose_angles(
        yaw: float,
        pitch: float,
        roll: float,
    ) -> tuple[float, float, float]:
        """Convert ambiguous Euler output into webcam-friendly pose angles."""
        return (
            _fold_face_angle(yaw),
            _fold_face_angle(pitch),
            _normalize_signed_angle(roll),
        )

    @staticmethod
    def _orthonormalize_rotation(rot_scale: np.ndarray) -> np.ndarray:
        """Strip scale/skew from a 3x3 transform block and keep a proper rotation."""
        u, _, vt = np.linalg.svd(rot_scale)
        rmat = u @ vt
        if np.linalg.det(rmat) < 0:
            u[:, -1] *= -1
            rmat = u @ vt
        return rmat

    @classmethod
    def pose_from_face_transform(
        cls,
        transform_matrix: Optional[np.ndarray],
    ) -> Optional[tuple[float, float, float]]:
        """Estimate yaw/pitch/roll from a MediaPipe 4x4 facial transform matrix."""
        if transform_matrix is None:
            return None

        matrix = np.asarray(transform_matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return None

        rmat = cls._orthonormalize_rotation(matrix[:3, :3])
        yaw, pitch, roll = _rotation_matrix_to_euler(rmat)
        return cls.normalize_pose_angles(yaw, pitch, roll)

    def classify_attention(
        self,
        yaw: float,
        pitch: float,
    ) -> tuple[bool, str, str]:
        """Classify attention state from yaw/pitch."""
        look_away = False
        severity = "none"
        if abs(yaw) > self.severe_yaw:
            look_away = True
            severity = "severe"
        elif abs(yaw) > self.moderate_yaw or abs(pitch) > self.pitch_warning:
            look_away = True
            severity = "moderate"
        return look_away, severity, self._direction(yaw, pitch)

    @staticmethod
    def _pixel_distance(
        pts_norm: np.ndarray,
        idx_a: int,
        idx_b: int,
        width: int,
        height: int,
    ) -> float:
        ax, ay = pts_norm[idx_a, 0] * width, pts_norm[idx_a, 1] * height
        bx, by = pts_norm[idx_b, 0] * width, pts_norm[idx_b, 1] * height
        return float(math.hypot(ax - bx, ay - by))

    @classmethod
    def measure_mouth_open_ratio(
        cls,
        landmark_result: Optional[LandmarkResult],
        image_shape: tuple[int, int, int] | tuple[int, int],
    ) -> Optional[float]:
        if landmark_result is None or not landmark_result.raw_landmarks:
            return None

        pts_norm = landmark_result.raw_landmarks[0]
        required_idx = max(
            _MOUTH_WIDTH_IDX[0],
            _MOUTH_WIDTH_IDX[1],
            max(max(pair) for pair in _MOUTH_GAP_PAIRS),
        )
        if pts_norm.shape[0] <= required_idx:
            return None

        height, width = image_shape[:2]
        mouth_width = cls._pixel_distance(
            pts_norm,
            _MOUTH_WIDTH_IDX[0],
            _MOUTH_WIDTH_IDX[1],
            width,
            height,
        )
        if mouth_width <= 1e-6:
            return None

        gaps = [
            cls._pixel_distance(pts_norm, upper_idx, lower_idx, width, height)
            for upper_idx, lower_idx in _MOUTH_GAP_PAIRS
        ]
        mean_gap = float(sum(gaps) / len(gaps))
        return mean_gap / mouth_width

    def classify_talking(
        self,
        mouth_open_ratio: Optional[float],
        baseline_mouth_open_ratio: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> tuple[bool, str, float, Optional[float]]:
        if mouth_open_ratio is None:
            return False, "none", 0.0, None

        baseline = baseline_mouth_open_ratio or 0.0
        delta = mouth_open_ratio - baseline

        yaw_val = abs(yaw or 0.0)
        pitch_val = abs(pitch or 0.0)
        pose_supported = (
            yaw is None
            or pitch is None
            or (
                yaw_val <= self.talking_max_abs_yaw
                and pitch_val <= self.talking_max_abs_pitch
            )
        )

        ratio_score = max(
            0.0,
            (mouth_open_ratio - self.talking_ratio_threshold)
            / max(1e-6, self.talking_likely_ratio_threshold - self.talking_ratio_threshold),
        )
        delta_score = max(
            0.0,
            (delta - self.talking_delta_threshold)
            / max(1e-6, self.talking_likely_delta_threshold - self.talking_delta_threshold),
        )
        confidence = min(1.0, 0.45 * ratio_score + 0.45 * delta_score + 0.1)
        if not pose_supported:
            confidence *= 0.35

        likely = (
            pose_supported
            and mouth_open_ratio >= self.talking_likely_ratio_threshold
            and delta >= self.talking_likely_delta_threshold
        )
        possible = (
            pose_supported
            and mouth_open_ratio >= self.talking_ratio_threshold
            and delta >= self.talking_delta_threshold
        )

        if likely:
            return True, "likely", round(confidence, 3), round(delta, 4)
        if possible:
            return True, "possible", round(confidence, 3), round(delta, 4)
        return False, "none", round(confidence, 3), round(delta, 4)

    def process(
        self,
        image: np.ndarray,
        landmark_result: Optional[LandmarkResult] = None,
    ) -> PoseGazeResult:
        if landmark_result is None or not landmark_result.raw_landmarks:
            return PoseGazeResult(
                method="head_pose_proxy",
                error="no landmarks available",
                severity="none",
            )

        try:
            pts_norm = landmark_result.raw_landmarks[0]  # first face
            h, w = image.shape[:2]
            yaw = pitch = roll = None
            method = "face_transform_matrix"

            if landmark_result.face_transforms:
                pose = self.pose_from_face_transform(landmark_result.face_transforms[0])
                if pose is not None:
                    yaw, pitch, roll = pose

            if yaw is None or pitch is None or roll is None:
                if pts_norm.shape[0] > max(_MP_IDX):
                    model_3d = _MODEL_3D
                    landmark_idx = _MP_IDX
                    method = "head_pose_proxy"
                    pnp_flag = cv2.SOLVEPNP_ITERATIVE
                elif pts_norm.shape[0] >= len(_MTCNN_IDX):
                    model_3d = _MTCNN_MODEL_3D
                    landmark_idx = _MTCNN_IDX
                    method = "mtcnn_head_pose_proxy"
                    pnp_flag = cv2.SOLVEPNP_EPNP
                else:
                    return PoseGazeResult(
                        method="head_pose_proxy",
                        error=f"insufficient landmarks: {pts_norm.shape[0]}",
                    )

                image_points = np.array(
                    [[pts_norm[i, 0] * w, pts_norm[i, 1] * h] for i in landmark_idx],
                    dtype=np.float64,
                )
                focal = w
                cam_matrix = np.array(
                    [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]],
                    dtype=np.float64,
                )
                dist_coeffs = np.zeros((4, 1), dtype=np.float64)

                success, rvec, tvec = cv2.solvePnP(
                    model_3d,
                    image_points,
                    cam_matrix,
                    dist_coeffs,
                    flags=pnp_flag,
                )
                if not success:
                    return PoseGazeResult(method="head_pose_proxy", error="solvePnP failed")

                rmat, _ = cv2.Rodrigues(rvec)
                yaw, pitch, roll = _rotation_matrix_to_euler(rmat)
                yaw, pitch, roll = self.normalize_pose_angles(yaw, pitch, roll)

            mouth_open_ratio = self.measure_mouth_open_ratio(landmark_result, image.shape)

            # Apply calibration baseline correction
            eff_yaw = yaw - (self.baseline_yaw or 0.0)
            eff_pitch = pitch - (self.baseline_pitch or 0.0)

            look_away, severity, direction = self.classify_attention(eff_yaw, eff_pitch)

            return PoseGazeResult(
                yaw=round(yaw, 2),
                pitch=round(pitch, 2),
                roll=round(roll, 2),
                gaze_direction=direction,
                look_away_flag=look_away,
                severity=severity,
                mouth_open_ratio=round(mouth_open_ratio, 4) if mouth_open_ratio is not None else None,
                confidence=0.92 if method == "face_transform_matrix" else 0.85,
                method=method,
            )
        except Exception as exc:
            logger.error("HeadPoseEstimator error: %s", exc)
            return PoseGazeResult(method="head_pose_proxy", error=str(exc))

    @staticmethod
    def _direction(yaw: float, pitch: float) -> str:
        if abs(yaw) <= 15 and abs(pitch) <= 15:
            return "center"
        if abs(yaw) >= abs(pitch):
            return "right" if yaw > 0 else "left"
        return "up" if pitch > 0 else "down"


class NullPoseGazeEstimator(BasePoseGazeEstimator):
    """No-op estimator for when the feature is explicitly disabled."""

    def process(
        self,
        image: np.ndarray,
        landmark_result: Optional[LandmarkResult] = None,
    ) -> PoseGazeResult:
        return PoseGazeResult(method="none", error="pose/gaze disabled")
