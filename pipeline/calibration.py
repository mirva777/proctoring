"""
Per-student baseline calibration.

Uses the first N valid frames of an attempt to estimate the student's natural
head pose (accounting for webcam angle, posture, etc.).  Subsequent frames are
evaluated relative to this baseline, reducing false positives caused by
non-centred camera placement.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class BaselineCalibrator:
    """
    Accumulates yaw/pitch from the first ``n_frames`` valid frames per
    (student_id, attempt_id) and returns mean values as the calibration offset.
    """

    def __init__(self, n_frames: int = 5) -> None:
        self.n_frames = n_frames
        self._yaw_buffer: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._pitch_buffer: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._mouth_buffer: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._calibrated: dict[tuple[str, str], tuple[float, float, float | None]] = {}

    def update(
        self,
        student_id: str,
        attempt_id: str,
        yaw: Optional[float],
        pitch: Optional[float],
        mouth_open_ratio: Optional[float] = None,
    ) -> None:
        key = (student_id, attempt_id)
        if key in self._calibrated:
            return
        if yaw is None or pitch is None:
            return

        yaw_buf = self._yaw_buffer[key]
        pitch_buf = self._pitch_buffer[key]
        mouth_buf = self._mouth_buffer[key]

        if len(yaw_buf) < self.n_frames:
            yaw_buf.append(yaw)
            pitch_buf.append(pitch)
            if mouth_open_ratio is not None:
                mouth_buf.append(mouth_open_ratio)

        if len(yaw_buf) >= self.n_frames:
            mean_yaw = sum(yaw_buf) / len(yaw_buf)
            mean_pitch = sum(pitch_buf) / len(pitch_buf)
            mean_mouth = (sum(mouth_buf) / len(mouth_buf)) if mouth_buf else None
            self._calibrated[key] = (mean_yaw, mean_pitch, mean_mouth)
            logger.info(
                "Baseline calibrated for student=%s attempt=%s yaw=%.1f pitch=%.1f mouth=%s",
                student_id,
                attempt_id,
                mean_yaw,
                mean_pitch,
                f"{mean_mouth:.3f}" if mean_mouth is not None else "n/a",
            )

    def get_baseline(
        self, student_id: str, attempt_id: str
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Return calibrated pose + mouth baselines if available."""
        key = (student_id, attempt_id)
        if key in self._calibrated:
            return self._calibrated[key]
        return None, None, None

    def is_calibrated(self, student_id: str, attempt_id: str) -> bool:
        return (student_id, attempt_id) in self._calibrated
