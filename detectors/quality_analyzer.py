"""
Image quality / tampering signal analyzer.

All computations use OpenCV/NumPy – no additional model required.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2  # type: ignore
import numpy as np

from .base import BaseQualityAnalyzer, QualityResult

logger = logging.getLogger(__name__)


class CVQualityAnalyzer(BaseQualityAnalyzer):
    """
    Pure OpenCV quality analyzer.

    Computes:
      - Blur score    : Laplacian variance (higher = sharper).
      - Brightness    : Mean of grayscale image [0–255].
      - Glare score   : Fraction of pixels with value > glare_threshold.
      - low_quality   : True if blur < blur_threshold or brightness < brightness_min
                        or brightness > brightness_max or glare > glare_fraction_max.
    """

    def __init__(
        self,
        blur_threshold: float = 60.0,
        brightness_min: float = 30.0,
        brightness_max: float = 230.0,
        glare_threshold: int = 245,
        glare_fraction_max: float = 0.05,
        roi_fraction: float = 0.6,
        max_global_glare_fraction: float = 0.2,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.glare_threshold = glare_threshold
        self.glare_fraction_max = glare_fraction_max
        self.roi_fraction = min(1.0, max(0.2, roi_fraction))
        self.max_global_glare_fraction = max_global_glare_fraction

    def _center_crop(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        crop_h = max(1, int(h * self.roi_fraction))
        crop_w = max(1, int(w * self.roi_fraction))
        y1 = max(0, (h - crop_h) // 2)
        x1 = max(0, (w - crop_w) // 2)
        return gray[y1:y1 + crop_h, x1:x1 + crop_w]

    def _face_roi(
        self,
        gray: np.ndarray,
        face_bboxes: Optional[list[dict[str, Any]]],
    ) -> np.ndarray:
        if not face_bboxes:
            return self._center_crop(gray)

        best_face = max(
            face_bboxes,
            key=lambda bb: max(1, (int(bb["x2"]) - int(bb["x1"])) * (int(bb["y2"]) - int(bb["y1"]))),
        )
        h, w = gray.shape[:2]
        x1 = int(best_face["x1"])
        y1 = int(best_face["y1"])
        x2 = int(best_face["x2"])
        y2 = int(best_face["y2"])
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)

        crop_x1 = max(0, int(x1 - 0.35 * face_w))
        crop_y1 = max(0, int(y1 - 0.2 * face_h))
        crop_x2 = min(w, int(x2 + 0.35 * face_w))
        crop_y2 = min(h, int(y2 + 0.45 * face_h))
        return gray[crop_y1:crop_y2, crop_x1:crop_x2]

    def process(
        self,
        image: np.ndarray,
        face_bboxes: Optional[list[dict[str, Any]]] = None,
    ) -> QualityResult:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            roi = self._face_roi(gray, face_bboxes)

            # Use the central region as a proxy for the face / student area.
            blur_score = float(cv2.Laplacian(roi, cv2.CV_64F).var())

            brightness = float(roi.mean())

            global_glare_score = float((gray > self.glare_threshold).mean())
            glare_score = float((roi > self.glare_threshold).mean())

            low_quality = (
                blur_score < self.blur_threshold
                or brightness < self.brightness_min
                or brightness > self.brightness_max
                or glare_score > self.glare_fraction_max
                or global_glare_score > self.max_global_glare_fraction
            )

            return QualityResult(
                blur_score=round(blur_score, 2),
                brightness_score=round(brightness, 2),
                glare_score=round(glare_score, 4),
                low_quality_flag=low_quality,
            )
        except Exception as exc:
            logger.error("CVQualityAnalyzer error: %s", exc)
            return QualityResult(error=str(exc))
