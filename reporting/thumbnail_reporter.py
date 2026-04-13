"""
Annotated thumbnail reporter.

Draws risk score, reasons, and face bounding boxes onto copies of flagged
frames and saves them to ``output_dir/flagged_thumbnails/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2  # type: ignore
import numpy as np

from scoring.aggregator import FrameRecord

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_RED = (0, 0, 255)
_ORANGE = (0, 165, 255)
_GREEN = (0, 200, 0)
_WHITE = (255, 255, 255)
_BG = (30, 30, 30)


def _risk_colour(score: float) -> tuple[int, int, int]:
    if score >= 70:
        return _RED
    if score >= 40:
        return _ORANGE
    return _GREEN


class ThumbnailReporter:
    """Saves annotated thumbnail images for flagged frames."""

    def __init__(
        self,
        output_dir: str | Path,
        score_threshold: float = 30.0,
        max_thumbnails: int = 500,
        thumb_max_dim: int = 640,
    ) -> None:
        self.output_dir = Path(output_dir) / "flagged_thumbnails"
        self.score_threshold = score_threshold
        self.max_thumbnails = max_thumbnails
        self.thumb_max_dim = thumb_max_dim
        self._count = 0

    def process_record(
        self,
        record: FrameRecord,
        image: Optional[np.ndarray] = None,
        base_dir: Optional[str | Path] = None,
    ) -> None:
        if record.risk_score < self.score_threshold:
            return
        if self._count >= self.max_thumbnails:
            return

        if image is None:
            from data_io.image_loader import load_image
            image = load_image(record.image_path, base_dir=base_dir, max_dim=self.thumb_max_dim)
        if image is None:
            return

        annotated = self._annotate(image.copy(), record)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = (
            f"{record.student_id}_{record.attempt_id}_{self._count:05d}.jpg"
            .replace("/", "_").replace("\\", "_")
        )
        out_path = self.output_dir / safe_name
        cv2.imwrite(str(out_path), annotated)
        self._count += 1

    @staticmethod
    def _annotate(image: np.ndarray, record: FrameRecord) -> np.ndarray:
        h, w = image.shape[:2]
        colour = _risk_colour(record.risk_score)

        # Top banner
        cv2.rectangle(image, (0, 0), (w, 40), _BG, -1)
        label = f"RISK: {record.risk_score:.0f}  {' | '.join(record.reasons[:4])}"
        cv2.putText(image, label, (6, 28), _FONT, 0.55, colour, 1, cv2.LINE_AA)

        # Bottom info
        cv2.rectangle(image, (0, h - 30), (w, h), _BG, -1)
        info = f"{record.student_id} | {record.timestamp}"
        cv2.putText(image, info, (6, h - 8), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)

        # Severity border
        thickness = 4 if record.risk_score >= 70 else 2
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), colour, thickness)

        return image
