"""
Robust image loader with error handling.

Supports JPEG, PNG and any format decodable by OpenCV.
Returns None (with log warning) for corrupt or missing files rather than
crashing the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


def load_image(
    image_path: str | Path,
    base_dir: Optional[str | Path] = None,
    max_dim: int = 1280,
) -> Optional[np.ndarray]:
    """
    Load an image from disk.

    Args:
        image_path: Absolute or relative path to the image file.
        base_dir:   If provided, relative paths are resolved relative to this.
        max_dim:    Downscale so the longest edge ≤ max_dim (preserves aspect).

    Returns:
        BGR numpy array, or None if the file is missing / corrupt.
    """
    path = Path(image_path)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path

    if not path.exists():
        logger.warning("Image not found: %s", path)
        return None

    try:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("OpenCV failed to decode: %s", path)
            return None

        # Optional downscale
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return img
    except Exception as exc:
        logger.warning("Unexpected error loading image %s: %s", path, exc)
        return None
