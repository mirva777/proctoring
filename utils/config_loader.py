"""
YAML configuration loader with validation and sensible defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults – these are used when a key is absent from config.yaml
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "pipeline": {
        "enable_face_detection": True,
        "enable_landmarks": True,
        "enable_pose_gaze": True,
        "enable_identity": True,
        "enable_objects": True,
        "enable_quality": True,
        "enable_thumbnails": False,
        "suspicious_score_threshold": 30.0,
        "num_workers": 1,
        "batch_size": 8,
        "image_max_dim": 1280,
    },
    "detectors": {
        "face": {
            "backend": "mediapipe",           # mediapipe | mtcnn | opencv_haar
            "runtime_fallback_to_mediapipe": True,
            "runtime_fallback_to_haar": True,
            "mtcnn": {
                "min_detection_confidence": 0.9,
                "min_face_area_ratio": 0.02,
                "dedupe_iou_threshold": 0.45,
                "secondary_face_min_primary_ratio": 0.55,
            },
            "mediapipe": {
                "delegate": "auto",
                "min_detection_confidence": 0.5,
                "model_selection": 0,
            },
        },
        "landmarks": {
            "backend": "mediapipe",           # mediapipe | mtcnn
            "runtime_fallback_to_mediapipe": True,
            "mtcnn": {
                "min_detection_confidence": 0.9,
            },
            "mediapipe": {
                "delegate": "auto",
                "max_num_faces": 2,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
        },
        "pose_gaze": {
            "backend": "head_pose",           # head_pose | none
            "head_pose": {
                "moderate_yaw_deg": 20.0,
                "severe_yaw_deg": 35.0,
                "pitch_warning_deg": 20.0,
                "talking_ratio_threshold": 0.08,
                "talking_likely_ratio_threshold": 0.12,
                "talking_delta_threshold": 0.018,
                "talking_likely_delta_threshold": 0.035,
                "talking_max_abs_yaw_deg": 35.0,
                "talking_max_abs_pitch_deg": 35.0,
            },
        },
        "identity": {
            "backend": "deepface",            # deepface | none
            "deepface": {
                "delegate": "auto",
                "model_name": "ArcFace",
                "distance_metric": "cosine",
                "mismatch_threshold": 0.4,
            },
        },
        "objects": {
            "backend": "yolo",                # yolo | none
            "yolo": {
                "model_name": "yolov8n",
                "confidence": 0.35,
                "device": "auto",
            },
        },
        "quality": {
            "backend": "opencv",              # opencv
            "opencv": {
                "blur_threshold": 60.0,
                "brightness_min": 30.0,
                "brightness_max": 230.0,
                "glare_threshold": 245,
                "glare_fraction_max": 0.05,
                "roi_fraction": 0.6,
                "max_global_glare_fraction": 0.2,
            },
        },
    },
    "scoring": {
        "weights": {
            "no_face": 40.0,
            "face_hidden": 55.0,
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
        },
    },
    "aggregation": {
        "suspicious_threshold": 30.0,
        "incident_window_seconds": 30,
        "min_frames_per_incident": 2,
        "single_frame_high_risk_threshold": 50.0,
        "calibration_frames": 5,
    },
    "logging": {
        "level": "INFO",
        "json": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load config.yaml and merge with defaults.

    Returns the merged configuration dict.
    """
    config: dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            try:
                import yaml  # type: ignore

                with path.open("r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                config = loaded
                logger.info("Loaded config from %s", path)
            except ImportError:
                logger.warning(
                    "PyYAML not installed – using defaults. "
                    "Install with: pip install pyyaml"
                )
            except Exception as exc:
                logger.error("Failed to load config %s: %s – using defaults", path, exc)
        else:
            logger.warning("Config file %s not found – using defaults", path)

    return _deep_merge(_DEFAULTS, config)
