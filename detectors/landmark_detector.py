"""
MediaPipe Face Mesh / FaceLandmarker landmark detector.

Supports both:
  - MediaPipe 0.10+ Tasks API  (face_landmarker.task)
  - MediaPipe 0.9.x Solutions API (legacy, auto-detected)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseLandmarkDetector, LandmarkResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")


class MediaPipeLandmarkDetector(BaseLandmarkDetector):
    """
    468-point facial landmark detector supporting MediaPipe 0.9 and 0.10+.

    Accuracy:  High for near-frontal faces; degrades at extreme poses.
    Speed:     ~10–25 ms per frame on CPU.
    License:   Apache 2.0.
    """

    def __init__(
        self,
        max_num_faces: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = _DEFAULT_MODEL,
        delegate: str = "cpu",
    ) -> None:
        self._max_num_faces = max_num_faces
        self._min_det_conf = min_detection_confidence
        self._min_trk_conf = min_tracking_confidence
        self._model_path = model_path
        self._delegate = (delegate or "cpu").lower()
        self._mesh: Any = None
        self._use_tasks: bool = False
        self._active_delegate: str = "cpu"

    def _create_tasks_landmarker(self, mp: Any, delegate: str) -> Any:
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}\n"
                "Run: python download_models.py"
            )

        delegate_enum = mp.tasks.BaseOptions.Delegate.CPU
        if delegate == "gpu":
            delegate_enum = mp.tasks.BaseOptions.Delegate.GPU

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self._model_path,
                delegate=delegate_enum,
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=self._max_num_faces,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def warmup(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            tasks_api_available = (
                hasattr(mp, "tasks")
                and hasattr(mp.tasks, "vision")
                and hasattr(mp.tasks.vision, "FaceLandmarker")
            )
            if tasks_api_available:
                delegates = [self._delegate]
                if self._delegate == "gpu":
                    delegates.append("cpu")

                for delegate in delegates:
                    try:
                        self._mesh = self._create_tasks_landmarker(mp, delegate)
                        self._use_tasks = True
                        self._active_delegate = delegate
                        logger.info(
                            "MediaPipeLandmarkDetector loaded (Tasks API 0.10+, delegate=%s)",
                            delegate,
                        )
                        return
                    except Exception as exc:
                        if delegate == "gpu":
                            logger.warning(
                                "MediaPipeLandmarkDetector GPU delegate failed (%s); retrying on CPU.",
                                exc,
                            )
                        else:
                            logger.warning("MediaPipeLandmarkDetector Tasks API warmup failed: %s", exc)

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                # Legacy 0.9.x CPU path
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=self._max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=self._min_det_conf,
                    min_tracking_confidence=self._min_trk_conf,
                )
                self._use_tasks = False
                self._active_delegate = "cpu"
                logger.info("MediaPipeLandmarkDetector loaded (legacy solutions API, delegate=cpu)")
        except ImportError:
            logger.warning("mediapipe not installed – landmark detection skipped.")
        except Exception as exc:
            logger.warning("MediaPipeLandmarkDetector warmup failed: %s", exc)

    def process(
        self,
        image: np.ndarray,
        face_bboxes: Optional[list[dict[str, Any]]] = None,
    ) -> LandmarkResult:
        if self._mesh is None:
            self.warmup()
        if self._mesh is None:
            return LandmarkResult(error="mediapipe unavailable", detector_name="mediapipe_mesh")

        try:
            import mediapipe as mp  # type: ignore

            h, w = image.shape[:2]
            rgb = image[:, :, ::-1].copy() if image.ndim == 3 and image.shape[2] == 3 else image
            landmarks: list[dict[str, Any]] = []
            raw: list[np.ndarray] = []
            transforms: list[np.ndarray] = []

            if self._use_tasks:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self._mesh.detect(mp_image)
                for face_lms in (result.face_landmarks or []):
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_lms])
                    raw.append(pts)
                    landmarks.append({
                        "count": len(face_lms),
                        "sample_nose_tip": {
                            "x": face_lms[4].x * w,
                            "y": face_lms[4].y * h,
                        },
                    })
                for transform in (result.facial_transformation_matrixes or []):
                    transforms.append(np.array(transform, dtype=np.float64))
            else:
                result = self._mesh.process(rgb)
                if result.multi_face_landmarks:
                    for face_lms in result.multi_face_landmarks:
                        pts = np.array([[lm.x, lm.y, lm.z] for lm in face_lms.landmark])
                        raw.append(pts)
                        landmarks.append({
                            "count": len(face_lms.landmark),
                            "sample_nose_tip": {
                                "x": face_lms.landmark[4].x * w,
                                "y": face_lms.landmark[4].y * h,
                            },
                        })

            return LandmarkResult(
                landmarks=landmarks,
                raw_landmarks=raw,
                face_transforms=transforms,
                detector_name="mediapipe_mesh",
            )
        except Exception as exc:
            logger.error("MediaPipeLandmarkDetector error: %s", exc)
            return LandmarkResult(error=str(exc), detector_name="mediapipe_mesh")

    def close(self) -> None:
        """Release the underlying MediaPipe mesh to avoid __del__ teardown errors."""
        if self._mesh is not None and hasattr(self._mesh, "close"):
            try:
                self._mesh.close()
            except Exception:
                pass
        self._mesh = None


class MTCNNLandmarkDetector(BaseLandmarkDetector):
    """
    5-point MTCNN facial landmark detector.

    This gives a GPU-capable fallback when MediaPipe's wheel is CPU-only, but it
    does not provide dense mouth landmarks or FaceLandmarker transform matrices.
    """

    _ORDER = ("nose", "left_eye", "right_eye", "mouth_left", "mouth_right")

    def __init__(
        self,
        min_detection_confidence: float = 0.85,
        device: str = "cpu",
    ) -> None:
        self._conf = min_detection_confidence
        self._requested_device = "GPU:0" if str(device).lower() in {"gpu", "cuda"} else "CPU:0"
        self._device = self._requested_device
        self._detector: Any = None
        self._available: Optional[bool] = None

    def warmup(self) -> None:
        if self._available is not None:
            return

        try:
            import tensorflow as tf  # type: ignore
            from mtcnn import MTCNN  # type: ignore

            devices = [self._requested_device]
            if self._requested_device != "CPU:0":
                devices.append("CPU:0")

            for device in devices:
                try:
                    with tf.device(f"/{device}"):
                        self._detector = MTCNN(stages="face_and_landmarks_detection", device=device)
                    self._device = device
                    self._available = True
                    logger.info("MTCNNLandmarkDetector loaded (device=%s)", self._device)
                    return
                except Exception as exc:
                    if device != "CPU:0":
                        logger.warning(
                            "MTCNNLandmarkDetector GPU warmup failed (%s); retrying on CPU.",
                            exc,
                        )
                    else:
                        logger.warning("MTCNNLandmarkDetector CPU warmup failed: %s", exc)

            self._detector = None
            self._available = False
        except Exception as exc:
            logger.warning("MTCNNLandmarkDetector warmup failed: %s", exc)
            self._detector = None
            self._available = False

    def process(
        self,
        image: np.ndarray,
        face_bboxes: Optional[list[dict[str, Any]]] = None,
    ) -> LandmarkResult:
        del face_bboxes
        if self._available is None:
            self.warmup()
        if not self._available or self._detector is None:
            return LandmarkResult(error="mtcnn unavailable", detector_name="mtcnn_5pt")

        try:
            h, w = image.shape[:2]
            rgb = image[:, :, ::-1].copy() if image.ndim == 3 and image.shape[2] == 3 else image
            faces = self._detector.detect_faces(rgb, box_format="xyxy") or []
            landmarks: list[dict[str, Any]] = []
            raw: list[np.ndarray] = []

            for face in faces:
                conf = float(face.get("confidence", 0.0))
                if conf < self._conf:
                    continue
                keypoints = face.get("keypoints") or {}
                if not all(name in keypoints for name in self._ORDER):
                    continue

                pts = np.zeros((len(self._ORDER), 3), dtype=np.float64)
                for idx, name in enumerate(self._ORDER):
                    x, y = keypoints[name]
                    pts[idx, 0] = float(x) / max(1.0, float(w))
                    pts[idx, 1] = float(y) / max(1.0, float(h))
                raw.append(pts)
                nose_x, nose_y = keypoints["nose"]
                landmarks.append(
                    {
                        "count": len(self._ORDER),
                        "sample_nose_tip": {
                            "x": float(nose_x),
                            "y": float(nose_y),
                        },
                    }
                )

            return LandmarkResult(
                landmarks=landmarks,
                raw_landmarks=raw,
                detector_name="mtcnn_5pt",
            )
        except Exception as exc:
            logger.error("MTCNNLandmarkDetector error: %s", exc)
            return LandmarkResult(error=str(exc), detector_name="mtcnn_5pt")


class FallbackLandmarkDetector(BaseLandmarkDetector):
    """Try the primary landmark backend first and fall back if it returns no points."""

    def __init__(
        self,
        primary: BaseLandmarkDetector,
        fallback: BaseLandmarkDetector,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def warmup(self) -> None:
        if hasattr(self.primary, "warmup"):
            self.primary.warmup()
        if hasattr(self.fallback, "warmup"):
            self.fallback.warmup()

    def process(
        self,
        image: np.ndarray,
        face_bboxes: Optional[list[dict[str, Any]]] = None,
    ) -> LandmarkResult:
        primary_result = self.primary.process(image, face_bboxes)
        if primary_result.error is None and primary_result.raw_landmarks:
            return primary_result

        fallback_result = self.fallback.process(image, face_bboxes)
        if fallback_result.error is None and fallback_result.raw_landmarks:
            fallback_result.detector_name = f"{primary_result.detector_name}+fallback"
            return fallback_result

        if primary_result.error is None:
            return primary_result
        return fallback_result

    def close(self) -> None:
        for detector in (self.primary, self.fallback):
            if hasattr(detector, "close"):
                try:
                    detector.close()
                except Exception:
                    pass
