"""
MediaPipe-based face detector.

Supports both:
  - MediaPipe 0.10+ Tasks API  (blaze_face_short_range.tflite)
  - MediaPipe 0.9.x Solutions API (legacy, auto-detected)

Falls back to OpenCV Haar when MediaPipe is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseFaceDetector, FaceDetectionResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = str(Path(__file__).parent.parent / "models" / "blaze_face_short_range.tflite")


class MediaPipeFaceDetector(BaseFaceDetector):
    """
    BlazeFace detector supporting MediaPipe 0.9 (solutions) and 0.10+ (Tasks).

    Accuracy:  Good for frontal/near-frontal faces in typical webcam frames.
    Speed:     ~5–15 ms per frame on CPU.
    License:   Apache 2.0.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
        model_path: str = _DEFAULT_MODEL,
        delegate: str = "cpu",
    ) -> None:
        self._conf = min_detection_confidence
        self._model_selection = model_selection
        self._model_path = model_path
        self._delegate = (delegate or "cpu").lower()
        self._detector: Any = None
        self._use_tasks: bool = False
        self._active_delegate: str = "cpu"

    def _create_tasks_detector(self, mp: Any, delegate: str) -> Any:
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}\n"
                "Run: python download_models.py"
            )

        delegate_enum = mp.tasks.BaseOptions.Delegate.CPU
        if delegate == "gpu":
            delegate_enum = mp.tasks.BaseOptions.Delegate.GPU

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self._model_path,
            delegate=delegate_enum,
        )
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self._conf,
        )
        return mp.tasks.vision.FaceDetector.create_from_options(options)

    def warmup(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            tasks_api_available = (
                hasattr(mp, "tasks")
                and hasattr(mp.tasks, "vision")
                and hasattr(mp.tasks.vision, "FaceDetector")
            )
            if tasks_api_available:
                delegates = [self._delegate]
                if self._delegate == "gpu":
                    delegates.append("cpu")

                for delegate in delegates:
                    try:
                        self._detector = self._create_tasks_detector(mp, delegate)
                        self._use_tasks = True
                        self._active_delegate = delegate
                        logger.info(
                            "MediaPipeFaceDetector loaded (Tasks API 0.10+, delegate=%s)",
                            delegate,
                        )
                        return
                    except Exception as exc:
                        if delegate == "gpu":
                            logger.warning(
                                "MediaPipeFaceDetector GPU delegate failed (%s); retrying on CPU.",
                                exc,
                            )
                        else:
                            logger.warning("MediaPipeFaceDetector Tasks API warmup failed: %s", exc)

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                # Legacy 0.9.x CPU path
                self._detector = mp.solutions.face_detection.FaceDetection(
                    min_detection_confidence=self._conf,
                    model_selection=self._model_selection,
                )
                self._use_tasks = False
                self._active_delegate = "cpu"
                logger.info("MediaPipeFaceDetector loaded (legacy solutions API, delegate=cpu)")
        except ImportError:
            logger.warning(
                "mediapipe not installed – face detection will be skipped. "
                "Install with: pip install mediapipe"
            )
        except Exception as exc:
            logger.warning("MediaPipeFaceDetector warmup failed: %s", exc)

    def process(self, image: np.ndarray) -> FaceDetectionResult:
        if self._detector is None:
            self.warmup()
        if self._detector is None:
            return FaceDetectionResult(error="mediapipe unavailable", detector_name="mediapipe")

        try:
            import mediapipe as mp  # type: ignore

            h, w = image.shape[:2]
            rgb = image[:, :, ::-1].copy() if image.shape[2] == 3 else image
            bboxes: list[dict[str, Any]] = []

            if self._use_tasks:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self._detector.detect(mp_image)
                for det in (result.detections or []):
                    bb = det.bounding_box
                    x1 = max(0, int(bb.origin_x))
                    y1 = max(0, int(bb.origin_y))
                    x2 = min(w, int(bb.origin_x + bb.width))
                    y2 = min(h, int(bb.origin_y + bb.height))
                    score = det.categories[0].score if det.categories else 0.0
                    bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": float(score)})
            else:
                result = self._detector.process(rgb)
                if result.detections:
                    for det in result.detections:
                        bb = det.location_data.relative_bounding_box
                        x1 = max(0, int(bb.xmin * w))
                        y1 = max(0, int(bb.ymin * h))
                        x2 = min(w, int((bb.xmin + bb.width) * w))
                        y2 = min(h, int((bb.ymin + bb.height) * h))
                        bboxes.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "confidence": det.score[0] if det.score else 0.0,
                        })

            return FaceDetectionResult(face_count=len(bboxes), bboxes=bboxes, detector_name="mediapipe")
        except Exception as exc:
            logger.error("MediaPipeFaceDetector error: %s", exc)
            return FaceDetectionResult(error=str(exc), detector_name="mediapipe")

    def close(self) -> None:
        """Release the underlying MediaPipe detector to avoid __del__ teardown errors."""
        if self._detector is not None and hasattr(self._detector, "close"):
            try:
                self._detector.close()
            except Exception:
                pass
        self._detector = None


class MTCNNFaceDetector(BaseFaceDetector):
    """
    MTCNN face detector backed by TensorFlow and bundled package weights.

    Accuracy:  Stronger than Haar on webcam faces and available without downloads.
    Speed:     GPU-accelerated when TensorFlow sees CUDA and device="gpu".
    License:   MIT.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.85,
        min_face_area_ratio: float = 0.02,
        dedupe_iou_threshold: float = 0.45,
        secondary_face_min_primary_ratio: float = 0.55,
        device: str = "cpu",
    ) -> None:
        self._conf = min_detection_confidence
        self._min_face_area_ratio = min_face_area_ratio
        self._dedupe_iou_threshold = dedupe_iou_threshold
        self._secondary_face_min_primary_ratio = secondary_face_min_primary_ratio
        self._requested_device = "GPU:0" if str(device).lower() in {"gpu", "cuda"} else "CPU:0"
        self._device = self._requested_device
        self._detector: Any = None
        self._available: Optional[bool] = None

    @staticmethod
    def _bbox_iou(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
        ax1, ay1, ax2, ay2 = [int(box_a[k]) for k in ("x1", "y1", "x2", "y2")]
        bx1, by1, bx2, by2 = [int(box_b[k]) for k in ("x1", "y1", "x2", "y2")]
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / float(area_a + area_b - inter_area)

    def _dedupe_bboxes(self, bboxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        ordered = sorted(bboxes, key=lambda box: float(box.get("confidence", 0.0)), reverse=True)
        for bbox in ordered:
            if any(self._bbox_iou(bbox, existing) >= self._dedupe_iou_threshold for existing in kept):
                continue
            kept.append(bbox)
        return kept

    def _filter_secondary_faces(self, bboxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(bboxes) <= 1:
            return bboxes

        ordered = sorted(
            bboxes,
            key=lambda box: (
                max(0, int(box["x2"]) - int(box["x1"]))
                * max(0, int(box["y2"]) - int(box["y1"]))
                * float(box.get("confidence", 0.0))
            ),
            reverse=True,
        )
        primary = ordered[0]
        primary_area = max(1, (int(primary["x2"]) - int(primary["x1"])) * (int(primary["y2"]) - int(primary["y1"])))
        filtered = [primary]
        for bbox in ordered[1:]:
            area = max(0, (int(bbox["x2"]) - int(bbox["x1"])) * (int(bbox["y2"]) - int(bbox["y1"])))
            if area / primary_area >= self._secondary_face_min_primary_ratio:
                filtered.append(bbox)
        return filtered

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
                        self._detector = MTCNN(stages="face_detection_only", device=device)
                    self._device = device
                    self._available = True
                    logger.info("MTCNNFaceDetector loaded (device=%s)", self._device)
                    return
                except Exception as exc:
                    if device != "CPU:0":
                        logger.warning(
                            "MTCNNFaceDetector GPU warmup failed (%s); retrying on CPU.",
                            exc,
                        )
                    else:
                        logger.warning("MTCNNFaceDetector CPU warmup failed: %s", exc)

            self._detector = None
            self._available = False
        except Exception as exc:
            logger.warning("MTCNNFaceDetector warmup failed: %s", exc)
            self._detector = None
            self._available = False

    def process(self, image: np.ndarray) -> FaceDetectionResult:
        if self._available is None:
            self.warmup()
        if not self._available or self._detector is None:
            return FaceDetectionResult(error="mtcnn unavailable", detector_name="mtcnn")

        try:
            h, w = image.shape[:2]
            if h < 40 or w < 40:
                return FaceDetectionResult(face_count=0, bboxes=[], detector_name="mtcnn")
            rgb = image[:, :, ::-1].copy() if image.ndim == 3 and image.shape[2] == 3 else image
            if rgb.size == 0:
                return FaceDetectionResult(face_count=0, bboxes=[], detector_name="mtcnn")
            faces = self._detector.detect_faces(rgb, box_format="xyxy") or []
            bboxes: list[dict[str, Any]] = []

            for face in faces:
                conf = float(face.get("confidence", 0.0))
                if conf < self._conf:
                    continue
                x1, y1, x2, y2 = [int(v) for v in face.get("box", [0, 0, 0, 0])]
                face_area = max(0, x2 - x1) * max(0, y2 - y1)
                if face_area / max(1, h * w) < self._min_face_area_ratio:
                    continue
                bboxes.append(
                    {
                        "x1": max(0, x1),
                        "y1": max(0, y1),
                        "x2": min(w, x2),
                        "y2": min(h, y2),
                        "confidence": conf,
                    }
                )

            bboxes = self._dedupe_bboxes(bboxes)
            bboxes = self._filter_secondary_faces(bboxes)
            return FaceDetectionResult(face_count=len(bboxes), bboxes=bboxes, detector_name="mtcnn")
        except Exception as exc:
            logger.error("MTCNNFaceDetector error: %s", exc)
            return FaceDetectionResult(error=str(exc), detector_name="mtcnn")


class FallbackFaceDetector(BaseFaceDetector):
    """
    Runtime fallback wrapper for face detection.

    Tries the primary detector first, then a secondary detector if the primary
    errors or returns no faces. This helps recover visible faces that a single
    backend misses on difficult real-world frames.
    """

    def __init__(
        self,
        primary: BaseFaceDetector,
        fallback: BaseFaceDetector,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def warmup(self) -> None:
        if hasattr(self.primary, "warmup"):
            self.primary.warmup()
        if hasattr(self.fallback, "warmup"):
            self.fallback.warmup()

    def process(self, image: np.ndarray) -> FaceDetectionResult:
        primary_result = self.primary.process(image)
        if primary_result.error is None and primary_result.face_count > 0:
            return primary_result

        fallback_result = self.fallback.process(image)
        if fallback_result.error is None and fallback_result.face_count > 0:
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


class OpenCVHaarFaceDetector(BaseFaceDetector):
    """
    Fallback face detector using OpenCV Haar cascades.

    Accuracy:  Lower than deep-learning methods; acceptable for easy cases.
    Speed:     Very fast on CPU.
    License:   BSD (OpenCV).
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._cascade: Any = None

    def warmup(self) -> None:
        import cv2  # type: ignore

        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        logger.info("OpenCVHaarFaceDetector loaded")

    def process(self, image: np.ndarray) -> FaceDetectionResult:
        if self._cascade is None:
            self.warmup()

        import cv2  # type: ignore

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
            )
            bboxes: list[dict[str, Any]] = []
            if len(faces):
                for x, y, fw, fh in faces:
                    bboxes.append(
                        {"x1": int(x), "y1": int(y), "x2": int(x + fw), "y2": int(y + fh), "confidence": 1.0}
                    )
            return FaceDetectionResult(face_count=len(bboxes), bboxes=bboxes, detector_name="opencv_haar")
        except Exception as exc:
            logger.error("OpenCVHaarFaceDetector error: %s", exc)
            return FaceDetectionResult(error=str(exc), detector_name="opencv_haar")
