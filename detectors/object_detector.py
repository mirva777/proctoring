"""
Suspicious object / scene detector – default: Ultralytics YOLOv8.

Detects: phone, book/notebook, extra person, face obstruction.

Fallback:  If ultralytics is not installed the detector returns empty results
           and the pipeline continues with remaining detectors.

Trade-offs:
  YOLOv8n  – 3 ms/frame on CPU, good recall for phones/persons.
  YOLOv8s  – Better accuracy, ~8 ms on CPU.
  Custom fine-tuned – Best for academic cheating objects.
  License:  Ultralytics AGPL-3.0.  For commercial use consider YOLOv5 MIT or
            a custom-trained RT-DETR under Apache 2.0.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .base import BaseObjectDetector, ObjectDetectionResult

logger = logging.getLogger(__name__)

# COCO class IDs relevant to proctoring
_COCO_PERSON = 0
_COCO_CELL_PHONE = 67
_COCO_BOOK = 73
_COCO_LAPTOP = 63   # additional suspicious device


class YOLOObjectDetector(BaseObjectDetector):
    """
    YOLOv8 object detector for suspicious scene events.

    model_name: 'yolov8n' | 'yolov8s' | 'yolov8m' | path to .pt file.
    confidence: minimum detection confidence threshold.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence: float = 0.35,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self._model: Any = None
        self._available: Optional[bool] = None

    def warmup(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_name)
            # Force model to the right device
            self._available = True
            logger.info("YOLOObjectDetector loaded model=%s device=%s", self.model_name, self.device)
        except ImportError:
            logger.warning(
                "ultralytics not installed – object detection disabled. "
                "Install with: pip install ultralytics"
            )
            self._available = False
        except Exception as exc:
            logger.error("YOLOObjectDetector warmup error: %s", exc)
            self._available = False

    def process(self, image: np.ndarray) -> ObjectDetectionResult:
        if self._model is None:
            self.warmup()
        if not self._available:
            return ObjectDetectionResult(
                error="ultralytics/YOLO unavailable",
                detector_name="yolo",
            )
        try:
            rgb = image[:, :, ::-1].copy() if image.shape[2] == 3 else image
            results = self._model(
                rgb,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )

            phone = False; phone_conf = 0.0
            book = False; book_conf = 0.0
            person_count = 0; person_conf = 0.0
            raw: list[dict[str, Any]] = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    label = result.names.get(cls_id, str(cls_id))
                    raw.append({"class": label, "class_id": cls_id, "conf": round(conf, 3),
                                "bbox": [x1, y1, x2, y2]})

                    if cls_id == _COCO_CELL_PHONE and conf > phone_conf:
                        phone = True; phone_conf = conf
                    elif cls_id == _COCO_BOOK and conf > book_conf:
                        book = True; book_conf = conf
                    elif cls_id == _COCO_PERSON:
                        person_count += 1
                        person_conf = max(person_conf, conf)
                    elif cls_id == _COCO_LAPTOP and conf > phone_conf:
                        phone = True; phone_conf = conf  # treat laptop as suspicious device

            # > 1 person = extra person present (the student counts as 1)
            extra_person = person_count > 1

            return ObjectDetectionResult(
                phone_detected=phone,
                phone_confidence=round(phone_conf, 3),
                book_detected=book,
                book_confidence=round(book_conf, 3),
                extra_person_detected=extra_person,
                extra_person_confidence=round(person_conf, 3),
                person_count=person_count,
                person_confidence=round(person_conf, 3),
                raw_detections=raw,
                detector_name="yolov8",
            )
        except Exception as exc:
            logger.error("YOLOObjectDetector.process error: %s", exc)
            return ObjectDetectionResult(error=str(exc), detector_name="yolov8")


class NullObjectDetector(BaseObjectDetector):
    """No-op detector when no backend is configured."""

    def process(self, image: np.ndarray) -> ObjectDetectionResult:
        return ObjectDetectionResult(
            error="object detection disabled",
            detector_name="none",
        )
