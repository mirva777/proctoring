from __future__ import annotations

import numpy as np

from detectors.base import FaceDetectionResult
from detectors.face_detector import FallbackFaceDetector, MTCNNFaceDetector
from detectors.quality_analyzer import CVQualityAnalyzer
from pipeline.processor import FrameProcessor


class _StubFaceDetector:
    def __init__(self, result: FaceDetectionResult) -> None:
        self.result = result

    def warmup(self) -> None:
        pass

    def process(self, image: np.ndarray) -> FaceDetectionResult:
        return self.result


def test_fallback_face_detector_recovers_visible_face_from_secondary_backend():
    primary = _StubFaceDetector(FaceDetectionResult(face_count=0, detector_name="mediapipe"))
    fallback = _StubFaceDetector(
        FaceDetectionResult(
            face_count=1,
            bboxes=[{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "confidence": 1.0}],
            detector_name="opencv_haar",
        )
    )
    detector = FallbackFaceDetector(primary=primary, fallback=fallback)

    result = detector.process(np.zeros((64, 64, 3), dtype=np.uint8))

    assert result.face_count == 1
    assert result.detector_name == "mediapipe+fallback"


def test_quality_analyzer_ignores_background_lights_outside_center_roi():
    rng = np.random.default_rng(123)
    base = rng.integers(90, 150, size=(200, 300, 1), dtype=np.uint8)
    image = np.repeat(base, 3, axis=2)
    image[:40, :60] = 255
    image[:40, -60:] = 255
    image[-40:, :60] = 255
    image[-40:, -60:] = 255

    analyzer = CVQualityAnalyzer(glare_fraction_max=0.05, roi_fraction=0.6, max_global_glare_fraction=0.2)
    result = analyzer.process(image)

    assert result.low_quality_flag is False
    assert result.glare_score < 0.01


def test_quality_analyzer_flags_central_glare():
    rng = np.random.default_rng(456)
    base = rng.integers(90, 150, size=(200, 300, 1), dtype=np.uint8)
    image = np.repeat(base, 3, axis=2)
    image[70:130, 110:190] = 255

    analyzer = CVQualityAnalyzer(glare_fraction_max=0.05, roi_fraction=0.6, max_global_glare_fraction=0.2)
    result = analyzer.process(image)

    assert result.low_quality_flag is True
    assert result.glare_score > 0.05


def test_infer_person_scene_dedupes_same_student_boxes():
    face_bboxes = [{"x1": 110, "y1": 40, "x2": 170, "y2": 110}]
    raw_detections = [
        {"class_id": 0, "conf": 0.92, "bbox": [60, 10, 230, 190]},
        {"class_id": 0, "conf": 0.88, "bbox": [65, 15, 225, 188]},
    ]

    person_count, extra_person, person_conf = FrameProcessor._infer_person_scene(
        face_bboxes,
        raw_detections,
        (240, 320),
    )

    assert person_count == 1
    assert extra_person is False
    assert round(person_conf, 2) == 0.92


def test_infer_person_scene_keeps_second_distinct_person():
    face_bboxes = [{"x1": 110, "y1": 40, "x2": 170, "y2": 110}]
    raw_detections = [
        {"class_id": 0, "conf": 0.92, "bbox": [60, 10, 230, 190]},
        {"class_id": 0, "conf": 0.86, "bbox": [235, 30, 315, 210]},
    ]

    person_count, extra_person, person_conf = FrameProcessor._infer_person_scene(
        face_bboxes,
        raw_detections,
        (240, 320),
    )

    assert person_count == 2
    assert extra_person is True
    assert round(person_conf, 2) == 0.92


def test_mtcnn_face_detector_dedupes_overlapping_face_boxes():
    detector = MTCNNFaceDetector(dedupe_iou_threshold=0.4)
    boxes = detector._dedupe_bboxes(
        [
            {"x1": 10, "y1": 10, "x2": 110, "y2": 110, "confidence": 0.98},
            {"x1": 12, "y1": 12, "x2": 108, "y2": 108, "confidence": 0.95},
            {"x1": 200, "y1": 20, "x2": 280, "y2": 120, "confidence": 0.93},
        ]
    )

    assert len(boxes) == 2
    assert boxes[0]["confidence"] == 0.98


def test_mtcnn_face_detector_suppresses_tiny_secondary_faces():
    detector = MTCNNFaceDetector(secondary_face_min_primary_ratio=0.5)
    boxes = detector._filter_secondary_faces(
        [
            {"x1": 10, "y1": 10, "x2": 110, "y2": 110, "confidence": 0.98},
            {"x1": 220, "y1": 20, "x2": 245, "y2": 45, "confidence": 0.99},
        ]
    )

    assert len(boxes) == 1
    assert boxes[0]["x1"] == 10
