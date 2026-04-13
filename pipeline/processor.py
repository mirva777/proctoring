"""
Main per-frame analysis pipeline processor.

Orchestrates all detector calls in order and assembles a FrameRecord.
Each detector is called with fallback logic:
  - If a detector raises an unexpected exception it is logged and skipped.
  - If a detector reports its own ``error`` field, processing continues.
  - The pipeline never hard-fails on a single corrupt frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from detectors.base import (
    BaseFaceDetector,
    BaseLandmarkDetector,
    BaseObjectDetector,
    BasePoseGazeEstimator,
    BaseIdentityVerifier,
    BaseQualityAnalyzer,
    FaceDetectionResult,
    LandmarkResult,
    PoseGazeResult,
    IdentityResult,
    ObjectDetectionResult,
    QualityResult,
)
from detectors.pose_gaze_estimator import HeadPoseEstimator
from scoring.risk_scorer import RiskScore, RiskScorer
from scoring.aggregator import FrameRecord

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Runtime flags that control which pipeline stages are active."""

    enable_face_detection: bool = True
    enable_landmarks: bool = True
    enable_pose_gaze: bool = True
    enable_identity: bool = True
    enable_objects: bool = True
    enable_quality: bool = True
    enable_thumbnails: bool = False
    suspicious_score_threshold: float = 30.0


class FrameProcessor:
    """
    Runs all detectors on a single image and returns a FrameRecord.

    All dependencies are injected – no default model is instantiated here.
    """

    def __init__(
        self,
        face_detector: BaseFaceDetector,
        landmark_detector: BaseLandmarkDetector,
        pose_gaze_estimator: BasePoseGazeEstimator,
        identity_verifier: BaseIdentityVerifier,
        object_detector: BaseObjectDetector,
        quality_analyzer: BaseQualityAnalyzer,
        risk_scorer: RiskScorer,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        self.pose_gaze_estimator = pose_gaze_estimator
        self.identity_verifier = identity_verifier
        self.object_detector = object_detector
        self.quality_analyzer = quality_analyzer
        self.risk_scorer = risk_scorer
        self.config = config or PipelineConfig()

    def process(
        self,
        image: np.ndarray,
        image_path: str,
        student_id: str,
        attempt_id: str,
        timestamp: str,
        course_id: str,
        quiz_id: str = "",
        quiz_name: str = "",
        quiz_page: str = "",
        question_id: str = "",
        question_slot: str = "",
        question_name: str = "",
        question_label: str = "",
        source_log_id: int | None = None,
        baseline_yaw: Optional[float] = None,
        baseline_pitch: Optional[float] = None,
        baseline_mouth_open: Optional[float] = None,
    ) -> FrameRecord:
        """Analyse one frame and return a fully populated FrameRecord."""
        cfg = self.config

        # ---- Face detection ---
        face_result = FaceDetectionResult(error="skipped")
        if cfg.enable_face_detection:
            face_result = self._safe_run(
                lambda: self.face_detector.process(image),
                FaceDetectionResult(error="face detector failed"),
                "face_detector",
            )

        # ---- Quality (uses face ROI when available) ---
        quality_result = QualityResult(error="skipped")
        if cfg.enable_quality:
            quality_result = self._safe_run(
                lambda: self.quality_analyzer.process(image, face_result.bboxes),
                QualityResult(error="quality analyzer failed"),
                "quality_analyzer",
            )

        # ---- Landmarks (needed for pose) ---
        landmark_result = LandmarkResult(error="skipped")
        if cfg.enable_landmarks and face_result.face_count > 0:
            landmark_result = self._safe_run(
                lambda: self.landmark_detector.process(image, face_result.bboxes),
                LandmarkResult(error="landmark detector failed"),
                "landmark_detector",
            )

        # ---- Pose / Gaze ---
        pose_result = PoseGazeResult(method="none", error="skipped")
        if cfg.enable_pose_gaze and face_result.face_count > 0:
            pose_result = self._safe_run(
                lambda: self.pose_gaze_estimator.process(image, landmark_result),
                PoseGazeResult(method="none", error="pose/gaze estimator failed"),
                "pose_gaze_estimator",
            )

        # Apply per-student baseline calibration if provided
        if baseline_yaw is not None and pose_result.yaw is not None:
            eff_yaw = pose_result.yaw - baseline_yaw
            eff_pitch = (pose_result.pitch or 0.0) - (baseline_pitch or 0.0)
            # Re-evaluate severity with calibrated angles
            if isinstance(self.pose_gaze_estimator, HeadPoseEstimator):
                look_away, severity, direction = self.pose_gaze_estimator.classify_attention(
                    eff_yaw,
                    eff_pitch,
                )
                pose_result.yaw = round(eff_yaw, 2)
                pose_result.pitch = round(eff_pitch, 2)
                pose_result.gaze_direction = direction
                pose_result.severity = severity
                pose_result.look_away_flag = look_away

        if (
            landmark_result.error is None
            and pose_result.error is None
            and isinstance(self.pose_gaze_estimator, HeadPoseEstimator)
        ):
            talking_flag, talking_severity, talking_confidence, mouth_open_delta = (
                self.pose_gaze_estimator.classify_talking(
                    pose_result.mouth_open_ratio,
                    baseline_mouth_open_ratio=baseline_mouth_open,
                    yaw=pose_result.yaw,
                    pitch=pose_result.pitch,
                )
            )
            pose_result.talking_flag = talking_flag
            pose_result.talking_severity = talking_severity
            pose_result.talking_confidence = talking_confidence
            pose_result.mouth_open_delta = mouth_open_delta

        # ---- Identity ---
        identity_result = IdentityResult(error="skipped")
        if cfg.enable_identity:
            identity_result = self._safe_run(
                lambda: self.identity_verifier.process(image, student_id),
                IdentityResult(error="identity verifier failed"),
                "identity_verifier",
            )

        # ---- Objects ---
        object_result = ObjectDetectionResult(error="skipped")
        if cfg.enable_objects:
            object_result = self._safe_run(
                lambda: self.object_detector.process(image),
                ObjectDetectionResult(error="object detector failed"),
                "object_detector",
            )
            if object_result.error is None:
                person_count, extra_person, person_conf = self._infer_person_scene(
                    face_result.bboxes,
                    object_result.raw_detections,
                    image.shape[:2],
                )
                object_result.person_count = person_count
                object_result.extra_person_detected = extra_person
                object_result.person_confidence = round(person_conf, 3)
                object_result.extra_person_confidence = round(person_conf, 3)

                if face_result.face_count > 0:
                    obstructed, overlap = self._infer_face_obstruction(
                        face_result.bboxes,
                        object_result.raw_detections,
                    )
                    object_result.face_obstructed = obstructed
                    object_result.obstruction_confidence = round(overlap, 3)

        # ---- Risk score ---
        risk: RiskScore = self.risk_scorer.score(
            face_result, pose_result, identity_result, object_result, quality_result
        )

        return FrameRecord(
            image_path=image_path,
            student_id=student_id,
            attempt_id=attempt_id,
            timestamp=timestamp,
            course_id=course_id,
            face_count=face_result.face_count,
            look_away_flag=pose_result.look_away_flag,
            severity=pose_result.severity,
            identity_mismatch=identity_result.mismatch_flag,
            identity_similarity=identity_result.similarity_score,
            phone_detected=object_result.phone_detected,
            extra_person_detected=object_result.extra_person_detected,
            book_detected=object_result.book_detected,
            face_obstructed=object_result.face_obstructed,
            talking_flag=pose_result.talking_flag,
            talking_severity=pose_result.talking_severity,
            talking_confidence=pose_result.talking_confidence,
            mouth_open_ratio=pose_result.mouth_open_ratio,
            mouth_open_delta=pose_result.mouth_open_delta,
            person_detected=object_result.person_count > 0,
            low_quality=quality_result.low_quality_flag,
            blur_score=quality_result.blur_score,
            brightness_score=quality_result.brightness_score,
            glare_score=quality_result.glare_score,
            risk_score=risk.score,
            reasons=risk.reasons,
            yaw=pose_result.yaw,
            pitch=pose_result.pitch,
            roll=pose_result.roll,
            gaze_direction=pose_result.gaze_direction,
            pose_method=pose_result.method,
            quiz_id=quiz_id,
            quiz_name=quiz_name,
            quiz_page=quiz_page,
            question_id=question_id,
            question_slot=question_slot,
            question_name=question_name,
            question_label=question_label,
            source_log_id=source_log_id,
            error=None,
        )

    def close_all(self) -> None:
        """Explicitly release MediaPipe detector resources before interpreter shutdown.

        Prevents ``__del__`` TypeError errors caused by Python GC running after
        module-level names have already been set to None.
        """
        for attr in ("face_detector", "landmark_detector", "identity_verifier"):
            det = getattr(self, attr, None)
            if det is not None and hasattr(det, "close"):
                try:
                    det.close()
                except Exception:
                    pass

    @staticmethod
    def _face_overlap_ratio(face_bbox: dict, obj_bbox: list[int]) -> float:
        fx1, fy1, fx2, fy2 = (
            int(face_bbox["x1"]),
            int(face_bbox["y1"]),
            int(face_bbox["x2"]),
            int(face_bbox["y2"]),
        )
        ox1, oy1, ox2, oy2 = obj_bbox
        inter_w = max(0, min(fx2, ox2) - max(fx1, ox1))
        inter_h = max(0, min(fy2, oy2) - max(fy1, oy1))
        inter_area = inter_w * inter_h
        face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        return inter_area / face_area

    @staticmethod
    def _bbox_area_ratio(bbox: list[int], image_shape: tuple[int, int]) -> float:
        h, w = image_shape
        x1, y1, x2, y2 = bbox
        return max(0, (x2 - x1) * (y2 - y1)) / max(1, h * w)

    @staticmethod
    def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / float(area_a + area_b - inter_area)

    @classmethod
    def _dedupe_boxes(cls, detections: list[dict], iou_threshold: float) -> list[dict]:
        kept: list[dict] = []
        ordered = sorted(detections, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
        for det in ordered:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            if any(cls._bbox_iou(bbox, existing["bbox"]) >= iou_threshold for existing in kept):
                continue
            kept.append(det)
        return kept

    @classmethod
    def _infer_person_scene(
        cls,
        face_bboxes: list[dict],
        raw_detections: list[dict],
        image_shape: tuple[int, int],
    ) -> tuple[int, bool, float]:
        person_detections = []
        for det in raw_detections:
            if det.get("class_id") != 0:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            area_ratio = cls._bbox_area_ratio(bbox, image_shape)
            conf = float(det.get("conf", 0.0))
            if conf < 0.35 or area_ratio < 0.015:
                continue
            person_detections.append(det)

        distinct_people = cls._dedupe_boxes(person_detections, iou_threshold=0.65)
        if not distinct_people:
            return 0, False, 0.0

        strict_people = [
            det for det in distinct_people
            if float(det.get("conf", 0.0)) >= 0.55
            and cls._bbox_area_ratio(det["bbox"], image_shape) >= 0.03
        ]
        if len(strict_people) <= 1:
            max_conf = max(float(det.get("conf", 0.0)) for det in distinct_people)
            return len(distinct_people), False, max_conf

        if not face_bboxes:
            max_conf = max(float(det.get("conf", 0.0)) for det in strict_people)
            return len(distinct_people), len(strict_people) > 1, max_conf

        primary_person = max(
            strict_people,
            key=lambda det: max(
                (cls._face_overlap_ratio(face_bbox, det["bbox"]) for face_bbox in face_bboxes),
                default=0.0,
            ),
        )
        extras = []
        for det in strict_people:
            if det is primary_person:
                continue
            if cls._bbox_iou(det["bbox"], primary_person["bbox"]) >= 0.35:
                continue
            extras.append(det)

        max_conf = max(float(det.get("conf", 0.0)) for det in strict_people)
        return len(distinct_people), bool(extras), max_conf

    @classmethod
    def _infer_face_obstruction(
        cls,
        face_bboxes: list[dict],
        raw_detections: list[dict],
    ) -> tuple[bool, float]:
        if not face_bboxes or not raw_detections:
            return False, 0.0

        suspicious_ids = {63, 67, 73}  # laptop, phone, book
        best_overlap = 0.0
        for face_bbox in face_bboxes:
            for det in raw_detections:
                if det.get("class_id") not in suspicious_ids:
                    continue
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                best_overlap = max(best_overlap, cls._face_overlap_ratio(face_bbox, bbox))

        return best_overlap >= 0.18, best_overlap

    @staticmethod
    def _safe_run(fn, fallback, label: str):
        try:
            return fn()
        except Exception as exc:
            logger.error("FrameProcessor: %s raised %s", label, exc, exc_info=True)
            return fallback
