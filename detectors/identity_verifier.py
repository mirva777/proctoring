"""
Face identity verifier – pluggable backend.

Default backend: DeepFace with ArcFace model.
Alternative:     InsightFace (buffalo_l or similar).
Fallback:        Skip silently when neither is available.

Trade-offs:
  DeepFace   – Pythonic wrapper, multiple backends (VGG-Face, ArcFace, Facenet),
                GPL-compatible (MIT).  Slightly slower than raw InsightFace.
  InsightFace – ONNX-based, faster, highly accurate (ArcFace).  Weights for
                non-commercial use; check InsightFace licence before production.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .base import BaseIdentityVerifier, IdentityResult

logger = logging.getLogger(__name__)


class DeepFaceVerifier(BaseIdentityVerifier):
    """
    Uses DeepFace library with configurable model backend.

    Recommended model: ArcFace (accuracy > Facenet > VGG-Face).
    Speed: 50–200 ms per comparison on CPU depending on backend.
    License: MIT (DeepFace).  Note: underlying model weights may differ.
    """

    def __init__(
        self,
        model_name: str = "ArcFace",
        distance_metric: str = "cosine",
        mismatch_threshold: float = 0.4,
        face_detector_delegate: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.mismatch_threshold = mismatch_threshold
        self.face_detector_delegate = (face_detector_delegate or "cpu").lower()
        self._references: dict[str, Any] = {}  # student_id -> embedding
        self._available: Optional[bool] = None
        self._face_detector: Any = None
        self._fallback_face_detector: Any = None

    def _check_available(self) -> bool:
        if self._available is None:
            try:
                from deepface import DeepFace  # type: ignore  # noqa: F401
                self._available = True
            except (ImportError, ValueError, Exception) as exc:
                logger.warning(
                    "deepface not usable – identity verification disabled: %s", exc
                )
                self._available = False
        return self._available

    def warmup(self) -> None:
        self._check_available()

    def _ensure_face_detectors(self) -> None:
        if self._face_detector is not None or self._fallback_face_detector is not None:
            return
        try:
            from .face_detector import MediaPipeFaceDetector, OpenCVHaarFaceDetector

            self._face_detector = MediaPipeFaceDetector(
                min_detection_confidence=0.35,
                model_selection=0,
                delegate=self.face_detector_delegate,
            )
            self._face_detector.warmup()
            self._fallback_face_detector = OpenCVHaarFaceDetector()
            self._fallback_face_detector.warmup()
        except Exception as exc:
            logger.debug("Identity face-detector warmup failed: %s", exc)
            self._face_detector = None
            self._fallback_face_detector = None

    @staticmethod
    def _expand_bbox(
        bbox: dict[str, Any],
        image_shape: tuple[int, ...],
        pad_ratio: float = 0.18,
    ) -> tuple[int, int, int, int]:
        h, w = image_shape[:2]
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        )

    def _extract_face_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        self._ensure_face_detectors()
        detections: list[dict[str, Any]] = []
        for detector in (self._face_detector, self._fallback_face_detector):
            if detector is None:
                continue
            try:
                result = detector.process(image)
            except Exception as exc:
                logger.debug("Identity face crop detector failed: %s", exc)
                continue
            if result.error is None and result.bboxes:
                detections = result.bboxes
                break

        if not detections:
            return None

        largest = max(
            detections,
            key=lambda b: (int(b["x2"]) - int(b["x1"])) * (int(b["y2"]) - int(b["y1"])),
        )
        x1, y1, x2, y2 = self._expand_bbox(largest, image.shape)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def build_reference(
        self,
        reference_images: list[np.ndarray],
        student_id: str,
    ) -> None:
        if not self._check_available():
            return
        try:
            from deepface import DeepFace  # type: ignore

            embeddings = []
            for img in reference_images:
                try:
                    face_crop = self._extract_face_crop(img)
                    if face_crop is None:
                        logger.debug("Skipping reference image without a usable face crop for %s", student_id)
                        continue
                    rep = DeepFace.represent(
                        face_crop,
                        model_name=self.model_name,
                        enforce_detection=False,
                    )
                    if rep:
                        embeddings.append(np.array(rep[0]["embedding"]))
                except Exception as exc:
                    logger.debug("Reference embedding failed: %s", exc)

            if embeddings:
                self._references[student_id] = np.mean(embeddings, axis=0)
                logger.info(
                    "Built reference embedding for student %s from %d images",
                    student_id,
                    len(embeddings),
                )
        except Exception as exc:
            logger.error("build_reference error for %s: %s", student_id, exc)

    def process(self, image: np.ndarray, student_id: str) -> IdentityResult:
        if not self._check_available():
            return IdentityResult(
                reference_available=False,
                verifier_name="deepface",
                error="deepface unavailable",
            )
        if student_id not in self._references:
            return IdentityResult(
                reference_available=False,
                verifier_name="deepface",
                error="no reference for student",
            )
        try:
            from deepface import DeepFace  # type: ignore

            face_crop = self._extract_face_crop(image)
            if face_crop is None:
                return IdentityResult(
                    similarity_score=0.0,
                    mismatch_flag=True,
                    reference_available=True,
                    verifier_name="deepface",
                    error="no usable face crop in frame",
                )

            rep = DeepFace.represent(
                face_crop,
                model_name=self.model_name,
                enforce_detection=False,
            )
            if not rep:
                return IdentityResult(
                    similarity_score=0.0,
                    mismatch_flag=True,
                    reference_available=True,
                    verifier_name="deepface",
                    error="no face in frame for embedding",
                )

            query_emb = np.array(rep[0]["embedding"])
            ref_emb = self._references[student_id]

            if self.distance_metric == "cosine":
                norm_q = np.linalg.norm(query_emb)
                norm_r = np.linalg.norm(ref_emb)
                if norm_q == 0 or norm_r == 0:
                    distance = 1.0
                else:
                    distance = float(
                        1.0 - np.dot(query_emb, ref_emb) / (norm_q * norm_r)
                    )
            else:
                distance = float(np.linalg.norm(query_emb - ref_emb))

            similarity = max(0.0, 1.0 - distance)
            mismatch = distance > self.mismatch_threshold
            return IdentityResult(
                similarity_score=round(similarity, 4),
                mismatch_flag=mismatch,
                reference_available=True,
                verifier_name="deepface",
            )
        except Exception as exc:
            logger.error("DeepFaceVerifier.process error: %s", exc)
            return IdentityResult(
                reference_available=True,
                verifier_name="deepface",
                error=str(exc),
            )

    def close(self) -> None:
        for detector_attr in ("_face_detector", "_fallback_face_detector"):
            detector = getattr(self, detector_attr, None)
            if detector is not None and hasattr(detector, "close"):
                try:
                    detector.close()
                except Exception:
                    pass
            setattr(self, detector_attr, None)


class NullIdentityVerifier(BaseIdentityVerifier):
    """No-op verifier when no backend is configured."""

    def build_reference(self, reference_images: list[np.ndarray], student_id: str) -> None:
        pass

    def process(self, image: np.ndarray, student_id: str) -> IdentityResult:
        return IdentityResult(
            reference_available=False,
            verifier_name="none",
            error="identity verification disabled",
        )

    def close(self) -> None:
        pass
