#!/usr/bin/env python3
"""
analyze_exam_snapshots.py
=========================
Production-ready CLI for batch proctoring analysis of webcam snapshots.

Usage:
    python analyze_exam_snapshots.py \\
        --metadata ./metadata.csv \\
        --output   ./results \\
        --config   ./config.yaml \\
        --reference-faces ./reference_faces/ \\
        --device   auto

See README.md for full documentation.
"""

from __future__ import annotations

import os
# Suppress TensorFlow C++ / oneDNN / MediaPipe absl log noise before any heavy imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("GLOG_minloglevel", "2")      # suppress MediaPipe absl INFO + WARNING

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm  # type: ignore

# ---------------------------------------------------------------------------
# Local imports (all with fallback-safe design)
# ---------------------------------------------------------------------------
from utils.logging_setup import configure_logging
from utils.config_loader import load_config

from data_io.metadata_loader import load_metadata
from data_io.image_loader import load_image

from detectors.face_detector import (
    MediaPipeFaceDetector,
    MTCNNFaceDetector,
    OpenCVHaarFaceDetector,
    FallbackFaceDetector,
)
from detectors.landmark_detector import (
    MediaPipeLandmarkDetector,
    MTCNNLandmarkDetector,
    FallbackLandmarkDetector,
)
from detectors.pose_gaze_estimator import HeadPoseEstimator, NullPoseGazeEstimator
from detectors.identity_verifier import DeepFaceVerifier, NullIdentityVerifier
from detectors.object_detector import YOLOObjectDetector, NullObjectDetector
from detectors.quality_analyzer import CVQualityAnalyzer

from pipeline.processor import FrameProcessor, PipelineConfig
from pipeline.calibration import BaselineCalibrator

from scoring.risk_scorer import RiskScorer
from scoring.aggregator import StudentAggregator, FrameRecord

from reporting.csv_reporter import write_image_results_csv, write_student_summary_csv
from reporting.jsonl_reporter import write_image_results_jsonl
from reporting.thumbnail_reporter import ThumbnailReporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory helpers – build detector instances from config
# ---------------------------------------------------------------------------

def _resolve_mediapipe_delegate(cfg: dict, device: str) -> str:
    delegate = str(cfg.get("delegate", "auto") or "auto").lower()
    if delegate in {"cpu", "gpu"}:
        return delegate
    return "gpu" if device == "cuda" else "cpu"


def build_face_detector(cfg: dict, device: str = "cpu"):
    backend = cfg.get("backend", "mediapipe")
    if backend == "mediapipe":
        mp_cfg = cfg.get("mediapipe", {})
        primary = MediaPipeFaceDetector(
            min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
            model_selection=mp_cfg.get("model_selection", 0),
            delegate=_resolve_mediapipe_delegate(mp_cfg, device),
        )
        if cfg.get("runtime_fallback_to_haar", True):
            det = FallbackFaceDetector(primary=primary, fallback=OpenCVHaarFaceDetector())
        else:
            det = primary
    elif backend == "mtcnn":
        mtcnn_cfg = cfg.get("mtcnn", {})
        primary = MTCNNFaceDetector(
            min_detection_confidence=mtcnn_cfg.get("min_detection_confidence", 0.85),
            min_face_area_ratio=mtcnn_cfg.get("min_face_area_ratio", 0.02),
            dedupe_iou_threshold=mtcnn_cfg.get("dedupe_iou_threshold", 0.45),
            secondary_face_min_primary_ratio=mtcnn_cfg.get("secondary_face_min_primary_ratio", 0.55),
            device=device,
        )
        if cfg.get("runtime_fallback_to_mediapipe", True):
            mp_cfg = cfg.get("mediapipe", {})
            fallback = MediaPipeFaceDetector(
                min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
                model_selection=mp_cfg.get("model_selection", 0),
                delegate=_resolve_mediapipe_delegate(mp_cfg, device),
            )
            if cfg.get("runtime_fallback_to_haar", True):
                fallback = FallbackFaceDetector(
                    primary=fallback,
                    fallback=OpenCVHaarFaceDetector(),
                )
            det = FallbackFaceDetector(primary=primary, fallback=fallback)
        elif cfg.get("runtime_fallback_to_haar", True):
            det = FallbackFaceDetector(primary=primary, fallback=OpenCVHaarFaceDetector())
        else:
            det = primary
    else:
        det = OpenCVHaarFaceDetector()
    det.warmup()
    return det


def build_landmark_detector(cfg: dict, device: str = "cpu"):
    backend = cfg.get("backend", "mediapipe")
    if backend == "mediapipe":
        mp_cfg = cfg.get("mediapipe", {})
        det = MediaPipeLandmarkDetector(
            max_num_faces=mp_cfg.get("max_num_faces", 2),
            min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
            delegate=_resolve_mediapipe_delegate(mp_cfg, device),
        )
        det.warmup()
        return det
    if backend == "mtcnn":
        mtcnn_cfg = cfg.get("mtcnn", {})
        primary = MTCNNLandmarkDetector(
            min_detection_confidence=mtcnn_cfg.get("min_detection_confidence", 0.85),
            device=device,
        )
        if cfg.get("runtime_fallback_to_mediapipe", True):
            mp_cfg = cfg.get("mediapipe", {})
            fallback = MediaPipeLandmarkDetector(
                max_num_faces=mp_cfg.get("max_num_faces", 2),
                min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
                min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
                delegate=_resolve_mediapipe_delegate(mp_cfg, device),
            )
            det = FallbackLandmarkDetector(primary=primary, fallback=fallback)
        else:
            det = primary
        det.warmup()
        return det
    return MediaPipeLandmarkDetector()


def build_pose_gaze_estimator(cfg: dict):
    backend = cfg.get("backend", "head_pose")
    if backend == "none":
        return NullPoseGazeEstimator()
    hp_cfg = cfg.get("head_pose", {})
    return HeadPoseEstimator(
        moderate_yaw_deg=hp_cfg.get("moderate_yaw_deg", 20.0),
        severe_yaw_deg=hp_cfg.get("severe_yaw_deg", 35.0),
        pitch_warning_deg=hp_cfg.get("pitch_warning_deg", 20.0),
        talking_ratio_threshold=hp_cfg.get("talking_ratio_threshold", 0.08),
        talking_likely_ratio_threshold=hp_cfg.get("talking_likely_ratio_threshold", 0.12),
        talking_delta_threshold=hp_cfg.get("talking_delta_threshold", 0.018),
        talking_likely_delta_threshold=hp_cfg.get("talking_likely_delta_threshold", 0.035),
        talking_max_abs_yaw_deg=hp_cfg.get("talking_max_abs_yaw_deg", 35.0),
        talking_max_abs_pitch_deg=hp_cfg.get("talking_max_abs_pitch_deg", 35.0),
    )


def build_identity_verifier(cfg: dict, device: str = "cpu"):
    backend = cfg.get("backend", "deepface")
    if backend == "none":
        return NullIdentityVerifier()
    if backend == "deepface":
        df_cfg = cfg.get("deepface", {})
        ver = DeepFaceVerifier(
            model_name=df_cfg.get("model_name", "ArcFace"),
            distance_metric=df_cfg.get("distance_metric", "cosine"),
            mismatch_threshold=df_cfg.get("mismatch_threshold", 0.4),
            face_detector_delegate=_resolve_mediapipe_delegate(df_cfg, device),
        )
        try:
            ver.warmup()
        except Exception as exc:
            logger.warning("Identity verifier warmup failed (%s); disabling.", exc)
            return NullIdentityVerifier()
        return ver
    return NullIdentityVerifier()


def build_object_detector(cfg: dict, device: str = "cpu"):
    backend = cfg.get("backend", "yolo")
    if backend == "none":
        return NullObjectDetector()
    if backend == "yolo":
        y_cfg = cfg.get("yolo", {})
        det = YOLOObjectDetector(
            model_name=y_cfg.get("model_name", "yolov8n"),
            confidence=y_cfg.get("confidence", 0.35),
            device=device,
        )
        det.warmup()
        return det
    return NullObjectDetector()


def _resolve_device(requested: str) -> str:
    cuda_available = False
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch = None  # type: ignore

    if requested == "auto":
        if cuda_available:
            return "cuda"
        return "cpu"
    if requested == "cpu" and cuda_available:
        logger.warning(
            "CUDA is available on this machine, but the run was forced to CPU. "
            "Use --device auto or --device cuda to put YOLO on GPU."
        )
    return requested


# ---------------------------------------------------------------------------
# Reference face loading
# ---------------------------------------------------------------------------

def load_reference_faces(ref_dir: Optional[str | Path], identity_verifier) -> None:
    """Load reference face images per student from ref_dir/<student_id>/ folders."""
    if ref_dir is None:
        return
    ref_path = Path(ref_dir)
    if not ref_path.exists():
        logger.warning("Reference faces directory not found: %s", ref_path)
        return

    if isinstance(identity_verifier, NullIdentityVerifier):
        return

    loaded = 0
    for student_dir in sorted(ref_path.iterdir()):
        if not student_dir.is_dir():
            continue
        student_id = student_dir.name
        imgs = []
        for img_file in sorted(student_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                img = load_image(img_file)
                if img is not None:
                    imgs.append(img)
        if imgs:
            identity_verifier.build_reference(imgs, student_id)
            loaded += 1

    logger.info("Loaded reference faces for %d students from %s", loaded, ref_path)


def _should_use_for_calibration(record: FrameRecord) -> bool:
    """
    Only use clean frames for baseline pose calibration.

    Otherwise early suspicious frames can become the student's baseline and
    suppress later detections.
    """
    return (
        record.error is None
        and record.face_count == 1
        and not record.look_away_flag
        and not record.low_quality
        and not record.phone_detected
        and not record.extra_person_detected
        and not record.face_obstructed
        and not record.talking_flag
        and (record.mouth_open_ratio is None or record.mouth_open_ratio < 0.11)
    )


def _parse_source_log_id(raw_value: str) -> int | None:
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _question_label_from_metadata(row) -> str:
    if row.question_label:
        return row.question_label
    bits = []
    if row.question_slot:
        bits.append(f"Q{row.question_slot}")
    elif row.quiz_page:
        bits.append(f"Page {row.quiz_page}")
    if row.question_name:
        bits.append(row.question_name)
    return ": ".join(bits)


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    configure_logging(
        level=cfg["logging"].get("level", "INFO"),
        log_file=Path(args.output) / "analysis.log" if args.output else None,
        json=cfg["logging"].get("json", False),
    )

    logger.info("Starting exam proctoring analysis")
    logger.info("Metadata: %s | Output: %s", args.metadata, args.output)

    # Load metadata
    try:
        metadata_rows = load_metadata(args.metadata)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load metadata: %s", exc)
        return 1

    if not metadata_rows:
        logger.error("No valid metadata rows found. Exiting.")
        return 1

    # Resolve device
    device = _resolve_device(getattr(args, "device", "cpu") or "cpu")
    logger.info("Compute device: %s", device)
    logger.info(
        "GPU note: YOLO uses the selected device, MediaPipe face/landmarks request the GPU delegate when "
        "device=cuda, DeepFace/TensorFlow can use CUDA automatically, and pose uses FaceLandmarker transforms "
        "with solvePnP only as fallback."
    )

    # Build detectors from config
    det_cfg = cfg["detectors"]
    pipe_cfg_dict = cfg["pipeline"]

    face_detector = build_face_detector(det_cfg["face"], device=device)
    landmark_detector = build_landmark_detector(det_cfg["landmarks"], device=device)
    pose_gaze_estimator = build_pose_gaze_estimator(det_cfg["pose_gaze"])
    identity_verifier = build_identity_verifier(det_cfg["identity"], device=device)
    object_detector = build_object_detector(det_cfg["objects"], device=device)
    quality_analyzer = CVQualityAnalyzer(**det_cfg["quality"].get("opencv", {}))

    # Load reference faces
    load_reference_faces(getattr(args, "reference_faces", None), identity_verifier)

    # Risk scorer
    risk_scorer = RiskScorer(weights=cfg["scoring"].get("weights"))

    # Pipeline config
    pipeline_config = PipelineConfig(
        enable_face_detection=pipe_cfg_dict.get("enable_face_detection", True),
        enable_landmarks=pipe_cfg_dict.get("enable_landmarks", True),
        enable_pose_gaze=pipe_cfg_dict.get("enable_pose_gaze", True),
        enable_identity=pipe_cfg_dict.get("enable_identity", True),
        enable_objects=pipe_cfg_dict.get("enable_objects", True),
        enable_quality=pipe_cfg_dict.get("enable_quality", True),
        enable_thumbnails=pipe_cfg_dict.get("enable_thumbnails", False),
        suspicious_score_threshold=pipe_cfg_dict.get("suspicious_score_threshold", 30.0),
    )

    processor = FrameProcessor(
        face_detector=face_detector,
        landmark_detector=landmark_detector,
        pose_gaze_estimator=pose_gaze_estimator,
        identity_verifier=identity_verifier,
        object_detector=object_detector,
        quality_analyzer=quality_analyzer,
        risk_scorer=risk_scorer,
        config=pipeline_config,
    )

    # Aggregation settings
    agg_cfg = cfg["aggregation"]
    calibrator = BaselineCalibrator(n_frames=agg_cfg.get("calibration_frames", 5))
    aggregator = StudentAggregator(
        suspicious_threshold=agg_cfg.get("suspicious_threshold", 30.0),
        window_seconds=agg_cfg.get("incident_window_seconds", 30),
        min_frames_per_incident=agg_cfg.get("min_frames_per_incident", 2),
        single_frame_high_risk_threshold=agg_cfg.get("single_frame_high_risk_threshold", 50.0),
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    thumbnail_reporter: Optional[ThumbnailReporter] = None
    if pipeline_config.enable_thumbnails:
        thumbnail_reporter = ThumbnailReporter(
            output_dir=output_dir,
            score_threshold=pipeline_config.suspicious_score_threshold,
        )

    all_records: list[FrameRecord] = []
    image_max_dim = pipe_cfg_dict.get("image_max_dim", 1280)
    metadata_dir = Path(args.metadata).parent

    logger.info("Processing %d frames…", len(metadata_rows))

    for row in tqdm(metadata_rows, desc="Analysing frames", unit="frame"):
        image = load_image(row.image_path, base_dir=metadata_dir, max_dim=image_max_dim)

        if image is None:
            # Record a failed/missing frame
            record = FrameRecord(
                image_path=row.image_path,
                student_id=row.student_id,
                attempt_id=row.attempt_id,
                timestamp=row.timestamp,
                course_id=row.course_id,
                quiz_id=row.quiz_id,
                quiz_name=row.quiz_name,
                quiz_page=row.quiz_page,
                question_id=row.question_id,
                question_slot=row.question_slot,
                question_name=row.question_name,
                question_label=_question_label_from_metadata(row),
                source_log_id=_parse_source_log_id(row.source_log_id),
                face_count=0,
                look_away_flag=False,
                severity="none",
                identity_mismatch=False,
                identity_similarity=1.0,
                phone_detected=False,
                extra_person_detected=False,
                book_detected=False,
                face_obstructed=False,
                talking_flag=False,
                talking_severity="none",
                talking_confidence=0.0,
                mouth_open_ratio=None,
                mouth_open_delta=None,
                low_quality=True,
                blur_score=0.0,
                brightness_score=0.0,
                glare_score=0.0,
                risk_score=0.0,
                reasons=[],
                error="image_load_failed",
            )
            all_records.append(record)
            aggregator.add(record)
            continue

        # Baseline calibration lookup
        base_yaw, base_pitch, base_mouth = calibrator.get_baseline(row.student_id, row.attempt_id)

        record = processor.process(
            image=image,
            image_path=row.image_path,
            student_id=row.student_id,
            attempt_id=row.attempt_id,
            timestamp=row.timestamp,
            course_id=row.course_id,
            quiz_id=row.quiz_id,
            quiz_name=row.quiz_name,
            quiz_page=row.quiz_page,
            question_id=row.question_id,
            question_slot=row.question_slot,
            question_name=row.question_name,
            question_label=_question_label_from_metadata(row),
            source_log_id=_parse_source_log_id(row.source_log_id),
            baseline_yaw=base_yaw,
            baseline_pitch=base_pitch,
            baseline_mouth_open=base_mouth,
        )

        # Update calibrator after processing using only clean frames
        if _should_use_for_calibration(record):
            calibrator.update(
                row.student_id,
                row.attempt_id,
                record.yaw,
                record.pitch,
                record.mouth_open_ratio,
            )

        if thumbnail_reporter:
            thumbnail_reporter.process_record(record, image=image)

        all_records.append(record)
        aggregator.add(record)

    # ---- Write outputs ---
    logger.info("Writing outputs to %s", output_dir)
    write_image_results_csv(all_records, output_dir / "image_level_results.csv")
    write_image_results_jsonl(all_records, output_dir / "image_level_results.jsonl")

    summaries = aggregator.summaries()
    write_student_summary_csv(summaries, output_dir / "student_summary.csv")

    logger.info(
        "Analysis complete. %d frames processed, %d students summarised.",
        len(all_records),
        len(summaries),
    )

    # Explicitly close MediaPipe detectors to avoid __del__ TypeError during interpreter shutdown
    processor.close_all()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_exam_snapshots",
        description="Batch proctoring analysis of Moodle quiz webcam snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metadata", required=True, help="Path to metadata CSV file.")
    p.add_argument("--output", default="./results", help="Directory for output files.")
    p.add_argument("--config", default=None, help="Path to config.yaml (optional).")
    p.add_argument(
        "--reference-faces",
        dest="reference_faces",
        default=None,
        help="Directory of reference face images per student (subfolders named by student_id).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device for object detection ('auto' selects GPU if available).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run_analysis(args))


if __name__ == "__main__":
    main()
