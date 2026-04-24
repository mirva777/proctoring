#!/usr/bin/env python3
"""
Realtime Moodle proctoring worker.

This service polls ``quizaccess_proctoring_logs`` for new snapshots, enriches
them with quiz/page/question metadata, runs the frame analysis pipeline, and
writes live results into a SQLite store consumed by ``review_dashboard.py``.

Database credentials are read from environment variables (or an optional
``.env`` file) so secrets never need to be hardcoded in source.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from analyze_exam_snapshots import (
    _resolve_device,
    _should_use_for_calibration,
    build_face_detector,
    build_identity_verifier,
    build_landmark_detector,
    build_object_detector,
    build_pose_gaze_estimator,
    load_reference_faces,
)
from data_io.image_loader import load_image
from data_io.live_moodle_source import LiveMoodleSource, MoodleDBConfig, db_config_from_env
from data_io.live_result_store import LiveResultStore
from data_io.ssh_moodle_bridge import ManagedSSHBridge, ssh_bridge_config_from_env
from pipeline.calibration import BaselineCalibrator
from pipeline.processor import FrameProcessor, PipelineConfig
from scoring.aggregator import FrameRecord
from scoring.risk_scorer import RiskScorer
from utils.config_loader import load_config
from utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _load_env_file(path: str | Path | None) -> None:
    if path is None:
        return
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_processor(cfg: dict, device: str) -> FrameProcessor:
    det_cfg = cfg["detectors"]
    pipe_cfg_dict = cfg["pipeline"]

    face_detector = build_face_detector(det_cfg["face"], device=device)
    landmark_detector = build_landmark_detector(det_cfg["landmarks"], device=device)
    pose_gaze_estimator = build_pose_gaze_estimator(det_cfg["pose_gaze"])
    identity_verifier = build_identity_verifier(det_cfg["identity"], device=device)
    object_detector = build_object_detector(det_cfg["objects"], device=device)

    from detectors.quality_analyzer import CVQualityAnalyzer

    quality_analyzer = CVQualityAnalyzer(**det_cfg["quality"].get("opencv", {}))
    risk_scorer = RiskScorer(weights=cfg["scoring"].get("weights"))
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
    return FrameProcessor(
        face_detector=face_detector,
        landmark_detector=landmark_detector,
        pose_gaze_estimator=pose_gaze_estimator,
        identity_verifier=identity_verifier,
        object_detector=object_detector,
        quality_analyzer=quality_analyzer,
        risk_scorer=risk_scorer,
        config=pipeline_config,
    )


def _restore_calibration(calibrator: BaselineCalibrator, store: LiveResultStore) -> None:
    restored = 0
    for record in store.fetch_frames():
        if _should_use_for_calibration(record):
            calibrator.update(
                record.student_id,
                record.attempt_id,
                record.yaw,
                record.pitch,
                record.mouth_open_ratio,
            )
            restored += 1
    if restored:
        logger.info("Restored calibration state from %d historical clean frames", restored)


def _verify_source_db_connection(source: LiveMoodleSource) -> None:
    with source.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    logger.info("Verified Moodle PostgreSQL connectivity through the active source connection")


def _failed_record(snapshot, image_path: str, error: str) -> FrameRecord:
    return FrameRecord(
        image_path=image_path,
        student_id=snapshot.student_id,
        attempt_id=snapshot.attempt_id,
        timestamp=snapshot.timestamp_iso,
        course_id=snapshot.course_id,
        quiz_id=snapshot.quiz_id,
        quiz_name=snapshot.quiz_name,
        quiz_page=snapshot.quiz_page,
        question_id=snapshot.question_id,
        question_slot=snapshot.question_slot,
        question_name=snapshot.question_name,
        question_label=snapshot.question_label,
        source_log_id=snapshot.source_log_id,
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
        pose_method="none",
        error=error,
    )


def _attach_source_snapshot_fields(
    record: FrameRecord,
    source: LiveMoodleSource,
    snapshot,
) -> FrameRecord:
    record.source_webcampicture = snapshot.webcampicture
    record.source_filename = snapshot.filename
    record.source_contenthash = snapshot.contenthash
    record.source_moodledata_path = source.source_file_path_for_snapshot(snapshot)
    return record


def _process_one_batch(
    *,
    source: LiveMoodleSource,
    store: LiveResultStore,
    processor: FrameProcessor,
    calibrator: BaselineCalibrator,
    image_max_dim: int,
    last_source_log_id: int,
    course_id: int | None,
    quiz_id: int | None,
    batch_limit: int,
) -> tuple[int, int]:
    snapshots = source.fetch_new_snapshots(
        last_source_log_id=last_source_log_id,
        course_id=course_id,
        quiz_id=quiz_id,
        limit=batch_limit,
    )
    if not snapshots:
        return 0, last_source_log_id

    touched_attempts: set[tuple[str, str]] = set()
    max_log_id = last_source_log_id

    for snapshot in snapshots:
        rel_path = source.materialize_snapshot(snapshot)
        image = None
        if rel_path:
            image = load_image(
                rel_path,
                base_dir=source.output_dir,
                max_dim=image_max_dim,
            )

        if rel_path is None or image is None:
            record = _failed_record(
                snapshot,
                rel_path or f"snapshots/{snapshot.student_id}/{snapshot.attempt_id}/{snapshot.filename}",
                "image_materialize_failed" if rel_path is None else "image_load_failed",
            )
        else:
            base_yaw, base_pitch, base_mouth = calibrator.get_baseline(
                snapshot.student_id,
                snapshot.attempt_id,
            )
            record = processor.process(
                image=image,
                image_path=rel_path,
                student_id=snapshot.student_id,
                attempt_id=snapshot.attempt_id,
                timestamp=snapshot.timestamp_iso,
                course_id=snapshot.course_id,
                quiz_id=snapshot.quiz_id,
                quiz_name=snapshot.quiz_name,
                quiz_page=snapshot.quiz_page,
                question_id=snapshot.question_id,
                question_slot=snapshot.question_slot,
                question_name=snapshot.question_name,
                question_label=snapshot.question_label,
                source_log_id=snapshot.source_log_id,
                baseline_yaw=base_yaw,
                baseline_pitch=base_pitch,
                baseline_mouth_open=base_mouth,
            )
            if _should_use_for_calibration(record):
                calibrator.update(
                    snapshot.student_id,
                    snapshot.attempt_id,
                    record.yaw,
                    record.pitch,
                    record.mouth_open_ratio,
                )

        record = _attach_source_snapshot_fields(record, source, snapshot)
        store.upsert_frame(record)
        touched_attempts.add((snapshot.student_id, snapshot.attempt_id))
        max_log_id = max(max_log_id, snapshot.source_log_id)

    for student_id, attempt_id in touched_attempts:
        store.update_attempt_summary(student_id, attempt_id)

    store.set_last_source_log_id(max_log_id)
    return len(snapshots), max_log_id


def run_live_pipeline(args: argparse.Namespace) -> int:
    _load_env_file(args.env_file)

    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(
        level=cfg["logging"].get("level", "INFO"),
        log_file=output_dir / "live_pipeline.log",
        json=cfg["logging"].get("json", False),
    )

    db_cfg = db_config_from_env(prefix=args.db_env_prefix, table_prefix=args.db_prefix)
    if not db_cfg.password:
        logger.error(
            "Database password env var %sPASSWORD is empty. "
            "Set it in your shell or in --env-file.",
            args.db_env_prefix,
        )
        return 1

    ssh_cfg = ssh_bridge_config_from_env(prefix=args.ssh_env_prefix)
    ssh_bridge = ManagedSSHBridge(ssh_cfg) if ssh_cfg is not None else None

    device = _resolve_device(args.device)
    logger.info(
        "Starting live Moodle polling | db=%s:%s/%s | course_id=%s quiz_id=%s | device=%s",
        db_cfg.host,
        db_cfg.port,
        db_cfg.dbname,
        args.course_id,
        args.quiz_id,
        device,
    )

    store = LiveResultStore(args.store_db or (output_dir / "live_results.sqlite3"))
    processor = _build_processor(cfg, device=device)
    load_reference_faces(args.reference_faces, processor.identity_verifier)

    agg_cfg = cfg["aggregation"]
    calibrator = BaselineCalibrator(n_frames=agg_cfg.get("calibration_frames", 5))
    _restore_calibration(calibrator, store)

    if ssh_bridge is not None:
        try:
            ssh_bridge.start()
        except Exception as exc:
            logger.error("Failed to start SSH bridge: %s", exc, exc_info=True)
            processor.close_all()
            return 1
        db_cfg = MoodleDBConfig(
            host=ssh_bridge.local_db_host,
            port=ssh_bridge.local_db_port,
            dbname=db_cfg.dbname,
            user=db_cfg.user,
            password=db_cfg.password,
            table_prefix=db_cfg.table_prefix,
        )
        logger.info(
            "Live worker will use SSH bridge for PostgreSQL and remote Moodle files"
        )

    source = LiveMoodleSource(
        db_config=db_cfg,
        output_dir=output_dir,
        moodledata_dir=args.moodledata,
        ssh_bridge=ssh_bridge,
    )

    try:
        _verify_source_db_connection(source)
    except Exception as exc:
        logger.error("Initial Moodle DB connectivity check failed: %s", exc, exc_info=True)
        processor.close_all()
        if ssh_bridge is not None:
            ssh_bridge.close()
        return 1

    image_max_dim = cfg["pipeline"].get("image_max_dim", 1280)
    last_log_id = store.get_last_source_log_id()
    logger.info("Resuming from source log id %s", last_log_id)

    try:
        while True:
            try:
                processed_count, last_log_id = _process_one_batch(
                    source=source,
                    store=store,
                    processor=processor,
                    calibrator=calibrator,
                    image_max_dim=image_max_dim,
                    last_source_log_id=last_log_id,
                    course_id=args.course_id,
                    quiz_id=args.quiz_id,
                    batch_limit=args.batch_limit,
                )
                if processed_count:
                    logger.info(
                        "Processed %d new snapshots; latest log_id=%s",
                        processed_count,
                        last_log_id,
                    )
                if args.once:
                    break
                if not processed_count:
                    time.sleep(args.poll_seconds)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("Live poll cycle failed: %s", exc, exc_info=True)
                if args.once:
                    return 1
                time.sleep(args.poll_seconds)
    finally:
        processor.close_all()
        if ssh_bridge is not None:
            ssh_bridge.close()

    logger.info("Live Moodle worker stopped cleanly")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live_moodle_pipeline",
        description="Realtime Moodle proctoring analyzer with live SQLite storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config_gpu.yaml", help="Pipeline config file.")
    parser.add_argument("--env-file", default=".env", help="Optional KEY=VALUE env file.")
    parser.add_argument(
        "--db-env-prefix",
        default="MOODLE_DB_",
        help="Environment variable prefix for DB connection settings.",
    )
    parser.add_argument(
        "--ssh-env-prefix",
        default="MOODLE_SSH_",
        help="Environment variable prefix for optional SSH bridge settings.",
    )
    parser.add_argument("--db-prefix", default="mdl_", help="Moodle table prefix.")
    parser.add_argument("--course-id", type=int, default=None, help="Course filter.")
    parser.add_argument("--quiz-id", type=int, default=None, help="Quiz filter.")
    parser.add_argument(
        "--moodledata",
        required=True,
        help="Path to Moodle moodledata directory containing filedir/.",
    )
    parser.add_argument(
        "--reference-faces",
        default=None,
        help="Reference face directory with one subdirectory per student_id.",
    )
    parser.add_argument(
        "--output",
        default="./live_moodle_export",
        help="Local directory for copied snapshots and logs.",
    )
    parser.add_argument(
        "--store-db",
        default=None,
        help="SQLite database path. Defaults to <output>/live_results.sqlite3.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Sleep time when no new rows appear.")
    parser.add_argument("--batch-limit", type=int, default=250, help="Maximum rows fetched per polling cycle.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device for GPU-capable detectors.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one polling cycle and exit (useful for smoke tests).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.course_id is None and args.quiz_id is None:
        parser.error("At least one of --course-id or --quiz-id must be provided.")
    sys.exit(run_live_pipeline(args))


if __name__ == "__main__":
    main()
