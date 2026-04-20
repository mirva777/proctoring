# Exam Proctoring Snapshot Analyser

A production-ready Python batch analysis tool for webcam snapshots captured during Moodle quizzes.  Detects suspicious events per image, aggregates temporal incidents per student, and produces machine-readable CSV/JSONL reports.

> **Non-goal**: This tool provides evidence for **human reviewers only**.  It never auto-fails students.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Moodle Integration](#moodle-integration)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Model / Backend Selection](#model--backend-selection)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Architecture](#architecture)
- [Risk Scoring](#risk-scoring)
- [Temporal Incident Logic](#temporal-incident-logic)
- [Baseline Calibration](#baseline-calibration)
- [Reference Faces](#reference-faces)
- [Running Tests](#running-tests)
- [Extending the Pipeline](#extending-the-pipeline)
- [Licensing Cautions](#licensing-cautions)

---

## Features

| Detection | Method | Fallback |
|-----------|--------|---------|
| Face presence (0 / 1 / multiple) | MediaPipe BlazeFace | OpenCV Haar cascade |
| Head pose / attention | OpenCV solvePnP + MediaPipe landmarks | None (skip) |
| Identity consistency | DeepFace ArcFace embeddings | Skip silently |
| Suspicious objects | YOLOv8 (phone, book, extra person) | Skip silently |
| Image quality | Laplacian blur + brightness + glare heuristics | Always runs |

Additional capabilities:
- Temporal incident grouping (reduces false positives from single-frame anomalies)
- Per-student baseline calibration for head-pose drift correction
- Configurable weighted risk score (0–100, never auto-fails)
- Annotated thumbnail output for human review
- Full YAML configuration with per-course override support
- Structured logging (JSON-compatible)
- Progressive progress bar

---

## Project Structure

```
proctoring/
├── analyze_exam_snapshots.py   # CLI entry point
├── config.yaml                 # Default configuration template
├── requirements.txt
├── sample_metadata.csv
├── sample_image_level_results.jsonl
├── sample_student_summary.csv
├── detectors/
│   ├── base.py                 # Abstract interfaces + result dataclasses
│   ├── face_detector.py        # MediaPipe / OpenCV Haar
│   ├── landmark_detector.py    # MediaPipe FaceMesh
│   ├── pose_gaze_estimator.py  # OpenCV solvePnP head-pose proxy
│   ├── identity_verifier.py    # DeepFace ArcFace / Null verifier
│   ├── object_detector.py      # YOLOv8 / Null detector
│   └── quality_analyzer.py     # OpenCV quality heuristics
├── scoring/
│   ├── risk_scorer.py          # Per-frame risk scoring (pure function)
│   └── aggregator.py           # Student-level aggregation + incidents
├── pipeline/
│   ├── processor.py            # Orchestrates all detectors per frame
│   └── calibration.py          # Per-student head-pose baseline
├── io/
│   ├── metadata_loader.py      # CSV metadata ingestion
│   └── image_loader.py         # Robust image loading (handles corrupt files)
├── reporting/
│   ├── csv_reporter.py         # CSV output writers
│   ├── jsonl_reporter.py       # JSONL output writer
│   └── thumbnail_reporter.py   # Annotated thumbnail generator
├── utils/
│   ├── config_loader.py        # YAML config loader with defaults
│   └── logging_setup.py        # Structured logging configuration
└── tests/
    └── test_scoring.py         # Unit tests for scoring & aggregation
```

---

## Quickstart

For Windows Server + WSL deployment with systemd services, use
[`deploy/wsl/README.md`](deploy/wsl/README.md). The deployment path runs the
live Moodle poller as `proctoring-live.service` and the dashboard as
`proctoring-dashboard.service`.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Minimum viable install** (CPU, no identity verification, no object detection):
> ```bash
> pip install numpy opencv-python mediapipe tqdm pyyaml
> ```

### 3. Prepare your metadata CSV

See `sample_metadata.csv` for the required format:

```
image_path,student_id,attempt_id,timestamp,course_id
snapshots/student_001/frame_001.jpg,student_001,attempt_5001,2024-06-10T09:00:00,course_CS101
```

### 4. Run analysis

```bash
python analyze_exam_snapshots.py \
  --metadata ./sample_metadata.csv \
  --output   ./results \
  --config   ./config.yaml \
  --device   auto
```

With reference faces (for identity verification):

```bash
python analyze_exam_snapshots.py \
  --metadata        ./metadata.csv \
  --output          ./results \
  --config          ./config.yaml \
  --reference-faces ./reference_faces/ \
  --device          auto
```

`reference_faces/` must contain one subdirectory per student, named by `student_id`, containing one or more face photos:

```
reference_faces/
├── student_001/
│   └── id_photo.jpg
└── student_002/
    ├── registration_photo.jpg
    └── another_photo.jpg
```

---

## Moodle Integration

This pipeline works with the [quizaccess_proctoring](https://github.com/eLearning-BS23/moodle-quizaccess_proctoring) Moodle plugin, which automatically captures webcam snapshots during quizzes at configurable intervals (default 30 seconds).

### How the Plugin Stores Images

1. **JavaScript** captures frames via `canvas.toDataURL('image/png')` → base64 data
2. **AJAX** sends the data to `quizaccess_proctoring_send_camshot` web service
3. **Moodle File API** stores the image as a PNG file:
   - Physical file in `moodledata/filedir/{xx}/{yy}/{contenthash}` (SHA1 content-addressed)
   - Metadata in `mdl_files` table (`component='quizaccess_proctoring'`, `filearea='picture'`)
   - Pluginfile URL saved in `mdl_quizaccess_proctoring_logs.webcampicture`
4. **Reference faces** are in the `user_photo` filearea, linked via `mdl_quizaccess_proctoring_user_images`

### Extracting Images from Moodle

Use `moodle_extractor.py` to pull images out of Moodle and generate a pipeline-compatible `metadata.csv`:

#### Method 1: Filesystem mode (on the Moodle server)

Best when you have direct access to the database and `moodledata/` directory:

```bash
python moodle_extractor.py \
    --db-engine mysql \
    --db-host localhost --db-name moodle --db-user moodle --db-pass secret \
    --moodledata /var/www/moodledata \
    --output ./moodle_export \
    --quiz-id 42
```

#### Method 2: Download mode (remote access)

When you can connect to the database but not the filesystem. Requires a Moodle web-service token:

```bash
python moodle_extractor.py \
    --db-engine mysql \
    --db-host db.example.com --db-name moodle --db-user reader --db-pass s3cret \
    --method download \
    --moodle-url https://moodle.example.com \
    --token abc123def456 \
    --output ./moodle_export \
    --quiz-id 42
```

#### Method 3: URL list mode (no DB access)

If you have a CSV of pluginfile URLs (e.g., exported from Moodle's proctoring report page):

```bash
python moodle_ws_extractor.py \
    --url-list exported_urls.csv \
    --moodle-url https://moodle.example.com \
    --token abc123 \
    --output ./moodle_export
```

### After Extraction — Run Analysis

The extractor produces a standard output compatible with the pipeline:

```
moodle_export/
├── metadata.csv                    # Pipeline-compatible metadata
├── snapshots/{student_id}/*.png    # Webcam captures
└── reference_faces/{student_id}/   # Admin-uploaded reference photos
```

```bash
python analyze_exam_snapshots.py \
    --metadata moodle_export/metadata.csv \
    --output moodle_export/results \
    --config config.yaml \
    --reference-faces moodle_export/reference_faces
```

### Realtime Moodle Pipeline

For live monitoring, run the long-running worker and point the dashboard at the
SQLite store it maintains.

1. Create a local `.env` file from `.env.example` and fill the Moodle DB values:

```bash
cp .env.example .env
```

2. Start the live worker on a machine that can reach PostgreSQL and Moodle's
`moodledata/filedir`:

```bash
python live_moodle_pipeline.py \
  --config config_gpu.yaml \
  --env-file ./.env \
  --course-id 1782 \
  --quiz-id 88195 \
  --moodledata /var/www/moodledata \
  --reference-faces ./real_moodle_export/reference_faces \
  --output ./live_moodle_export \
  --store-db ./live_moodle_export/live_results.sqlite3 \
  --device auto
```

3. Open the dashboard in live mode:

```bash
python review_dashboard.py \
  --live-db ./live_moodle_export/live_results.sqlite3 \
  --snapshots ./live_moodle_export \
  --host 127.0.0.1 \
  --port 5001
```

In live mode the UI polls `/api/live/state` every 3 seconds and reloads when a
new Moodle log row has been analyzed. Frame cards show quiz page and question
labels when Moodle question metadata can be resolved.

### Filtering

| Flag | Description |
|------|-------------|
| `--quiz-id 42` | Export a specific quiz (by course module ID) |
| `--course-id 5` | Export all quizzes in a course |
| Both flags | Combine for precise filtering |

### Database Engines

| `--db-engine` | Driver | Install |
|---------------|--------|---------|
| `mysql` (default) | mysql-connector-python | `pip install mysql-connector-python` |
| `pgsql` | psycopg2 | `pip install psycopg2-binary` |
| `sqlite` | Built-in | (for local testing with `generate_mock_moodle_db.py`) |

### Testing with Mock Data

Generate a mock Moodle database for testing without a real Moodle installation:

```bash
# Generate mock Moodle data (SQLite + filedir structure)
python generate_mock_moodle_db.py --output ./mock_moodle --use-test-data ./test_data

# Extract from mock data
python moodle_extractor.py \
    --db-engine sqlite --db-name ./mock_moodle/moodle.db \
    --moodledata ./mock_moodle/moodledata \
    --output ./moodle_export --quiz-id 42

# Run analysis
python analyze_exam_snapshots.py \
    --metadata moodle_export/metadata.csv \
    --output moodle_export/results \
    --config config.yaml
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--metadata` | *(required)* | Path to metadata CSV |
| `--output` | `./results` | Output directory |
| `--config` | `None` | Path to YAML config file |
| `--reference-faces` | `None` | Directory of per-student reference images |
| `--device` | `auto` | Compute device: `auto` \| `cpu` \| `cuda` \| `mps` |

---

## Configuration

All behaviour is controlled through `config.yaml`.  The file uses deep-merge semantics — you only need to include keys you want to override.

Key sections:

```yaml
pipeline:
  enable_thumbnails: true        # generate annotated images for flagged frames
  suspicious_score_threshold: 30.0

detectors:
  face:
    backend: mediapipe            # swap to opencv_haar for no-dependency fallback
  identity:
    backend: none                 # disable identity check entirely
  objects:
    backend: none                 # disable YOLO (removes AGPL dependency)
  pose_gaze:
    head_pose:
      severe_yaw_deg: 30.0        # make threshold stricter for a specific course

scoring:
  weights:
    phone: 60.0                   # increase phone penalty for this deployment

aggregation:
  incident_window_seconds: 60     # longer window = fewer, larger incidents
  min_frames_per_incident: 3      # require 3 consecutive hits before flagging
```

---

## Model / Backend Selection

### Face Detection

| Backend | Accuracy | Speed (CPU) | Licence | Notes |
|---------|----------|-------------|---------|-------|
| **MediaPipe BlazeFace** ✓ default | Good | 5–15 ms | Apache 2.0 | Works well for webcam distance |
| OpenCV Haar | Fair | < 1 ms | BSD | Use when MediaPipe unavailable |

**How to switch**: set `detectors.face.backend: opencv_haar` in `config.yaml`.

### Head Pose / Gaze

| Backend | Method | Accuracy | Licence | Notes |
|---------|--------|----------|---------|-------|
| **OpenCV solvePnP** ✓ default | 3-D head pose proxy from landmarks | ±5–10° | BSD + Apache 2.0 | No extra model download |
| Direct gaze model | CNN / NN gaze regression | Higher | Varies | Implement `BasePoseGazeEstimator` and set backend key |

**Recommended gaze alternatives** (require custom `BasePoseGazeEstimator` implementation):
- [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) – good accuracy, MIT
- [GazeML](https://github.com/swook/GazeML) – TensorFlow-based
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) – comprehensive, academic licence

**How to switch**: implement `BasePoseGazeEstimator` in `detectors/`, register in `build_pose_gaze_estimator()` in `analyze_exam_snapshots.py`, and set `detectors.pose_gaze.backend: your_key`.

### Identity Verification

| Backend | Model | Accuracy | Licence | Notes |
|---------|-------|----------|---------|-------|
| **DeepFace ArcFace** ✓ default | ArcFace (MS-Celeb) | Excellent | MIT wrapper; model non-commercial | See licensing caution below |
| DeepFace Facenet | FaceNet | Very good | MIT wrapper; model Apache 2.0 | Set `model_name: Facenet512` |
| InsightFace | buffalo_l | Excellent | Non-commercial | Faster ONNX runtime |
| Disable | — | — | — | Set `backend: none` |

**How to switch**: implement `BaseIdentityVerifier`, add to `build_identity_verifier()`, and set `detectors.identity.backend`.

### Object Detection

| Backend | Model | Accuracy | Licence | Notes |
|---------|-------|----------|---------|-------|
| **YOLOv8n** ✓ default | COCO | Good for phone/person | AGPL-3.0 | Fast on CPU, auto-downloads weights |
| YOLOv8s / YOLOv8m | COCO | Better | AGPL-3.0 | Set `model_name: yolov8s` |
| Custom fine-tuned | — | Best | Your licence | Path to `.pt` file in `model_name` |
| Disable | — | — | — | Set `backend: none` |

**How to switch**: set `detectors.objects.backend: none` to remove AGPL dependency, or provide path to a custom model in `model_name`.

---

## Input Format

Metadata CSV columns (all required):

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | string | Absolute or relative path to PNG/JPG snapshot |
| `student_id` | string | Unique student identifier |
| `attempt_id` | string | Unique quiz attempt identifier |
| `timestamp` | ISO 8601 | Capture time (`YYYY-MM-DDTHH:MM:SS`) |
| `course_id` | string | Course identifier |

---

## Output Format

### `image_level_results.csv` / `image_level_results.jsonl`

One record per analysed image.  Key fields:

| Field | Description |
|-------|-------------|
| `face_count` | Number of faces detected |
| `look_away_flag` | True if head pose indicates look-away |
| `severity` | `none` \| `moderate` \| `severe` |
| `yaw` / `pitch` / `roll` | Head angles in degrees |
| `identity_mismatch` | True if face doesn't match reference |
| `identity_similarity` | Cosine similarity vs reference [0–1] |
| `phone_detected` | YOLO phone detection |
| `extra_person_detected` | More than one person in frame |
| `risk_score` | Weighted risk 0–100 |
| `reasons` | JSON list of triggered reason codes |

### `student_summary.csv`

One row per (student_id, attempt_id):

| Field | Description |
|-------|-------------|
| `total_frames` | All frames in this attempt |
| `valid_frames` | Frames with no load error |
| `suspicious_frames` | Frames with risk_score ≥ threshold |
| `percentage_suspicious` | `suspicious / valid × 100` |
| `max_risk_score` | Peak risk in this attempt |
| `mean_risk_score` | Average risk across valid frames |
| `top_reasons` | Most frequent reason codes |
| `incident_count` | Temporal clusters of suspicious frames |
| `identity_stability_score` | Mean identity similarity |
| `overall_risk_level` | `low` \| `medium` \| `high` (for human triage) |
| `flagged_timeline` | JSON list of `{timestamp, risk_score, reasons}` |

### `flagged_thumbnails/` (optional)

Annotated JPEG copies of frames with `risk_score ≥ threshold`.  Includes:
- Risk score and reason codes on banner
- Colour-coded border (green / orange / red)
- Student ID and timestamp footer

---

## Architecture

```
metadata.csv
      │
      ▼
MetadataLoader ──► ImageLoader
                        │
                        ▼
              ┌─────────────────────────────┐
              │       FrameProcessor        │
              │  ┌─────────────────────┐   │
              │  │  QualityAnalyzer    │   │
              │  │  FaceDetector       │   │
              │  │  LandmarkDetector   │   │
              │  │  PoseGazeEstimator  │   │
              │  │  IdentityVerifier   │   │
              │  │  ObjectDetector     │   │
              │  └─────────────────────┘   │
              │         ▼                  │
              │    RiskScorer              │
              └─────────┬───────────────────┘
                        │  FrameRecord
                        ▼
          ┌─────────────────────────┐
          │  StudentAggregator      │
          │  BaselineCalibrator     │
          │  Incident detection     │
          └────────────┬────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   image_level   student_summary  flagged_
   _results.csv  .csv             thumbnails/
   + .jsonl
```

Each detector implements one of these interfaces from `detectors/base.py`:

- `BaseFaceDetector`
- `BaseLandmarkDetector`
- `BasePoseGazeEstimator`
- `BaseIdentityVerifier`
- `BaseObjectDetector`
- `BaseQualityAnalyzer`

---

## Risk Scoring

Scores are weighted sums capped at 100, computed per frame:

```
score = min(100,
    no_face×40 + multi_face×60 + phone×50 + extra_person×70 +
    look_away_severe×35 + look_away_moderate×20 +
    identity_mismatch×80 + low_quality×10
)
```

All weights are configurable in `config.yaml` under `scoring.weights`.  The score is **never used to auto-fail a student** — it is labelled `overall_risk_level: low|medium|high` for human reviewers.

---

## Temporal Incident Logic

A single suspicious frame rarely indicates cheating.  The aggregator groups consecutive flagged frames into **incidents**:

1. Sort all frames with `risk_score ≥ suspicious_threshold` by timestamp.
2. Group frames within `incident_window_seconds` of each other into one incident.
3. Discard groups with fewer than `min_frames_per_incident` frames.

This means a single accidental phone appearance does not become an incident, but three consecutive phone frames within 30 seconds creates one incident.  Configure via `aggregation.*` in `config.yaml`.

---

## Baseline Calibration

Students seated at different angles or with off-axis webcams will show a constant yaw/pitch offset that is NOT cheating.  The calibrator:

1. Collects head pose from the first `calibration_frames` (default: 5) valid frames per attempt.
2. Computes the mean yaw and pitch as the "natural" baseline.
3. All subsequent frames measure deviation from this baseline, not from the absolute 0°.

Enable/configure via `aggregation.calibration_frames` in `config.yaml`.

---

## Reference Faces

For identity verification, prepare a directory:

```
reference_faces/
├── student_001/  ← folder name = student_id in metadata
│   └── photo.jpg
└── student_002/
    └── photo.jpg
```

Pass with `--reference-faces ./reference_faces/`.

If a student doesn't have a reference folder, identity checks are silently skipped for that student while all other detections continue normally.

---

## Running Tests

```bash
pytest tests/ -v
```

For coverage:

```bash
pytest tests/ --cov=scoring --cov=detectors --cov-report=term-missing
```

The test suite covers:
- Risk scorer edge cases (zero score, cap at 100, every reason code)
- Aggregator grouping and temporal incident detection
- Empty input handling
- Custom weights

---

## Extending the Pipeline

### Add a new face detector

1. Create a class in `detectors/face_detector.py` that inherits `BaseFaceDetector` and implements `process(image) -> FaceDetectionResult`.
2. Add a branch in `build_face_detector()` in `analyze_exam_snapshots.py`.
3. Set `detectors.face.backend: your_key` in `config.yaml`.

### Add a direct gaze model

1. Implement `BasePoseGazeEstimator.process(image, landmark_result) -> PoseGazeResult`.
2. Set `gaze_vector` or `gaze_direction` in the result.
3. Register in `build_pose_gaze_estimator()` and update `config.yaml`.

### Add a new object class

1. Map the new COCO or custom class ID in `detectors/object_detector.py`.
2. Add the corresponding boolean field to `ObjectDetectionResult` in `detectors/base.py`.
3. Add a weight in `DEFAULT_WEIGHTS` in `scoring/risk_scorer.py` and expose it in `config.yaml`.

---

## Licensing Cautions

| Component | Licence | Production Use |
|-----------|---------|----------------|
| MediaPipe | Apache 2.0 | ✅ Commercial OK |
| OpenCV | BSD | ✅ Commercial OK |
| DeepFace (library) | MIT | ✅ Commercial OK |
| ArcFace weights | Research / non-commercial | ⚠️ Check before commercial deployment |
| FaceNet weights | Apache 2.0 (varies by variant) | ⚠️ Verify specific checkpoint licence |
| InsightFace weights | Non-commercial | ❌ Not for commercial use without license |
| YOLOv8 / Ultralytics | AGPL-3.0 | ⚠️ Requires open-source compliance if distributing |
| YOLOv5 | MIT | ✅ Commercial OK |

**Recommendation for commercial / institutional deployment**:
- Replace ArcFace with a commercially licensed face-recognition model.
- Replace Ultralytics YOLO with a MIT-licensed alternative (YOLOv5, RT-DETR Apache 2.0) or obtain an Ultralytics Enterprise licence.
- Consult your institution's legal team regarding biometric data processing requirements (GDPR, FERPA, etc.).
