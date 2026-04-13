# Copilot Instructions – Exam Proctoring Snapshot Analyser

## Purpose & Non-Goal
Batch-analyses webcam snapshots from Moodle quizzes to detect suspicious events. **Never auto-fails students** — all output is evidence for human reviewers only.

---

## Architecture Overview

```
analyze_exam_snapshots.py  ← CLI entry point; builds all components via factory helpers
        │
        ▼
pipeline/processor.py      ← FrameProcessor: orchestrates all detectors per image
        │                     (dependency-injected; never instantiates models itself)
        ├── detectors/     ← One concrete class per detection type; all inherit from detectors/base.py
        ├── scoring/risk_scorer.py    ← Pure stateless function: detector results → RiskScore (0–100)
        └── scoring/aggregator.py    ← Temporal grouping: FrameRecords → StudentSummary + Incidents
                │
                ▼
        reporting/         ← CSV, JSONL, and annotated thumbnail writers
```

**Key design decisions:**
- All detectors are injected into `FrameProcessor` — to add a backend, implement the abstract base in `detectors/base.py` and wire it in `analyze_exam_snapshots.py`'s factory helpers (`build_face_detector`, etc.).
- `RiskScorer` and `aggregator.py` are pure functions with no I/O — unit test them directly without mocking.
- Every detector call in `FrameProcessor.process()` is wrapped in a try/except; a single corrupt frame never aborts the pipeline.

---

## Data Flow

1. `data_io/metadata_loader.py` reads a CSV with columns: `image_path, student_id, attempt_id, timestamp, course_id`
2. Each row becomes a frame processed by `FrameProcessor`, which produces a `FrameRecord` (`scoring/aggregator.py`)
3. `FrameRecord` objects are passed to `StudentAggregator` → `StudentSummary` with `Incident` clusters
4. `reporting/` writes `image_level_results.{csv,jsonl}` and `student_summary.csv` to `--output`

---

## Detector / Backend Pattern

Each detector has:
- An abstract base in `detectors/base.py` (e.g. `BaseFaceDetector`)
- A result dataclass with an `error: Optional[str]` field (never raises into pipeline)
- A real implementation (e.g. `MediaPipeFaceDetector`) and a `Null*` stub for disabled stages

To **add a new detector backend**: subclass the relevant `Base*` class, implement `process(image: np.ndarray)`, return the result dataclass, then update the factory helper in `analyze_exam_snapshots.py` and the backend key in `config.yaml`.

To **disable a detector entirely**: set `backend: none` in `config.yaml` — the factory returns the `Null*` stub automatically.

---

## Risk Scoring & Weights

Default weights live in `scoring/risk_scorer.py::DEFAULT_WEIGHTS` and are mirrored in `config.yaml` under `scoring.weights`. Override per deployment:

```yaml
scoring:
  weights:
    identity_mismatch: 80.0   # highest signal
    multi_face: 60.0
    phone: 50.0
    no_face: 40.0
    look_away_severe: 35.0
    look_away_moderate: 20.0
    low_quality: 10.0
```

Scores are **additive and capped at 100**. Frames above `pipeline.suspicious_score_threshold` (default `30.0`) are marked suspicious.

---

## Head-Pose Calibration

`pipeline/calibration.py::BaselineCalibrator` collects yaw/pitch from the first N frames (default 5) per `(student_id, attempt_id)` to correct for off-centre webcam placement. Adjust via `calibration.n_baseline_frames` in `config.yaml`.

---

## Developer Workflows

**Install (full):**
```bash
pip install -r requirements.txt
```

**Minimum viable install** (no identity check, no YOLO):
```bash
pip install numpy opencv-python mediapipe tqdm pyyaml
```

**Run analysis:**
```bash
python analyze_exam_snapshots.py --metadata ./test_data/metadata.csv --output ./results --config ./config.yaml --reference-faces ./reference_faces/
```

**Run tests** (scoring & aggregation only — no model calls):
```bash
pytest tests/ -v
```

**Generate synthetic test data:**
```bash
python generate_test_data.py
```

**Download ML models** (BlazeFace `.tflite`, FaceLandmarker `.task`):
```bash
python download_models.py
```

---

## Configuration Conventions

- `config.yaml` uses **deep-merge semantics** — only include keys to override.
- `config_test.yaml` is a lightweight config for CI/test runs (disables heavy detectors).
- Per-course overrides: pass a separate `--config` pointing to a course-specific YAML that overrides only the relevant thresholds.

---

## Key Files

| File | Role |
|------|------|
| `analyze_exam_snapshots.py` | CLI entry, all factory wiring |
| `pipeline/processor.py` | `FrameProcessor` — core orchestration |
| `detectors/base.py` | Abstract interfaces + all result dataclasses |
| `scoring/risk_scorer.py` | Stateless per-frame scorer; pure function |
| `scoring/aggregator.py` | Temporal incident logic; `FrameRecord`, `StudentSummary` |
| `pipeline/calibration.py` | Head-pose baseline per student/attempt |
| `config.yaml` | Canonical defaults for all tuneable parameters |
| `tests/test_scoring.py` | Unit tests using `_ok_face()` / `_ok_pose()` helper pattern |
| `moodle_extractor.py` | Extract images from Moodle DB + moodledata (filesystem or download) |
| `moodle_ws_extractor.py` | Lightweight URL-list based downloader (no direct DB) |
| `generate_mock_moodle_db.py` | Create mock Moodle SQLite DB + filedir for testing |
| `review_dashboard.py` | Flask web dashboard for reviewing results |
