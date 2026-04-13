"""
download_real_test_data.py
==========================
Builds realistic exam-proctor test scenarios from pre-extracted LFW images
(Labeled Faces in the Wild, already in ~/scikit_learn_data/lfw_home/lfw_funneled).

Why this beats the synthetic data
----------------------------------
The cartoon faces in generate_test_data.py are invisible to MediaPipe
BlazeFace, so every frame scores NO_FACE → 40 pts → every student is
flagged "high" regardless of scenario.  With real face images the
detectors actually fire, so different scenarios produce meaningfully
different risk scores.

Scenarios
---------
  student_r01  Normal        Frontal face every frame                 → expect LOW  risk
  student_r02  Looking-away  Frames 4-8 have face rotated left 45°   → expect MED  risk
  student_r03  No-face       Frames 3-7 are blank (student left desk) → expect HIGH risk
  student_r04  Multi-face    Every frame has *two* faces side-by-side  → expect HIGH risk
  student_r05  Face-hidden   Frames 3-8: body visible but face covered → expect HIGH risk (FACE_HIDDEN)

Requirements
------------
  LFW must already be extracted to:
    ~/scikit_learn_data/lfw_home/lfw_funneled/
  (This happens automatically after sklearn.datasets.fetch_lfw_people() runs
   once, OR if you previously ran:  python download_real_test_data.py)

  If the directory is missing, the script falls back to downloading a small
  set of images via scikit-learn.

Output
------
    real_test_data/
    ├── metadata.csv
    └── snapshots/
        ├── student_r01/ … student_r04/

    real_reference_faces/
    ├── student_r01/ … student_r04/   (one reference photo each)
"""

from __future__ import annotations

import csv
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR = Path("real_test_data")
REF_DIR = Path("real_reference_faces")
FRAME_W, FRAME_H = 640, 480
FRAMES = 12
COURSE = "course_CS101"
BASE_TS = datetime(2024, 6, 10, 9, 0, 0)

# ---------------------------------------------------------------------------
# Locate and load LFW images
# ---------------------------------------------------------------------------

# Common locations where sklearn caches the extracted LFW dataset
_LFW_CANDIDATES = [
    Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled",
    Path.home() / "scikit_learn_data" / "lfw_home" / "lfw",
    Path("/tmp/lfw_funneled"),
]


def find_lfw_root() -> Optional[Path]:
    for p in _LFW_CANDIDATES:
        if p.exists() and any(p.iterdir()):
            return p
    return None


def load_lfw_direct(min_images: int = 15) -> dict[str, list[np.ndarray]]:
    """
    Reads face images straight from the extracted LFW directory tree.
    Returns {person_name: [bgr_image, ...]} for people with ≥ min_images photos.
    Falls back to sklearn download if the local cache is absent.
    """
    root = find_lfw_root()

    if root is None:
        print("LFW not found locally – downloading via scikit-learn (~200 MB)…")
        try:
            from sklearn.datasets import fetch_lfw_people  # type: ignore
        except ImportError:
            raise SystemExit("Run: pip install scikit-learn")
        lfw = fetch_lfw_people(
            min_faces_per_person=min_images, color=True, resize=0.5
        )
        root = find_lfw_root()
        if root is None:
            raise SystemExit("LFW download completed but directory not found.")

    print(f"Reading LFW from {root} …")
    per_person: dict[str, list[np.ndarray]] = {}

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            bgr = cv2.imread(str(img_path))
            if bgr is not None:
                imgs.append(bgr)
        if len(imgs) >= min_images:
            per_person[person_dir.name] = imgs

    people_sorted = sorted(per_person.keys(),
                           key=lambda k: len(per_person[k]), reverse=True)
    print(f"  {len(people_sorted)} people with ≥{min_images} images. "
          f"Top 6: " + ", ".join(
              f"{n}({len(per_person[n])})" for n in people_sorted[:6]))
    return {n: per_person[n] for n in people_sorted}


# ---------------------------------------------------------------------------
# Frame-building helpers
# ---------------------------------------------------------------------------

BACKGROUND_COLOR = (60, 62, 68)   # dark neutral, webcam-style


def make_blank_frame() -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), BACKGROUND_COLOR, dtype=np.uint8)


def embed_face(
    frame: np.ndarray,
    face_bgr: np.ndarray,
    cx: int,
    cy: int,
    target_h: int = 220,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Resize and paste a face image centred at (cx, cy), optionally rotated."""
    h0, w0 = face_bgr.shape[:2]
    scale = target_h / h0
    new_w = int(w0 * scale)
    resized = cv2.resize(face_bgr, (new_w, target_h),
                         interpolation=cv2.INTER_LANCZOS4)

    # Optional in-plane rotation (simulates head turn)
    if rotation_deg != 0.0:
        M = cv2.getRotationMatrix2D(
            (new_w // 2, target_h // 2), rotation_deg, 1.0
        )
        resized = cv2.warpAffine(
            resized, M, (new_w, target_h),
            borderMode=cv2.BORDER_REPLICATE,
        )

    x1 = cx - new_w // 2
    y1 = cy - target_h // 2
    x2 = x1 + new_w
    y2 = y1 + target_h

    # Clamp to frame
    fx1 = max(x1, 0);  fy1 = max(y1, 0)
    fx2 = min(x2, FRAME_W); fy2 = min(y2, FRAME_H)
    rx1 = fx1 - x1;  ry1 = fy1 - y1
    rx2 = rx1 + (fx2 - fx1); ry2 = ry1 + (fy2 - fy1)

    if fx2 > fx1 and fy2 > fy1:
        frame[fy1:fy2, fx1:fx2] = resized[ry1:ry2, rx1:rx2]
    return frame


def overlay_phone(frame: np.ndarray,
                  x: int = 460, y: int = 330) -> np.ndarray:
    """Draw a stylised smartphone onto the frame."""
    cv2.rectangle(frame, (x, y), (x + 55, y + 100), (20, 20, 25), -1)
    cv2.rectangle(frame, (x, y), (x + 55, y + 100), (70, 70, 80), 2)
    cv2.rectangle(frame, (x + 5, y + 12), (x + 50, y + 82), (80, 150, 200), -1)
    cv2.circle(frame, (x + 27, y + 93), 5, (90, 90, 100), -1)
    return frame


def add_timestamp_text(frame: np.ndarray, ts: str) -> np.ndarray:
    cv2.putText(frame, ts, (10, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
                cv2.LINE_AA)
    return frame


def add_webcam_vignette(frame: np.ndarray) -> np.ndarray:
    """Subtle corner darkening to make it look more webcam-like."""
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    cv2.ellipse(mask, (FRAME_W // 2, FRAME_H // 2),
                (FRAME_W // 2, FRAME_H // 2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    vignette = (mask[:, :, np.newaxis] * 0.25 + 0.75).astype(np.float32)
    return np.clip((frame.astype(np.float32) * vignette), 0, 255).astype(np.uint8)


def add_noise(frame: np.ndarray, sigma: int = 6) -> np.ndarray:
    noise = np.random.normal(0, sigma, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

MetaRows = list[dict]


def make_normal_scenario(
    images: list[np.ndarray],
    student_id: str,
    attempt_id: str,
    out_dir: Path,
) -> MetaRows:
    """Frontal face in every frame → should score LOW risk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: MetaRows = []
    ts = BASE_TS

    for i in range(FRAMES):
        frame = make_blank_frame()
        face = images[i % len(images)]
        # Slight random jitter to simulate natural movement
        jx = int(np.random.uniform(-15, 15))
        jy = int(np.random.uniform(-8, 8))
        embed_face(frame, face, FRAME_W // 2 + jx, FRAME_H // 2 + jy)
        add_timestamp_text(frame, ts.isoformat(timespec="seconds"))
        frame = add_webcam_vignette(add_noise(frame))

        fname = f"frame_{i + 1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(_row(student_id, attempt_id, fname, ts))
        ts += timedelta(seconds=5)
    return rows


def make_looking_away_scenario(
    images: list[np.ndarray],
    student_id: str,
    attempt_id: str,
    out_dir: Path,
) -> MetaRows:
    """
    Frames 4-9 have the face rotated ~45° (simulates student looking away),
    and a phone is visible in frames 5-8.
    Should score MEDIUM-HIGH risk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: MetaRows = []
    ts = BASE_TS

    for i in range(FRAMES):
        frame = make_blank_frame()
        face = images[i % len(images)]
        looking_away = 3 <= i <= 8
        phone_visible = 4 <= i <= 7

        rotation = 40.0 if looking_away else 0.0
        cx = (FRAME_W // 2 + 55) if looking_away else FRAME_W // 2

        embed_face(frame, face, cx, FRAME_H // 2, rotation_deg=rotation)
        if phone_visible:
            overlay_phone(frame)

        add_timestamp_text(frame, ts.isoformat(timespec="seconds"))
        frame = add_webcam_vignette(add_noise(frame))

        fname = f"frame_{i + 1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(_row(student_id, attempt_id, fname, ts))
        ts += timedelta(seconds=5)
    return rows


def make_noface_scenario(
    images: list[np.ndarray],
    student_id: str,
    attempt_id: str,
    out_dir: Path,
) -> MetaRows:
    """
    Frames 3-7: face disappears (student left desk or covered camera).
    Should score HIGH risk for those frames (NO_FACE).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: MetaRows = []
    ts = BASE_TS

    for i in range(FRAMES):
        frame = make_blank_frame()
        face = images[i % len(images)]
        no_face = 2 <= i <= 6

        if not no_face:
            embed_face(frame, face, FRAME_W // 2, FRAME_H // 2)
        else:
            # draw an empty desk to look realistic
            cv2.rectangle(frame, (0, FRAME_H - 80), (FRAME_W, FRAME_H),
                          (80, 70, 60), -1)

        add_timestamp_text(frame, ts.isoformat(timespec="seconds"))
        frame = add_webcam_vignette(add_noise(frame))

        fname = f"frame_{i + 1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(_row(student_id, attempt_id, fname, ts))
        ts += timedelta(seconds=5)
    return rows


def make_multiface_scenario(
    images_a: list[np.ndarray],
    images_b: list[np.ndarray],
    student_id: str,
    attempt_id: str,
    out_dir: Path,
) -> MetaRows:
    """
    Two faces visible in every frame → should score HIGH risk (MULTI_FACE).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: MetaRows = []
    ts = BASE_TS

    for i in range(FRAMES):
        frame = make_blank_frame()
        face_a = images_a[i % len(images_a)]
        face_b = images_b[i % len(images_b)]
        embed_face(frame, face_a, FRAME_W // 3,     FRAME_H // 2, target_h=200)
        embed_face(frame, face_b, 2 * FRAME_W // 3, FRAME_H // 2, target_h=200)

        add_timestamp_text(frame, ts.isoformat(timespec="seconds"))
        frame = add_webcam_vignette(add_noise(frame))

        fname = f"frame_{i + 1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(_row(student_id, attempt_id, fname, ts))
        ts += timedelta(seconds=5)
    return rows


def make_face_hidden_scenario(
    images: list[np.ndarray],
    student_id: str,
    attempt_id: str,
    out_dir: Path,
) -> MetaRows:
    """
    Frames 3-8: person body is still visible but face is covered/hidden
    (simulates student looking down at cheat sheet under desk, or covering
    the camera with a hand, or leaning far forward).

    YOLO should still detect the human body → person_count ≥ 1.
    MediaPipe will fail to find a face → face_count == 0.
    Result: FACE_HIDDEN flag (55 pts) instead of NO_FACE (40 pts).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: MetaRows = []
    ts = BASE_TS

    for i in range(FRAMES):
        frame = make_blank_frame()
        face = images[i % len(images)]
        hide_face = 2 <= i <= 7

        embed_face(frame, face, FRAME_W // 2, FRAME_H // 2, target_h=240)

        if hide_face:
            # Cover the face region with a dark rectangle.
            # The embedded face centre is (~320, ~240) with height 240.
            # The face occupies roughly the top 55% of the embedded image.
            face_top    = FRAME_H // 2 - 120          # ~120
            face_bottom = FRAME_H // 2 - 120 + 140    # ~260  (covers eyes/nose/mouth)
            face_left   = FRAME_W // 2 - 90           # ~230
            face_right  = FRAME_W // 2 + 90           # ~410
            cv2.rectangle(frame,
                          (face_left, face_top),
                          (face_right, face_bottom),
                          (30, 25, 20), -1)            # near-black box
            # Add subtle text hint that person bent forward
            cv2.putText(frame, "student bent forward",
                        (face_left, face_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (120, 120, 120), 1, cv2.LINE_AA)

        add_timestamp_text(frame, ts.isoformat(timespec="seconds"))
        frame = add_webcam_vignette(add_noise(frame))

        fname = f"frame_{i + 1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(_row(student_id, attempt_id, fname, ts))
        ts += timedelta(seconds=5)
    return rows



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(student_id: str, attempt_id: str, fname: str,
         ts: datetime) -> dict:
    return {
        "image_path": f"snapshots/{student_id}/{fname}",
        "student_id": student_id,
        "attempt_id": attempt_id,
        "timestamp": ts.isoformat(timespec="seconds"),
        "course_id": COURSE,
    }


def save_reference_face(face_bgr: np.ndarray,
                        student_id: str) -> None:
    ref_dir = REF_DIR / student_id
    ref_dir.mkdir(parents=True, exist_ok=True)
    # Upscale to a reasonable portrait size
    h, w = face_bgr.shape[:2]
    scale = 200 / h
    ref = cv2.resize(face_bgr, (int(w * scale), 200),
                     interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(ref_dir / "reference.jpg"), ref,
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def write_metadata(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "student_id", "attempt_id",
                           "timestamp", "course_id"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    per_person = load_lfw_direct(min_images=15)
    people = list(per_person.keys())  # sorted by image count descending

    if len(people) < 5:
        raise SystemExit(
            f"Need at least 5 people in LFW subset, got {len(people)}. "
            "Try lowering min_faces_per_person in fetch_lfw()."
        )

    # Assign people to roles
    person_a, person_b, person_c, person_d, person_e = (
        people[0], people[1], people[2], people[3], people[4]
    )
    print(f"\nScenario assignments:")
    print(f"  student_r01 (normal)       → {person_a}")
    print(f"  student_r02 (looking-away) → {person_b}")
    print(f"  student_r03 (no-face)      → {person_c}")
    print(f"  student_r04 (multi-face)   → {person_d} + {person_a}")
    print(f"  student_r05 (face-hidden)  → {person_e}")

    snaps = OUT_DIR / "snapshots"
    all_rows: list[dict] = []

    print("\nBuilding test frames…")

    # student_r01 – normal
    all_rows += make_normal_scenario(
        per_person[person_a], "student_r01", "attempt_r01",
        snaps / "student_r01",
    )
    save_reference_face(per_person[person_a][FRAMES], "student_r01")

    # student_r02 – looking away + phone
    all_rows += make_looking_away_scenario(
        per_person[person_b], "student_r02", "attempt_r02",
        snaps / "student_r02",
    )
    save_reference_face(per_person[person_b][FRAMES], "student_r02")

    # student_r03 – no face
    all_rows += make_noface_scenario(
        per_person[person_c], "student_r03", "attempt_r03",
        snaps / "student_r03",
    )
    save_reference_face(per_person[person_c][FRAMES], "student_r03")

    # student_r04 – multi-face
    all_rows += make_multiface_scenario(
        per_person[person_d], per_person[person_a],
        "student_r04", "attempt_r04",
        snaps / "student_r04",
    )
    save_reference_face(per_person[person_d][FRAMES], "student_r04")

    # student_r05 – face hidden (person visible, face covered)
    all_rows += make_face_hidden_scenario(
        per_person[person_e], "student_r05", "attempt_r05",
        snaps / "student_r05",
    )
    save_reference_face(per_person[person_e][FRAMES], "student_r05")

    write_metadata(all_rows, OUT_DIR / "metadata.csv")

    print(f"\nDone!  {len(all_rows)} frames across 5 students.")
    print(f"Test data  → {OUT_DIR}/")
    print(f"Ref faces  → {REF_DIR}/")
    print()
    print("Run the pipeline:")
    print(f"  python analyze_exam_snapshots.py \\")
    print(f"    --metadata {OUT_DIR}/metadata.csv \\")
    print(f"    --output   real_results \\")
    print(f"    --config   config_test.yaml")
    print()
    print("Expected scores (approx):")
    print("  student_r01  LOW   risk  (frontal face every frame)")
    print("  student_r02  MED   risk  (look-away + phone in ~half the frames)")
    print("  student_r03  HIGH  risk  (NO_FACE for 5 frames — empty desk)")
    print("  student_r04  HIGH  risk  (MULTI_FACE every frame)")
    print("  student_r05  HIGH  risk  (FACE_HIDDEN — body visible, face covered)")


if __name__ == "__main__":
    main()
