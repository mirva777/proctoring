"""
generate_test_data.py
=====================
Creates synthetic webcam-style test images and a matching metadata CSV
so you can run analyze_exam_snapshots.py without real photos.

Usage:
    python generate_test_data.py

Output:
    test_data/
    ├── snapshots/
    │   ├── student_001/  (normal student – should score low)
    │   ├── student_002/  (phone visible + look-away – should score high)
    │   └── student_003/  (no face frames – should score high)
    └── metadata.csv
    
    reference_faces/
    ├── student_001/
    ├── student_002/
    └── student_003/
"""

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("test_data")
REF_DIR = Path("reference_faces")
FRAME_W, FRAME_H = 640, 480

# How many frames per student
FRAMES = 10


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def blank(bg_color=(60, 60, 60)):
    """Return a plain dark-grey webcam-sized frame."""
    return np.full((FRAME_H, FRAME_W, 3), bg_color, dtype=np.uint8)


def draw_face(img, cx, cy, size=90, skin=(180, 140, 110), looking_away=False):
    """
    Draw a cartoon face oval at (cx, cy).
    If looking_away=True, shift the pupils to the side to simulate gaze-off.
    """
    # Head oval
    cv2.ellipse(img, (cx, cy), (size, int(size * 1.25)), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (cx, cy), (size, int(size * 1.25)), 0, 0, 360, (90, 70, 50), 2)

    # Eyes
    eye_y = cy - size // 4
    for ex in [cx - size // 3, cx + size // 3]:
        cv2.circle(img, (ex, eye_y), size // 8, (255, 255, 255), -1)
        # Pupil – shift sideways if looking away
        pupil_offset = size // 5 if looking_away else 0
        cv2.circle(img, (ex + pupil_offset, eye_y), size // 14, (30, 30, 30), -1)

    # Mouth
    mouth_y = cy + size // 3
    cv2.ellipse(img, (cx, mouth_y), (size // 3, size // 8), 0, 0, 180, (120, 80, 60), 2)

    # Simple hair
    cv2.ellipse(img, (cx, cy - size // 2), (size, size // 3), 0, 180, 360, (60, 40, 20), -1)

    return img


def draw_phone(img, x=480, y=350):
    """Draw a simple rectangle representing a mobile phone."""
    cv2.rectangle(img, (x, y), (x + 50, y + 90), (30, 30, 30), -1)
    cv2.rectangle(img, (x + 5, y + 10), (x + 45, y + 75), (100, 180, 220), -1)
    cv2.circle(img, (x + 25, y + 82), 5, (80, 80, 80), -1)
    return img


def draw_book(img, x=30, y=300):
    """Draw a simple open book."""
    cv2.rectangle(img, (x, y), (x + 120, y + 90), (210, 180, 130), -1)
    cv2.line(img, (x + 60, y), (x + 60, y + 90), (140, 110, 80), 2)
    for line_y in range(y + 15, y + 80, 12):
        cv2.line(img, (x + 8, line_y), (x + 52, line_y), (160, 140, 100), 1)
        cv2.line(img, (x + 68, line_y), (x + 112, line_y), (160, 140, 100), 1)
    return img


def add_timestamp(img, ts: str):
    cv2.putText(img, ts, (10, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def add_noise(img, amount=8):
    noise = np.random.randint(-amount, amount, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Student scenarios
# ─────────────────────────────────────────────────────────────────────────────

def make_normal_student(out_dir: Path, student_id="student_001",
                        attempt_id="attempt_001", start_ts=None):
    """Normal student: face present, looking forward, no objects."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ts = start_ts or datetime(2024, 6, 10, 9, 0, 0)

    for i in range(FRAMES):
        img = blank((55, 55, 65))
        # Slight random head position drift
        jitter_x = np.random.randint(-10, 10)
        jitter_y = np.random.randint(-5, 5)
        draw_face(img, FRAME_W // 2 + jitter_x, FRAME_H // 2 + jitter_y, size=85)
        add_timestamp(img, ts.strftime("%Y-%m-%dT%H:%M:%S"))
        add_noise(img)

        fname = f"frame_{i+1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), img)
        rows.append({
            "image_path": f"snapshots/{student_id}/{fname}",
            "student_id": student_id,
            "attempt_id": attempt_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "course_id": "course_CS101",
        })
        ts += timedelta(seconds=5)

    return rows


def make_suspicious_student(out_dir: Path, student_id="student_002",
                             attempt_id="attempt_002", start_ts=None):
    """Suspicious student: phone appears in frames 4-7, look-away in frames 5-8."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ts = start_ts or datetime(2024, 6, 10, 9, 0, 0)

    for i in range(FRAMES):
        img = blank((55, 55, 65))
        phone = (3 <= i <= 7)
        looking_away = (4 <= i <= 8)

        # Draw face (slightly off-centre when looking away)
        cx = FRAME_W // 2 + (60 if looking_away else 0)
        draw_face(img, cx, FRAME_H // 2, size=85, looking_away=looking_away)

        if phone:
            draw_phone(img)

        add_timestamp(img, ts.strftime("%Y-%m-%dT%H:%M:%S"))
        add_noise(img)

        fname = f"frame_{i+1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), img)
        rows.append({
            "image_path": f"snapshots/{student_id}/{fname}",
            "student_id": student_id,
            "attempt_id": attempt_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "course_id": "course_CS101",
        })
        ts += timedelta(seconds=5)

    return rows


def make_noface_student(out_dir: Path, student_id="student_003",
                        attempt_id="attempt_003", start_ts=None):
    """Problematic student: face missing in several frames, book visible."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ts = start_ts or datetime(2024, 6, 10, 9, 0, 0)

    for i in range(FRAMES):
        img = blank((55, 55, 65))
        no_face = (2 <= i <= 5)  # face leaves frame for ~20 seconds

        if not no_face:
            draw_face(img, FRAME_W // 2, FRAME_H // 2, size=85)

        # Book always visible
        draw_book(img)

        add_timestamp(img, ts.strftime("%Y-%m-%dT%H:%M:%S"))
        add_noise(img)

        fname = f"frame_{i+1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), img)
        rows.append({
            "image_path": f"snapshots/{student_id}/{fname}",
            "student_id": student_id,
            "attempt_id": attempt_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "course_id": "course_MATH201",
        })
        ts += timedelta(seconds=5)

    return rows


def make_low_quality_student(out_dir: Path, student_id="student_004",
                              attempt_id="attempt_004", start_ts=None):
    """Low quality: blurry / dark / over-exposed frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ts = start_ts or datetime(2024, 6, 10, 9, 0, 0)

    for i in range(FRAMES):
        img = blank((55, 55, 65))
        draw_face(img, FRAME_W // 2, FRAME_H // 2, size=85)

        if i < 3:
            # Very dark
            img = (img * 0.15).astype(np.uint8)
        elif i < 6:
            # Blurry
            img = cv2.GaussianBlur(img, (31, 31), 15)
        else:
            # Over-exposed / glare
            img = np.clip(img.astype(np.int32) + 180, 0, 255).astype(np.uint8)

        add_timestamp(img, ts.strftime("%Y-%m-%dT%H:%M:%S"))

        fname = f"frame_{i+1:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), img)
        rows.append({
            "image_path": f"snapshots/{student_id}/{fname}",
            "student_id": student_id,
            "attempt_id": attempt_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "course_id": "course_MATH201",
        })
        ts += timedelta(seconds=5)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Reference faces (single clean portrait per student)
# ─────────────────────────────────────────────────────────────────────────────

def make_reference_face(ref_dir: Path, student_id: str, skin_color=(180, 140, 110)):
    ref_path = ref_dir / student_id
    ref_path.mkdir(parents=True, exist_ok=True)
    img = blank((70, 70, 80))
    draw_face(img, FRAME_W // 2, FRAME_H // 2, size=100, skin=skin_color)
    cv2.imwrite(str(ref_path / "reference.jpg"), img)
    print(f"  Reference face: {ref_path / 'reference.jpg'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic test data...\n")

    all_rows = []

    scenarios = [
        ("student_001", "attempt_001", make_normal_student,    "NORMAL – should score LOW"),
        ("student_002", "attempt_002", make_suspicious_student,"SUSPICIOUS – phone + look-away"),
        ("student_003", "attempt_003", make_noface_student,    "NO FACE + book visible"),
        ("student_004", "attempt_004", make_low_quality_student,"LOW QUALITY frames"),
    ]

    for sid, aid, fn, desc in scenarios:
        print(f"[{sid}] {desc}")
        snap_dir = OUT_DIR / "snapshots" / sid
        rows = fn(snap_dir, student_id=sid, attempt_id=aid)
        all_rows.extend(rows)
        print(f"  → {len(rows)} frames in {snap_dir}")

    # Write metadata CSV
    csv_path = OUT_DIR / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_path", "student_id", "attempt_id", "timestamp", "course_id"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nMetadata CSV: {csv_path}  ({len(all_rows)} rows)")

    # Write reference faces
    print("\nGenerating reference faces...")
    skin_tones = [
        (180, 140, 110),
        (160, 120, 90),
        (200, 160, 130),
        (140, 100, 75),
    ]
    for i, (sid, _, _, _) in enumerate(scenarios):
        make_reference_face(REF_DIR, sid, skin_color=skin_tones[i])

    print(f"\n✓ Done!\n")
    print("Run the analyser with:")
    print(f"  python analyze_exam_snapshots.py \\")
    print(f"    --metadata {csv_path} \\")
    print(f"    --output   ./results \\")
    print(f"    --config   ./config.yaml \\")
    print(f"    --reference-faces {REF_DIR}")


if __name__ == "__main__":
    main()
