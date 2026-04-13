#!/usr/bin/env python3
"""
generate_mock_moodle_db.py
==========================
Creates a local SQLite database that mimics the Moodle quizaccess_proctoring
plugin's schema, populated with test data. Also creates a fake moodledata/filedir
tree so you can test moodle_extractor.py without a real Moodle installation.

This script reuses existing test_data/ snapshots (or generates simple images)
and stores them in the Moodle content-addressed format.

Usage:
    python generate_mock_moodle_db.py [--output ./mock_moodle]

Output structure:
    mock_moodle/
        moodle.db               ← SQLite database mimicking Moodle tables
        moodledata/filedir/     ← Content-addressed file store
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Check if we have cv2 for generating fallback images
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def create_moodle_schema(conn: sqlite3.Connection):
    """Create the MDL tables we need for testing."""
    cur = conn.cursor()

    # User table (simplified)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_user (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            firstname TEXT NOT NULL,
            lastname TEXT NOT NULL,
            email TEXT NOT NULL,
            picture INTEGER DEFAULT 0
        )
    """)

    # Course table (simplified)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_course (
            id INTEGER PRIMARY KEY,
            shortname TEXT NOT NULL,
            fullname TEXT NOT NULL
        )
    """)

    # Course modules (simplified)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_course_modules (
            id INTEGER PRIMARY KEY,
            course INTEGER NOT NULL,
            module INTEGER NOT NULL,
            instance INTEGER NOT NULL
        )
    """)

    # Context table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_context (
            id INTEGER PRIMARY KEY,
            contextlevel INTEGER NOT NULL,
            instanceid INTEGER NOT NULL
        )
    """)

    # Proctoring logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_quizaccess_proctoring_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            courseid INTEGER NOT NULL,
            quizid INTEGER NOT NULL,
            userid INTEGER NOT NULL,
            webcampicture TEXT DEFAULT '',
            status INTEGER DEFAULT 0,
            awsscore INTEGER DEFAULT 0,
            awsflag INTEGER DEFAULT 0,
            deletionprogress INTEGER DEFAULT 0,
            timemodified INTEGER DEFAULT 0
        )
    """)

    # Proctoring settings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_quizaccess_proctoring (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quizid INTEGER NOT NULL,
            proctoringrequired INTEGER DEFAULT 1
        )
    """)

    # User images (reference photos)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_quizaccess_proctoring_user_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            photo_draft_id INTEGER DEFAULT 0
        )
    """)

    # Face images
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_quizaccess_proctoring_face_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_type TEXT DEFAULT '',
            parentid INTEGER NOT NULL,
            faceimage TEXT DEFAULT '',
            facefound INTEGER DEFAULT 0,
            timemodified INTEGER DEFAULT 0
        )
    """)

    # Files table (simplified Moodle files table)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mdl_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contenthash TEXT NOT NULL,
            pathnamehash TEXT NOT NULL,
            contextid INTEGER NOT NULL,
            component TEXT NOT NULL,
            filearea TEXT NOT NULL,
            itemid INTEGER NOT NULL,
            filepath TEXT DEFAULT '/',
            filename TEXT NOT NULL,
            userid INTEGER DEFAULT 0,
            filesize INTEGER DEFAULT 0,
            mimetype TEXT DEFAULT 'image/png',
            timecreated INTEGER DEFAULT 0,
            timemodified INTEGER DEFAULT 0
        )
    """)

    conn.commit()


def store_file_moodle_style(
    file_data: bytes,
    moodledata: Path,
    conn: sqlite3.Connection,
    contextid: int,
    component: str,
    filearea: str,
    itemid: int,
    filename: str,
    userid: int = 0,
) -> str:
    """Store a file in Moodle's content-addressed format and register it in mdl_files."""
    # Calculate content hash (SHA1)
    contenthash = hashlib.sha1(file_data).hexdigest()

    # Create filedir path
    dir1 = contenthash[:2]
    dir2 = contenthash[2:4]
    file_dir = moodledata / "filedir" / dir1 / dir2
    file_dir.mkdir(parents=True, exist_ok=True)

    file_path = file_dir / contenthash
    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(file_data)

    # Calculate pathnamehash
    pathname = f"/{contextid}/quizaccess_proctoring/{filearea}/{itemid}/{filename}"
    pathnamehash = hashlib.sha1(pathname.encode()).hexdigest()

    # Insert into files table
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO mdl_files (contenthash, pathnamehash, contextid, component,
                               filearea, itemid, filepath, filename, userid,
                               filesize, timecreated, timemodified)
        VALUES (?, ?, ?, ?, ?, ?, '/', ?, ?, ?, ?, ?)
    """, (contenthash, pathnamehash, contextid, component, filearea, itemid,
          filename, userid, len(file_data), int(time.time()), int(time.time())))

    # Also insert the directory entry ('.' filename)
    dir_pathname = f"/{contextid}/quizaccess_proctoring/{filearea}/{itemid}/."
    dir_pathnamehash = hashlib.sha1(dir_pathname.encode()).hexdigest()
    cur.execute("""
        INSERT OR IGNORE INTO mdl_files (contenthash, pathnamehash, contextid, component,
                               filearea, itemid, filepath, filename, userid,
                               filesize, timecreated, timemodified)
        VALUES (?, ?, ?, ?, ?, ?, '/', '.', ?, 0, ?, ?)
    """, (hashlib.sha1(b"").hexdigest(), dir_pathnamehash, contextid, component, filearea, itemid,
          userid, int(time.time()), int(time.time())))

    conn.commit()

    return contenthash


def generate_simple_image(text: str = "TEST", w: int = 320, h: int = 240) -> bytes:
    """Generate a simple test image with text overlay."""
    if HAS_CV2:
        img = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
        cv2.putText(img, text, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        _, buf = cv2.imencode(".png", img)
        return bytes(buf)
    else:
        # Minimal 1x1 PNG if no cv2
        import struct
        import zlib
        raw = b'\x00\x80\x80\x80'
        compressed = zlib.compress(raw)
        ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        chunks = b''
        for ctype, data in [(b'IHDR', ihdr), (b'IDAT', compressed), (b'IEND', b'')]:
            chunk = struct.pack('>I', len(data)) + ctype + data
            chunk += struct.pack('>I', zlib.crc32(ctype + data) & 0xffffffff)
            chunks += chunk
        return b'\x89PNG\r\n\x1a\n' + chunks


def main():
    parser = argparse.ArgumentParser(description="Generate mock Moodle proctoring data for testing")
    parser.add_argument("--output", default="./mock_moodle", help="Output directory")
    parser.add_argument("--use-test-data", default=None,
                        help="Path to existing test_data/ directory to reuse real images")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    moodledata = output / "moodledata"
    moodledata.mkdir(exist_ok=True)

    db_path = output / "moodle.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    create_moodle_schema(conn)

    cur = conn.cursor()

    # Create test data
    COURSE_ID = 5
    QUIZ_CMID = 42  # cmid (course module id)
    MODULE_CONTEXT_ID = 100
    SYSTEM_CONTEXT_ID = 1
    MOODLE_URL = "https://moodle.example.com"

    # Insert contexts
    cur.execute("INSERT INTO mdl_context (id, contextlevel, instanceid) VALUES (?, 10, 0)", (SYSTEM_CONTEXT_ID,))
    cur.execute("INSERT INTO mdl_context (id, contextlevel, instanceid) VALUES (?, 70, ?)",
                (MODULE_CONTEXT_ID, QUIZ_CMID))

    # Insert course
    cur.execute("INSERT INTO mdl_course (id, shortname, fullname) VALUES (?, 'CS101', 'Introduction to CS')",
                (COURSE_ID,))

    # Insert course module
    cur.execute("INSERT INTO mdl_course_modules (id, course, module, instance) VALUES (?, ?, 1, 1)",
                (QUIZ_CMID, COURSE_ID))

    # Insert proctoring setting
    cur.execute("INSERT INTO mdl_quizaccess_proctoring (quizid, proctoringrequired) VALUES (?, 1)",
                (QUIZ_CMID,))

    # Define test students
    students = [
        (101, "jsmith", "John", "Smith", "jsmith@example.com"),
        (102, "jdoe", "Jane", "Doe", "jdoe@example.com"),
        (103, "mbrown", "Mike", "Brown", "mbrown@example.com"),
        (104, "awilson", "Alice", "Wilson", "awilson@example.com"),
    ]

    for uid, uname, fname, lname, email in students:
        cur.execute("INSERT INTO mdl_user (id, username, firstname, lastname, email) VALUES (?, ?, ?, ?, ?)",
                    (uid, uname, fname, lname, email))

    conn.commit()

    # Try to load existing test images
    test_data_dir = Path(args.use_test_data) if args.use_test_data else None
    existing_images: dict[str, list[Path]] = {}

    if test_data_dir and test_data_dir.exists():
        snap_dir = test_data_dir / "snapshots"
        if snap_dir.exists():
            for student_dir in sorted(snap_dir.iterdir()):
                if student_dir.is_dir():
                    imgs = sorted(student_dir.glob("*.jpg")) + sorted(student_dir.glob("*.png"))
                    if imgs:
                        existing_images[student_dir.name] = imgs
            logger.info("Found existing test images for %d students", len(existing_images))

    # Map test students to existing image directories
    image_sources = list(existing_images.values()) if existing_images else []

    # Generate proctoring snapshots
    base_time = int(datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc).timestamp())
    snapshot_interval = 30  # 30 seconds between captures
    frames_per_student = 12

    log_id = 0
    for student_idx, (uid, uname, fname, lname, email) in enumerate(students):
        for frame_idx in range(frames_per_student):
            log_id += 1
            frame_time = base_time + (student_idx * frames_per_student + frame_idx) * snapshot_interval

            # Get or generate image data
            if image_sources and student_idx < len(image_sources):
                img_path = image_sources[student_idx][frame_idx % len(image_sources[student_idx])]
                with open(img_path, "rb") as f:
                    img_data = f.read()
            else:
                img_data = generate_simple_image(f"{uname} #{frame_idx + 1}")

            # Store image in Moodle-style file storage
            filename = f"webcam-{log_id}-{uid}-{COURSE_ID}-{frame_time}{frame_idx}.png"
            contenthash = store_file_moodle_style(
                file_data=img_data,
                moodledata=moodledata,
                conn=conn,
                contextid=MODULE_CONTEXT_ID,
                component="quizaccess_proctoring",
                filearea="picture",
                itemid=log_id,
                filename=filename,
                userid=uid,
            )

            # Build pluginfile URL (as the real plugin would store)
            webcam_url = (
                f"{MOODLE_URL}/pluginfile.php/{MODULE_CONTEXT_ID}"
                f"/quizaccess_proctoring/picture/{log_id}/{filename}"
            )

            # Insert proctoring log
            cur.execute("""
                INSERT INTO mdl_quizaccess_proctoring_logs
                    (id, courseid, quizid, userid, webcampicture, status, timemodified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (log_id, COURSE_ID, QUIZ_CMID, uid, webcam_url, 1, frame_time))

    conn.commit()

    # Generate reference face images
    for student_idx, (uid, uname, fname, lname, email) in enumerate(students):
        # Use the first image as reference, or generate one
        if image_sources and student_idx < len(image_sources):
            with open(image_sources[student_idx][0], "rb") as f:
                ref_data = f.read()
        else:
            ref_data = generate_simple_image(f"REF {uname}")

        ref_filename = f"userpic-{uid}.png"
        store_file_moodle_style(
            file_data=ref_data,
            moodledata=moodledata,
            conn=conn,
            contextid=SYSTEM_CONTEXT_ID,
            component="quizaccess_proctoring",
            filearea="user_photo",
            itemid=uid,
            filename=ref_filename,
            userid=uid,
        )

        cur.execute("""
            INSERT INTO mdl_quizaccess_proctoring_user_images (user_id, photo_draft_id)
            VALUES (?, ?)
        """, (uid, 0))

    conn.commit()
    conn.close()

    total_snapshots = len(students) * frames_per_student
    print(f"\n  Mock Moodle data created in: {output.resolve()}")
    print(f"  Database: {db_path.resolve()}")
    print(f"  Moodledata: {moodledata.resolve()}")
    print(f"  Students: {len(students)}")
    print(f"  Snapshots: {total_snapshots}")
    print(f"  Reference images: {len(students)}")
    print(f"\n  Test the extractor with SQLite (requires modifying moodle_extractor.py for SQLite)")
    print(f"  Or use the mock data directly — see generate_mock_moodle_export.py")


if __name__ == "__main__":
    main()
