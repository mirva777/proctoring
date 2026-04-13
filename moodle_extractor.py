#!/usr/bin/env python3
"""
moodle_extractor.py
===================
Extract proctoring webcam snapshots from a Moodle installation that uses the
quizaccess_proctoring plugin (eLearning-BS23/moodle-quizaccess_proctoring).

Two extraction methods are supported:

  1. **filesystem** (default) – Direct access to Moodle DB *and* the
     ``moodledata/`` directory.  Fastest; copies images from Moodle's
     content-addressed file store to a local snapshot tree.

  2. **download** – DB access only.  The ``webcampicture`` column already
     contains a full pluginfile.php URL; we download each image over HTTP
     using a Moodle web-service token.

In both cases the script produces:
  • ``snapshots/<student_id>/`` directories with PNG files
  • ``metadata.csv`` matching the pipeline's expected schema
  • ``reference_faces/<student_id>/`` with admin-uploaded reference photos

Usage examples
--------------
  # Filesystem mode (Moodle on same server):
  python moodle_extractor.py \\
      --db-host localhost --db-name moodle --db-user moodle --db-pass secret \\
      --moodledata /var/www/moodledata \\
      --output ./moodle_export \\
      --quiz-id 42

  # Download mode (remote Moodle):
  python moodle_extractor.py \\
      --db-host db.example.com --db-name moodle --db-user reader --db-pass s3cret \\
      --method download \\
      --moodle-url https://moodle.example.com \\
      --token abc123def456 \\
      --output ./moodle_export \\
      --quiz-id 42

  # Export all quizzes for a course:
  python moodle_extractor.py \\
      --db-host localhost --db-name moodle --db-user moodle --db-pass secret \\
      --moodledata /var/www/moodledata \\
      --output ./moodle_export \\
      --course-id 5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies – fail late with helpful messages
# ---------------------------------------------------------------------------
try:
    import mysql.connector  # type: ignore
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False

try:
    import psycopg2  # type: ignore
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import requests  # type: ignore
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

import sqlite3  # Always available — used for local testing with mock data


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MoodleSnapshot:
    """One webcam snapshot row from quizaccess_proctoring_logs."""
    log_id: int
    course_id: int
    quiz_id: int
    user_id: int
    webcampicture: str          # pluginfile URL or empty
    attempt_status: int         # attempt id stored in 'status' column
    timestamp: int              # UNIX epoch
    # Populated after user join
    username: str = ""
    firstname: str = ""
    lastname: str = ""
    email: str = ""


@dataclass
class MoodleRefImage:
    """Reference (admin-uploaded) face image for a student."""
    user_id: int
    photo_draft_id: int
    username: str = ""


@dataclass
class ExportStats:
    snapshots_found: int = 0
    snapshots_exported: int = 0
    snapshots_failed: int = 0
    ref_images_found: int = 0
    ref_images_exported: int = 0
    students: int = 0


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_connection(args):
    """Create a DB connection based on --db-engine."""
    engine = args.db_engine.lower()
    prefix = getattr(args, "db_prefix", "mdl_")

    if engine == "sqlite":
        # Local testing with SQLite — expects --db-name to be a file path
        conn = sqlite3.connect(args.db_name)
        return conn, prefix

    elif engine == "mysql":
        if not HAS_MYSQL:
            sys.exit("ERROR: mysql-connector-python is required. Run: pip install mysql-connector-python")
        conn = mysql.connector.connect(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_pass,
            charset="utf8mb4",
        )
        return conn, prefix

    elif engine == "pgsql":
        if not HAS_PSYCOPG2:
            sys.exit("ERROR: psycopg2 is required. Run: pip install psycopg2-binary")
        conn = psycopg2.connect(
            host=args.db_host,
            port=args.db_port,
            dbname=args.db_name,
            user=args.db_user,
            password=args.db_pass,
        )
        return conn, prefix

    else:
        sys.exit(f"ERROR: Unsupported db-engine '{engine}'. Use mysql, pgsql, or sqlite.")


def _placeholder(conn) -> str:
    """Return the parameter placeholder for the connection type."""
    if isinstance(conn, sqlite3.Connection):
        return "?"
    return "%s"


def query_snapshots(conn, prefix: str, quiz_id: Optional[int], course_id: Optional[int]) -> list[MoodleSnapshot]:
    """Query quizaccess_proctoring_logs joined with user table."""
    cur = conn.cursor()
    ph = _placeholder(conn)

    sql = f"""
        SELECT
            pl.id,
            pl.courseid,
            pl.quizid,
            pl.userid,
            pl.webcampicture,
            pl.status,
            pl.timemodified,
            u.username,
            u.firstname,
            u.lastname,
            u.email
        FROM {prefix}quizaccess_proctoring_logs pl
        INNER JOIN {prefix}user u ON u.id = pl.userid
        WHERE pl.webcampicture != ''
          AND pl.webcampicture IS NOT NULL
    """
    params = []

    if quiz_id is not None:
        sql += f" AND pl.quizid = {ph}"
        params.append(quiz_id)
    if course_id is not None:
        sql += f" AND pl.courseid = {ph}"
        params.append(course_id)

    sql += " ORDER BY pl.userid, pl.timemodified ASC"

    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()

    snapshots = []
    for r in rows:
        snapshots.append(MoodleSnapshot(
            log_id=r[0],
            course_id=r[1],
            quiz_id=r[2],
            user_id=r[3],
            webcampicture=r[4] or "",
            attempt_status=r[5] or 0,
            timestamp=r[6] or 0,
            username=r[7] or "",
            firstname=r[8] or "",
            lastname=r[9] or "",
            email=r[10] or "",
        ))
    return snapshots


def query_reference_images(conn, prefix: str, user_ids: set[int]) -> list[MoodleRefImage]:
    """Get reference face image records for given user IDs."""
    if not user_ids:
        return []

    cur = conn.cursor()
    ph = _placeholder(conn)

    placeholders = ",".join([ph] * len(user_ids))
    sql = f"""
        SELECT
            ui.user_id,
            ui.photo_draft_id,
            u.username
        FROM {prefix}quizaccess_proctoring_user_images ui
        INNER JOIN {prefix}user u ON u.id = ui.user_id
        WHERE ui.user_id IN ({placeholders})
    """
    cur.execute(sql, list(user_ids))
    rows = cur.fetchall()
    cur.close()

    return [MoodleRefImage(user_id=r[0], photo_draft_id=r[1], username=r[2] or "") for r in rows]


def resolve_file_path_from_db(conn, prefix: str, component: str, filearea: str,
                               itemid: int, contextid: Optional[int] = None) -> Optional[tuple[str, str]]:
    """
    Look up a stored file's contenthash from mdl_files.

    Returns (contenthash, filename) or None.
    """
    cur = conn.cursor()
    ph = _placeholder(conn)

    sql = f"""
        SELECT contenthash, filename
        FROM {prefix}files
        WHERE component = {ph}
          AND filearea = {ph}
          AND itemid = {ph}
          AND filename != '.'
    """
    params = [component, filearea, itemid]

    if contextid is not None:
        sql += f" AND contextid = {ph}"
        params.append(contextid)

    sql += " LIMIT 1"

    cur.execute(sql, params)
    row = cur.fetchone()
    cur.close()

    if row:
        return (row[0], row[1])
    return None


def contenthash_to_path(moodledata: Path, contenthash: str) -> Path:
    """Convert a Moodle contenthash to its physical path under filedir."""
    return moodledata / "filedir" / contenthash[:2] / contenthash[2:4] / contenthash


def resolve_context_id_for_cm(conn, prefix: str, cmid: int) -> Optional[int]:
    """Get the context ID for a course module (contextlevel=70)."""
    cur = conn.cursor()
    ph = _placeholder(conn)
    sql = f"""
        SELECT id FROM {prefix}context
        WHERE contextlevel = 70 AND instanceid = {ph}
        LIMIT 1
    """
    cur.execute(sql, [cmid])
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def resolve_system_context_id(conn, prefix: str) -> int:
    """Get the system context ID (contextlevel=10)."""
    cur = conn.cursor()
    sql = f"""
        SELECT id FROM {prefix}context
        WHERE contextlevel = 10
        LIMIT 1
    """
    cur.execute(sql)
    row = cur.fetchone()
    cur.close()
    return row[0] if row else 1


# ---------------------------------------------------------------------------
# Image extraction — filesystem mode
# ---------------------------------------------------------------------------

def extract_image_filesystem(
    conn, prefix: str, moodledata: Path, snapshot: MoodleSnapshot,
    output_dir: Path, context_cache: dict[int, Optional[int]]
) -> Optional[str]:
    """
    Copy the proctoring image from moodledata/filedir/ to the output tree.

    Returns the relative image path (for metadata.csv) or None on failure.
    """
    if not snapshot.webcampicture:
        return None

    # Parse the pluginfile URL to extract itemid
    # Typical URL: .../pluginfile.php/{contextid}/quizaccess_proctoring/picture/{itemid}/filename.png
    try:
        url_path = urlparse(snapshot.webcampicture).path
        parts = url_path.split("/")

        # Find 'pluginfile.php' in the path and extract components after it
        if "pluginfile.php" in parts:
            idx = parts.index("pluginfile.php")
            # parts after pluginfile.php: contextid, component, filearea, itemid, [subpath/], filename
            if len(parts) > idx + 4:
                url_contextid = int(parts[idx + 1])
                url_itemid = int(parts[idx + 4])
            else:
                logger.warning("Cannot parse pluginfile URL: %s", snapshot.webcampicture)
                return None
        else:
            # Fallback: try using the log_id as itemid
            url_contextid = None
            url_itemid = snapshot.log_id
    except (ValueError, IndexError) as e:
        logger.warning("Error parsing URL %s: %s", snapshot.webcampicture, e)
        return None

    # Look up the actual file in mdl_files
    result = resolve_file_path_from_db(
        conn, prefix,
        component="quizaccess_proctoring",
        filearea="picture",
        itemid=url_itemid,
        contextid=url_contextid if url_contextid else None,
    )

    if result is None:
        # Try with log_id as fallback
        result = resolve_file_path_from_db(
            conn, prefix,
            component="quizaccess_proctoring",
            filearea="picture",
            itemid=snapshot.log_id,
        )

    if result is None:
        logger.warning("File record not found for snapshot %d (user=%d)", snapshot.log_id, snapshot.user_id)
        return None

    contenthash, original_filename = result
    src_path = contenthash_to_path(moodledata, contenthash)

    if not src_path.exists():
        logger.warning("Physical file missing: %s (hash=%s)", src_path, contenthash)
        return None

    # Build student identifier (prefer username, fall back to user_id)
    student_id = _student_id(snapshot)
    student_dir = output_dir / "snapshots" / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    # Build a meaningful filename
    ts_str = datetime.fromtimestamp(snapshot.timestamp, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest_filename = f"snap_{snapshot.log_id}_{ts_str}.png"
    dest_path = student_dir / dest_filename

    shutil.copy2(str(src_path), str(dest_path))

    return f"snapshots/{student_id}/{dest_filename}"


# ---------------------------------------------------------------------------
# Image extraction — download mode
# ---------------------------------------------------------------------------

def extract_image_download(
    snapshot: MoodleSnapshot,
    output_dir: Path,
    moodle_url: str,
    token: str,
    session: Optional["requests.Session"] = None,
) -> Optional[str]:
    """
    Download the proctoring image via pluginfile URL using a web-service token.

    Returns the relative image path or None on failure.
    """
    if not snapshot.webcampicture:
        return None

    if not HAS_REQUESTS:
        sys.exit("ERROR: requests library required for download mode. Run: pip install requests")

    if session is None:
        session = requests.Session()

    # Append token to the pluginfile URL (skip if no token — some servers serve publicly)
    url = snapshot.webcampicture
    if token:
        separator = "&" if "?" in url else "?"
        download_url = f"{url}{separator}token={token}"
    else:
        download_url = url

    student_id = _student_id(snapshot)
    student_dir = output_dir / "snapshots" / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    ts_str = datetime.fromtimestamp(snapshot.timestamp, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest_filename = f"snap_{snapshot.log_id}_{ts_str}.png"
    dest_path = student_dir / dest_filename

    try:
        resp = session.get(download_url, timeout=30, stream=True)
        resp.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"snapshots/{student_id}/{dest_filename}"

    except Exception as e:
        logger.warning("Failed to download snapshot %d: %s", snapshot.log_id, e)
        return None


# ---------------------------------------------------------------------------
# Reference face extraction
# ---------------------------------------------------------------------------

def extract_reference_face_filesystem(
    conn, prefix: str, moodledata: Path, ref: MoodleRefImage, output_dir: Path
) -> bool:
    """Copy the admin-uploaded reference face image to reference_faces/<student_id>/."""
    # Reference images are stored with component='quizaccess_proctoring', filearea='user_photo'
    # System context, itemid = user_id
    result = resolve_file_path_from_db(
        conn, prefix,
        component="quizaccess_proctoring",
        filearea="user_photo",
        itemid=ref.user_id,
    )

    if result is None:
        logger.debug("No reference file record for user %d", ref.user_id)
        return False

    contenthash, original_filename = result
    src_path = contenthash_to_path(moodledata, contenthash)

    if not src_path.exists():
        logger.warning("Reference image file missing: %s", src_path)
        return False

    student_id = ref.username if ref.username else f"user_{ref.user_id}"
    ref_dir = output_dir / "reference_faces" / student_id
    ref_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(original_filename).suffix or ".png"
    dest_path = ref_dir / f"reference{ext}"
    shutil.copy2(str(src_path), str(dest_path))
    return True


def extract_reference_face_download(
    conn, prefix: str, ref: MoodleRefImage, output_dir: Path,
    moodle_url: str, token: str, session: Optional["requests.Session"] = None,
) -> bool:
    """Download the reference face via Moodle's pluginfile URL."""
    if not HAS_REQUESTS:
        return False

    if session is None:
        session = requests.Session()

    # Build the pluginfile URL for reference images
    # user_photo area, system context, itemid = user_id
    sys_ctx = 1  # System context is typically ID 1
    url = f"{moodle_url}/pluginfile.php/{sys_ctx}/quizaccess_proctoring/user_photo/{ref.user_id}/"

    # We need to find the filename — query from DB
    result = resolve_file_path_from_db(
        conn, prefix,
        component="quizaccess_proctoring",
        filearea="user_photo",
        itemid=ref.user_id,
    )

    if result is None:
        return False

    _, filename = result
    if token:
        download_url = f"{url}{filename}?token={token}"
    else:
        download_url = f"{url}{filename}"

    student_id = ref.username if ref.username else f"user_{ref.user_id}"
    ref_dir = output_dir / "reference_faces" / student_id
    ref_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = session.get(download_url, timeout=30)
        resp.raise_for_status()

        ext = Path(filename).suffix or ".png"
        dest_path = ref_dir / f"reference{ext}"
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        logger.warning("Failed to download reference for user %d: %s", ref.user_id, e)
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _student_id(snap: MoodleSnapshot) -> str:
    """Build a student identifier from snapshot data."""
    if snap.username:
        return snap.username
    return f"user_{snap.user_id}"


def _attempt_id(snap: MoodleSnapshot) -> str:
    """Build an attempt identifier."""
    return f"quiz{snap.quiz_id}_attempt{snap.attempt_status}"


def _fix_attempt_ids(records: list[tuple["MoodleSnapshot", str]]) -> None:
    """Fix attempt IDs where the Moodle plugin stored timestamps instead of real attempt IDs.

    Detects this by checking if status == timemodified (a known plugin bug in
    some versions).  For those snapshots, groups all frames from the same user +
    quiz that are within ``SESSION_GAP`` seconds and assigns a synthetic attempt
    ID based on the session start time.
    """
    SESSION_GAP = 300  # 5-minute gap between sessions = new attempt

    # Separate snapshots into "good" (real attempt id) and "bad" (timestamp as status)
    bad_keys: set[tuple[int, int]] = set()  # (user_id, quiz_id)
    for snap, _ in records:
        if snap.attempt_status == snap.timestamp:
            bad_keys.add((snap.user_id, snap.quiz_id))

    if not bad_keys:
        return  # all attempt IDs look correct

    # Group bad snapshots by (user_id, quiz_id), sorted by time
    from collections import defaultdict
    groups: dict[tuple[int, int], list[MoodleSnapshot]] = defaultdict(list)
    for snap, _ in records:
        key = (snap.user_id, snap.quiz_id)
        if key in bad_keys:
            groups[key].append(snap)

    for key, snaps in groups.items():
        snaps.sort(key=lambda s: s.timestamp)
        session_start = snaps[0].timestamp
        attempt_num = 1
        prev_ts = snaps[0].timestamp

        for snap in snaps:
            if snap.timestamp - prev_ts > SESSION_GAP:
                attempt_num += 1
                session_start = snap.timestamp
            snap.attempt_status = int(f"{key[1]}{attempt_num:04d}")  # synthetic ID
            prev_ts = snap.timestamp

    fixed = sum(len(v) for v in groups.values())
    logger.info("Fixed %d snapshot attempt IDs across %d user/quiz groups",
                fixed, len(groups))


def write_metadata_csv(
    records: list[tuple[MoodleSnapshot, str]],  # (snapshot, relative_image_path)
    output_path: Path,
):
    """Write the pipeline-compatible metadata CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "student_id", "attempt_id", "timestamp", "course_id"])

        for snap, rel_path in records:
            # Format timestamp as ISO string
            ts = datetime.fromtimestamp(snap.timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            student_id = _student_id(snap)
            attempt_id = _attempt_id(snap)

            writer.writerow([rel_path, student_id, attempt_id, ts, str(snap.course_id)])

    logger.info("Wrote metadata CSV with %d rows to %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Main extraction workflow
# ---------------------------------------------------------------------------

def run_extraction(args: argparse.Namespace) -> int:
    stats = ExportStats()

    # Connect to Moodle DB
    logger.info("Connecting to Moodle database (%s @ %s:%d / %s)",
                args.db_engine, args.db_host, args.db_port, args.db_name)
    conn, prefix = get_db_connection(args)
    logger.info("Connected. Table prefix: '%s'", prefix)

    # Query snapshots
    logger.info("Querying proctoring snapshots (quiz_id=%s, course_id=%s)",
                args.quiz_id, args.course_id)
    snapshots = query_snapshots(conn, prefix, args.quiz_id, args.course_id)
    stats.snapshots_found = len(snapshots)
    logger.info("Found %d snapshot records", len(snapshots))

    if not snapshots:
        logger.warning("No snapshots found. Check --quiz-id and --course-id filters.")
        conn.close()
        return 0

    # Collect unique user IDs
    user_ids = {s.user_id for s in snapshots}
    stats.students = len(user_ids)
    logger.info("Found snapshots for %d unique students", len(user_ids))

    # Query reference images
    ref_images = query_reference_images(conn, prefix, user_ids)
    stats.ref_images_found = len(ref_images)
    logger.info("Found %d reference face images", len(ref_images))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    method = args.method

    # Extract snapshots
    logger.info("Extracting snapshots using method '%s'...", method)
    session = None
    if method == "download":
        if not HAS_REQUESTS:
            sys.exit("ERROR: requests library required. Run: pip install requests")
        session = requests.Session()

    exported_records: list[tuple[MoodleSnapshot, str]] = []

    for i, snap in enumerate(snapshots):
        if (i + 1) % 100 == 0:
            logger.info("  Progress: %d / %d snapshots", i + 1, len(snapshots))

        rel_path = None
        if method == "filesystem":
            moodledata = Path(args.moodledata)
            if not moodledata.exists():
                sys.exit(f"ERROR: moodledata directory not found: {moodledata}")
            context_cache: dict[int, Optional[int]] = {}
            rel_path = extract_image_filesystem(conn, prefix, moodledata, snap, output_dir, context_cache)
        elif method == "download":
            rel_path = extract_image_download(snap, output_dir, args.moodle_url, args.token, session)

        if rel_path:
            exported_records.append((snap, rel_path))
            stats.snapshots_exported += 1
        else:
            stats.snapshots_failed += 1

    # Extract reference faces
    logger.info("Extracting reference face images...")
    for ref in ref_images:
        ok = False
        if method == "filesystem":
            moodledata = Path(args.moodledata)
            ok = extract_reference_face_filesystem(conn, prefix, moodledata, ref, output_dir)
        elif method == "download":
            ok = extract_reference_face_download(conn, prefix, ref, output_dir,
                                                  args.moodle_url, args.token, session)
        if ok:
            stats.ref_images_exported += 1

    # Fix attempt IDs where the plugin stored timestamps instead of real attempt IDs
    _fix_attempt_ids(exported_records)

    # Write metadata CSV
    metadata_path = output_dir / "metadata.csv"
    write_metadata_csv(exported_records, metadata_path)

    conn.close()

    # Print summary
    print("\n" + "=" * 60)
    print("  Moodle Proctoring Export Summary")
    print("=" * 60)
    print(f"  Students:              {stats.students}")
    print(f"  Snapshots found:       {stats.snapshots_found}")
    print(f"  Snapshots exported:    {stats.snapshots_exported}")
    print(f"  Snapshots failed:      {stats.snapshots_failed}")
    print(f"  Reference images:      {stats.ref_images_found}")
    print(f"  Ref images exported:   {stats.ref_images_exported}")
    print(f"  Output directory:      {output_dir.resolve()}")
    print(f"  Metadata CSV:          {metadata_path.resolve()}")
    print("=" * 60)

    if stats.snapshots_exported > 0:
        print(f"\n  Next step — run the analysis pipeline:")
        print(f"    python analyze_exam_snapshots.py \\")
        print(f"      --metadata {metadata_path} \\")
        print(f"      --output {output_dir / 'results'} \\")
        print(f"      --config config.yaml \\")
        if stats.ref_images_exported > 0:
            print(f"      --reference-faces {output_dir / 'reference_faces'}")
        else:
            print(f"      --reference-faces {output_dir / 'reference_faces'}  # (no ref images found)")
        print()

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="moodle_extractor",
        description="Extract proctoring webcam snapshots from Moodle's quizaccess_proctoring plugin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filesystem mode (on Moodle server or with NFS mount):
  python moodle_extractor.py \\
      --db-host localhost --db-name moodle --db-user moodle --db-pass secret \\
      --moodledata /var/www/moodledata \\
      --output ./moodle_export --quiz-id 42

  # Download mode (remote with web-service token):
  python moodle_extractor.py \\
      --db-host db.example.com --db-name moodle --db-user reader --db-pass s3cret \\
      --method download --moodle-url https://moodle.example.com --token abc123 \\
      --output ./moodle_export --quiz-id 42
""",
    )

    # Database connection
    db = p.add_argument_group("Database connection")
    db.add_argument("--db-engine", default="mysql", choices=["mysql", "pgsql", "sqlite"],
                    help="Database engine (default: mysql). Use sqlite for local testing.")
    db.add_argument("--db-host", default="localhost", help="Database host.")
    db.add_argument("--db-port", type=int, default=None,
                    help="Database port (default: 3306 for mysql, 5432 for pgsql).")
    db.add_argument("--db-name", required=True,
                    help="Database name (or path to .db file for sqlite).")
    db.add_argument("--db-user", default="", help="Database user (not needed for sqlite).")
    db.add_argument("--db-pass", default="", help="Database password (not needed for sqlite).")
    db.add_argument("--db-prefix", default="mdl_", help="Moodle table prefix (default: mdl_).")

    # Extraction method
    ext = p.add_argument_group("Extraction method")
    ext.add_argument("--method", default="filesystem", choices=["filesystem", "download"],
                     help="How to extract images (default: filesystem).")
    ext.add_argument("--moodledata", default=None,
                     help="Path to moodledata directory (required for filesystem method).")
    ext.add_argument("--moodle-url", default=None,
                     help="Moodle base URL (required for download method).")
    ext.add_argument("--token", default=None,
                     help="Moodle web-service token (required for download method).")

    # Filters
    flt = p.add_argument_group("Filters")
    flt.add_argument("--quiz-id", type=int, default=None,
                     help="Only export snapshots for this quiz (course_modules.id / cmid).")
    flt.add_argument("--course-id", type=int, default=None,
                     help="Only export snapshots for this course.")

    # Output
    p.add_argument("--output", default="./moodle_export",
                   help="Output directory for exported data.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug logging.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Set default ports
    if args.db_port is None:
        args.db_port = 3306 if args.db_engine == "mysql" else 5432

    # Validate method-specific requirements
    if args.method == "filesystem" and not args.moodledata:
        parser.error("--moodledata is required for filesystem method")
    if args.method == "download":
        if not args.moodle_url:
            parser.error("--moodle-url is required for download method")
        # token is optional — some Moodle instances serve images publicly

    if not args.quiz_id and not args.course_id:
        parser.error("At least one of --quiz-id or --course-id is required")

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sys.exit(run_extraction(args))


if __name__ == "__main__":
    main()
