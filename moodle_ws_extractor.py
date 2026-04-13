#!/usr/bin/env python3
"""
moodle_ws_extractor.py
======================
Lightweight extractor that uses **only Moodle Web Services** (no direct DB access).

This is useful when you can't access the Moodle database directly but have
a web-service token with appropriate permissions.

Requirements:
  - A Moodle web-service token (Site admin → Server → Web services → Manage tokens)
  - The token must belong to a user with 'quizaccess/proctoring:viewreport' capability
  - The 'quizaccess_proctoring_send_camshot' service must be enabled

Since the quizaccess_proctoring plugin doesn't expose a listing API, this script
uses Moodle's core web services to:
  1. Get enrolled users in the course
  2. Download images from the pluginfile URLs stored in proctoring logs

If you have DB access, use ``moodle_extractor.py`` instead — it's more reliable.

Alternative: Export snapshots from Moodle's proctoring report page manually and
use the provided CSV import mode.

Usage:
  # From a manually exported list of pluginfile URLs:
  python moodle_ws_extractor.py \\
      --url-list exported_urls.csv \\
      --moodle-url https://moodle.example.com \\
      --token abc123 \\
      --output ./moodle_export

  # The url-list CSV should have columns:
  #   url,student_id,attempt_id,timestamp,course_id
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlencode

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def download_image(url: str, token: str, dest_path: Path, session: requests.Session) -> bool:
    """Download an image from a Moodle pluginfile URL using a token."""
    separator = "&" if "?" in url else "?"
    download_url = f"{url}{separator}token={token}"

    try:
        resp = session.get(download_url, timeout=30, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            logger.warning("Got HTML instead of image for %s (auth issue?)", url)
            return False

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.warning("Download failed for %s: %s", url, e)
        return False


def call_moodle_ws(moodle_url: str, token: str, function: str, params: dict) -> dict:
    """Call a Moodle web service function."""
    ws_url = f"{moodle_url}/webservice/rest/server.php"
    data = {
        "wstoken": token,
        "wsfunction": function,
        "moodlewsrestformat": "json",
        **params,
    }
    resp = requests.post(ws_url, data=data, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    if isinstance(result, dict) and "exception" in result:
        logger.error("Moodle WS error: %s — %s", result.get("errorcode"), result.get("message"))
        return {}

    return result


def get_enrolled_users(moodle_url: str, token: str, course_id: int) -> list[dict]:
    """Get list of enrolled users in a course via Moodle WS."""
    result = call_moodle_ws(moodle_url, token, "core_enrol_get_enrolled_users", {
        "courseid": course_id,
    })
    if isinstance(result, list):
        return result
    return []


def run_url_list_import(args: argparse.Namespace) -> int:
    """Import from a CSV file containing pluginfile URLs."""
    if not HAS_REQUESTS:
        sys.exit("ERROR: requests library required. Run: pip install requests")

    url_list_path = Path(args.url_list)
    if not url_list_path.exists():
        sys.exit(f"ERROR: URL list file not found: {url_list_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    # Read URL list
    exported = []
    failed = 0

    with open(url_list_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"url", "student_id"}
        if reader.fieldnames is None:
            sys.exit("ERROR: URL list CSV is empty")

        cols = set(reader.fieldnames)
        missing = required - cols
        if missing:
            sys.exit(f"ERROR: URL list CSV missing columns: {missing}. "
                     f"Expected: url,student_id,attempt_id,timestamp,course_id")

        rows = list(reader)

    logger.info("Processing %d URLs from %s", len(rows), url_list_path)

    metadata_rows = []

    for i, row in enumerate(rows):
        url = row.get("url", "").strip()
        student_id = row.get("student_id", "").strip()
        attempt_id = row.get("attempt_id", "").strip() or "attempt_1"
        timestamp = row.get("timestamp", "").strip()
        course_id = row.get("course_id", "").strip() or "0"

        if not url or not student_id:
            logger.warning("Skipping row %d: missing url or student_id", i + 2)
            continue

        # Create student directory
        student_dir = output_dir / "snapshots" / student_id
        student_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename
        dest_filename = f"snap_{i:04d}.png"
        dest_path = student_dir / dest_filename

        if download_image(url, args.token, dest_path, session):
            rel_path = f"snapshots/{student_id}/{dest_filename}"
            metadata_rows.append({
                "image_path": rel_path,
                "student_id": student_id,
                "attempt_id": attempt_id,
                "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "course_id": course_id,
            })
            exported.append(rel_path)
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            logger.info("  Progress: %d / %d", i + 1, len(rows))

    # Write metadata CSV
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "student_id", "attempt_id", "timestamp", "course_id"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\n  Exported: {len(exported)} snapshots, Failed: {failed}")
    print(f"  Metadata: {metadata_path.resolve()}")
    print(f"\n  Run analysis:")
    print(f"    python analyze_exam_snapshots.py --metadata {metadata_path} --output {output_dir / 'results'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="moodle_ws_extractor",
        description="Download proctoring snapshots from Moodle using web service tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--url-list", required=True,
                   help="CSV file with columns: url,student_id[,attempt_id,timestamp,course_id]")
    p.add_argument("--moodle-url", required=True,
                   help="Moodle base URL (e.g., https://moodle.example.com)")
    p.add_argument("--token", required=True,
                   help="Moodle web-service token.")
    p.add_argument("--output", default="./moodle_export",
                   help="Output directory.")
    p.add_argument("-v", "--verbose", action="store_true")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sys.exit(run_url_list_import(args))


if __name__ == "__main__":
    main()
