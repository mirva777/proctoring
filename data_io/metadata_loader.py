"""
Metadata CSV loader with validation.

Expected columns: image_path, student_id, attempt_id, timestamp, course_id
Optional columns: quiz_id, quiz_name, quiz_page, question_id, question_slot,
                  question_name, question_label, source_log_id
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"image_path", "student_id", "attempt_id", "timestamp", "course_id"}


@dataclass
class MetadataRow:
    image_path: str
    student_id: str
    attempt_id: str
    timestamp: str
    course_id: str
    quiz_id: str = ""
    quiz_name: str = ""
    quiz_page: str = ""
    question_id: str = ""
    question_slot: str = ""
    question_name: str = ""
    question_label: str = ""
    source_log_id: str = ""


def load_metadata(csv_path: str | Path) -> list[MetadataRow]:
    """
    Load and validate a metadata CSV.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if required columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    rows: list[MetadataRow] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("Metadata CSV is empty")
        cols = set(reader.fieldnames)
        missing = REQUIRED_COLUMNS - cols
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {missing}")

        for i, row in enumerate(reader):
            ip = row.get("image_path", "").strip()
            sid = row.get("student_id", "").strip()
            aid = row.get("attempt_id", "").strip()
            ts = row.get("timestamp", "").strip()
            cid = row.get("course_id", "").strip()
            qid = row.get("quiz_id", "").strip()
            qname = row.get("quiz_name", "").strip()
            qpage = row.get("quiz_page", "").strip()
            question_id = row.get("question_id", "").strip()
            question_slot = row.get("question_slot", "").strip()
            question_name = row.get("question_name", "").strip()
            question_label = row.get("question_label", "").strip()
            source_log_id = row.get("source_log_id", "").strip()

            if not ip or not sid or not aid:
                logger.warning("Skipping metadata row %d: missing required field", i + 2)
                continue

            rows.append(MetadataRow(
                image_path=ip,
                student_id=sid,
                attempt_id=aid,
                timestamp=ts,
                course_id=cid,
                quiz_id=qid,
                quiz_name=qname,
                quiz_page=qpage,
                question_id=question_id,
                question_slot=question_slot,
                question_name=question_name,
                question_label=question_label,
                source_log_id=source_log_id,
            ))

    logger.info("Loaded %d metadata rows from %s", len(rows), path)
    return rows
