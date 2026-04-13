#!/usr/bin/env python3
"""
Rebuild a Moodle proctoring export from a PostgreSQL custom dump plus filedir.tar.gz.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import urlparse

from moodle_extractor import MoodleSnapshot, _attempt_id, _fix_attempt_ids, _student_id

logger = logging.getLogger(__name__)


def _pg_unescape(value: str | None) -> Optional[str]:
    if value is None or value == r"\N":
        return None
    return (
        value.replace(r"\\", "\\")
        .replace(r"\t", "\t")
        .replace(r"\n", "\n")
        .replace(r"\r", "\r")
    )


def iter_pg_restore_copy_rows(dump_path: Path, table_name: str) -> Iterator[dict[str, Optional[str]]]:
    cmd = ["pg_restore", "-a", "-t", table_name, "-f", "-", str(dump_path)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    in_copy = False
    columns: list[str] = []

    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not in_copy:
                prefix = f"COPY public.{table_name} ("
                if not line.startswith(prefix):
                    continue
                in_copy = True
                columns_part = line[len(prefix) : line.index(") FROM stdin;")]
                columns = [col.strip() for col in columns_part.split(",")]
                continue

            if line == r"\.":
                break

            values = [_pg_unescape(part) for part in line.split("\t")]
            if len(values) < len(columns):
                values.extend([None] * (len(columns) - len(values)))
            yield dict(zip(columns, values))
    finally:
        stderr = proc.stderr.read()
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"pg_restore failed for table {table_name}: exit={return_code} stderr={stderr.strip()}"
            )


def load_table(dump_path: Path, table_name: str) -> list[dict[str, Optional[str]]]:
    rows = list(iter_pg_restore_copy_rows(dump_path, table_name))
    logger.info("Loaded %d rows from %s", len(rows), table_name)
    return rows


def _to_int(value: Optional[str], default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _filename_from_url(url: str) -> str:
    if not url:
        return ""
    path = urlparse(url).path
    return path.rsplit("/", 1)[-1]


def _safe_question_mapping(
    slot_rows: Iterable[dict[str, Optional[str]]],
    question_rows: dict[int, dict[str, Optional[str]]],
    slot_question_rows: dict[int, dict[str, Optional[str]]],
) -> tuple[str, str, str, str]:
    question_ids: list[str] = []
    question_slots: list[str] = []
    question_names: list[str] = []
    question_labels: list[str] = []

    seen_ids: set[str] = set()
    seen_slots: set[str] = set()
    seen_names: set[str] = set()
    seen_labels: set[str] = set()

    for slot in sorted(slot_rows, key=lambda row: _to_int(row.get("slot"))):
        slot_id = _to_int(slot.get("id"))
        slot_num = str(slot.get("slot") or "").strip()
        question_id = _to_int(slot.get("questionid"))

        question_name = ""
        if question_id and question_id in question_rows:
            question_name = str(question_rows[question_id].get("name") or "").strip()
        elif slot_id and slot_id in slot_question_rows:
            question_id = _to_int(slot_question_rows[slot_id].get("question_id"))
            question_name = str(slot_question_rows[slot_id].get("question_name") or "").strip()

        question_id_str = str(question_id) if question_id else ""
        question_label = ""
        if slot_num and question_name:
            question_label = f"Q{slot_num}: {question_name}"
        elif slot_num:
            question_label = f"Q{slot_num}"
        elif question_name:
            question_label = question_name

        if question_id_str and question_id_str not in seen_ids:
            question_ids.append(question_id_str)
            seen_ids.add(question_id_str)
        if slot_num and slot_num not in seen_slots:
            question_slots.append(slot_num)
            seen_slots.add(slot_num)
        if question_name and question_name not in seen_names:
            question_names.append(question_name)
            seen_names.add(question_name)
        if question_label and question_label not in seen_labels:
            question_labels.append(question_label)
            seen_labels.add(question_label)

    return (
        ",".join(question_ids),
        ",".join(question_slots),
        " | ".join(question_labels),
        " | ".join(question_names),
    )


def _match_attempt_id(
    log_row: dict[str, Optional[str]],
    attempts_by_user_quiz: dict[tuple[int, int], list[dict[str, Optional[str]]]],
    resolved_quiz_id: int,
) -> int:
    user_id = _to_int(log_row.get("userid"))
    ts = _to_int(log_row.get("timemodified"))
    candidates = attempts_by_user_quiz.get((user_id, resolved_quiz_id), [])
    for attempt in candidates:
        start = _to_int(attempt.get("timestart"))
        finish = _to_int(attempt.get("timefinish"))
        if ts < start:
            continue
        if finish and ts > finish + 300:
            continue
        return _to_int(attempt.get("id"))
    status = _to_int(log_row.get("status"))
    return status or _to_int(log_row.get("id"))


@dataclass
class ExportedFrame:
    snapshot: MoodleSnapshot
    rel_path: str
    quiz_id: str
    quiz_name: str
    quiz_page: str
    question_id: str
    question_slot: str
    question_label: str
    question_name: str
    source_log_id: str


def build_export(args: argparse.Namespace) -> int:
    dump_path = Path(args.dump).resolve()
    filedir_tar_path = Path(args.filedir_tar).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_tables = {
        "mdl_quizaccess_proctoring_logs": load_table(dump_path, "mdl_quizaccess_proctoring_logs"),
        "mdl_files": load_table(dump_path, "mdl_files"),
        "mdl_user": load_table(dump_path, "mdl_user"),
        "mdl_quiz": load_table(dump_path, "mdl_quiz"),
        "mdl_quiz_attempts": load_table(dump_path, "mdl_quiz_attempts"),
        "mdl_quiz_slots": load_table(dump_path, "mdl_quiz_slots"),
        "mdl_question": load_table(dump_path, "mdl_question"),
        "mdl_question_references": load_table(dump_path, "mdl_question_references"),
        "mdl_question_versions": load_table(dump_path, "mdl_question_versions"),
        "mdl_quizaccess_proctoring_user_images": load_table(
            dump_path,
            "mdl_quizaccess_proctoring_user_images",
        ),
    }
    course_module_rows: list[dict[str, Optional[str]]] = []
    try:
        course_module_rows = load_table(dump_path, "mdl_course_modules")
    except RuntimeError as exc:
        logger.warning("Could not load mdl_course_modules; continuing without cm mapping: %s", exc)

    users_by_id = {_to_int(row.get("id")): row for row in required_tables["mdl_user"]}
    quizzes_by_id = {_to_int(row.get("id")): row for row in required_tables["mdl_quiz"]}
    course_modules_by_id = {_to_int(row.get("id")): row for row in course_module_rows}
    questions_by_id = {_to_int(row.get("id")): row for row in required_tables["mdl_question"]}

    attempts_by_user_quiz: dict[tuple[int, int], list[dict[str, Optional[str]]]] = defaultdict(list)
    for row in required_tables["mdl_quiz_attempts"]:
        attempts_by_user_quiz[(_to_int(row.get("userid")), _to_int(row.get("quiz")))].append(row)
    for rows in attempts_by_user_quiz.values():
        rows.sort(key=lambda item: (_to_int(item.get("timestart")), _to_int(item.get("id"))))

    files_by_filename: dict[str, dict[str, Optional[str]]] = {}
    user_photo_files_by_itemid: dict[int, dict[str, Optional[str]]] = {}
    for row in required_tables["mdl_files"]:
        if (row.get("component") or "") != "quizaccess_proctoring":
            continue
        filearea = row.get("filearea") or ""
        filename = row.get("filename") or ""
        if filename in ("", "."):
            continue
        if filearea == "picture":
            files_by_filename[filename] = row
        elif filearea == "user_photo":
            itemid = _to_int(row.get("itemid"))
            if itemid and itemid not in user_photo_files_by_itemid:
                user_photo_files_by_itemid[itemid] = row

    slot_rows_by_quiz_page: dict[tuple[int, int], list[dict[str, Optional[str]]]] = defaultdict(list)
    for row in required_tables["mdl_quiz_slots"]:
        quizid = _to_int(row.get("quizid"))
        page = _to_int(row.get("page"), -1)
        slot_rows_by_quiz_page[(quizid, page)].append(row)

    question_versions_by_entry: dict[int, list[dict[str, Optional[str]]]] = defaultdict(list)
    for row in required_tables["mdl_question_versions"]:
        question_versions_by_entry[_to_int(row.get("questionbankentryid"))].append(row)
    for rows in question_versions_by_entry.values():
        rows.sort(key=lambda item: _to_int(item.get("version")), reverse=True)

    slot_question_rows: dict[int, dict[str, Optional[str]]] = {}
    for row in required_tables["mdl_question_references"]:
        if (row.get("component") or "") != "mod_quiz":
            continue
        if (row.get("questionarea") or "") != "slot":
            continue

        slot_id = _to_int(row.get("itemid"))
        entry_id = _to_int(row.get("questionbankentryid"))
        target_version = _to_int(row.get("version"), -1)
        candidates = question_versions_by_entry.get(entry_id, [])

        chosen: Optional[dict[str, Optional[str]]] = None
        for candidate in candidates:
            if (candidate.get("status") or "").lower() == "draft":
                continue
            if target_version != -1 and _to_int(candidate.get("version"), -1) != target_version:
                continue
            chosen = candidate
            break
        if chosen is None:
            for candidate in candidates:
                if (candidate.get("status") or "").lower() != "draft":
                    chosen = candidate
                    break

        if chosen is None:
            continue

        question_id = _to_int(chosen.get("questionid"))
        if not question_id:
            continue

        question_name = str(questions_by_id.get(question_id, {}).get("name") or "").strip()
        slot_question_rows[slot_id] = {
            "question_id": str(question_id),
            "question_name": question_name,
        }

    selected_logs: list[dict[str, Optional[str]]] = []
    for row in required_tables["mdl_quizaccess_proctoring_logs"]:
        if not (row.get("webcampicture") or "").strip():
            continue
        if _to_int(row.get("deletionprogress")) != 0:
            continue
        if args.course_id is not None and _to_int(row.get("courseid")) != args.course_id:
            continue
        if args.quiz_id is not None and _to_int(row.get("quizid")) != args.quiz_id:
            continue
        selected_logs.append(row)
    selected_logs.sort(key=lambda item: _to_int(item.get("id")))

    exported_frames: list[ExportedFrame] = []
    exported_snapshot_user_ids: set[int] = set()
    missing_files = 0

    with tarfile.open(filedir_tar_path, "r:gz") as tar:
        for row in selected_logs:
            user_id = _to_int(row.get("userid"))
            quiz_id = _to_int(row.get("quizid"))
            resolved_quiz_id = quiz_id
            resolved_quiz = quizzes_by_id.get(quiz_id)
            if resolved_quiz is None and quiz_id in course_modules_by_id:
                resolved_quiz_id = _to_int(course_modules_by_id[quiz_id].get("instance"))
                resolved_quiz = quizzes_by_id.get(resolved_quiz_id)

            username = str(users_by_id.get(user_id, {}).get("username") or f"user_{user_id}")
            attempt_key = _match_attempt_id(row, attempts_by_user_quiz, resolved_quiz_id)
            timestamp_epoch = _to_int(row.get("timemodified"))
            timestamp_iso = datetime.fromtimestamp(
                timestamp_epoch,
                tz=timezone.utc,
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

            webcampicture = str(row.get("webcampicture") or "")
            filename = _filename_from_url(webcampicture) or f"log_{_to_int(row.get('id'))}.png"
            file_row = files_by_filename.get(filename)
            if file_row is None:
                missing_files += 1
                continue

            contenthash = str(file_row.get("contenthash") or "")
            if not contenthash:
                missing_files += 1
                continue

            member_name = f"filedir/{contenthash[:2]}/{contenthash[2:4]}/{contenthash}"
            try:
                member = tar.getmember(member_name)
            except KeyError:
                logger.warning("Missing filedir member for log_id=%s hash=%s", row.get("id"), contenthash)
                missing_files += 1
                continue

            snapshot = MoodleSnapshot(
                log_id=_to_int(row.get("id")),
                course_id=_to_int(row.get("courseid")),
                quiz_id=resolved_quiz_id,
                user_id=user_id,
                webcampicture=webcampicture,
                attempt_status=attempt_key,
                timestamp=timestamp_epoch,
                username=username,
                firstname=str(users_by_id.get(user_id, {}).get("firstname") or ""),
                lastname=str(users_by_id.get(user_id, {}).get("lastname") or ""),
                email=str(users_by_id.get(user_id, {}).get("email") or ""),
            )

            student_id = _student_id(snapshot)
            attempt_id = _attempt_id(snapshot)
            dest_rel = Path("snapshots") / student_id / attempt_id / filename
            dest_path = output_dir / dest_rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if not dest_path.exists():
                with tar.extractfile(member) as src, dest_path.open("wb") as dst:
                    if src is None:
                        raise FileNotFoundError(f"Could not read {member_name} from tar archive")
                    shutil.copyfileobj(src, dst)

            quiz_page = str(row.get("quizpage") or "0")
            question_id, question_slot, question_label, question_name = _safe_question_mapping(
                slot_rows_by_quiz_page.get((resolved_quiz_id, _to_int(row.get("quizpage"))), []),
                questions_by_id,
                slot_question_rows,
            )
            if not question_label and quiz_page not in ("", "0", "-1"):
                question_label = f"Page {quiz_page}"

            exported_frames.append(
                ExportedFrame(
                    snapshot=snapshot,
                    rel_path=dest_rel.as_posix(),
                    quiz_id=str(resolved_quiz_id),
                    quiz_name=str((resolved_quiz or {}).get("name") or ""),
                    quiz_page=quiz_page,
                    question_id=question_id,
                    question_slot=question_slot,
                    question_label=question_label,
                    question_name=question_name,
                    source_log_id=str(row.get("id") or ""),
                )
            )
            exported_snapshot_user_ids.add(user_id)

        exported_user_rows = {
            _to_int(row.get("user_id")): row
            for row in required_tables["mdl_quizaccess_proctoring_user_images"]
        }
        for user_id in sorted(exported_snapshot_user_ids):
            file_row = user_photo_files_by_itemid.get(user_id)
            if file_row is None and user_id in exported_user_rows:
                file_row = user_photo_files_by_itemid.get(user_id)
            if file_row is None:
                continue

            contenthash = str(file_row.get("contenthash") or "")
            if not contenthash:
                continue

            member_name = f"filedir/{contenthash[:2]}/{contenthash[2:4]}/{contenthash}"
            try:
                member = tar.getmember(member_name)
            except KeyError:
                logger.warning("Missing reference face member for user_id=%s hash=%s", user_id, contenthash)
                continue

            username = str(users_by_id.get(user_id, {}).get("username") or f"user_{user_id}")
            suffix = Path(str(file_row.get("filename") or "")).suffix or ".png"
            dest_path = output_dir / "reference_faces" / username / f"reference{suffix}"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if not dest_path.exists():
                with tar.extractfile(member) as src, dest_path.open("wb") as dst:
                    if src is None:
                        raise FileNotFoundError(f"Could not read {member_name} from tar archive")
                    shutil.copyfileobj(src, dst)

    record_pairs = [(frame.snapshot, frame.rel_path) for frame in exported_frames]
    _fix_attempt_ids(record_pairs)

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "image_path",
                "student_id",
                "attempt_id",
                "timestamp",
                "course_id",
                "quiz_id",
                "quiz_name",
                "quiz_page",
                "question_id",
                "question_slot",
                "question_name",
                "question_label",
                "source_log_id",
            ]
        )
        for frame in exported_frames:
            snapshot = frame.snapshot
            writer.writerow(
                [
                    frame.rel_path,
                    _student_id(snapshot),
                    _attempt_id(snapshot),
                    datetime.fromtimestamp(snapshot.timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    str(snapshot.course_id),
                    frame.quiz_id,
                    frame.quiz_name,
                    frame.quiz_page,
                    frame.question_id,
                    frame.question_slot,
                    frame.question_name,
                    frame.question_label,
                    frame.source_log_id,
                ]
            )

    logger.info("Exported %d frames across %d students", len(exported_frames), len(exported_snapshot_user_ids))
    logger.info("Metadata written to %s", metadata_path)
    logger.info("Missing file records or tar members: %d", missing_files)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offline_moodle_dump_export",
        description="Rebuild a Moodle proctoring export from a PostgreSQL dump and filedir.tar.gz.",
    )
    parser.add_argument("--dump", required=True, help="Path to PostgreSQL custom dump (.dump).")
    parser.add_argument("--filedir-tar", required=True, help="Path to Moodle filedir tar.gz archive.")
    parser.add_argument("--output", required=True, help="Output directory for reconstructed export.")
    parser.add_argument("--course-id", type=int, default=None, help="Optional course filter.")
    parser.add_argument("--quiz-id", type=int, default=None, help="Optional quiz filter.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    return build_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
