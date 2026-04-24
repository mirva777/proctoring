"""
Realtime Moodle proctoring source for PostgreSQL-backed quizaccess_proctoring.

The poller reads new rows from ``mdl_quizaccess_proctoring_logs``, enriches them
with user/quiz/question context, and exports image bytes from Moodle's
content-addressed file store when ``moodledata`` is available locally.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

from data_io.ssh_moodle_bridge import ManagedSSHBridge

logger = logging.getLogger(__name__)

try:
    import psycopg2  # type: ignore
except ImportError as exc:  # pragma: no cover
    psycopg2 = None  # type: ignore
    _PSYCOPG_IMPORT_ERROR = exc
else:
    _PSYCOPG_IMPORT_ERROR = None


@dataclass(frozen=True)
class MoodleDBConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    table_prefix: str = "mdl_"


@dataclass
class MoodleLiveSnapshot:
    source_log_id: int
    course_id: str
    quiz_id: str
    quiz_name: str
    student_user_id: int
    student_id: str
    student_name: str
    attempt_id: str
    quiz_page: str
    question_id: str
    question_slot: str
    question_name: str
    question_label: str
    timestamp_epoch: int
    timestamp_iso: str
    webcampicture: str
    filename: str
    contenthash: str
    filesize: int


def db_config_from_env(prefix: str = "MOODLE_DB_", table_prefix: str = "mdl_") -> MoodleDBConfig:
    """Read Moodle DB connection settings from environment variables."""
    host = os.getenv(f"{prefix}HOST", "127.0.0.1")
    port = int(os.getenv(f"{prefix}PORT", "6432"))
    dbname = os.getenv(f"{prefix}NAME", "moodle")
    user = os.getenv(f"{prefix}USER", "moodle")
    password = os.getenv(f"{prefix}PASSWORD", "")
    return MoodleDBConfig(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        table_prefix=os.getenv(f"{prefix}PREFIX", table_prefix),
    )


class LiveMoodleSource:
    """Polls Moodle proctoring rows and materializes snapshots into local files."""

    def __init__(
        self,
        db_config: MoodleDBConfig,
        output_dir: str | Path,
        moodledata_dir: str | Path | None = None,
        ssh_bridge: ManagedSSHBridge | None = None,
    ) -> None:
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2-binary is required for live Moodle polling"
            ) from _PSYCOPG_IMPORT_ERROR

        self.db_config = db_config
        self.prefix = db_config.table_prefix
        self.output_dir = Path(output_dir)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.ssh_bridge = ssh_bridge
        self.remote_moodledata_dir: PurePosixPath | None = None
        self.moodledata_dir: Path | None = None
        if moodledata_dir:
            if ssh_bridge is not None:
                # Keep remote Linux paths POSIX on Windows; Path.resolve() would
                # turn /var/www/... into C:/var/www/... before SFTP.
                remote_path = str(moodledata_dir).replace("\\", "/")
                self.remote_moodledata_dir = PurePosixPath(remote_path)
            else:
                self.moodledata_dir = Path(moodledata_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def connect(self):
        if self.ssh_bridge is not None:
            self.ssh_bridge.ensure_ready()
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            dbname=self.db_config.dbname,
            user=self.db_config.user,
            password=self.db_config.password,
            connect_timeout=10,
        )

    def fetch_new_snapshots(
        self,
        last_source_log_id: int,
        course_id: int | None = None,
        quiz_id: int | None = None,
        limit: int = 200,
    ) -> list[MoodleLiveSnapshot]:
        """Fetch rows newer than ``last_source_log_id``."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        self._sql_with_question_metadata(course_id, quiz_id),
                        self._query_params(last_source_log_id, course_id, quiz_id, limit),
                    )
                except Exception as exc:
                    logger.warning(
                        "Legacy quiz_slots.questionid query failed (%s); retrying Moodle 4 question_references mapping.",
                        exc,
                    )
                    conn.rollback()
                    try:
                        cur.execute(
                            self._sql_with_question_references(course_id, quiz_id),
                            self._query_params(last_source_log_id, course_id, quiz_id, limit),
                        )
                    except Exception as exc_2:
                        logger.warning(
                            "question_references query failed (%s); retrying with page-only metadata.",
                            exc_2,
                        )
                        conn.rollback()
                        cur.execute(
                            self._sql_without_question_metadata(course_id, quiz_id),
                            self._query_params(last_source_log_id, course_id, quiz_id, limit),
                        )
                rows = cur.fetchall()

        return [self._row_to_snapshot(row) for row in rows]

    def fetch_snapshots_by_log_ids(self, source_log_ids: list[int]) -> list[MoodleLiveSnapshot]:
        """Fetch Moodle snapshot metadata for existing processed log IDs."""
        unique_ids = sorted({int(log_id) for log_id in source_log_ids if log_id is not None})
        if not unique_ids:
            return []

        params = {"source_log_ids": unique_ids}
        with self.connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        self._sql_with_question_metadata(None, None, by_source_log_ids=True),
                        params,
                    )
                except Exception as exc:
                    logger.warning(
                        "Legacy quiz_slots.questionid source backfill query failed (%s); retrying Moodle 4 question_references mapping.",
                        exc,
                    )
                    conn.rollback()
                    try:
                        cur.execute(
                            self._sql_with_question_references(None, None, by_source_log_ids=True),
                            params,
                        )
                    except Exception as exc_2:
                        logger.warning(
                            "question_references source backfill query failed (%s); retrying with page-only metadata.",
                            exc_2,
                        )
                        conn.rollback()
                        cur.execute(
                            self._sql_without_question_metadata(None, None, by_source_log_ids=True),
                            params,
                        )
                rows = cur.fetchall()

        return [self._row_to_snapshot(row) for row in rows]

    def materialize_snapshot(self, snapshot: MoodleLiveSnapshot) -> Optional[str]:
        """
        Copy one snapshot into ``<output_dir>/snapshots/<student>/<attempt>/...``.

        Returns a path relative to ``output_dir`` or None if the Moodle file is
        missing locally.
        """
        if (self.moodledata_dir is None and self.remote_moodledata_dir is None) or not snapshot.contenthash:
            logger.warning(
                "Skipping snapshot %s because --moodledata was not provided or contenthash is missing",
                snapshot.source_log_id,
            )
            return None

        src_path = self._contenthash_to_path(snapshot.contenthash)

        safe_name = snapshot.filename or f"log_{snapshot.source_log_id}.png"
        dest_rel = Path("snapshots") / snapshot.student_id / snapshot.attempt_id / safe_name
        dest_path = self.output_dir / dest_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_rel.as_posix()

        if isinstance(src_path, Path) and src_path.exists():
            shutil.copy2(src_path, dest_path)
            return dest_rel.as_posix()

        if self.ssh_bridge is not None:
            try:
                self.ssh_bridge.fetch_file(src_path.as_posix(), dest_path)
                return dest_rel.as_posix()
            except Exception as exc:
                logger.warning(
                    "Remote Moodle file fetch failed for log_id=%s src=%s err=%s",
                    snapshot.source_log_id,
                    src_path,
                    exc,
                )
                return None

        logger.warning("Moodle file missing for log_id=%s: %s", snapshot.source_log_id, src_path)
        return None

    def source_file_path_for_snapshot(self, snapshot: MoodleLiveSnapshot) -> str:
        """Return the original Moodle ``moodledata/filedir`` path for a snapshot."""
        if not snapshot.contenthash:
            return ""
        return self._contenthash_to_path(snapshot.contenthash).as_posix()

    @staticmethod
    def _query_params(
        last_source_log_id: int,
        course_id: int | None,
        quiz_id: int | None,
        limit: int,
    ) -> dict[str, int]:
        params = {
            "last_source_log_id": int(last_source_log_id),
            "limit": int(limit),
        }
        if course_id is not None:
            params["course_id"] = int(course_id)
        if quiz_id is not None:
            params["quiz_id"] = int(quiz_id)
        return params

    def _sql_with_question_metadata(
        self,
        course_id: int | None,
        quiz_id: int | None,
        *,
        by_source_log_ids: bool = False,
    ) -> str:
        where_sql = self._where_sql(course_id, quiz_id, by_source_log_ids=by_source_log_ids)
        limit_sql = "" if by_source_log_ids else "LIMIT %(limit)s"
        p = self.prefix
        return f"""
            SELECT
                l.id,
                l.courseid,
                l.quizid,
                COALESCE(qz_direct.name, qz_cm.name, '') AS quiz_name,
                l.userid,
                COALESCE(NULLIF(u.username, ''), 'user_' || l.userid::text) AS student_code,
                TRIM(COALESCE(u.firstname, '') || ' ' || COALESCE(u.lastname, '')) AS student_name,
                COALESCE(qa.id, NULLIF(l.status, 0), l.id) AS attempt_key,
                COALESCE(l.quizpage, 0) AS quiz_page,
                COALESCE(
                    STRING_AGG(DISTINCT q.id::text, ',' ORDER BY q.id::text),
                    ''
                ) AS question_id,
                COALESCE(
                    STRING_AGG(DISTINCT qs.slot::text, ',' ORDER BY qs.slot::text),
                    ''
                ) AS question_slot,
                COALESCE(
                    STRING_AGG(
                        DISTINCT CONCAT('Q', qs.slot::text, ': ', q.name),
                        ' | '
                        ORDER BY CONCAT('Q', qs.slot::text, ': ', q.name)
                    ),
                    ''
                ) AS question_label,
                COALESCE(
                    STRING_AGG(DISTINCT q.name, ' | ' ORDER BY q.name),
                    ''
                ) AS question_name,
                l.timemodified,
                COALESCE(l.webcampicture, '') AS webcampicture,
                SUBSTRING(COALESCE(l.webcampicture, '') FROM '[^/]+$') AS filename,
                COALESCE(f.contenthash, '') AS contenthash,
                COALESCE(f.filesize, 0) AS filesize
            FROM {p}quizaccess_proctoring_logs l
            INNER JOIN {p}user u
                ON u.id = l.userid
            LEFT JOIN {p}quiz qz_direct
                ON qz_direct.id = l.quizid
            LEFT JOIN {p}course_modules cm
                ON cm.id = l.quizid
               AND cm.course = l.courseid
            LEFT JOIN {p}quiz qz_cm
                ON qz_cm.id = cm.instance
            LEFT JOIN {p}quiz_attempts qa
                ON qa.quiz = COALESCE(qz_direct.id, qz_cm.id, l.quizid)
               AND qa.userid = l.userid
               AND l.timemodified >= qa.timestart
               AND (
                    qa.timefinish = 0
                    OR l.timemodified <= qa.timefinish + 300
               )
            LEFT JOIN {p}quiz_slots qs
                ON qs.quizid = COALESCE(qz_direct.id, qz_cm.id, l.quizid)
               AND qs.page = l.quizpage
            LEFT JOIN {p}question q
                ON q.id = qs.questionid
            LEFT JOIN {p}files f
                ON f.component = 'quizaccess_proctoring'
               AND f.filearea = 'picture'
               AND f.filename = SUBSTRING(COALESCE(l.webcampicture, '') FROM '[^/]+$')
            WHERE {where_sql}
            GROUP BY
                l.id, l.courseid, l.quizid, qz_direct.name, qz_cm.name, l.userid,
                u.username, u.firstname, u.lastname,
                qa.id, l.status, l.quizpage, l.timemodified,
                l.webcampicture, f.contenthash, f.filesize
            ORDER BY l.id ASC
            {limit_sql}
        """

    def _sql_without_question_metadata(
        self,
        course_id: int | None,
        quiz_id: int | None,
        *,
        by_source_log_ids: bool = False,
    ) -> str:
        where_sql = self._where_sql(course_id, quiz_id, by_source_log_ids=by_source_log_ids)
        limit_sql = "" if by_source_log_ids else "LIMIT %(limit)s"
        p = self.prefix
        return f"""
            SELECT
                l.id,
                l.courseid,
                l.quizid,
                COALESCE(qz_direct.name, qz_cm.name, '') AS quiz_name,
                l.userid,
                COALESCE(NULLIF(u.username, ''), 'user_' || l.userid::text) AS student_code,
                TRIM(COALESCE(u.firstname, '') || ' ' || COALESCE(u.lastname, '')) AS student_name,
                COALESCE(qa.id, NULLIF(l.status, 0), l.id) AS attempt_key,
                COALESCE(l.quizpage, 0) AS quiz_page,
                '' AS question_id,
                '' AS question_slot,
                '' AS question_label,
                '' AS question_name,
                l.timemodified,
                COALESCE(l.webcampicture, '') AS webcampicture,
                SUBSTRING(COALESCE(l.webcampicture, '') FROM '[^/]+$') AS filename,
                COALESCE(f.contenthash, '') AS contenthash,
                COALESCE(f.filesize, 0) AS filesize
            FROM {p}quizaccess_proctoring_logs l
            INNER JOIN {p}user u
                ON u.id = l.userid
            LEFT JOIN {p}quiz qz_direct
                ON qz_direct.id = l.quizid
            LEFT JOIN {p}course_modules cm
                ON cm.id = l.quizid
               AND cm.course = l.courseid
            LEFT JOIN {p}quiz qz_cm
                ON qz_cm.id = cm.instance
            LEFT JOIN {p}quiz_attempts qa
                ON qa.quiz = COALESCE(qz_direct.id, qz_cm.id, l.quizid)
               AND qa.userid = l.userid
               AND l.timemodified >= qa.timestart
               AND (
                    qa.timefinish = 0
                    OR l.timemodified <= qa.timefinish + 300
               )
            LEFT JOIN {p}files f
                ON f.component = 'quizaccess_proctoring'
               AND f.filearea = 'picture'
               AND f.filename = SUBSTRING(COALESCE(l.webcampicture, '') FROM '[^/]+$')
            WHERE {where_sql}
            ORDER BY l.id ASC
            {limit_sql}
        """

    def _sql_with_question_references(
        self,
        course_id: int | None,
        quiz_id: int | None,
        *,
        by_source_log_ids: bool = False,
    ) -> str:
        where_sql = self._where_sql(course_id, quiz_id, by_source_log_ids=by_source_log_ids)
        limit_sql = "" if by_source_log_ids else "LIMIT %(limit)s"
        p = self.prefix
        return f"""
            SELECT
                l.id,
                l.courseid,
                l.quizid,
                COALESCE(qz_direct.name, qz_cm.name, '') AS quiz_name,
                l.userid,
                COALESCE(NULLIF(u.username, ''), 'user_' || l.userid::text) AS student_code,
                TRIM(COALESCE(u.firstname, '') || ' ' || COALESCE(u.lastname, '')) AS student_name,
                COALESCE(qa.id, NULLIF(l.status, 0), l.id) AS attempt_key,
                COALESCE(l.quizpage, 0) AS quiz_page,
                COALESCE(
                    STRING_AGG(DISTINCT qmeta.question_id::text, ',' ORDER BY qmeta.question_id::text),
                    ''
                ) AS question_id,
                COALESCE(
                    STRING_AGG(DISTINCT qs.slot::text, ',' ORDER BY qs.slot::text),
                    ''
                ) AS question_slot,
                COALESCE(
                    STRING_AGG(
                        DISTINCT CONCAT('Q', qs.slot::text, ': ', qmeta.question_name),
                        ' | '
                        ORDER BY CONCAT('Q', qs.slot::text, ': ', qmeta.question_name)
                    ),
                    ''
                ) AS question_label,
                COALESCE(
                    STRING_AGG(DISTINCT qmeta.question_name, ' | ' ORDER BY qmeta.question_name),
                    ''
                ) AS question_name,
                l.timemodified,
                COALESCE(l.webcampicture, '') AS webcampicture,
                SUBSTRING(COALESCE(l.webcampicture, '') FROM '[^/]+$') AS filename,
                COALESCE(f.contenthash, '') AS contenthash,
                COALESCE(f.filesize, 0) AS filesize
            FROM {p}quizaccess_proctoring_logs l
            INNER JOIN {p}user u
                ON u.id = l.userid
            LEFT JOIN {p}quiz qz_direct
                ON qz_direct.id = l.quizid
            LEFT JOIN {p}course_modules cm
                ON cm.id = l.quizid
               AND cm.course = l.courseid
            LEFT JOIN {p}quiz qz_cm
                ON qz_cm.id = cm.instance
            LEFT JOIN {p}quiz_attempts qa
                ON qa.quiz = COALESCE(qz_direct.id, qz_cm.id, l.quizid)
               AND qa.userid = l.userid
               AND l.timemodified >= qa.timestart
               AND (
                    qa.timefinish = 0
                    OR l.timemodified <= qa.timefinish + 300
               )
            LEFT JOIN {p}quiz_slots qs
                ON qs.quizid = COALESCE(qz_direct.id, qz_cm.id, l.quizid)
               AND qs.page = l.quizpage
            LEFT JOIN LATERAL (
                SELECT
                    q.id AS question_id,
                    q.name AS question_name
                FROM {p}question_references qr
                INNER JOIN {p}question_versions qv
                    ON qv.questionbankentryid = qr.questionbankentryid
                   AND qv.status <> 'draft'
                   AND (
                        qr.version IS NULL
                        OR qv.version = qr.version
                   )
                INNER JOIN {p}question q
                    ON q.id = qv.questionid
                WHERE qr.component = 'mod_quiz'
                  AND qr.questionarea = 'slot'
                  AND qr.itemid = qs.id
                ORDER BY qv.version DESC
                LIMIT 1
            ) qmeta ON TRUE
            LEFT JOIN {p}files f
                ON f.component = 'quizaccess_proctoring'
               AND f.filearea = 'picture'
               AND f.filename = SUBSTRING(COALESCE(l.webcampicture, '') FROM '[^/]+$')
            WHERE {where_sql}
            GROUP BY
                l.id, l.courseid, l.quizid, qz_direct.name, qz_cm.name, l.userid,
                u.username, u.firstname, u.lastname,
                qa.id, l.status, l.quizpage, l.timemodified,
                l.webcampicture, f.contenthash, f.filesize
            ORDER BY l.id ASC
            {limit_sql}
        """

    @staticmethod
    def _where_sql(
        course_id: int | None,
        quiz_id: int | None,
        *,
        by_source_log_ids: bool = False,
    ) -> str:
        clauses = [
            "l.id = ANY(%(source_log_ids)s)" if by_source_log_ids else "l.id > %(last_source_log_id)s",
            "COALESCE(l.webcampicture, '') <> ''",
            "COALESCE(l.deletionprogress, 0) = 0",
        ]
        if course_id is not None:
            clauses.append("l.courseid = %(course_id)s")
        if quiz_id is not None:
            clauses.append("l.quizid = %(quiz_id)s")
        return " AND ".join(clauses)

    def _row_to_snapshot(self, row) -> MoodleLiveSnapshot:
        quiz_page = str(row[8] or 0)
        question_slot = str(row[10] or "")
        question_name = str(row[12] or "").strip()
        question_label = str(row[11] or "").strip()
        if not question_label:
            question_label = f"Page {quiz_page}" if quiz_page else ""

        timestamp_epoch = int(row[13] or 0)
        timestamp_iso = datetime.fromtimestamp(
            timestamp_epoch,
            tz=timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        attempt_raw = int(row[7] or row[0])
        attempt_id = f"quiz{int(row[2] or 0)}_attempt{attempt_raw}"

        return MoodleLiveSnapshot(
            source_log_id=int(row[0]),
            course_id=str(row[1] or ""),
            quiz_id=str(row[2] or ""),
            quiz_name=str(row[3] or ""),
            student_user_id=int(row[4] or 0),
            student_id=str(row[5] or f"user_{row[4]}"),
            student_name=str(row[6] or "").strip(),
            attempt_id=attempt_id,
            quiz_page=quiz_page,
            question_id=str(row[9] or ""),
            question_slot=question_slot,
            question_name=question_name,
            question_label=question_label,
            timestamp_epoch=timestamp_epoch,
            timestamp_iso=timestamp_iso,
            webcampicture=str(row[14] or ""),
            filename=str(row[15] or f"log_{row[0]}.png"),
            contenthash=str(row[16] or ""),
            filesize=int(row[17] or 0),
        )

    def _contenthash_to_path(self, contenthash: str) -> Path | PurePosixPath:
        base_path: Path | PurePosixPath
        if self.remote_moodledata_dir is not None:
            base_path = self.remote_moodledata_dir
        else:
            assert self.moodledata_dir is not None
            base_path = self.moodledata_dir
        return (
            base_path
            / "filedir"
            / contenthash[:2]
            / contenthash[2:4]
            / contenthash
        )
