from __future__ import annotations

from data_io.live_moodle_source import LiveMoodleSource


def _source() -> LiveMoodleSource:
    source = LiveMoodleSource.__new__(LiveMoodleSource)
    source.prefix = "mdl_"
    return source


def test_live_source_filters_directly_by_log_quizid():
    where_sql = LiveMoodleSource._where_sql(course_id=1782, quiz_id=88195)

    assert "l.courseid = %(course_id)s" in where_sql
    assert "l.quizid = %(quiz_id)s" in where_sql
    assert "qa." not in where_sql
    assert "attempt" not in where_sql.lower()


def test_live_source_sql_selects_log_quizid_as_exam_id():
    sql = _source()._sql_without_question_metadata(course_id=1782, quiz_id=88195)

    assert "l.quizid" in sql
    assert "WHERE" in sql
    assert "l.quizid = %(quiz_id)s" in sql
