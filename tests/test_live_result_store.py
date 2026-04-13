from __future__ import annotations

from data_io.live_result_store import LiveResultStore
from scoring.aggregator import FrameRecord


def _record(
    *,
    source_log_id: int = 101,
    student_id: str = "student_1",
    attempt_id: str = "quiz10_attempt99",
    risk_score: float = 50.0,
) -> FrameRecord:
    return FrameRecord(
        image_path=f"snapshots/{student_id}/{attempt_id}/frame.png",
        student_id=student_id,
        attempt_id=attempt_id,
        timestamp="2026-04-03T10:00:00Z",
        course_id="1782",
        quiz_id="88195",
        quiz_name="Midterm",
        quiz_page="3",
        question_id="45",
        question_slot="7",
        question_name="Binary Search",
        question_label="Q7: Binary Search",
        source_log_id=source_log_id,
        face_count=1,
        look_away_flag=True,
        severity="moderate",
        identity_mismatch=False,
        identity_similarity=0.98,
        phone_detected=False,
        extra_person_detected=False,
        book_detected=False,
        face_obstructed=False,
        talking_flag=False,
        talking_severity="none",
        talking_confidence=0.0,
        mouth_open_ratio=0.02,
        mouth_open_delta=0.0,
        low_quality=False,
        blur_score=100.0,
        brightness_score=120.0,
        glare_score=0.0,
        risk_score=risk_score,
        reasons=["LOOK_AWAY_MODERATE"],
        person_detected=True,
        yaw=22.0,
        pitch=4.0,
        roll=1.0,
        gaze_direction="left",
        pose_method="test",
        error=None,
    )


def test_live_result_store_roundtrip_and_summary(tmp_path):
    store = LiveResultStore(tmp_path / "live.sqlite3")
    store.upsert_frame(_record())
    store.update_attempt_summary("student_1", "quiz10_attempt99")
    store.set_last_source_log_id(101)

    frames = store.fetch_frame_dicts("student_1", "quiz10_attempt99")
    assert len(frames) == 1
    assert frames[0]["quiz_id"] == "88195"
    assert frames[0]["question_label"] == "Q7: Binary Search"
    assert frames[0]["source_log_id"] == 101

    summaries = store.fetch_summaries()
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["student_id"] == "student_1"
    assert summary["attempt_id"] == "quiz10_attempt99"
    assert summary["quiz_name"] == "Midterm"
    assert summary["question_overview"] == ["Q7: Binary Search"]
    assert summary["incident_count"] == 1

    state = store.fetch_state()
    assert state["last_source_log_id"] == 101
    assert state["frame_count"] == 1
    assert state["summary_count"] == 1


def test_live_result_store_replaces_same_source_log_id(tmp_path):
    store = LiveResultStore(tmp_path / "live.sqlite3")
    store.upsert_frame(_record(source_log_id=7, risk_score=20.0))
    store.upsert_frame(_record(source_log_id=7, risk_score=80.0))
    store.update_attempt_summary("student_1", "quiz10_attempt99")

    frames = store.fetch_frames("student_1", "quiz10_attempt99")
    assert len(frames) == 1
    assert frames[0].risk_score == 80.0

    summary = store.fetch_summary("student_1", "quiz10_attempt99")
    assert summary is not None
    assert summary["max_risk_score"] == 80.0
