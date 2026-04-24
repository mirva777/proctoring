from __future__ import annotations

from data_io.live_result_store import LiveResultStore
from data_io.review_label_store import ReviewLabelStore
from review_dashboard import create_app
from scoring.aggregator import FrameRecord


def _record(
    *,
    source_log_id: int,
    student_id: str,
    attempt_id: str,
    timestamp: str,
    risk_score: float,
    reasons: list[str],
    phone_detected: bool = False,
    talking_flag: bool = False,
    look_away_flag: bool = False,
    low_quality: bool = False,
) -> FrameRecord:
    return FrameRecord(
        image_path=f"snapshots/{student_id}/{attempt_id}/{source_log_id}.png",
        student_id=student_id,
        attempt_id=attempt_id,
        timestamp=timestamp,
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
        look_away_flag=look_away_flag,
        severity="moderate" if look_away_flag else "none",
        identity_mismatch=False,
        identity_similarity=0.98,
        phone_detected=phone_detected,
        extra_person_detected=False,
        book_detected=False,
        face_obstructed=False,
        talking_flag=talking_flag,
        talking_severity="likely" if talking_flag else "none",
        talking_confidence=0.85 if talking_flag else 0.0,
        mouth_open_ratio=0.08 if talking_flag else 0.02,
        mouth_open_delta=0.04 if talking_flag else 0.0,
        low_quality=low_quality,
        blur_score=45.0 if low_quality else 100.0,
        brightness_score=120.0,
        glare_score=0.0,
        risk_score=risk_score,
        reasons=reasons,
        person_detected=True,
        yaw=22.0 if look_away_flag else 2.0,
        pitch=4.0,
        roll=1.0,
        gaze_direction="left" if look_away_flag else "center",
        pose_method="test",
        error=None,
    )


def _seed_store(store: LiveResultStore) -> None:
    store.upsert_frame(
        _record(
            source_log_id=101,
            student_id="student_1",
            attempt_id="quiz10_attempt99",
            timestamp="2026-04-21T10:00:00",
            risk_score=70.0,
            reasons=["PHONE", "LOOK_AWAY_MODERATE"],
            phone_detected=True,
            look_away_flag=True,
        )
    )
    store.upsert_frame(
        _record(
            source_log_id=102,
            student_id="student_1",
            attempt_id="quiz10_attempt99",
            timestamp="2026-04-21T10:01:00",
            risk_score=50.0,
            reasons=["TALKING_LIKELY"],
            talking_flag=True,
        )
    )
    store.upsert_frame(
        _record(
            source_log_id=201,
            student_id="student_2",
            attempt_id="quiz10_attempt100",
            timestamp="2026-04-21T10:02:00",
            risk_score=15.0,
            reasons=["LOW_QUALITY"],
            low_quality=True,
        )
    )
    store.update_attempt_summary("student_1", "quiz10_attempt99")
    store.update_attempt_summary("student_2", "quiz10_attempt100")


def test_results_api_filters_by_flag_and_returns_nested_metadata(tmp_path):
    live_db = tmp_path / "live.sqlite3"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    review_db = tmp_path / "review.sqlite3"

    store = LiveResultStore(live_db)
    _seed_store(store)
    review_store = ReviewLabelStore(review_db)
    review_store.save_label(
        student_id="student_1",
        attempt_id="quiz10_attempt99",
        image_path="snapshots/student_1/quiz10_attempt99/101.png",
        source_log_id=101,
        labels=["PHONE"],
        notes="confirmed by reviewer",
    )

    app = create_app(live_db=live_db, snapshots=snapshots, review_db=review_db)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/api/results?course_id=1782&quiz_id=88195&flag=PHONE")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["meta"]["total_matching_frames"] == 1
    assert payload["meta"]["total_matching_attempts"] == 1
    assert payload["attempts"][0]["student_id"] == "student_1"
    assert payload["attempts"][0]["matched_frame_count"] == 1

    frame = payload["frames"][0]
    assert frame["student"]["student_id"] == "student_1"
    assert frame["exam"]["quiz_id"] == "88195"
    assert frame["analysis_flags"]["phone"] is True
    assert "PHONE" in frame["analysis_flag_names"]
    assert frame["review"]["labels"] == ["PHONE"]


def test_results_api_accepts_post_body_and_attempt_shortcut_route(tmp_path):
    live_db = tmp_path / "live.sqlite3"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    store = LiveResultStore(live_db)
    _seed_store(store)

    app = create_app(live_db=live_db, snapshots=snapshots)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/api/results",
        json={
            "exam": {"course_id": "1782", "quiz_id": "88195"},
            "student": {"student_id": "student_1"},
            "flag": ["LOOK_AWAY", "PHONE"],
            "flag_mode": "all",
            "sort_by": "risk_score",
            "sort_order": "desc",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["meta"]["total_matching_frames"] == 1
    assert payload["frames"][0]["analysis_flags"]["look_away"] is True
    assert payload["frames"][0]["analysis_flags"]["phone"] is True

    shortcut = client.get("/api/results/student_1/quiz10_attempt99")
    assert shortcut.status_code == 200
    shortcut_payload = shortcut.get_json()
    assert shortcut_payload["filters"]["student_id"] == ["student_1"]
    assert shortcut_payload["filters"]["attempt_id"] == ["quiz10_attempt99"]
    assert shortcut_payload["meta"]["total_matching_attempts"] == 1
    assert len(shortcut_payload["frames"]) == 2


def test_openapi_and_swagger_docs_routes(tmp_path):
    live_db = tmp_path / "live.sqlite3"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    store = LiveResultStore(live_db)
    _seed_store(store)

    app = create_app(live_db=live_db, snapshots=snapshots)
    app.config["TESTING"] = True
    client = app.test_client()

    spec_response = client.get("/openapi.json")
    assert spec_response.status_code == 200
    spec = spec_response.get_json()
    assert spec["openapi"] == "3.0.3"
    assert spec["info"]["title"] == "Exam Proctoring Review API"
    assert "/api/results" in spec["paths"]
    assert "/api/review-labels" in spec["paths"]
    result_examples = spec["paths"]["/api/results/{student_id}/{attempt_id}"]["get"]["responses"]["200"]["content"]["application/json"]["examples"]
    example = result_examples["specificStudentAttempt"]["value"]
    assert example["attempts"][0]["student_id"] == "ps2124-11487"
    assert example["frames"][0]["snapshot_url"].startswith("/snapshot/")
    assert example["frames"][0]["analysis_flags"]["extra_person"] is True

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    html = docs_response.get_data(as_text=True)
    assert "SwaggerUIBundle" in html
    assert "/openapi.json" in html
