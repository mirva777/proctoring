from __future__ import annotations

from data_io.review_dataset_builder import build_review_dataset_rows


def test_build_review_dataset_rows_joins_by_source_log_id():
    image_rows = [
        {
            "student_id": "student_1",
            "attempt_id": "quiz10_attempt99",
            "image_path": "snapshots/student_1/quiz10_attempt99/frame.png",
            "source_log_id": "123",
            "timestamp": "2026-04-06T11:00:00Z",
            "course_id": "2",
            "quiz_id": "4",
            "quiz_name": "Midterm",
            "quiz_page": "2",
            "question_label": "Q2: Arrays",
            "risk_score": "70",
            "reasons_list": ["EXTRA_PERSON", "PHONE"],
            "face_count": "2",
            "gaze_direction": "left",
            "severity": "moderate",
            "talking_flag": "False",
            "phone_detected": "True",
            "extra_person_detected": "True",
            "book_detected": "False",
            "face_obstructed": "False",
            "person_detected": "True",
            "low_quality": "False",
        }
    ]
    review_rows = [
        {
            "student_id": "student_1",
            "attempt_id": "quiz10_attempt99",
            "image_path": "snapshots/student_1/quiz10_attempt99/frame.png",
            "source_log_id": 123,
            "labels": ["PHONE", "EXTRA_PERSON"],
            "notes": "clear second person",
            "updated_at": "2026-04-06T12:00:00Z",
        }
    ]

    rows = build_review_dataset_rows(image_rows, review_rows)
    assert len(rows) == 1
    row = rows[0]
    assert row["human_labels_pipe"] == "PHONE|EXTRA_PERSON"
    assert row["model_reasons_pipe"] == "EXTRA_PERSON|PHONE"
    assert row["reviewed"] == "true"
    assert row["human_notes"] == "clear second person"
