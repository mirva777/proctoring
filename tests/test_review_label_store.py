from __future__ import annotations

from data_io.review_label_store import ReviewLabelStore


def test_review_label_store_roundtrip_and_counts(tmp_path):
    store = ReviewLabelStore(tmp_path / "review.sqlite3")
    store.save_label(
        student_id="student_1",
        attempt_id="quiz10_attempt99",
        image_path="snapshots/student_1/quiz10_attempt99/frame.png",
        source_log_id=123,
        labels=["PHONE", "LOOK_AWAY", "PHONE"],
        notes="phone visible on desk",
    )

    labels = store.fetch_attempt_labels("student_1", "quiz10_attempt99")
    assert len(labels) == 1
    record = next(iter(labels.values()))
    assert record["labels"] == ["LOOK_AWAY", "PHONE"]
    assert record["notes"] == "phone visible on desk"

    counts = store.fetch_attempt_review_counts()
    assert counts[("student_1", "quiz10_attempt99")] == 1

    exported = store.export_rows()
    assert len(exported) == 1
    assert exported[0]["labels_pipe"] == "LOOK_AWAY|PHONE"


def test_review_label_store_clears_empty_labels_and_notes(tmp_path):
    store = ReviewLabelStore(tmp_path / "review.sqlite3")
    store.save_label(
        student_id="student_1",
        attempt_id="quiz10_attempt99",
        image_path="snapshots/student_1/quiz10_attempt99/frame.png",
        source_log_id=123,
        labels=["EXTRA_PERSON"],
        notes="",
    )
    store.save_label(
        student_id="student_1",
        attempt_id="quiz10_attempt99",
        image_path="snapshots/student_1/quiz10_attempt99/frame.png",
        source_log_id=123,
        labels=[],
        notes="   ",
    )

    labels = store.fetch_attempt_labels("student_1", "quiz10_attempt99")
    assert labels == {}
