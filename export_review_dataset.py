#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from data_io.review_dataset_builder import (
    build_review_dataset_rows,
    load_image_results_csv,
    load_review_rows_from_csv,
    summarize_review_dataset,
)
from data_io.review_label_store import ReviewLabelStore


def default_review_db_for_results(results_dir: Path) -> Path:
    return Path("review_data") / results_dir.name / "review_labels.sqlite3"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join human review labels with model outputs for training and evaluation."
    )
    parser.add_argument("--results", required=True, help="Directory containing image_level_results.csv")
    parser.add_argument("--output", required=True, help="Output CSV path for the joined review dataset")
    parser.add_argument("--snapshots", default=None, help="Optional snapshots root for absolute image paths")
    parser.add_argument("--review-db", default=None, help="SQLite DB created by the review dashboard")
    parser.add_argument("--labels-csv", default=None, help="Exported review_labels.csv from the dashboard")
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include model rows that do not yet have human labels",
    )
    args = parser.parse_args()

    results_dir = Path(args.results).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict]
    if args.labels_csv:
        review_rows = load_review_rows_from_csv(args.labels_csv)
    else:
        review_db_path = Path(args.review_db).resolve() if args.review_db else default_review_db_for_results(results_dir).resolve()
        if not review_db_path.exists():
            legacy = results_dir / "review_labels.sqlite3"
            if legacy.exists():
                review_db_path = legacy.resolve()
            else:
                raise SystemExit(f"Review DB not found: {review_db_path}")
        store = ReviewLabelStore(review_db_path)
        review_rows = store.export_rows()
        store.close()

    image_rows = load_image_results_csv(results_dir)
    dataset_rows = build_review_dataset_rows(
        image_rows,
        review_rows,
        snapshots_dir=args.snapshots,
        include_unreviewed=args.include_unreviewed,
    )

    fieldnames = [
        "student_id",
        "attempt_id",
        "image_path",
        "absolute_image_path",
        "timestamp",
        "course_id",
        "quiz_id",
        "quiz_name",
        "quiz_page",
        "question_label",
        "source_log_id",
        "risk_score",
        "model_reasons_pipe",
        "face_count",
        "gaze_direction",
        "severity",
        "talking_flag",
        "phone_detected",
        "extra_person_detected",
        "book_detected",
        "face_obstructed",
        "person_detected",
        "low_quality",
        "human_labels_pipe",
        "human_notes",
        "review_updated_at",
        "reviewed",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)

    summary_path = output_path.with_suffix(".summary.json")
    summary = summarize_review_dataset(dataset_rows)
    summary["results_dir"] = str(results_dir)
    summary["output_csv"] = str(output_path)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"review_rows={len(review_rows)}")
    print(f"dataset_rows={len(dataset_rows)}")
    print(f"output_csv={output_path}")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
