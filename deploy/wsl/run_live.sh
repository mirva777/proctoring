#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-/etc/proctoring/proctoring.env}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

APP_DIR="${PROCTORING_APP_DIR:-/opt/proctoring}"
OUTPUT_DIR="${PROCTORING_OUTPUT_DIR:-/var/lib/proctoring/live}"
LIVE_DB="${PROCTORING_LIVE_DB:-$OUTPUT_DIR/live_results.sqlite3}"
CONFIG="${PROCTORING_CONFIG:-$APP_DIR/config_wsl_live.yaml}"
MOODLEDATA="${PROCTORING_MOODLEDATA:-/var/moodledata}"
DEVICE="${PROCTORING_DEVICE:-auto}"
BATCH_LIMIT="${PROCTORING_BATCH_LIMIT:-250}"
POLL_SECONDS="${PROCTORING_POLL_SECONDS:-2}"

cd "$APP_DIR"
mkdir -p "$OUTPUT_DIR" "$(dirname "$LIVE_DB")"

cmd=(
  "$APP_DIR/.venv/bin/python"
  "$APP_DIR/live_moodle_pipeline.py"
  --config "$CONFIG"
  --env-file "$ENV_FILE"
  --moodledata "$MOODLEDATA"
  --output "$OUTPUT_DIR"
  --store-db "$LIVE_DB"
  --device "$DEVICE"
  --batch-limit "$BATCH_LIMIT"
  --poll-seconds "$POLL_SECONDS"
)

if [[ -n "${PROCTORING_COURSE_ID:-}" ]]; then
  cmd+=(--course-id "$PROCTORING_COURSE_ID")
fi
if [[ -n "${PROCTORING_QUIZ_ID:-}" ]]; then
  cmd+=(--quiz-id "$PROCTORING_QUIZ_ID")
fi
if [[ -n "${PROCTORING_REFERENCE_FACES:-}" ]]; then
  cmd+=(--reference-faces "$PROCTORING_REFERENCE_FACES")
fi

cmd+=("$@")

exec "${cmd[@]}"
