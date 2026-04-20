#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-/etc/proctoring/proctoring.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

APP_DIR="${PROCTORING_APP_DIR:-/opt/proctoring}"
HOST="${PROCTORING_DASHBOARD_HOST:-0.0.0.0}"
PORT="${PROCTORING_DASHBOARD_PORT:-5001}"
WORKERS="${PROCTORING_DASHBOARD_WORKERS:-2}"
OUTPUT_DIR="${PROCTORING_OUTPUT_DIR:-/var/lib/proctoring/live}"
LIVE_DB="${PROCTORING_LIVE_DB:-$OUTPUT_DIR/live_results.sqlite3}"
REVIEW_DB="${PROCTORING_REVIEW_DB:-/var/lib/proctoring/review_labels.sqlite3}"

export PROCTORING_OUTPUT_DIR="$OUTPUT_DIR"
export PROCTORING_LIVE_DB="$LIVE_DB"
export PROCTORING_REVIEW_DB="$REVIEW_DB"
export PROCTORING_SNAPSHOTS_DIR="${PROCTORING_SNAPSHOTS_DIR:-$OUTPUT_DIR}"
export PROCTORING_RESULTS_DIR="${PROCTORING_RESULTS_DIR:-$OUTPUT_DIR}"

cd "$APP_DIR"
mkdir -p "$OUTPUT_DIR" "$(dirname "$LIVE_DB")" "$(dirname "$REVIEW_DB")"

exec "$APP_DIR/.venv/bin/gunicorn" \
  --workers "$WORKERS" \
  --bind "$HOST:$PORT" \
  --access-logfile - \
  --error-logfile - \
  wsgi:app
