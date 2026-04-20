#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-/etc/proctoring/proctoring.env}"
APP_DIR="${PROCTORING_APP_DIR:-/opt/proctoring}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  APP_DIR="${PROCTORING_APP_DIR:-/opt/proctoring}"
fi

"$APP_DIR/deploy/wsl/run_live.sh" "$ENV_FILE" --once
