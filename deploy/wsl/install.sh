#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/proctoring}"
APP_USER="${APP_USER:-proctoring}"
ENV_DIR="${ENV_DIR:-/etc/proctoring}"
DATA_DIR="${DATA_DIR:-/var/lib/proctoring}"
INSTALL_CUDA_TORCH="${INSTALL_CUDA_TORCH:-1}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu130}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run with sudo inside WSL/Linux: sudo bash deploy/wsl/install.sh" >&2
  exit 1
fi

echo "Installing OS packages..."
apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip rsync git curl ca-certificates \
  build-essential libgl1 libglib2.0-0 libpq-dev

if ! getent group "$APP_USER" >/dev/null 2>&1; then
  groupadd --system "$APP_USER"
fi
if ! id "$APP_USER" >/dev/null 2>&1; then
  useradd --system --gid "$APP_USER" --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

mkdir -p "$APP_DIR" "$ENV_DIR" "$DATA_DIR/live" "$DATA_DIR/reference_faces"

echo "Copying application to $APP_DIR..."
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.venv-win/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'server_dumps/' \
  --exclude 'review_data/' \
  --exclude 'live_moodle_export*/' \
  --exclude 'wsl_run_real_results*/' \
  --exclude 'windows_smoke_results*/' \
  --exclude 'real_moodle_export/' \
  --exclude 'real_test_data/' \
  --exclude 'test_data/' \
  --exclude 'reference_faces/' \
  --exclude 'real_reference_faces/' \
  --exclude '*.sqlite3' \
  --exclude '*.dump' \
  --exclude '*.tar.gz' \
  --exclude 'yolov8n.pt' \
  "$SOURCE_DIR/" "$APP_DIR/"

if [[ ! -f "$ENV_DIR/proctoring.env" ]]; then
  cp "$APP_DIR/deploy/wsl/proctoring.env.example" "$ENV_DIR/proctoring.env"
  chmod 640 "$ENV_DIR/proctoring.env"
  echo "Created $ENV_DIR/proctoring.env. Edit it before starting services."
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$APP_DIR/.venv/bin/python" -m pip install --prefer-binary -r "$APP_DIR/requirements-wsl.txt"

if [[ "$INSTALL_CUDA_TORCH" == "1" ]]; then
  "$APP_DIR/.venv/bin/python" -m pip install --index-url "$TORCH_CUDA_INDEX" torch torchvision
fi

(cd "$APP_DIR" && "$APP_DIR/.venv/bin/python" "$APP_DIR/download_models.py") || true

chmod +x "$APP_DIR"/deploy/wsl/*.sh
chown -R "$APP_USER:$APP_USER" "$APP_DIR" "$DATA_DIR"
chown root:"$APP_USER" "$ENV_DIR/proctoring.env"

cp "$APP_DIR/deploy/wsl/proctoring-live.service" /etc/systemd/system/proctoring-live.service
cp "$APP_DIR/deploy/wsl/proctoring-dashboard.service" /etc/systemd/system/proctoring-dashboard.service

systemctl daemon-reload || true

cat <<EOF

Install complete.

1. Edit secrets and filters:
   sudo nano $ENV_DIR/proctoring.env

2. Test one live pull:
   sudo -u $APP_USER $APP_DIR/deploy/wsl/run_live.sh $ENV_DIR/proctoring.env --once

3. Start services:
   sudo systemctl enable --now proctoring-live.service
   sudo systemctl enable --now proctoring-dashboard.service

4. Check logs:
   journalctl -u proctoring-live.service -f
   journalctl -u proctoring-dashboard.service -f

Dashboard URL:
   http://<server-ip>:\${PROCTORING_DASHBOARD_PORT:-5001}
EOF
