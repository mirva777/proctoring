# WSL / Linux Deployment

Deploy inside the WSL/Linux filesystem, not directly from `/mnt/c`, for better file I/O and service reliability.

## 1. Enable systemd in WSL

Inside WSL:

```bash
cat /etc/wsl.conf
```

If it does not contain systemd support, add:

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true
EOF
```

Then from Windows PowerShell:

```powershell
wsl --shutdown
```

Open WSL again.

## 2. Install the App

From this repo inside WSL:

```bash
cd /mnt/c/Users/Mirvohid/Desktop/proctoring
sudo bash deploy/wsl/install.sh
```

The installer copies only source/deployment files into `/opt/proctoring`, creates `/opt/proctoring/.venv`, and installs service files.

## 3. Configure Secrets and Moodle Filters

```bash
sudo nano /etc/proctoring/proctoring.env
```

Set at least:

```bash
MOODLE_DB_HOST=127.0.0.1
MOODLE_DB_PORT=5432
MOODLE_DB_NAME=moodle
MOODLE_DB_USER=moodle
MOODLE_DB_PASSWORD=change-me
MOODLE_DB_PREFIX=mdl_
PROCTORING_COURSE_ID=16
PROCTORING_QUIZ_ID=51
PROCTORING_MOODLEDATA=/var/moodledata
```

If the service connects to another Linux host through SSH, also set `MOODLE_SSH_*`.

## 4. Smoke Test One Batch

```bash
sudo -u proctoring /opt/proctoring/deploy/wsl/run_live.sh /etc/proctoring/proctoring.env --once
```

Expected output includes:

```text
Verified Moodle PostgreSQL connectivity
Processed ... new snapshots
```

## 5. Start Services

```bash
sudo systemctl enable --now proctoring-live.service
sudo systemctl enable --now proctoring-dashboard.service
```

Logs:

```bash
journalctl -u proctoring-live.service -f
journalctl -u proctoring-dashboard.service -f
```

Dashboard:

```text
http://<server-ip>:5001
```

## Useful Commands

Restart after config edits:

```bash
sudo systemctl restart proctoring-live.service proctoring-dashboard.service
```

Check local results DB:

```bash
sqlite3 /var/lib/proctoring/live/live_results.sqlite3 \
  'select count(*) from frame_results; select count(*) from attempt_summaries;'
```

Check CUDA:

```bash
nvidia-smi
/opt/proctoring/.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```
