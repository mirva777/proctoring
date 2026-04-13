"""
Managed SSH bridge for Moodle DB tunneling and remote file fetches.

This is built for environments where direct access to PostgreSQL and
``moodledata/filedir`` is unavailable from the worker machine.  The bridge keeps
the SSH transport alive, reconnects on drop, forwards a local TCP port to the
remote PostgreSQL socket, and downloads snapshot files over SFTP.
"""

from __future__ import annotations

import logging
import os
import select
import shlex
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import paramiko  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSHBridgeConfig:
    host: str
    port: int = 22
    username: str = ""
    password: str = ""
    key_path: str = ""
    remote_db_host: str = "127.0.0.1"
    remote_db_port: int = 5432
    keepalive_seconds: int = 3
    connect_timeout_seconds: float = 15.0
    connect_retries: int = 5
    connect_retry_sleep_seconds: float = 3.0


def ssh_bridge_config_from_env(prefix: str = "MOODLE_SSH_") -> SSHBridgeConfig | None:
    """Return SSH settings from environment variables, or None when disabled."""
    host = os.getenv(f"{prefix}HOST", "").strip()
    if not host:
        return None

    return SSHBridgeConfig(
        host=host,
        port=int(os.getenv(f"{prefix}PORT", "22")),
        username=os.getenv(f"{prefix}USER", "").strip(),
        password=os.getenv(f"{prefix}PASSWORD", ""),
        key_path=os.getenv(f"{prefix}KEY_PATH", "").strip(),
        remote_db_host=os.getenv(f"{prefix}REMOTE_DB_HOST", "127.0.0.1").strip(),
        remote_db_port=int(os.getenv(f"{prefix}REMOTE_DB_PORT", "5432")),
        keepalive_seconds=int(os.getenv(f"{prefix}KEEPALIVE_SECONDS", "3")),
        connect_timeout_seconds=float(os.getenv(f"{prefix}CONNECT_TIMEOUT_SECONDS", "15")),
        connect_retries=int(os.getenv(f"{prefix}CONNECT_RETRIES", "5")),
        connect_retry_sleep_seconds=float(
            os.getenv(f"{prefix}CONNECT_RETRY_SLEEP_SECONDS", "3")
        ),
    )


class _ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, bridge: "ManagedSSHBridge") -> None:
        self.bridge = bridge
        super().__init__(("127.0.0.1", 0), _ForwardHandler)


class _ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        channel = self.server.bridge.open_db_channel(self.client_address)  # type: ignore[attr-defined]
        if channel is None:
            return

        try:
            while True:
                readable, _, _ = select.select([self.request, channel], [], [], 1.0)

                if self.request in readable:
                    data = self.request.recv(32768)
                    if not data:
                        break
                    channel.sendall(data)

                if channel in readable:
                    data = channel.recv(32768)
                    if not data:
                        break
                    self.request.sendall(data)
        except OSError:
            pass
        finally:
            try:
                channel.close()
            except Exception:
                pass


class ManagedSSHBridge:
    """Reconnectable SSH tunnel + SFTP bridge."""

    def __init__(self, config: SSHBridgeConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._client: Optional[paramiko.SSHClient] = None
        self._transport: Optional[paramiko.Transport] = None
        self._forward_server: Optional[_ForwardServer] = None
        self._forward_thread: Optional[threading.Thread] = None
        self.local_db_host = "127.0.0.1"
        self.local_db_port = 0

        if not self.config.username:
            raise ValueError("SSH username is required when MOODLE_SSH_HOST is set")
        if not self.config.password and not self.config.key_path:
            raise ValueError("Provide MOODLE_SSH_PASSWORD or MOODLE_SSH_KEY_PATH")

    def start(self) -> None:
        with self._lock:
            self.ensure_ready()
            if self._forward_server is None:
                self._forward_server = _ForwardServer(self)
                self.local_db_port = int(self._forward_server.server_address[1])
                self._forward_thread = threading.Thread(
                    target=self._forward_server.serve_forever,
                    daemon=True,
                    name="moodle-ssh-forward",
                )
                self._forward_thread.start()
                logger.info(
                    "SSH DB tunnel ready: %s:%s -> %s:%s via %s@%s:%s",
                    self.local_db_host,
                    self.local_db_port,
                    self.config.remote_db_host,
                    self.config.remote_db_port,
                    self.config.username,
                    self.config.host,
                    self.config.port,
                )

    def close(self) -> None:
        with self._lock:
            if self._forward_server is not None:
                try:
                    self._forward_server.shutdown()
                    self._forward_server.server_close()
                except Exception:
                    pass
                self._forward_server = None

            self._disconnect_locked()
            self._forward_thread = None
            self.local_db_port = 0

    def ensure_ready(self) -> None:
        with self._lock:
            if self._transport is not None and self._transport.is_active():
                return
            self._connect_locked()

    def open_db_channel(self, client_address) -> Optional[paramiko.Channel]:
        try:
            self.ensure_ready()
            return self._open_db_channel_once(client_address)
        except Exception as exc:
            logger.warning("SSH DB channel open failed (%s); reconnecting once.", exc)
            with self._lock:
                self._disconnect_locked()
            try:
                self.ensure_ready()
                return self._open_db_channel_once(client_address)
            except Exception as exc_2:
                logger.error("SSH DB channel reopen failed: %s", exc_2)
                return None

    def fetch_file(self, remote_path: str | Path, local_path: str | Path) -> None:
        src = Path(remote_path).as_posix()
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.ensure_ready()
            self._fetch_file_once(src, dest)
            return
        except Exception as exc:
            logger.warning("SSH file fetch failed for %s (%s); reconnecting once.", src, exc)

        with self._lock:
            self._disconnect_locked()
        self.ensure_ready()
        self._fetch_file_once(src, dest)

    def _open_db_channel_once(self, client_address) -> paramiko.Channel:
        with self._lock:
            if self._transport is None or not self._transport.is_active():
                raise RuntimeError("SSH transport is not active")
            return self._transport.open_channel(
                "direct-tcpip",
                (self.config.remote_db_host, self.config.remote_db_port),
                client_address,
            )

    def _fetch_file_once(self, remote_path: str, local_path: Path) -> None:
        tmp_path = local_path.with_name(local_path.name + ".part")
        if tmp_path.exists():
            tmp_path.unlink()

        try:
            with self._lock:
                if self._client is None:
                    raise RuntimeError("SSH client is not connected")
                sftp = self._client.open_sftp()
            try:
                sftp.get(remote_path, str(tmp_path))
            finally:
                sftp.close()
        except PermissionError:
            self._fetch_file_with_sudo(remote_path, tmp_path)
        except OSError as exc:
            if "Permission denied" in str(exc):
                self._fetch_file_with_sudo(remote_path, tmp_path)
            else:
                raise

        os.replace(tmp_path, local_path)

    def _fetch_file_with_sudo(self, remote_path: str, local_path: Path) -> None:
        with self._lock:
            if self._client is None:
                raise RuntimeError("SSH client is not connected")
            _stdin, stdout, stderr = self._client.exec_command(
                f"sudo -S -p '' cat -- {shlex.quote(remote_path)}",
                timeout=self.config.connect_timeout_seconds,
            )
            if self.config.password:
                _stdin.write(self.config.password + "\n")
                _stdin.flush()

            data = stdout.read()
            err = stderr.read().decode("utf-8", errors="replace").strip()
            exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            raise RuntimeError(
                f"sudo cat failed for {remote_path} with exit={exit_status}: {err}"
            )
        local_path.write_bytes(data)

    def _connect_locked(self) -> None:
        self._disconnect_locked()

        attempts = max(1, self.config.connect_retries)
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=self.config.host,
                    port=self.config.port,
                    username=self.config.username,
                    password=self.config.password or None,
                    key_filename=self.config.key_path or None,
                    allow_agent=False,
                    look_for_keys=False,
                    timeout=self.config.connect_timeout_seconds,
                    banner_timeout=self.config.connect_timeout_seconds,
                    auth_timeout=self.config.connect_timeout_seconds,
                )

                transport = client.get_transport()
                if transport is None or not transport.is_active():
                    raise RuntimeError("SSH transport did not become active")

                if self.config.keepalive_seconds > 0:
                    transport.set_keepalive(self.config.keepalive_seconds)

                self._client = client
                self._transport = transport
                logger.info(
                    "SSH bridge connected to %s@%s:%s (keepalive=%ss)",
                    self.config.username,
                    self.config.host,
                    self.config.port,
                    self.config.keepalive_seconds,
                )
                return
            except Exception as exc:
                last_exc = exc
                try:
                    client.close()
                except Exception:
                    pass
                if attempt < attempts:
                    logger.warning(
                        "SSH connect attempt %s/%s failed (%s); retrying in %.1fs",
                        attempt,
                        attempts,
                        exc,
                        self.config.connect_retry_sleep_seconds,
                    )
                    time.sleep(self.config.connect_retry_sleep_seconds)

        raise RuntimeError(
            f"SSH connection failed after {attempts} attempts"
        ) from last_exc

    def _disconnect_locked(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._transport = None
