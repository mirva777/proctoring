from __future__ import annotations

from data_io.ssh_moodle_bridge import ssh_bridge_config_from_env


def test_ssh_bridge_config_disabled_without_host(monkeypatch):
    monkeypatch.delenv("MOODLE_SSH_HOST", raising=False)
    assert ssh_bridge_config_from_env() is None


def test_ssh_bridge_config_from_env(monkeypatch):
    monkeypatch.setenv("MOODLE_SSH_HOST", "moodle.example.internal")
    monkeypatch.setenv("MOODLE_SSH_PORT", "22")
    monkeypatch.setenv("MOODLE_SSH_USER", "moodle_user")
    monkeypatch.setenv("MOODLE_SSH_PASSWORD", "secret")
    monkeypatch.setenv("MOODLE_SSH_REMOTE_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("MOODLE_SSH_REMOTE_DB_PORT", "5432")
    monkeypatch.setenv("MOODLE_SSH_KEEPALIVE_SECONDS", "2")
    monkeypatch.setenv("MOODLE_SSH_CONNECT_TIMEOUT_SECONDS", "15")

    cfg = ssh_bridge_config_from_env()
    assert cfg is not None
    assert cfg.host == "moodle.example.internal"
    assert cfg.port == 22
    assert cfg.username == "moodle_user"
    assert cfg.password == "secret"
    assert cfg.remote_db_host == "127.0.0.1"
    assert cfg.remote_db_port == 5432
    assert cfg.keepalive_seconds == 2
    assert cfg.connect_timeout_seconds == 15.0
