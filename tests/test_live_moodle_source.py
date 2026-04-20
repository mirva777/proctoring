from __future__ import annotations

from pathlib import Path, PurePosixPath

from data_io.live_moodle_source import LiveMoodleSource, MoodleDBConfig
from data_io.ssh_moodle_bridge import ManagedSSHBridge, SSHBridgeConfig


def _db_config() -> MoodleDBConfig:
    return MoodleDBConfig(
        host="127.0.0.1",
        port=5432,
        dbname="moodle",
        user="moodle",
        password="secret",
    )


def test_remote_moodledata_path_stays_posix_on_windows(tmp_path):
    bridge = ManagedSSHBridge(
        SSHBridgeConfig(host="moodle.example.internal", username="user", password="secret")
    )
    source = LiveMoodleSource(
        db_config=_db_config(),
        output_dir=tmp_path,
        moodledata_dir="/var/www/moodledata",
        ssh_bridge=bridge,
    )

    path = source._contenthash_to_path("abcdef123456")

    assert isinstance(path, PurePosixPath)
    assert path.as_posix() == "/var/www/moodledata/filedir/ab/cd/abcdef123456"


def test_local_moodledata_path_uses_platform_path(tmp_path):
    moodledata = tmp_path / "moodledata"
    source = LiveMoodleSource(
        db_config=_db_config(),
        output_dir=tmp_path / "out",
        moodledata_dir=moodledata,
    )

    path = source._contenthash_to_path("abcdef123456")

    assert isinstance(path, Path)
    assert path == moodledata.resolve() / "filedir" / "ab" / "cd" / "abcdef123456"
