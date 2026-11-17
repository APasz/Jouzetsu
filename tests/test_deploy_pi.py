from __future__ import annotations

import stat
import subprocess
from pathlib import Path

from scripts.deploy_pi import rsync_arguments, sync_local_configuration


def test_deployment_updates_only_code_and_preserves_runtime_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    for root in (source, target):
        for directory in ("jouzetsu", ".venv", "data", "private", "deploy/private"):
            (root / directory).mkdir(parents=True)
    _ = (source / "jouzetsu" / "app.py").write_text("new code")
    _ = (source / "config.json").write_text("desktop settings")
    _ = (source / "data" / "chat.json").write_text("desktop chat")
    _ = (source / "private" / "secret.txt").write_text("desktop secret")
    _ = (source / "deploy" / "local.toml").write_text("local machine settings")
    _ = (source / "deploy" / "private" / "report.txt").write_text("private operational report")
    _ = (source / "deploy" / "public.py").write_text("public deployment code")
    protected = ("config.json", "data/chat.json", ".venv/python", "private/keep.txt",
                 "deploy/local.toml", "deploy/private/keep.txt")
    for name in protected:
        _ = (target / name).write_text("production")
    _ = (target / "jouzetsu" / "obsolete.py").write_text("removed code")

    subprocess.run(rsync_arguments(source, str(target)), check=True, capture_output=True)

    assert (target / "jouzetsu" / "app.py").read_text() == "new code"
    assert not (target / "jouzetsu" / "obsolete.py").exists()
    assert not (target / "private" / "secret.txt").exists()
    assert not (target / "deploy" / "private" / "report.txt").exists()
    assert (target / "deploy" / "public.py").read_text() == "public deployment code"
    for name in protected:
        assert (target / name).read_text() == "production"


def test_deployment_dry_run_preserves_every_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "jouzetsu").mkdir(parents=True)
    (target / "jouzetsu").mkdir(parents=True)
    _ = (source / "jouzetsu" / "new.py").write_text("new code")
    _ = (target / "jouzetsu" / "old.py").write_text("old code")

    subprocess.run(rsync_arguments(source, str(target), dry_run=True), check=True, capture_output=True)

    assert not (target / "jouzetsu" / "new.py").exists()
    assert (target / "jouzetsu" / "old.py").read_text() == "old code"


def test_explicit_local_configuration_transfer_is_owner_only(tmp_path: Path) -> None:
    source = tmp_path / "source.toml"
    target = tmp_path / "target.toml"
    _ = source.write_text("local installation settings")
    source.chmod(0o644)
    sync_local_configuration(source, str(target))
    assert target.read_text() == source.read_text()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
