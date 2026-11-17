from __future__ import annotations

import stat
import subprocess
from ipaddress import IPv4Network
from pathlib import Path
from typing import Literal

import pytest

from deploy.configuration import DeploymentSettings
from scripts import deploy_pi
from scripts.deploy_pi import committed_revision, export_committed_source, rsync_arguments, sync_local_configuration


def git(repository: Path, *arguments: str) -> str:
    return subprocess.run([
        "git", "-c", "user.name=Deployment Tests", "-c", "user.email=deploy@example.invalid",
        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *arguments,
    ], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "jouzetsu").mkdir(parents=True)
    _ = (root / "jouzetsu" / "app.py").write_text("committed code")
    _ = (root / ".gitignore").write_text("ignored.py\ndeploy/local.toml\ndata/\n")
    _ = git(root, "init")
    _ = git(root, "add", ".")
    _ = git(root, "commit", "-m", "Initial deployment test")
    return root


def test_clean_checkout_resolves_the_full_commit_and_allows_ignored_files(repository: Path) -> None:
    (repository / "deploy").mkdir()
    _ = (repository / "deploy" / "local.toml").write_text("private settings")
    _ = (repository / "ignored.py").write_text("uncommitted ignored code")
    assert committed_revision(repository) == git(repository, "rev-parse", "HEAD")


@pytest.mark.parametrize("change", ["unstaged", "staged", "untracked", "deleted"])
def test_uncommitted_changes_prevent_deployment(
    repository: Path, change: Literal["unstaged", "staged", "untracked", "deleted"]
) -> None:
    path = repository / "jouzetsu" / "app.py"
    if change == "deleted":
        path.unlink()
    elif change == "untracked":
        _ = (repository / "new.py").write_text("uncommitted code")
    else:
        _ = path.write_text("uncommitted edit")
        if change == "staged":
            _ = git(repository, "add", ".")
    with pytest.raises(RuntimeError, match="clean Git checkout"):
        _ = committed_revision(repository)


def test_export_uses_the_captured_commit_and_excludes_ignored_files(repository: Path, tmp_path: Path) -> None:
    commit = committed_revision(repository)
    _ = (repository / "jouzetsu" / "app.py").write_text("later committed code")
    _ = git(repository, "add", ".")
    _ = git(repository, "commit", "-m", "Later commit")
    _ = (repository / "jouzetsu" / "app.py").write_text("later working-tree edit")
    _ = (repository / "ignored.py").write_text("ignored private code")
    target = tmp_path / "export"
    export_committed_source(repository, commit, target)
    assert (target / "jouzetsu" / "app.py").read_text() == "committed code"
    assert not (target / "ignored.py").exists()
    assert not (target / ".git").exists()


@pytest.fixture
def simulated_deployment(repository: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    settings = DeploymentSettings("app-host.example", "operator", "eth0", IPv4Network("192.0.2.0/24"))
    commands: list[list[str]] = []

    def transfer(source: Path, _destination: str, *, dry_run: bool = False) -> list[str]:
        assert source != repository
        assert (source / "jouzetsu" / "app.py").read_text() == "committed code"
        return ["/usr/bin/true"]

    def remote(host: str, arguments: list[str], *, administrator: bool = False) -> None:
        assert host == settings.ssh_host
        commands.append(arguments)

    def sync_settings(_source: Path, _destination: str) -> None:
        return None

    monkeypatch.setattr(deploy_pi, "SOURCE_ROOT", repository)
    monkeypatch.setattr(deploy_pi, "load_deployment_settings", lambda: settings)
    monkeypatch.setattr(deploy_pi, "rsync_arguments", transfer)
    monkeypatch.setattr(deploy_pi, "remote", remote)
    monkeypatch.setattr(deploy_pi, "sync_local_configuration", sync_settings)
    return commands


@pytest.mark.parametrize("arguments", [[], ["--dry-run"], ["--sync-only"]])
def test_dirty_checkout_fails_before_contacting_the_host(
    repository: Path, simulated_deployment: list[list[str]], arguments: list[str],
) -> None:
    _ = (repository / "new.py").write_text("uncommitted code")
    with pytest.raises(SystemExit) as error:
        deploy_pi.main(arguments)
    assert error.value.code == 2
    assert simulated_deployment == []


def test_success_records_the_captured_commit_only_after_readiness(
    repository: Path, simulated_deployment: list[list[str]],
) -> None:
    commit = committed_revision(repository)
    deploy_pi.main([])
    assert simulated_deployment[-2][0] == "curl"
    assert "deploy.record_deployment" in simulated_deployment[-1][-1]
    assert simulated_deployment[-1][-1].endswith(commit)


@pytest.mark.parametrize("arguments", [["--dry-run"], ["--sync-only"]])
def test_preview_and_initial_transfer_never_restart_or_record(
    simulated_deployment: list[list[str]], arguments: list[str],
) -> None:
    deploy_pi.main(arguments)
    assert simulated_deployment == []


def test_readiness_failure_never_records_a_successful_deployment(
    simulated_deployment: list[list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = deploy_pi.remote

    def fail_readiness(host: str, arguments: list[str], *, administrator: bool = False) -> None:
        if arguments[0] == "curl":
            raise subprocess.CalledProcessError(7, arguments)
        remote(host, arguments, administrator=administrator)

    monkeypatch.setattr(deploy_pi, "remote", fail_readiness)
    with pytest.raises(subprocess.CalledProcessError):
        deploy_pi.main([])
    assert not any("deploy.record_deployment" in command[-1] for command in simulated_deployment)


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
    for root in (source, target):
        (root / "jouzetsu" / "web" / "static").mkdir(parents=True)
    _ = (source / "jouzetsu" / "web" / "static" / "icon.svg").write_text("desktop local icon")
    protected = ("config.json", "data/chat.json", ".venv/python", "private/keep.txt",
                 "deploy/local.toml", "deploy/private/keep.txt", "jouzetsu/web/static/icon.svg")
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
