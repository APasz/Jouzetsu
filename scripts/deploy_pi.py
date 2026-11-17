"""Deploy code from this checkout without copying desktop configuration or production data."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path

from deploy.configuration import LOCAL_CONFIGURATION, PROJECT_ROOT, load_deployment_settings

SOURCE_ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = ("jouzetsu", "scripts", "deploy", "tests", "pyproject.toml", "uv.lock", "README.md")


def committed_revision(source: Path) -> str:
    """Reject staged, unstaged and untracked changes before contacting the host."""

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
        cwd=source, check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("Deployment requires a clean Git checkout. Commit or remove local changes and untracked files first.")
    return subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=source, check=True, capture_output=True, text=True,
    ).stdout.strip()


def export_committed_source(source: Path, commit: str, destination: Path) -> None:
    """Export immutable committed files without ignored settings or later working-tree edits."""

    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=source, stdout=archive, check=True)
        _ = archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as package:
            package.extractall(destination, filter="data")


def rsync_arguments(source: Path, destination: str, *, dry_run: bool = False) -> list[str]:
    """Allow only deployable code and protect mutable files even inside the code directory."""

    arguments = [
        "rsync", "--archive", "--no-owner", "--no-group", "--delete", "--itemize-changes",
        "--chmod=D2750,F640", "--exclude=__pycache__/", "--exclude=*.py[cod]",
        "--exclude=/.venv/", "--exclude=/data/", "--exclude=/config.json",
        "--exclude=/.cache/", "--exclude=/.tools/", "--exclude=/.validation/",
        "--exclude=/deploy/local.toml", "--exclude=/deploy/private/",
        "--exclude=/jouzetsu/web/static/icon.svg",
    ]
    if dry_run:
        arguments.append("--dry-run")
    arguments.extend(f"--include=/{path}/***" if (source / path).is_dir() else f"--include=/{path}" for path in CODE_PATHS)
    arguments.extend(["--exclude=*", str(source) + "/", destination.rstrip("/") + "/"])
    return arguments


def remote(host: str, arguments: list[str], *, administrator: bool = False) -> None:
    command = ["ssh"]
    if administrator:
        command.append("-t")
    command.extend([host, shlex.join(arguments)])
    subprocess.run(command, check=True)


def sync_local_configuration(source: Path, destination: str) -> None:
    """Transfer installation metadata explicitly and retain owner-only access."""

    subprocess.run([
        "rsync", "--archive", "--no-owner", "--no-group", "--chmod=F600", str(source), destination,
    ], check=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--sync-only", action="store_true", help="initial transfer before the service exists")
    _ = parser.add_argument("--dry-run", action="store_true", help="preview the code transfer without stopping or changing anything")
    arguments = parser.parse_args(argv)
    try:
        commit = committed_revision(SOURCE_ROOT)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"{exc}\n")
    settings = load_deployment_settings()
    host = settings.ssh_host
    with tempfile.TemporaryDirectory(prefix="jouzetsu-deploy-") as temporary:
        source = Path(temporary)
        export_committed_source(SOURCE_ROOT, commit, source)
        print(f"{'Previewing' if arguments.dry_run else 'Deploying'} commit {commit}", flush=True)
        if not arguments.sync_only and not arguments.dry_run:
            remote(host, ["sudo", "systemctl", "stop", "jouzetsu.service"], administrator=True)
        subprocess.run(
            rsync_arguments(source, f"{host}:{PROJECT_ROOT / 'app'}", dry_run=arguments.dry_run),
            check=True,
        )
        if arguments.dry_run:
            return
    sync_local_configuration(LOCAL_CONFIGURATION, f"{host}:{PROJECT_ROOT / 'app' / 'deploy' / LOCAL_CONFIGURATION.name}")
    if arguments.sync_only:
        return
    remote(host, [
        "sh", "-c", f"cd {shlex.quote(str(PROJECT_ROOT / 'app'))} && python3 -m deploy.install_runtime",
    ])
    remote(host, [
        "sh", "-c", (
            f"cd {shlex.quote(str(PROJECT_ROOT / 'app'))} && "
            "sudo env PYTHONDONTWRITEBYTECODE=1 python3 -m deploy.setup_pi"
        ),
    ], administrator=True)
    remote(host, ["systemctl", "is-active", "jouzetsu.service"])
    remote(host, ["curl", "--fail", "--silent", "--output", "/dev/null", "--retry", "10",
                            "--retry-connrefused", "--retry-delay", "1", "http://127.0.0.1:6900/chats"])
    remote(host, [
        "sh", "-c", (
            f"cd {shlex.quote(str(PROJECT_ROOT / 'app'))} && "
            f"env JOUZETSU_SHARED_FILES=true .venv/bin/python -m deploy.record_deployment {shlex.quote(commit)}"
        ),
    ])
    print(f"Deployed commit {commit}; service and HTTP readiness checks passed.")


if __name__ == "__main__":
    main()
