"""Deploy code from this checkout without copying desktop configuration or production data."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from deploy.configuration import LOCAL_CONFIGURATION, PROJECT_ROOT, load_deployment_settings

SOURCE_ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = ("jouzetsu", "scripts", "deploy", "tests", "pyproject.toml", "uv.lock", "README.md")


def rsync_arguments(source: Path, destination: str, *, dry_run: bool = False) -> list[str]:
    """Allow only deployable code and protect mutable files even inside the code directory."""

    arguments = [
        "rsync", "--archive", "--no-owner", "--no-group", "--delete", "--itemize-changes",
        "--chmod=D2750,F640", "--exclude=__pycache__/", "--exclude=*.py[cod]",
        "--exclude=/.venv/", "--exclude=/data/", "--exclude=/config.json",
        "--exclude=/.cache/", "--exclude=/.tools/", "--exclude=/.validation/",
        "--exclude=/deploy/local.toml", "--exclude=/deploy/private/",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--sync-only", action="store_true", help="initial transfer before the service exists")
    _ = parser.add_argument("--dry-run", action="store_true", help="preview the code transfer without stopping or changing anything")
    arguments = parser.parse_args()
    settings = load_deployment_settings()
    host = settings.ssh_host
    if not arguments.sync_only and not arguments.dry_run:
        remote(host, ["sudo", "systemctl", "stop", "jouzetsu.service"], administrator=True)
    subprocess.run(
        rsync_arguments(SOURCE_ROOT, f"{host}:{PROJECT_ROOT / 'app'}", dry_run=arguments.dry_run),
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
    print("Jouzetsu is active and its local HTTP readiness check passed.")


if __name__ == "__main__":
    main()
