"""Prepare the service account, shared project storage and systemd units."""

from __future__ import annotations

import argparse
import grp
import os
import pwd
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from deploy.configuration import PROJECT_ROOT, DeploymentSettings, load_deployment_settings, render_firewall

SERVICE_USER: Final[str] = "jouzetsu"


def run(*arguments: str) -> None:
    subprocess.run(arguments, check=True)


def prepare_directory(path: Path, *, owner: int, group: int, mode: int, shared: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chown(path, owner, group)
    path.chmod(mode)
    if shared:
        run("setfacl", "-m", "d:u::rwx,d:g::rwx,d:m::rwx,d:o::---", str(path))


def prepare(settings: DeploymentSettings) -> None:
    if not Path("/srv").is_mount():
        raise RuntimeError("/srv must be the mounted SSD before preparing Jouzetsu")
    administrator = pwd.getpwnam(settings.admin_user)
    try:
        group = grp.getgrnam(SERVICE_USER)
    except KeyError:
        run("groupadd", "--system", SERVICE_USER)
        group = grp.getgrnam(SERVICE_USER)
    try:
        service = pwd.getpwnam(SERVICE_USER)
    except KeyError:
        run(
            "useradd", "--system", "--gid", SERVICE_USER, "--home-dir", str(PROJECT_ROOT),
            "--no-create-home", "--shell", "/usr/sbin/nologin", SERVICE_USER,
        )
        service = pwd.getpwnam(SERVICE_USER)
    if service.pw_uid >= 1000 or service.pw_shell != "/usr/sbin/nologin" or service.pw_gid != group.gr_gid:
        raise RuntimeError("the existing jouzetsu account is not the expected non-login system account")
    run("usermod", "--append", "--groups", SERVICE_USER, settings.admin_user)
    prepare_directory(PROJECT_ROOT, owner=administrator.pw_uid, group=group.gr_gid, mode=0o2770, shared=True)
    for name in ("app", ".tools", ".cache", ".validation"):
        prepare_directory(PROJECT_ROOT / name, owner=administrator.pw_uid, group=group.gr_gid, mode=0o2750)
        run("setfacl", "--remove-default", str(PROJECT_ROOT / name))
    for name in ("data", "tmp"):
        prepare_directory(PROJECT_ROOT / name, owner=service.pw_uid, group=group.gr_gid, mode=0o2770, shared=True)
    if shutil.which("nft") is None:
        run("apt-get", "update", "--quiet")
        run("apt-get", "install", "--yes", "--no-install-recommends", "nftables")
    print("Prepared SSD directories, service account, administrator group access and firewall tooling.", flush=True)


def install_service(settings: DeploymentSettings) -> None:
    source = PROJECT_ROOT / "app" / "deploy"
    python = PROJECT_ROOT / "app" / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("install the ARM Python environment before installing the service")
    firewall_directory = Path("/etc/nftables.d")
    firewall_directory.mkdir(exist_ok=True)
    destinations = (
        ("jouzetsu.service", Path("/etc/systemd/system/jouzetsu.service")),
        ("jouzetsu-firewall.service", Path("/etc/systemd/system/jouzetsu-firewall.service")),
    )
    for filename, destination in destinations:
        shutil.copyfile(source / filename, destination)
        os.chown(destination, 0, 0)
        destination.chmod(0o644)
    firewall = firewall_directory / "jouzetsu.nft"
    _ = firewall.write_text(render_firewall((source / "jouzetsu.nft.in").read_text(encoding="utf-8"), settings))
    os.chown(firewall, 0, 0)
    firewall.chmod(0o644)
    run("nft", "--check", "--file", str(firewall_directory / "jouzetsu.nft"))
    run("systemd-analyze", "--generators=yes", "verify", "/etc/systemd/system/jouzetsu.service", "/etc/systemd/system/jouzetsu-firewall.service")
    run("systemctl", "daemon-reload")
    run("systemctl", "enable", "jouzetsu-firewall.service", "jouzetsu.service")
    run("systemctl", "restart", "jouzetsu-firewall.service")
    run("systemctl", "restart", "jouzetsu.service")
    print("Installed and started Jouzetsu with its dedicated account and port firewall.", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise PermissionError("run this setup with sudo")
    settings = load_deployment_settings()
    prepare(settings)
    if not arguments.prepare_only:
        install_service(settings)


if __name__ == "__main__":
    main()
