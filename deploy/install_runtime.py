"""Install a project-local ARM Python runtime and frozen dependencies as the deployment administrator."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Final
from urllib.request import urlopen

from deploy.configuration import PROJECT_ROOT

UV_VERSION: Final[str] = "0.13.8"
PYTHON_VERSION: Final[str] = "3.14.4"


def runtime_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        UV_CACHE_DIR=str(PROJECT_ROOT / ".cache" / "uv"),
        UV_PYTHON_INSTALL_DIR=str(PROJECT_ROOT / ".tools" / "python"),
        UV_PYTHON_BIN_DIR=str(PROJECT_ROOT / ".tools" / "bin"),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(PROJECT_ROOT / "tmp"),
    )
    return environment


def install_uv() -> Path:
    binary_directory = PROJECT_ROOT / ".tools" / "bin"
    uv = binary_directory / "uv"
    if uv.exists():
        subprocess.run([str(uv), "--version"], check=True)
        return uv
    if platform.machine() != "aarch64":
        raise RuntimeError("the Pi bootstrap expects an aarch64 host")
    archive_name = "uv-aarch64-unknown-linux-gnu.tar.gz"
    release = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{archive_name}"
    with urlopen(release + ".sha256", timeout=60) as response:
        expected = response.read().decode("ascii").split()[0]
    with urlopen(release, timeout=60) as response:
        archive_bytes = response.read()
    if hashlib.sha256(archive_bytes).hexdigest() != expected:
        raise RuntimeError("the downloaded uv archive failed its release checksum")
    binary_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".cache") as temporary:
        directory = Path(temporary)
        archive = directory / archive_name
        _ = archive.write_bytes(archive_bytes)
        with tarfile.open(archive) as package:
            package.extractall(directory, filter="data")
        for name in ("uv", "uvx"):
            destination = binary_directory / name
            _ = shutil.copyfile(directory / "uv-aarch64-unknown-linux-gnu" / name, destination)
            destination.chmod(0o750)
    return uv


def main() -> None:
    if os.geteuid() == 0:
        raise PermissionError("install the application runtime as the deployment administrator, not root")
    if not Path("/srv").is_mount():
        raise RuntimeError("the SSD must be mounted at /srv")
    uv = install_uv()
    environment = runtime_environment()
    for arguments in (
        [str(uv), "python", "install", PYTHON_VERSION, "--no-bin"],
        [str(uv), "sync", "--frozen", "--no-dev", "--python", PYTHON_VERSION],
    ):
        subprocess.run(arguments, cwd=PROJECT_ROOT / "app", env=environment, check=True)
    print("Installed the self-contained ARM runtime and frozen production dependencies.", flush=True)


if __name__ == "__main__":
    main()
