"""Safe discovery and bounded reading of Jouzetsu log files."""

from __future__ import annotations

from pathlib import Path

MAX_LOG_TAIL_BYTES: int = 65_536


def discover_log_files(directory: Path) -> tuple[Path, ...]:
    """Return regular files in the configured Jouzetsu logging directory."""
    if not directory.is_dir():
        return ()
    files: list[Path] = [path for path in directory.iterdir() if path.is_file()]
    return tuple[Path, ...](sorted(files, key=lambda path: path.name.casefold()))


def read_log_tail(path: Path, *, max_bytes: int = MAX_LOG_TAIL_BYTES) -> str:
    """Return at most the final ``max_bytes`` of a UTF-8 log file."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    with path.open("rb") as log_file:
        _ = log_file.seek(0, 2)
        size: int = log_file.tell()
        _ = log_file.seek(max(size - max_bytes, 0))
        return log_file.read().decode("utf-8", errors="replace")
