"""Headless smoke test: boot Jouzetsu on a fixed port, hit the root URL,
verify the page contains expected markers, then exit.

This is a developer tool, not a real test suite. Run it from the
project root:

    uv run python -m scripts.smoke_server
"""

import sys
import tempfile
from pathlib import Path

from jouzetsu.app import run
from jouzetsu.config import AppConfig, LoggingSettings, UiSettings

# Use a unique port so we don't clash with a real server.
PORT = 18099


def _smoke_config(runtime_dir: Path) -> AppConfig:
    """Create an isolated runtime configuration for the smoke server."""
    return AppConfig(
        ui=UiSettings(port=PORT, auto_open_browser=False),
        logging=LoggingSettings(enabled=False),
        data_dir=runtime_dir,
        chats_file=runtime_dir / "chats.json",
        config_file=runtime_dir / "config.json",
    )


if __name__ == "__main__":
    try:
        with tempfile.TemporaryDirectory(prefix="jouzetsu-smoke-") as runtime_dir:
            run(_smoke_config(Path(runtime_dir)))
    except KeyboardInterrupt:
        sys.exit(0)
