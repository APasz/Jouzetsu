from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import uvicorn

from jouzetsu import app as app_module
from jouzetsu.app import (
    clear_unclean_shutdown_marker,
    mark_unclean_shutdown,
    unclean_shutdown_marker_path,
)
from jouzetsu.config import AppConfig, AppPaths


class _WebApplicationStub:
    """Minimal production-runner surface used to inspect Uvicorn configuration."""

    def __init__(self) -> None:
        self.app: object = object()
        self.exit_callback_set: bool = False

    def set_exit_callback(self, _callback: object) -> None:
        self.exit_callback_set = True


class ShutdownMarkerTests(unittest.TestCase):
    def test_incomplete_shutdown_marker_is_written_and_can_be_cleared_after_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            config = AppConfig(paths=AppPaths.for_home(home))

            mark_unclean_shutdown(config)

            marker_path = unclean_shutdown_marker_path(config)
            self.assertTrue(marker_path.exists())
            self.assertIn("recorded_at=", marker_path.read_text(encoding="utf-8"))

            clear_unclean_shutdown_marker(config)

            self.assertFalse(marker_path.exists())

    def test_server_config_bounds_graceful_shutdown_for_open_sse_streams(self) -> None:
        config = AppConfig()
        web_application = _WebApplicationStub()
        with (
            patch.object(app_module, "build_app", return_value=config),
            patch.object(app_module, "_log_access_urls"),
            patch.object(app_module, "_web_application", web_application),
            patch.object(uvicorn, "Server") as server_type,
        ):
            app_module.run(config)

        server_config: uvicorn.Config = cast(uvicorn.Config, server_type.call_args.args[0])
        self.assertEqual(
            server_config.timeout_graceful_shutdown,
            app_module._UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,  # pyright: ignore[reportPrivateUsage]
        )
        self.assertTrue(web_application.exit_callback_set)
