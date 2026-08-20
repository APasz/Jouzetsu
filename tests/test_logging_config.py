from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from jouzetsu.config import AppConfig, AppPaths, LoggingSettings, encode_config
from jouzetsu.logging_config import (
    APPLICATION_LOGGER_NAME,
    chat_logger,
    configure_bootstrap_logging,
    configure_logging,
)


class LoggingConfigTests(unittest.TestCase):
    def test_app_config_derives_fixed_log_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = AppConfig(paths=AppPaths.for_home(Path(tmp_dir)))
            log_dir = config.log_directory

            self.assertEqual(log_dir / "system.log", Path(tmp_dir) / "data" / "logs" / "system.log")
            self.assertEqual(log_dir / "error.log", Path(tmp_dir) / "data" / "logs" / "error.log")
            self.assertEqual(log_dir / "chat.log", Path(tmp_dir) / "data" / "logs" / "chat.log")

    def test_codec_keeps_app_home_relative_log_directory_portable(self) -> None:
        paths = AppPaths.for_home(Path("/tmp") / "jouzetsu-test-home")
        config = AppConfig(paths=paths, logging=LoggingSettings(directory=paths.log_directory))

        self.assertEqual(encode_config(config)["logging"], {"enabled": True, "directory": "data/logs"})

    def test_configure_logging_routes_system_error_and_chat_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "logs"
            config = AppConfig(logging=LoggingSettings(directory=log_dir))
            log_dir.mkdir()
            for log_name in ("system.log", "error.log", "chat.log"):
                _ = (log_dir / log_name).write_text("previous process log\n", encoding="utf-8")
            previous_log_dir = log_dir.with_name("logs.old")
            previous_log_dir.mkdir()
            _ = (previous_log_dir / "obsolete.log").write_text("older process log\n", encoding="utf-8")

            configure_logging(config)
            try:
                system_logger = logging.getLogger("jouzetsu.test_logging_config")
                system_logger.info("system marker")
                system_logger.error("error marker")

                events_logger = chat_logger()
                events_logger.info("chat marker")
                events_logger.error("chat error marker")

                for logger in (logging.getLogger(APPLICATION_LOGGER_NAME), events_logger):
                    for handler in logger.handlers:
                        handler.flush()

                system_log = (log_dir / "system.log").read_text(encoding="utf-8")
                error_log = (log_dir / "error.log").read_text(encoding="utf-8")
                chat_log = (log_dir / "chat.log").read_text(encoding="utf-8")

                self.assertIn("system marker", system_log)
                self.assertIn("error marker", system_log)
                self.assertNotIn("chat marker", system_log)
                self.assertNotIn("previous process log", system_log)

                self.assertIn("error marker", error_log)
                self.assertIn("chat error marker", error_log)
                self.assertNotIn("system marker", error_log)
                self.assertNotIn("previous process log", error_log)

                self.assertIn("chat marker", chat_log)
                self.assertIn("chat error marker", chat_log)
                self.assertNotIn("system marker", chat_log)
                self.assertNotIn("previous process log", chat_log)

                self.assertEqual(
                    (previous_log_dir / "system.log").read_text(encoding="utf-8"),
                    "previous process log\n",
                )
                self.assertEqual(
                    (previous_log_dir / "error.log").read_text(encoding="utf-8"),
                    "previous process log\n",
                )
                self.assertEqual(
                    (previous_log_dir / "chat.log").read_text(encoding="utf-8"),
                    "previous process log\n",
                )
                self.assertFalse((previous_log_dir / "obsolete.log").exists())
            finally:
                configure_logging(AppConfig(logging=LoggingSettings(enabled=False)))

    def test_bootstrap_records_are_written_to_system_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "logs"
            records = configure_bootstrap_logging()
            logging.getLogger("jouzetsu.config").warning("bootstrap configuration warning")

            configure_logging(AppConfig(logging=LoggingSettings(directory=log_dir)), startup_records=records)
            try:
                system_log = (log_dir / "system.log").read_text(encoding="utf-8")
                self.assertIn("bootstrap configuration warning", system_log)
            finally:
                configure_logging(AppConfig(logging=LoggingSettings(enabled=False)))

    def test_root_logger_records_do_not_enter_jouzetsu_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "logs"
            configure_logging(AppConfig(logging=LoggingSettings(directory=log_dir)))
            try:
                external_logger = logging.getLogger("third_party.test")
                external_logger.error("third-party marker")

                for handler in logging.getLogger(APPLICATION_LOGGER_NAME).handlers:
                    handler.flush()

                self.assertNotIn("third-party marker", (log_dir / "system.log").read_text(encoding="utf-8"))
                self.assertNotIn("third-party marker", (log_dir / "error.log").read_text(encoding="utf-8"))
            finally:
                configure_logging(AppConfig(logging=LoggingSettings(enabled=False)))
