from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jouzetsu.log_files import discover_log_files, read_log_tail


class LogFileTests(unittest.TestCase):
    def test_discovers_regular_files_in_name_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            _ = (directory / "chat.log").touch()
            _ = (directory / "error.log").touch()
            (directory / "archive").mkdir()

            self.assertEqual(
                discover_log_files(directory),
                (directory / "chat.log", directory / "error.log"),
            )

    def test_reads_a_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "system.log"
            _ = log_file.write_text("0123456789", encoding="utf-8")

            self.assertEqual(read_log_tail(log_file, max_bytes=4), "6789")
