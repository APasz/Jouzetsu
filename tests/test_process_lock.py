from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from jouzetsu.process_lock import DataDirectoryInUseError, LOCK_FILE_NAME, LinuxDataDirectoryLock


class LinuxDataDirectoryLockTests(unittest.TestCase):
    def test_prevents_a_second_lock_until_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir) / "data"
            first_lock = LinuxDataDirectoryLock.acquire(directory)
            try:
                with self.assertRaises(DataDirectoryInUseError):
                    _ = LinuxDataDirectoryLock.acquire(directory)

                metadata = (directory / LOCK_FILE_NAME).read_text(encoding="utf-8")
                self.assertIn(f"pid={os.getpid()}", metadata)
            finally:
                first_lock.release()

            second_lock = LinuxDataDirectoryLock.acquire(directory)
            second_lock.release()
