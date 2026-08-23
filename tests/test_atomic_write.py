from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from jouzetsu.atomic_write import atomic_write_text


def test_failed_atomic_write_preserves_destination_and_removes_temporary_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path: Path = Path(tmp_dir) / "config.json"
        _ = path.write_text("original", encoding="utf-8")

        with (
            patch(
                "jouzetsu.atomic_write.os.replace", side_effect=OSError("disk failure")
            ),
            pytest.raises(OSError, match="disk failure"),
        ):
            atomic_write_text(path, "replacement")

        assert path.read_text(encoding="utf-8") == "original"
        assert not list(path.parent.glob("config.json.*.tmp"))


def test_failed_atomic_write_records_destination_path() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path: Path = Path(tmp_dir) / "config.json"

        with (
            unittest.TestCase().assertLogs(
                "jouzetsu.atomic_write", level="ERROR"
            ) as logs,
            patch(
                "jouzetsu.atomic_write.os.replace", side_effect=OSError("disk failure")
            ),
            pytest.raises(OSError, match="disk failure"),
        ):
            atomic_write_text(path, "replacement")

        assert any(
            f"atomic write failed path={path}" in message for message in logs.output
        )
