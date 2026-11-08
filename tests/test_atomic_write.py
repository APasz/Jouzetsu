from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest

from jouzetsu.atomic_write import (
    CommitUncertainError,
    DurableWriteGuard,
    atomic_write_text,
    ensure_directory,
    move_path,
    remove_path,
)


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


def test_file_sync_failure_preserves_previous_document(tmp_path: Path) -> None:
    path = tmp_path / "document.json"
    _ = path.write_bytes(b"original")
    sync = os.fsync

    def fail_file_sync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("file sync failed")
        sync(descriptor)

    with (
        patch("jouzetsu.atomic_write.os.fsync", side_effect=fail_file_sync),
        pytest.raises(OSError, match="file sync failed"),
    ):
        atomic_write_text(path, "replacement")

    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp"))


def test_sync_order_confirms_file_before_replace_and_directory_after(
    tmp_path: Path,
) -> None:
    events: list[Literal["file", "directory", "replace"]] = []
    sync = os.fsync
    replace = os.replace

    def record_sync(descriptor: int) -> None:
        events.append(
            "file" if stat.S_ISREG(os.fstat(descriptor).st_mode) else "directory"
        )
        sync(descriptor)

    def record_replace(source: str, destination: Path) -> None:
        events.append("replace")
        replace(source, destination)

    with (
        patch("jouzetsu.atomic_write.os.fsync", side_effect=record_sync),
        patch("jouzetsu.atomic_write.os.replace", side_effect=record_replace),
    ):
        atomic_write_text(tmp_path / "document.json", "saved")

    assert events == ["file", "replace", "directory"]


def test_existing_directory_needs_no_parent_sync(tmp_path: Path) -> None:
    with patch("jouzetsu.atomic_write._sync_directory") as sync:
        ensure_directory(tmp_path)
    sync.assert_not_called()


def test_each_created_directory_syncs_its_parent_once(tmp_path: Path) -> None:
    directory = tmp_path / "one" / "two"
    synced: list[Path] = []
    with patch("jouzetsu.atomic_write._sync_directory", side_effect=synced.append):
        ensure_directory(directory)
    assert directory.is_dir()
    assert synced == [tmp_path, tmp_path / "one"]


def test_new_directory_sync_failure_is_an_uncertain_mutation(tmp_path: Path) -> None:
    directory = tmp_path / "new"
    with (
        patch(
            "jouzetsu.atomic_write._sync_directory", side_effect=OSError("sync failed")
        ),
        pytest.raises(CommitUncertainError),
    ):
        ensure_directory(directory)
    assert directory.is_dir()


def test_new_directory_then_file_syncs_in_durable_order(tmp_path: Path) -> None:
    directory = tmp_path / "new"
    events: list[str | Path] = []
    replace = os.replace

    def record_file_sync(_descriptor: int) -> None:
        events.append("file")

    def record_replace(source: str, destination: Path) -> None:
        events.append("replace")
        replace(source, destination)

    with (
        patch("jouzetsu.atomic_write._sync_directory", side_effect=events.append),
        patch(
            "jouzetsu.atomic_write._sync_file",
            side_effect=record_file_sync,
        ),
        patch("jouzetsu.atomic_write.os.replace", side_effect=record_replace),
    ):
        atomic_write_text(directory / "document.json", "saved")
    assert events == [tmp_path, "file", "replace", directory]


def test_move_syncs_both_directories_and_remove_syncs_its_parent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    destination_directory = tmp_path / "trash"
    destination_directory.mkdir()
    destination = destination_directory / source.name
    _ = source.write_text("saved")
    synced: list[Path] = []
    with patch("jouzetsu.atomic_write._sync_directory", side_effect=synced.append):
        move_path(source, destination)
        remove_path(destination)
    assert synced == [destination_directory, tmp_path, destination_directory]


def test_directory_sync_failure_reports_uncertain_commit_and_blocks_replay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "document.json"
    other_path = tmp_path / "other.json"
    _ = path.write_text("original")
    guard = DurableWriteGuard()

    with (
        patch(
            "jouzetsu.atomic_write._confirm_directories",
            side_effect=CommitUncertainError("sync failed"),
        ),
        pytest.raises(CommitUncertainError, match="sync failed"),
        guard.protect(path),
    ):
        atomic_write_text(path, "replacement")

    assert path.read_text() == "replacement"
    assert not list(tmp_path.glob("*.tmp"))
    with (
        pytest.raises(CommitUncertainError, match="requires recovery"),
        guard.protect(path),
    ):
        atomic_write_text(path, "replayed")
    with guard.protect(other_path):
        atomic_write_text(other_path, "unrelated write")

    assert path.read_text() == "replacement"
    assert other_path.read_text() == "unrelated write"


@pytest.mark.parametrize("operation", ("write", "move", "remove"))
def test_post_change_sync_failure_is_never_reported_as_a_safe_failure(
    tmp_path: Path, operation: Literal["write", "move", "remove"]
) -> None:
    path = tmp_path / "document.json"
    _ = path.write_text("original")

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory sync failed")

    # Directory creation/preflight completes before the injected post-change fault.
    with (
        patch("jouzetsu.atomic_write.ensure_directory"),
        patch("jouzetsu.atomic_write.os.fsync", side_effect=fail_directory_sync),
        pytest.raises(CommitUncertainError) as error,
    ):
        if operation == "write":
            atomic_write_text(path, "replacement")
        elif operation == "move":
            move_path(path, tmp_path / "moved.json")
        else:
            remove_path(path)

    assert isinstance(error.value.__cause__, OSError)
