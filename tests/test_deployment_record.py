from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from deploy.record_deployment import record_deployment


@pytest.mark.parametrize("length", [40, 64])
def test_deployment_record_contains_only_the_full_commit_and_utc_time(tmp_path: Path, length: int) -> None:
    path = tmp_path / "deployment.json"
    commit = "a" * length
    before = datetime.now(UTC)
    record_deployment(commit, path)
    decoded: object = json.loads(path.read_text())
    assert isinstance(decoded, dict)
    record = cast(dict[str, object], decoded)
    assert set(record) == {"commit", "deployed_at"}
    assert record["commit"] == commit
    timestamp = record["deployed_at"]
    assert isinstance(timestamp, str)
    deployed_at = datetime.fromisoformat(timestamp)
    assert deployed_at.utcoffset() == UTC.utcoffset(None)
    assert before <= deployed_at <= datetime.now(UTC)


@pytest.mark.parametrize("commit", ["HEAD", "abc123", "a" * 39, "a" * 41, "g" * 40])
def test_invalid_revision_preserves_the_previous_record(tmp_path: Path, commit: str) -> None:
    path = tmp_path / "deployment.json"
    _ = path.write_text("previous deployment")
    with pytest.raises(ValueError, match="full Git object ID"):
        record_deployment(commit, path)
    assert path.read_text() == "previous deployment"


def test_failed_record_replacement_preserves_the_previous_deployment(tmp_path: Path) -> None:
    path = tmp_path / "deployment.json"
    _ = path.write_text("previous deployment")
    with (
        patch("jouzetsu.atomic_write.os.replace", side_effect=OSError("disk failure")),
        pytest.raises(OSError, match="disk failure"),
    ):
        record_deployment("a" * 40, path)
    assert path.read_text() == "previous deployment"
    assert not list(tmp_path.glob("*.tmp"))
