"""Record the last deployment that completed installation and readiness checks."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from deploy.configuration import PROJECT_ROOT
from jouzetsu.atomic_write import atomic_write_text

DEPLOYMENT_RECORD: Final[Path] = PROJECT_ROOT / "deployment.json"


def record_deployment(commit: str, path: Path = DEPLOYMENT_RECORD) -> None:
    """Atomically replace the successful-deployment record with a full commit and UTC time."""

    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
        raise ValueError("deployment commit must be a full Git object ID")
    record = {"commit": commit, "deployed_at": datetime.now(UTC).isoformat()}
    atomic_write_text(path, json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("commit", help="full Git commit ID of the verified deployment")
    arguments = parser.parse_args()
    record_deployment(arguments.commit)


if __name__ == "__main__":
    main()
