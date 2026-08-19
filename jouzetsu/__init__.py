"""Jouzetsu application package.

An async FastHTML web interface for LM Studio.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tomllib import load
from typing import cast


_DISTRIBUTION_NAME = "jouzetsu"


def _source_tree_version() -> str:
    """Read the authoritative project version when running from a source checkout."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        project_data: object = load(pyproject_file).get("project")

    if not isinstance(project_data, dict):
        raise RuntimeError(f"{pyproject_path} must contain a [project] table")

    project = cast(dict[str, object], project_data)
    source_version: object = project.get("version")
    if not isinstance(source_version, str):
        raise RuntimeError(f"{pyproject_path} [project].version must be a string")

    return source_version


try:
    __version__ = version(_DISTRIBUTION_NAME)
except PackageNotFoundError:
    __version__ = _source_tree_version()
