from pathlib import Path
from tomllib import load
from typing import cast

import jouzetsu


def test_package_version_matches_project_metadata() -> None:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = cast(dict[str, object], load(pyproject_file))

    project_data: object = pyproject.get("project")
    assert isinstance(project_data, dict)
    project = cast(dict[str, object], project_data)
    expected_version: object = project.get("version")
    assert isinstance(expected_version, str)
    assert jouzetsu.__version__ == expected_version
