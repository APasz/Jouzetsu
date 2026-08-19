from pathlib import Path
from unittest.mock import patch

from jouzetsu.__main__ import main


def test_console_entry_point_starts_the_application() -> None:
    with patch("jouzetsu.__main__.run") as run:
        main([])

    run.assert_called_once_with()


def test_console_entry_point_accepts_an_explicit_application_home() -> None:
    home = Path("/tmp/jouzetsu-home")

    with patch("jouzetsu.__main__.run") as run:
        main(["--home", str(home)])

    run.assert_called_once_with(app_home=home)
