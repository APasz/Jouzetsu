"""`python -m jouzetsu` entry point."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .app import run


def main(argv: Sequence[str] | None = None) -> None:
    """Start Jouzetsu from a module or installed console command."""

    parser = argparse.ArgumentParser(description="Run the Jouzetsu chat interface.")
    _ = parser.add_argument(
        "--home",
        type=Path,
        metavar="DIRECTORY",
        help="store config.json and data/ in DIRECTORY (or set JOUZETSU_HOME)",
    )
    arguments = parser.parse_args(argv)
    if arguments.home is None:
        run()
    else:
        run(app_home=arguments.home)


if __name__ == "__main__":
    main()
