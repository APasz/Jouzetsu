"""Safe CommonMark rendering for chat messages."""

from __future__ import annotations

from typing import Final, cast

from fastcore.xml import Safe  # pyright: ignore[reportMissingTypeStubs]
from markdown_it import MarkdownIt

_RENDERER: Final[MarkdownIt] = MarkdownIt(
    "commonmark",
    {
        "breaks": True,
        "html": False,
        "linkify": False,
        "typographer": False,
    },
)


def render_markdown(content: str) -> Safe:
    """Render CommonMark without trusting HTML supplied by a message author."""
    rendered: str = cast(str, _RENDERER.render(content))
    return Safe(rendered)
