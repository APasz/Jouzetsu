from __future__ import annotations

from pathlib import Path

from jouzetsu.config import IconColorSettings
from jouzetsu.web.icon import (
    FALLBACK_ICON_FILENAME,
    ICON_FILENAME,
    configured_icon_svg,
    icon_artwork_path,
)


def test_icon_artwork_path_uses_release_safe_fallback_when_private_icon_is_missing(
    tmp_path: Path,
) -> None:
    assert icon_artwork_path(tmp_path) == tmp_path / FALLBACK_ICON_FILENAME


def test_icon_artwork_path_prefers_local_private_icon(tmp_path: Path) -> None:
    private_icon: Path = tmp_path / ICON_FILENAME
    private_icon.touch()

    assert icon_artwork_path(tmp_path) == private_icon


def test_fallback_icon_uses_the_configured_palette() -> None:
    asset_directory: Path = Path(__file__).parents[1] / "jouzetsu" / "web" / "static"
    svg: str = configured_icon_svg(
        asset_directory / FALLBACK_ICON_FILENAME,
        IconColorSettings(
            linework_color="#112233", accent_color="#445566", surface_color="#778899"
        ),
    )

    assert 'stroke="#112233"' in svg
    assert 'fill="#445566"' in svg
    assert 'fill="#778899"' in svg
