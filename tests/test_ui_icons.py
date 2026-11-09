"""Validate the packaged sprite used by server and browser-created controls."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from hashlib import sha256
from pathlib import Path
from typing import Final, cast, get_args

import pytest
from fastcore.xml import (  # pyright: ignore[reportMissingTypeStubs]
    to_xml,  # pyright: ignore[reportUnknownVariableType]
)

from jouzetsu.web.views.icons import (
    ICON_VIEW_BOX,
    IconName,
    icon_sprite_url,
    render_icon,
)

_ICON_DIRECTORY: Final[Path] = Path(__file__).parents[1] / "jouzetsu/web/static/icons"
_SVG_NAMESPACE: Final[str] = "{http://www.w3.org/2000/svg}"


def test_packaged_sprite_covers_the_interface_without_external_dependencies() -> None:
    sprite = ET.parse(_ICON_DIRECTORY / "phosphor.svg").getroot()
    symbols = sprite.findall(f"{_SVG_NAMESPACE}symbol")
    names = [symbol.attrib["id"] for symbol in symbols]
    assert len(names) == len(set(names))
    assert set(names) == set(get_args(IconName.__value__))
    assert sprite.attrib["fill"] == "currentColor"
    for symbol in symbols:
        assert symbol.attrib["viewBox"] == ICON_VIEW_BOX
        assert symbol.attrib["data-phosphor-name"]
        assert symbol.attrib["data-weight"] in {"regular", "bold"}
        assert len(symbol)
        for shape in symbol:
            assert shape.tag == f"{_SVG_NAMESPACE}path"
            assert set(shape.attrib) == {"d"}
            assert shape.attrib["d"]
    assert (
        "Permission is hereby granted"
        in (_ICON_DIRECTORY / "PHOSPHOR-LICENSE.txt").read_text()
    )


@pytest.mark.parametrize(
    "name", cast(tuple[IconName, ...], get_args(IconName.__value__))
)
def test_rendered_icons_reference_the_versioned_local_sprite(name: IconName) -> None:
    sprite = (_ICON_DIRECTORY / "phosphor.svg").read_bytes()
    expected_url = f"/static/icons/phosphor.svg?v={sha256(sprite).hexdigest()[:12]}"
    assert icon_sprite_url() == expected_url
    icon = ET.fromstring(to_xml(render_icon(name)))
    assert icon.tag == "svg"
    assert icon.attrib["viewbox"] == ICON_VIEW_BOX
    assert icon.attrib["fill"] == "currentColor"
    assert icon.attrib["aria-hidden"] == "true"
    assert icon.attrib["focusable"] == "false"
    use = icon.find("use")
    assert use is not None
    assert use.attrib["href"] == f"{expected_url}#{name}"


def test_armed_continue_has_distinct_regular_and_native_bold_artwork() -> None:
    icon = ET.fromstring(to_xml(render_icon("play", armed_name="play-bold")))
    uses = icon.findall("use")
    assert [use.attrib["href"].split("#")[1] for use in uses] == ["play", "play-bold"]
    sprite = ET.parse(_ICON_DIRECTORY / "phosphor.svg").getroot()
    regular = sprite.find(f"{_SVG_NAMESPACE}symbol[@id='play']")
    bold = sprite.find(f"{_SVG_NAMESPACE}symbol[@id='play-bold']")
    assert regular is not None and bold is not None
    assert bold.attrib["data-weight"] == "bold"
    assert regular[0].attrib["d"] != bold[0].attrib["d"]
