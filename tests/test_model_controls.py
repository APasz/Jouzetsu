"""Capability badges retain confirmed, unsupported, and unknown model metadata."""

from __future__ import annotations

from dataclasses import replace
from xml.etree import ElementTree as ET

from inference_support import backend as backend_config

from jouzetsu.backend_models import RegisteredModel
from jouzetsu.runtime import ModelCapabilities, ModelDescriptor
from jouzetsu.web.views.model_controls import model_choices, render_model_capabilities


def test_confirmed_capabilities_have_accessible_icons_in_display_order() -> None:
    markup = ET.fromstring(
        str(render_model_capabilities(ModelCapabilities(True, True, True)))
    )
    assert markup.attrib["role"] == "group"
    assert [badge.attrib["aria-label"] for badge in markup] == [
        "Vision supported",
        "Tool use supported",
        "Reasoning supported",
    ]
    assert [
        use.attrib["href"].split("#")[-1] for use in markup.findall("span/svg/use")
    ] == [
        "vision",
        "tool-use",
        "reasoning",
    ]


def test_only_explicitly_supported_capabilities_get_badges() -> None:
    markup = str(render_model_capabilities(ModelCapabilities(False, True, None)))
    assert 'aria-label="Tool use supported"' in markup
    assert "Unknown" not in markup
    assert "#vision" not in markup
    assert "#reasoning" not in markup


def test_basic_and_unknown_models_have_no_capability_markup() -> None:
    assert render_model_capabilities(ModelCapabilities()) is None
    assert render_model_capabilities(ModelCapabilities(False, False, False)) is None


def test_model_choices_keep_capabilities_and_do_not_invent_them_for_missing_models() -> (
    None
):
    capabilities = ModelCapabilities(True, True, True)
    backend = replace(
        backend_config(),
        models=(
            RegisteredModel(
                ModelDescriptor("example", "Example", capabilities=capabilities)
            ),
        ),
    )
    available, missing = model_choices(backend, "missing")
    assert available.capabilities is capabilities
    assert available.detail == ""
    assert missing.capabilities is None
    assert missing.detail == "Missing"
