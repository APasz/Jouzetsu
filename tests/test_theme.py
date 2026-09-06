"""Colour ownership, automatic shades, and contrast regression coverage."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.datastructures import FormData

from jouzetsu.access import AccessDecision
from jouzetsu.colors import contrast_ratio, mix_colors, readable_accent
from jouzetsu.config import (
    AppColorwaySettings,
    AppConfig,
    AppPaths,
    MessageAction,
    ThemeColorway,
    default_config,
)
from jouzetsu.state import AppState
from jouzetsu.web.form_data import (
    FormValues,
    key_visual_from_form,
    updated_theme_settings,
)
from jouzetsu.web.theme import render_theme_css, resolved_theme_colours
from jouzetsu.web.views.context import PageContext
from jouzetsu.web.views.global_settings import render_global_settings_dialog


def _root_colors(config: AppConfig) -> dict[str, str]:
    root = render_theme_css(config).split("}", maxsplit=1)[0]
    return dict(re.findall(r"(--jouzetsu-[\w-]+):\s*([^;]+);", root))


@pytest.mark.parametrize(
    "owner", [ThemeColorway.USER, ThemeColorway.ASSISTANT, ThemeColorway.SYSTEM]
)
def test_role_changes_do_not_recolour_other_roles_or_app(
    tmp_path: Path, owner: ThemeColorway
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    before = _root_colors(config)
    current = dict(config.theme.colorways())[owner]
    config.theme = replace(
        config.theme, **{owner.value: replace(current, accent="#257a61")}
    )

    after = _root_colors(config)

    prefix = f"--jouzetsu-{owner.value}-"
    changed = {name for name in before if before[name] != after[name]}
    assert {
        f"{prefix}{shade}" for shade in ("accent", "muted", "subtle", "hover")
    } <= changed
    assert all(name.startswith(prefix) for name in changed)


@pytest.mark.parametrize("dark_mode", [True, False])
@pytest.mark.parametrize(
    "accent", ["#000000", "#ffffff", "#777777", "#ff3048", "#ad6cff"]
)
def test_automatic_text_remains_readable_for_extreme_main_colours(
    tmp_path: Path, dark_mode: bool, accent: str
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.ui.dark_mode = dark_mode
    config.theme = replace(
        config.theme,
        **{
            owner.value: replace(settings, accent=accent)
            for owner, settings in config.theme.colorways()
        },
    )

    palette = _root_colors(config)

    for owner in ThemeColorway:
        prefix = f"--jouzetsu-{owner.value}-"
        for foreground, background in (
            ("on-accent", "accent"),
            ("on-hover", "hover"),
            ("foreground", "subtle"),
            ("body", "subtle"),
        ):
            assert (
                contrast_ratio(
                    palette[prefix + foreground], palette[prefix + background]
                )
                >= 4.5
            )


def test_light_mode_changes_app_surfaces_and_keeps_role_accents(tmp_path: Path) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    dark = _root_colors(config)
    config.ui.dark_mode = False

    light = _root_colors(config)

    for name in ("canvas", "surface", "surface-raised", "text"):
        assert dark[f"--jouzetsu-app-{name}"] != light[f"--jouzetsu-app-{name}"]
    for owner in ThemeColorway:
        assert (
            dark[f"--jouzetsu-{owner.value}-accent"]
            == light[f"--jouzetsu-{owner.value}-accent"]
        )


def test_default_colourways_use_distinct_role_accents_and_neutral_app_surfaces(
    tmp_path: Path,
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    dark = _root_colors(config)
    config.ui.dark_mode = False
    light = _root_colors(config)

    assert dark["--jouzetsu-user-accent"] == "#c92840"
    assert dark["--jouzetsu-assistant-accent"] == "#2a72c5"
    assert dark["--jouzetsu-system-accent"] == "#187d46"
    assert dark["--jouzetsu-app-accent"] == "#7439b0"
    assert dark["--jouzetsu-app-key-visual"] == "#7439b0"
    assert dark["--jouzetsu-app-message-actions"] == readable_accent(
        "#7439b0", "#000000"
    )
    assert {
        action: dark[f"--jouzetsu-app-{action.color_field.replace('_', '-')}"]
        for action in MessageAction
    } == {
        MessageAction.DELETE: "#e68e97",
        MessageAction.REGENERATE: "#e3a26b",
        MessageAction.MERGE: "#75a7df",
        MessageAction.EDIT: "#8fbe9b",
        MessageAction.CONTINUE: "#b89ae0",
    }
    assert dark["--jouzetsu-app-canvas"] == "#000000"
    assert dark["--jouzetsu-app-surface"] == "#000000"
    assert dark["--jouzetsu-app-surface-raised"] == "#0f0f0f"
    assert dark["--jouzetsu-app-border"] == "#333333"
    assert dark["--jouzetsu-app-border-strong"] == "#595959"
    assert dark["--jouzetsu-app-text"] == "#ffffff"
    assert dark["--jouzetsu-app-text-muted"] == "#ffffff"
    assert light["--jouzetsu-app-canvas"] == "#ffffff"
    assert light["--jouzetsu-app-surface"] == "#ffffff"
    assert light["--jouzetsu-app-surface-raised"] == "#f0f0f0"
    assert light["--jouzetsu-app-border"] == "#cccccc"
    assert light["--jouzetsu-app-border-strong"] == "#a6a6a6"
    assert light["--jouzetsu-app-text"] == "#000000"
    assert light["--jouzetsu-app-text-muted"] == "#000000"
    assert all(
        dark[f"--jouzetsu-{owner.value}-foreground"] == "#ffffff"
        and dark[f"--jouzetsu-{owner.value}-body"] == "#ffffff"
        and light[f"--jouzetsu-{owner.value}-foreground"] == "#000000"
        and light[f"--jouzetsu-{owner.value}-body"] == "#000000"
        for owner in ThemeColorway
    )
    assert all(
        color[1:3] == color[3:5] == color[5:7]
        for palette in (dark, light)
        for name in (
            "canvas",
            "surface",
            "surface-raised",
            "border",
            "border-strong",
            "text",
            "text-muted",
        )
        if (color := palette[f"--jouzetsu-app-{name}"]).startswith("#")
    )


def test_app_key_and_message_action_overrides_are_independent(tmp_path: Path) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.theme = replace(
        config.theme,
        app=replace(
            config.theme.app,
            key_visual="#112233",
            message_actions="#445566",
        ),
    )

    palette = _root_colors(config)

    assert palette["--jouzetsu-app-key-visual"] == "#112233"
    assert palette["--jouzetsu-app-message-actions"] == readable_accent(
        "#445566", "#000000"
    )


@pytest.mark.parametrize("dark_mode", [True, False])
def test_uniform_message_action_colour_remains_readable(
    tmp_path: Path, dark_mode: bool
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.ui.dark_mode = dark_mode

    palette = _root_colors(config)

    assert (
        contrast_ratio(
            palette["--jouzetsu-app-message-actions"],
            palette["--jouzetsu-app-canvas"],
        )
        >= 4.5
    )


def test_semantic_message_action_overrides_are_independent(tmp_path: Path) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.theme = replace(
        config.theme,
        app=replace(
            config.theme.app,
            message_action_delete="#ff3344",
            message_action_merge="#66bbff",
        ),
    )

    palette = _root_colors(config)

    assert palette["--jouzetsu-app-message-action-delete"] == "#ff3344"
    assert palette["--jouzetsu-app-message-action-merge"] == "#66bbff"
    assert palette["--jouzetsu-app-message-action-edit"] == "#8fbe9b"


@pytest.mark.parametrize("dark_mode", [True, False])
def test_semantic_message_action_colours_remain_readable(
    tmp_path: Path, dark_mode: bool
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.ui.dark_mode = dark_mode
    palette = _root_colors(config)
    surface = palette["--jouzetsu-app-canvas"]

    for action in MessageAction:
        colour = palette[f"--jouzetsu-app-{action.color_field.replace('_', '-')}"]
        assert contrast_ratio(colour, surface) >= 4.5


def test_semantic_message_actions_use_the_message_canvas_for_contrast(
    tmp_path: Path,
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.theme = replace(
        config.theme,
        app=replace(
            config.theme.app,
            canvas="#000000",
            surface="#ffffff",
            message_action_delete="#777777",
        ),
    )

    palette = _root_colors(config)

    assert palette["--jouzetsu-app-message-action-delete"] == "#777777"


def test_message_role_text_uses_the_canvas_when_app_surfaces_differ(
    tmp_path: Path,
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.theme = replace(
        config.theme,
        app=replace(config.theme.app, canvas="#000000", surface="#ffffff"),
    )

    palette = _root_colors(config)
    canvas = palette["--jouzetsu-app-canvas"]

    for owner in (ThemeColorway.USER, ThemeColorway.ASSISTANT, ThemeColorway.SYSTEM):
        assert contrast_ratio(palette[f"--jouzetsu-{owner.value}-body"], canvas) >= 4.5


def test_automatic_app_panel_shades_follow_the_panel_background(tmp_path: Path) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.theme = replace(
        config.theme,
        app=replace(config.theme.app, canvas="#000000", surface="#ffffff"),
    )

    palette = _root_colors(config)
    surface = palette["--jouzetsu-app-surface"]
    text = palette["--jouzetsu-app-text"]

    assert palette["--jouzetsu-app-surface-raised"] == mix_colors(
        surface, text, 0.06
    )
    assert palette["--jouzetsu-app-border"] == mix_colors(surface, text, 0.2)


def test_key_visual_picker_preserves_and_restores_main_colour_following() -> None:
    automatic = AppColorwaySettings("#7439b0")

    unchanged_picker = FormValues(FormData({"theme_app_key_visual": "#7439b0"}))
    assert (
        key_visual_from_form(
            automatic,
            unchanged_picker,
            prefix="theme_app",
            main_colour="#112233",
        )
        is None
    )

    separate_picker = FormValues(FormData({"theme_app_key_visual": "#445566"}))
    assert (
        key_visual_from_form(
            automatic,
            separate_picker,
            prefix="theme_app",
            main_colour="#112233",
        )
        == "#445566"
    )

    automatic_picker = FormValues(
        FormData(
            {
                "theme_app_key_visual": "#445566",
                "theme_app_key_visual_automatic": "true",
            }
        )
    )
    assert (
        key_visual_from_form(
            automatic,
            automatic_picker,
            prefix="theme_app",
            main_colour="#112233",
        )
        is None
    )

    separate = replace(automatic, key_visual="#445566")
    matching_picker = FormValues(FormData({"theme_app_key_visual": "#112233"}))
    assert (
        key_visual_from_form(
            separate,
            matching_picker,
            prefix="theme_app",
            main_colour="#112233",
        )
        is None
    )

    explicit_picker = FormValues(
        FormData(
            {
                "theme_app_key_visual": "#7439b0",
                "theme_app_key_visual_automatic_present": "true",
            }
        )
    )
    assert (
        key_visual_from_form(
            automatic,
            explicit_picker,
            prefix="theme_app",
            main_colour="#112233",
        )
        == "#7439b0"
    )


def test_messages_keep_the_canvas_background_and_share_app_actions() -> None:
    styles = (
        Path(__file__).resolve().parents[1]
        / "jouzetsu"
        / "web"
        / "static"
        / "theme"
        / "chat.css"
    ).read_text(encoding="utf-8")

    message_surface = styles.partition(".jouzetsu-message-surface {")[2].partition("}")[
        0
    ]
    message_action = styles.partition(".jouzetsu-message-action {")[2].partition("}")[0]

    assert "background: var(--jouzetsu-app-canvas);" in message_surface
    assert (
        "color: var(--jouzetsu-action-color, "
        "var(--jouzetsu-app-message-actions));" in message_action
    )
    assert "--jouzetsu-message-" not in message_action
    assert ".jouzetsu-message-actions.is-semantic" in styles


def test_placeholder_text_uses_the_app_text_colour() -> None:
    styles = (
        Path(__file__).resolve().parents[1]
        / "jouzetsu"
        / "web"
        / "static"
        / "theme"
        / "foundation.css"
    ).read_text(encoding="utf-8")

    placeholder = styles.partition("::placeholder {")[2].partition("}")[0]

    assert "color: var(--jouzetsu-app-text);" in placeholder
    assert "opacity: 1;" in placeholder


def test_override_stays_fixed_and_clearing_it_restores_automatic_shading(
    tmp_path: Path,
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.theme = replace(
        config.theme, assistant=replace(config.theme.assistant, muted="#123456")
    )
    config.theme = replace(
        config.theme, assistant=replace(config.theme.assistant, accent="#654321")
    )
    assert _root_colors(config)["--jouzetsu-assistant-muted"] == "#123456"

    config.theme = replace(
        config.theme, assistant=replace(config.theme.assistant, muted=None)
    )
    assert _root_colors(config)["--jouzetsu-assistant-muted"] != "#123456"


def test_advanced_picker_auto_toggle_clears_an_override(tmp_path: Path) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.theme = replace(
        config.theme, assistant=replace(config.theme.assistant, muted="#123456")
    )
    state = cast(AppState, SimpleNamespace(config=config))

    selected = updated_theme_settings(
        state, FormValues(FormData({"theme_assistant_muted": "#abcdef"}))
    )
    assert selected.assistant.muted == "#abcdef"

    automatic = updated_theme_settings(
        state,
        FormValues(
            FormData(
                {
                    "theme_assistant_muted": "#abcdef",
                    "theme_assistant_muted_automatic": "true",
                }
            )
        ),
    )
    assert automatic.assistant.muted is None
    assert automatic.assistant.subtle == config.theme.assistant.subtle


def test_resolved_theme_colours_match_the_automatic_picker_preview(
    tmp_path: Path,
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    config.theme = replace(
        config.theme, assistant=replace(config.theme.assistant, muted=None)
    )

    preview = resolved_theme_colours(config)[ThemeColorway.ASSISTANT]["muted"]

    assert preview == _root_colors(config)["--jouzetsu-assistant-muted"]


def test_colour_overrides_render_as_compact_pickers_with_auto_controls(
    tmp_path: Path,
) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    state = cast(AppState, SimpleNamespace(config=config))
    context = PageContext(
        decision=AccessDecision(
            access_allowed=True,
            can_use_global_settings=True,
            can_manage_access=False,
            is_localhost=True,
            device_id="",
            reason="localhost",
        ),
        client_label="Local browser",
        csrf_token="test-token",
    )

    markup = repr(render_global_settings_dialog(state, context, active_tab="appearance"))
    picker = re.search(
        r'<input(?=[^>]*data-testid="theme-assistant-muted-override")[^>]*>',
        markup,
    )
    automatic = re.search(
        r'<input(?=[^>]*data-testid="theme-assistant-muted-automatic")[^>]*>',
        markup,
    )
    key_visual_picker = re.search(
        r'<input(?=[^>]*data-testid="theme-app-key-visual-color")[^>]*>',
        markup,
    )
    key_visual_automatic = re.search(
        r'<input(?=[^>]*data-testid="theme-app-key-visual-color-automatic")[^>]*>',
        markup,
    )

    assert picker is not None
    assert 'type="color"' in picker.group()
    assert (
        f'value="{resolved_theme_colours(config)[ThemeColorway.ASSISTANT]["muted"]}"'
        in picker.group()
    )
    assert 'data-colour-override-picker="true"' in picker.group()
    assert automatic is not None
    assert 'type="checkbox"' in automatic.group()
    assert 'checked' in automatic.group()
    assert 'data-colour-override-automatic="true"' in automatic.group()
    assert key_visual_picker is not None
    assert 'type="color"' in key_visual_picker.group()
    assert 'data-colour-override-picker="true"' in key_visual_picker.group()
    assert key_visual_automatic is not None
    assert 'checked' in key_visual_automatic.group()
    assert 'name="theme_app_key_visual_automatic_present"' in markup


def test_styles_reference_defined_theme_variables(tmp_path: Path) -> None:
    config = default_config(AppPaths.for_home(tmp_path))
    directory = (
        Path(__file__).resolve().parents[1] / "jouzetsu" / "web" / "static" / "theme"
    )
    static = "\n".join(
        path.read_text(encoding="utf-8") for path in directory.glob("*.css")
    )
    styles = static + render_theme_css(config)
    declarations = set(re.findall(r"(--jouzetsu-[\w-]+)\s*:", styles))
    references_without_fallback = set(re.findall(r"var\((--jouzetsu-[\w-]+)\)", styles))

    assert references_without_fallback <= declarations
    assert not re.search(
        r"--jouzetsu-(?:primary|secondary|edit|action-delete)\b", static
    )
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", static)
