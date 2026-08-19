"""Character library and editor page renderers."""

from __future__ import annotations

from ...character_presets import CharacterFieldPreset, CharacterPresetCatalog
from ...models import Character, CharacterField
from ...state import AppState
from ..html import (
    H1,
    H2,
    HTML,
    A,
    Aside,
    Button,
    Details,
    Div,
    Form,
    Header,
    Img,
    Input,
    Label,
    Main,
    Option,
    P,
    Pre,
    Section,
    Select,
    Small,
    Span,
    Summary,
    Textarea,
)
from ..icon import app_icon_url
from ..security import CSRF_FORM_FIELD
from .context import PageContext
from .controls import csrf_context, field
from .feedback import render_notice_area


def render_character_page(
    state: AppState,
    context: PageContext,
    preset_catalog: CharacterPresetCatalog,
    *,
    selected_character_id: str = "",
    notice: str = "",
    error: str = "",
) -> HTML:
    """Render the reusable-character workspace and, optionally, one editor."""

    characters: list[Character] = state.characters
    selected_character: Character | None = next(
        (
            character
            for character in characters
            if character.id == selected_character_id
        ),
        None,
    )
    editor: HTML = (
        _render_character_editor(selected_character, context, preset_catalog)
        if selected_character is not None
        else _character_empty_state()
    )
    return Div(
        csrf_context(context),
        render_notice_area(notice=notice, error=error),
        _character_workspace_header(state),
        Div(
            _character_library(characters, selected_character_id),
            Main(editor, cls="jouzetsu-character-main"),
            cls="jouzetsu-character-layout",
        ),
        cls="jouzetsu-app jouzetsu-character-app",
        data_character_workspace="true",
    )


def _character_workspace_header(state: AppState) -> HTML:
    return Header(
        A(
            Img(
                src=app_icon_url(state.config.ui.icon_colors),
                alt="",
                cls="jouzetsu-workspace-brand-icon",
            ),
            Span("Jouzetsu", cls="jouzetsu-workspace-brand"),
            href="/",
            cls="jouzetsu-workspace-brand-lockup",
        ),
        Div(
            A("Chats", href="/chats", cls="jouzetsu-workspace-link"),
            A(
                "Characters",
                href="/characters",
                cls="jouzetsu-workspace-link is-active",
            ),
            cls="jouzetsu-workspace-nav",
        ),
        cls="jouzetsu-workspace-topbar",
    )


def _character_library(characters: list[Character], selected_character_id: str) -> HTML:
    character_links: tuple[HTML, ...] = tuple(
        A(
            character.name,
            href=f"/characters/{character.id}",
            cls=(
                "jouzetsu-character-library-item is-active"
                if character.id == selected_character_id
                else "jouzetsu-character-library-item"
            ),
            data_testid=f"character-library-item-{character.id}",
        )
        for character in characters
    )
    return Aside(
        Div(
            H2("Characters", cls="jouzetsu-panel-title"),
            Form(
                Button(
                    "New character",
                    type="submit",
                    cls="jouzetsu-button jouzetsu-button-primary",
                ),
                action="/characters/new",
                method="post",
            ),
            cls="jouzetsu-character-library-header",
        ),
        Div(
            *character_links,
            cls="jouzetsu-character-library-list",
            data_testid="character-library",
        ),
        Span(f"{len(characters)} character(s)", cls="jouzetsu-sheet-footnote"),
        cls="jouzetsu-character-library",
    )


def _character_empty_state() -> HTML:
    return Section(
        H1("Character library", cls="jouzetsu-character-title"),
        P(
            "Create a character, then add only the profile fields that matter for it.",
            cls="jouzetsu-character-intro",
        ),
        cls="jouzetsu-character-empty-state",
    )


def _render_character_editor(
    character: Character,
    context: PageContext,
    preset_catalog: CharacterPresetCatalog,
) -> HTML:
    """Render profile metadata from the character's own field definitions."""

    fields: tuple[HTML, ...] = tuple(
        _render_character_field(profile_field) for profile_field in character.fields
    )
    return Section(
        Form(
            Div(
                H1(character.name, cls="jouzetsu-character-title"),
                P(
                    f"Revision {character.revision} · chats keep the revision they started with.",
                    cls="jouzetsu-character-intro",
                ),
                cls="jouzetsu-character-editor-heading",
            ),
            Input(type="hidden", name="revision", value=character.revision),
            Input(type="hidden", name=CSRF_FORM_FIELD, value=context.csrf_token),
            field(
                "Name",
                Input(
                    name="name",
                    value=character.name,
                    autocomplete="off",
                    required=True,
                    data_testid="character-name-input",
                ),
            ),
            _render_character_preset_controls(character, preset_catalog),
            Div(
                Div(
                    H2("Profile fields", cls="jouzetsu-section-title"),
                    P(
                        "Add, remove, and choose the input style for the details this character actually needs.",
                        cls="jouzetsu-character-field-help",
                    ),
                    cls="jouzetsu-character-fields-heading",
                ),
                Div(
                    *fields,
                    cls="jouzetsu-character-fields",
                    data_character_fields="true",
                ),
                Button(
                    "Add field",
                    type="submit",
                    cls="jouzetsu-button",
                    formaction=f"/characters/{character.id}/fields/add",
                    formmethod="post",
                    formnovalidate=True,
                    data_character_add_field="true",
                ),
                cls="jouzetsu-character-fields-editor",
                data_character_fields_editor="true",
            ),
            Div(
                Button(
                    "Save character",
                    type="submit",
                    cls="jouzetsu-button jouzetsu-button-primary",
                ),
                Button(
                    "Save and start chat",
                    type="submit",
                    cls="jouzetsu-button",
                    formaction=f"/characters/{character.id}/chat",
                    formmethod="post",
                ),
                cls="jouzetsu-character-editor-actions",
            ),
            action=f"/characters/{character.id}",
            method="post",
            cls="jouzetsu-character-editor-form",
            data_character_editor_form="true",
            data_testid="character-editor-form",
        ),
        Details(
            Summary("Compiled chat prompt"),
            Pre(
                character.compiled_system_prompt(),
                cls="jouzetsu-character-prompt-preview",
            ),
            cls="jouzetsu-character-prompt-details",
        ),
        Form(
            Button(
                "Delete character",
                type="submit",
                cls="jouzetsu-button jouzetsu-button-danger",
                data_confirm=f"Delete {character.name} permanently? Existing chats will be retained.",
            ),
            action=f"/characters/{character.id}/delete",
            method="post",
            cls="jouzetsu-character-delete-form",
        ),
        cls="jouzetsu-character-editor",
    )


def _render_character_preset_controls(
    character: Character, preset_catalog: CharacterPresetCatalog
) -> HTML:
    """Render template choices from the shared preset catalogue."""

    base_options: tuple[HTML, ...] = (
        Option("No base preset", value=""),
        *(
            Option(
                preset.label,
                value=preset.id,
                title=preset.description,
                data_character_preset_fields=preset.fields_json(),
            )
            for preset in preset_catalog.base_presets
        ),
    )
    load_issue_notice: tuple[HTML, ...] = (
        (
            Details(
                Summary("Some custom preset packs could not be loaded"),
                *(
                    P(
                        f"{issue.path.name}: {issue.message}",
                        cls="jouzetsu-character-field-help",
                    )
                    for issue in preset_catalog.load_issues
                ),
                cls="jouzetsu-character-preset-load-issues",
            ),
        )
        if preset_catalog.load_issues
        else ()
    )
    return Div(
        Div(
            H2("Field presets", cls="jouzetsu-section-title"),
            P(
                "Choose one compatible base and any extra templates, then apply them as editable fields.",
                cls="jouzetsu-character-field-help",
            ),
            cls="jouzetsu-character-fields-heading",
        ),
        *load_issue_notice,
        field(
            "Base preset",
            Select(
                *base_options,
                name="base_preset",
                data_character_base_preset="true",
                data_testid="character-base-preset",
            ),
        ),
        Div(
            Span("Extra presets", cls="jouzetsu-field-label"),
            Div(
                *(
                    _render_character_extra_preset(preset)
                    for preset in preset_catalog.extra_presets
                ),
                cls="jouzetsu-character-extra-presets",
            ),
            cls="jouzetsu-character-preset-extras",
        ),
        Button(
            "Apply selected presets",
            type="submit",
            cls="jouzetsu-button",
            formaction=f"/characters/{character.id}/presets/apply",
            formmethod="post",
            formnovalidate=True,
            data_character_apply_presets="true",
            data_testid="character-apply-presets",
        ),
        cls="jouzetsu-character-preset-editor",
        data_character_preset_editor="true",
    )


def _render_character_extra_preset(preset: CharacterFieldPreset) -> HTML:
    return Label(
        Input(
            type="checkbox",
            name="extra_preset",
            value=preset.id,
            data_character_extra_preset="true",
            data_character_preset_fields=preset.fields_json(),
        ),
        Span(preset.label, cls="jouzetsu-character-extra-preset-label"),
        Small(preset.description, cls="jouzetsu-character-extra-preset-description"),
        cls="jouzetsu-character-extra-preset",
    )


def _render_character_field(profile_field: CharacterField) -> HTML:
    value_control: HTML = (
        Textarea(
            profile_field.value,
            name="field_value",
            rows=5,
            aria_label=f"{profile_field.label} value",
            data_character_field_value="true",
        )
        if profile_field.kind == "long_text"
        else Input(
            name="field_value",
            value=profile_field.value,
            autocomplete="off",
            aria_label=f"{profile_field.label} value",
            data_character_field_value="true",
        )
    )
    return Div(
        Input(type="hidden", name="field_id", value=profile_field.id),
        field(
            "Label",
            Input(
                name="field_label",
                value=profile_field.label,
                autocomplete="off",
                required=True,
            ),
        ),
        field(
            "Input style",
            Select(
                Option(
                    "Short text",
                    value="short_text",
                    selected=profile_field.kind == "short_text",
                ),
                Option(
                    "Long text",
                    value="long_text",
                    selected=profile_field.kind == "long_text",
                ),
                name="field_kind",
                aria_label=f"{profile_field.label} input style",
            ),
        ),
        field("Value", value_control),
        Button(
            "Remove field",
            type="button",
            cls="jouzetsu-button jouzetsu-button-muted",
            data_character_remove_field="true",
        ),
        cls="jouzetsu-character-field-row",
        data_character_field="true",
    )
