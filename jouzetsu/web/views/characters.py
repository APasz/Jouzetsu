"""Character library and editor page renderers."""

from __future__ import annotations

from ...character_presets import CharacterFieldPreset, CharacterPresetCatalog
from ...models import (
    CHARACTER_PROMPT_CLOSING,
    CHARACTER_PROMPT_EMPTY_PROFILE,
    CHARACTER_PROMPT_NAME_TEMPLATE,
    CHARACTER_PROMPT_PROFILE_HEADING,
    Character,
    CharacterField,
)
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
from .icons import render_icon


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
    selected_character_name: str = next(
        (
            character.name
            for character in characters
            if character.id == selected_character_id
        ),
        "Choose a character",
    )
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
        Button(
            Span("Characters", cls="jouzetsu-character-library-toggle-label"),
            Span(
                selected_character_name, cls="jouzetsu-character-library-toggle-value"
            ),
            type="button",
            cls="jouzetsu-character-library-toggle",
            aria_controls="character-library-picker",
            aria_expanded="true",
            data_character_library_toggle="true",
        ),
        Div(
            *character_links,
            id="character-library-picker",
            cls="jouzetsu-character-library-list",
            data_testid="character-library",
        ),
        Span(f"{len(characters)} character(s)", cls="jouzetsu-sheet-footnote"),
        cls="jouzetsu-character-library",
        data_character_library="true",
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

    is_onboarding: bool = (
        character.revision == 1
        and character.name == "New character"
        and not character.fields
    )
    fields: tuple[HTML, ...] = tuple(
        _render_character_field(profile_field) for profile_field in character.fields
    )
    return Section(
        Form(
            Div(
                H1(
                    "Create character" if is_onboarding else character.name,
                    cls="jouzetsu-character-title",
                ),
                P(
                    (
                        "Start with a name, choose the details that shape the conversation, "
                        "then make them your own."
                        if is_onboarding
                        else (
                            f"Revision {character.revision} · chats keep the revision they "
                            "started with."
                        )
                    ),
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
            _render_character_preset_controls(
                character, preset_catalog, is_onboarding=is_onboarding
            ),
            Div(
                Div(
                    H2("Profile fields", cls="jouzetsu-section-title"),
                    P(
                        "Add the details that affect the chat. Blank values are not included "
                        "in the prompt.",
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
                    "Add custom field",
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
            Summary(
                Span("Live compiled chat prompt"),
                Small(
                    "Updates as you edit · blank fields are omitted",
                    cls="jouzetsu-character-prompt-summary-help",
                ),
            ),
            Pre(
                character.compiled_system_prompt(),
                cls="jouzetsu-character-prompt-preview",
                data_character_prompt_preview="true",
                data_character_prompt_name_template=CHARACTER_PROMPT_NAME_TEMPLATE,
                data_character_prompt_profile_heading=CHARACTER_PROMPT_PROFILE_HEADING,
                data_character_prompt_empty_profile=CHARACTER_PROMPT_EMPTY_PROFILE,
                data_character_prompt_closing=CHARACTER_PROMPT_CLOSING,
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
    character: Character,
    preset_catalog: CharacterPresetCatalog,
    *,
    is_onboarding: bool,
) -> HTML:
    """Render durable template choices from the shared preset catalogue."""

    base_options: tuple[HTML, ...] = (
        Option(
            "No base template",
            value="",
            selected=not character.presets.base_id,
        ),
        *(
            Option(
                preset.label,
                value=preset.id,
                title=preset.description,
                selected=preset.id == character.presets.base_id,
                data_character_preset_fields=preset.fields_json(),
            )
            for preset in preset_catalog.base_presets
        ),
    )
    controls_id: str = f"character-{character.id}-preset-controls"
    section_title: str = "Quick start" if is_onboarding else "Profile templates"
    help_text: str = (
        "1. Name them above. 2. Choose their form. 3. Pick the details that "
        "will shape the conversations you want to have."
        if is_onboarding
        else (
            "Template choices are remembered. Adding selected fields never "
            "overwrites the profile you have already made."
        )
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
            Div(
                H2(section_title, cls="jouzetsu-section-title"),
                P(help_text, cls="jouzetsu-character-field-help"),
                cls="jouzetsu-character-preset-copy",
            ),
            Button(
                "Hide templates",
                type="button",
                cls="jouzetsu-character-preset-toggle",
                aria_controls=controls_id,
                aria_expanded="true",
                data_character_preset_toggle="true",
            ),
            cls="jouzetsu-character-preset-heading",
        ),
        Div(
            *load_issue_notice,
            field(
                "Character type" if is_onboarding else "Base template",
                Select(
                    *base_options,
                    name="base_preset",
                    data_character_base_preset="true",
                    data_testid="character-base-preset",
                ),
            ),
            Div(
                Span(
                    "What should shape the conversation?"
                    if is_onboarding
                    else "Extra templates",
                    cls="jouzetsu-field-label",
                ),
                Div(
                    *(
                        _render_character_extra_preset(
                            preset,
                            checked=preset.id in character.presets.extra_ids,
                        )
                        for preset in preset_catalog.extra_presets
                    ),
                    cls="jouzetsu-character-extra-presets",
                ),
                cls="jouzetsu-character-preset-extras",
            ),
            Button(
                "Add chosen fields" if is_onboarding else "Add selected fields",
                type="submit",
                cls="jouzetsu-button",
                formaction=f"/characters/{character.id}/presets/apply",
                formmethod="post",
                formnovalidate=True,
                data_character_apply_presets="true",
                data_testid="character-apply-presets",
            ),
            id=controls_id,
            cls="jouzetsu-character-preset-controls",
            data_character_preset_controls="true",
        ),
        cls="jouzetsu-character-preset-editor",
        data_character_preset_editor="true",
    )


def _render_character_extra_preset(
    preset: CharacterFieldPreset, *, checked: bool
) -> HTML:
    return Label(
        Input(
            type="checkbox",
            name="extra_preset",
            value=preset.id,
            checked=checked,
            data_character_extra_preset="true",
            data_character_preset_fields=preset.fields_json(),
        ),
        Span(preset.label, cls="jouzetsu-character-extra-preset-label"),
        Small(preset.description, cls="jouzetsu-character-extra-preset-description"),
        cls="jouzetsu-character-extra-preset",
    )


def _render_character_field(profile_field: CharacterField) -> HTML:
    options_id: str = f"character-field-options-{profile_field.id}"
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
            "Field",
            Input(
                name="field_label",
                value=profile_field.label,
                autocomplete="off",
                required=True,
                data_character_field_label="true",
            ),
            classes="jouzetsu-character-field-name",
        ),
        field("Value", value_control, classes="jouzetsu-character-field-value"),
        Div(
            Button(
                render_icon("more"),
                type="button",
                cls="jouzetsu-button jouzetsu-character-field-menu-toggle",
                title="Field options",
                aria_label=f"Options for {profile_field.label}",
                aria_controls=options_id,
                aria_expanded="false",
                data_character_field_menu_toggle="true",
            ),
            Div(
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
                Button(
                    "Remove field",
                    type="button",
                    cls="jouzetsu-button jouzetsu-button-muted",
                    data_character_remove_field="true",
                ),
                id=options_id,
                cls="jouzetsu-character-field-menu",
                data_character_field_menu="true",
                hidden=True,
            ),
            cls="jouzetsu-character-field-actions",
        ),
        cls="jouzetsu-character-field-row",
        data_character_field="true",
    )
