"""Global application-settings dialog rendering."""

from __future__ import annotations

from dataclasses import fields
from typing import Final, Literal, cast

from ...config import (
    AppColorwaySettings,
    ColorwaySettings,
    GenerationSettings,
    MessageAction,
    MessageActionStyle,
    SpellingReplacement,
    ThemeColorway,
    builtin_message_action_color,
)
from ...state import AppState
from ..html import (
    H2,
    H3,
    HTML,
    Button,
    Details,
    Dialog,
    Div,
    Form,
    Input,
    Label,
    Option,
    P,
    Section,
    Select,
    Small,
    Span,
    Summary,
    Textarea,
)
from ..theme import resolved_theme_colours
from .context import PageContext
from .controls import (
    checkbox,
    close_dialog_button,
    field,
    form_submit_actions,
    post_button,
    reset_form_button,
)
from .feedback import dialog_feedback
from .tabs import DialogTab, render_dialog_tab_list, render_dialog_tab_panel

type GlobalSettingsTab = Literal["behaviour", "appearance"]

_BEHAVIOUR_TAB: Final[DialogTab] = DialogTab("behaviour", "Behaviour")
_APPEARANCE_TAB: Final[DialogTab] = DialogTab("appearance", "Appearance")
_GLOBAL_SETTINGS_TABS: Final[tuple[DialogTab, ...]] = (
    _BEHAVIOUR_TAB,
    _APPEARANCE_TAB,
)
_COLOURWAY_DESCRIPTIONS: Final[dict[ThemeColorway, str]] = {
    ThemeColorway.APP: "Neutral surfaces and text, plus the shared accent for controls and key interface visuals.",
    ThemeColorway.USER: "Your messages, composer accents, and send button.",
    ThemeColorway.ASSISTANT: "Assistant replies, reasoning, and generation highlights.",
    ThemeColorway.SYSTEM: "System messages, notifications, and inline feedback.",
}
_APP_EDITOR_EXCLUDED_FIELDS: Final[frozenset[str]] = frozenset(
    {"accent", "key_visual", *(action.color_field for action in MessageAction)}
)


def render_global_settings_dialog(
    state: AppState,
    context: PageContext,
    *,
    active_tab: GlobalSettingsTab = "behaviour",
    notice: str = "",
    error: str = "",
) -> HTML:
    """Render application-wide behaviour and appearance defaults."""

    return Dialog(
        Div(
            H2(
                "Global settings",
                id="global-settings-dialog-title",
                cls="jouzetsu-modal-title",
            ),
            P(
                "Set chat-wide behaviour and the application's appearance.",
                id="global-settings-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            dialog_feedback(notice=notice, error=error),
            render_dialog_tab_list(
                "global-settings-dialog",
                _GLOBAL_SETTINGS_TABS,
                active_tab,
                label="Global settings sections",
            ),
            render_dialog_tab_panel(
                "global-settings-dialog",
                _BEHAVIOUR_TAB,
                active_tab,
                _system_prompt_form(state),
                _generation_defaults_form(state),
            ),
            render_dialog_tab_panel(
                "global-settings-dialog",
                _APPEARANCE_TAB,
                active_tab,
                _appearance_form(state),
            ),
            Div(
                close_dialog_button("global-settings-dialog"),
                cls="jouzetsu-dialog-actions",
            ),
            (
                post_button(
                    "Exit",
                    action="/app/exit",
                    marker="exit-process-button",
                    classes="jouzetsu-button jouzetsu-button-danger",
                    confirm="Stop Jouzetsu?",
                )
                if context.decision.can_manage_access
                else None
            ),
            cls="jouzetsu-modal-card jouzetsu-global-settings-card",
        ),
        id="global-settings-dialog",
        cls="jouzetsu-dialog",
        aria_labelledby="global-settings-dialog-title",
        aria_describedby="global-settings-dialog-description",
    )


def global_settings_tab_from_text(value: str) -> GlobalSettingsTab:
    """Return a supported Global Settings tab, defaulting to behaviour."""

    return "appearance" if value == "appearance" else "behaviour"


def _system_prompt_form(state: AppState) -> HTML:
    return Form(
        Section(
            H3("System prompt", cls="jouzetsu-section-title"),
            field(
                "Used when a chat has no override",
                Textarea(
                    state.config.generation.system_prompt,
                    name="system_prompt",
                    rows=6,
                    placeholder="Set the default assistant behaviour…",
                    data_testid="global-prompt",
                ),
            ),
            cls="jouzetsu-dialog-section",
        ),
        form_submit_actions(
            "Save system prompt",
            "save-global-prompt-button",
            reset_action="/settings/global/prompt/reset",
            reset_marker="reset-global-prompt-button",
            reset_confirmation="Reset the global system prompt to its default?",
        ),
        action="/settings/global/prompt",
        method="post",
        cls="jouzetsu-global-settings-form",
    )


def _appearance_form(state: AppState) -> HTML:
    """Present four main colours with optional overrides grouped by ownership."""

    colorways = state.config.theme.colorways()
    effective_colours = resolved_theme_colours(state.config)
    return Div(
        Form(
            Small(
                "Choose a main colour for each part of the interface. Automatic shades preserve readable text and keep App surfaces neutral.",
                cls="jouzetsu-form-help",
            ),
            *(
                _colourway_editor(
                    state,
                    owner,
                    settings,
                    effective_colours=effective_colours[owner],
                    reset_form_id=_colourway_reset_form_id(owner),
                )
                for owner, settings in colorways
            ),
            form_submit_actions(
                "Save appearance",
                "save-global-appearance-button",
                reset_action="/settings/global/appearance/reset",
                reset_marker="reset-global-appearance-button",
                reset_confirmation="Reset all appearance settings to their defaults?",
            ),
            action="/settings/global/appearance",
            method="post",
            cls="jouzetsu-global-settings-form",
        ),
        *(_colourway_reset_form(owner) for owner, _settings in colorways),
    )


def _colourway_editor(
    state: AppState,
    owner: ThemeColorway,
    settings: ColorwaySettings,
    *,
    effective_colours: dict[str, str],
    reset_form_id: str,
) -> HTML:
    is_app: bool = isinstance(settings, AppColorwaySettings)
    return Section(
        H3(owner.label, cls="jouzetsu-appearance-group-title"),
        Small(_COLOURWAY_DESCRIPTIONS[owner], cls="jouzetsu-form-help"),
        Div(
            field(
                "Main colour",
                Input(
                    value=settings.accent,
                    name=f"theme_{owner.value}_accent",
                    type="color",
                    data_testid=f"theme-{owner.value}-accent-color",
                ),
            ),
            (
                _key_visual_field(
                    settings, preview=effective_colours["key_visual"]
                )
                if is_app
                else None
            ),
            cls="jouzetsu-form-grid is-two-columns" if is_app else None,
        ),
        (
            Small(
                "Key visuals cover navigation, active tabs, and message actions. "
                "Use Auto to keep them following Main colour.",
                cls="jouzetsu-form-help",
            )
            if is_app
            else None
        ),
        (
            checkbox(
                "dark_mode",
                checked=state.config.ui.dark_mode,
                label="Dark appearance",
                marker="appearance-dark-mode",
            )
            if owner is ThemeColorway.APP
            else None
        ),
        Details(
            Summary("Advanced", cls="jouzetsu-appearance-advanced-title"),
            Small(
                "Pick an override colour, or use Auto to restore the generated colour. "
                "Uniform message actions follow the main App colour unless you "
                "override them.",
                cls="jouzetsu-form-help",
            ),
            Div(
                *(
                    _colour_override_field(
                        label=cast(str, item.metadata["label"]),
                        name=f"theme_{owner.value}_{item.name}",
                        value=cast(str | None, getattr(settings, item.name)),
                        preview=effective_colours[item.name],
                        marker=f"theme-{owner.value}-{item.name.replace('_', '-')}-override",
                    )
                    for item in fields(settings)
                    if item.name not in _APP_EDITOR_EXCLUDED_FIELDS
                ),
                cls="jouzetsu-form-grid is-two-columns",
            ),
            *(_app_appearance_controls(state) if owner is ThemeColorway.APP else ()),
            cls="jouzetsu-appearance-advanced",
        ),
        Div(
            reset_form_button(
                action=f"/settings/global/appearance/{owner.value}/reset",
                marker=f"reset-theme-{owner.value}-button",
                label=f"Reset {owner.label} colours",
                confirmation=(
                    f"Reset the {owner.label} colourway to its defaults? "
                    "Any unsaved appearance changes will be discarded."
                ),
                form_id=reset_form_id,
            ),
            cls="jouzetsu-dialog-actions",
        ),
        cls="jouzetsu-appearance-group",
        data_testid=f"theme-{owner.value}-colourway",
    )


def _colour_override_field(
    *,
    label: str,
    name: str,
    value: str | None,
    preview: str,
    marker: str,
) -> HTML:
    """Render one compact override picker with an explicit automatic state."""

    automatic: bool = value is None
    automatic_marker: str = f"{marker.removesuffix('-override')}-automatic"
    return Div(
        Span(label, cls="jouzetsu-field-label"),
        Div(
            Input(
                value=value or preview,
                name=name,
                type="color",
                aria_label=f"Select {label.lower()}",
                data_colour_override_picker="true",
                data_testid=marker,
            ),
            Input(
                type="hidden",
                name=f"{name}_automatic_present",
                value="true",
            ),
            Label(
                Input(
                    type="checkbox",
                    name=f"{name}_automatic",
                    value="true",
                    checked=automatic,
                    aria_label=f"Use automatic {label.lower()}",
                    data_colour_override_automatic="true",
                    data_testid=automatic_marker,
                ),
                Span("Auto"),
                cls="jouzetsu-colour-override-automatic",
            ),
            cls=(
                "jouzetsu-colour-override is-automatic"
                if automatic
                else "jouzetsu-colour-override"
            ),
            data_colour_override="true",
            role="group",
            aria_label=label,
        ),
        cls="jouzetsu-field",
    )


def _colourway_reset_form_id(owner: ThemeColorway) -> str:
    """Return the separate target form for a colourway-specific reset control."""

    return f"theme-{owner.value}-reset-form"


def _colourway_reset_form(owner: ThemeColorway) -> HTML:
    """Keep a colourway reset out of the shared form's implicit-submit order."""

    return Form(
        id=_colourway_reset_form_id(owner),
        action=f"/settings/global/appearance/{owner.value}/reset",
        method="post",
        hidden=True,
    )


def _key_visual_field(settings: AppColorwaySettings, *, preview: str) -> HTML:
    """Render the independently configurable App emphasis colour prominently."""

    return _colour_override_field(
        label="Key visual colour",
        name="theme_app_key_visual",
        value=settings.key_visual,
        preview=preview,
        marker="theme-app-key-visual-color",
    )


def _app_appearance_controls(state: AppState) -> tuple[HTML, ...]:
    app: AppColorwaySettings = state.config.theme.app
    icon = state.config.ui.icon_colors
    host = state.config.host_stats
    return (
        _message_action_style_control(state.config.ui.message_action_style),
        _appearance_feature(
            "Semantic message actions",
            tuple(
                (
                    action.label,
                    f"theme_app_{action.color_field}",
                    getattr(app, action.color_field)
                    or builtin_message_action_color(action),
                    f"theme-app-{action.color_field.replace('_', '-')}-color",
                )
                for action in MessageAction
            ),
        ),
        _appearance_feature(
            "App icon",
            (
                (
                    "Linework",
                    "icon_linework_color",
                    icon.linework_color,
                    "icon-linework-color",
                ),
                ("Accent", "icon_accent_color", icon.accent_color, "icon-accent-color"),
                (
                    "Surface",
                    "icon_surface_color",
                    icon.surface_color,
                    "icon-surface-color",
                ),
            ),
        ),
        _appearance_feature(
            "Host activity meter",
            (
                (
                    "Low activity",
                    "activity_start_color",
                    host.activity_start_color,
                    "host-stats-activity-start-color",
                ),
                (
                    "High activity",
                    "activity_end_color",
                    host.activity_end_color,
                    "host-stats-activity-end-color",
                ),
            ),
        ),
    )


def _message_action_style_control(style: MessageActionStyle) -> HTML:
    """Render the action-colour policy before its semantic palette."""

    return Div(
        field(
            "Message action style",
            Select(
                *(
                    Option(
                        candidate.label,
                        value=candidate.value,
                        selected=candidate is style,
                    )
                    for candidate in MessageActionStyle
                ),
                name="message_action_style",
                data_testid="message-action-style",
            ),
        ),
        Small(
            "Uniform uses the App message action colour. Semantic uses the action "
            "colours below; other message controls remain App-coloured.",
            cls="jouzetsu-form-help",
        ),
        cls="jouzetsu-dialog-section",
    )


def _appearance_feature(
    label: str, controls: tuple[tuple[str, str, str, str], ...]
) -> HTML:
    return Div(
        Span(label, cls="jouzetsu-field-label"),
        Div(
            *(
                field(
                    title,
                    Input(value=value, name=name, type="color", data_testid=marker),
                )
                for title, name, value, marker in controls
            ),
            cls="jouzetsu-form-grid",
        ),
        cls="jouzetsu-dialog-section",
    )


def _generation_defaults_form(state: AppState) -> HTML:
    settings: GenerationSettings = state.config.generation
    return Form(
        Section(
            H3("Generation defaults", cls="jouzetsu-section-title"),
            Div(
                field(
                    "Temperature",
                    Input(
                        value=settings.temperature,
                        name="temperature",
                        type="number",
                        min=GenerationSettings.MIN_TEMPERATURE,
                        max=GenerationSettings.MAX_TEMPERATURE,
                        step="0.05",
                        data_testid="global-temperature",
                    ),
                ),
                field(
                    "Top P",
                    Input(
                        value=settings.top_p,
                        name="top_p",
                        type="number",
                        min=GenerationSettings.MIN_TOP_P,
                        max=GenerationSettings.MAX_TOP_P,
                        step="0.05",
                        data_testid="global-top-p",
                    ),
                ),
                field(
                    "Maximum output tokens",
                    Input(
                        value=settings.max_tokens,
                        name="max_tokens",
                        type="number",
                        min=GenerationSettings.MIN_MAX_TOKENS,
                        step=1,
                        data_testid="global-max-tokens",
                    ),
                ),
                cls="jouzetsu-form-grid",
            ),
            Div(
                checkbox(
                    "continuity_review",
                    checked=settings.continuity_review,
                    label="Check reply continuity before sending",
                    marker="global-continuity-review",
                ),
                Small(
                    "Uses one extra model pass to correct concrete contradictions or missed constraints.",
                    cls="jouzetsu-form-help",
                ),
                cls="jouzetsu-dialog-section",
            ),
            _spelling_replacements_editor(settings.british_spelling_replacements),
            cls="jouzetsu-dialog-section",
        ),
        form_submit_actions(
            "Save generation defaults",
            "save-global-generation-button",
            reset_action="/settings/global/generation/reset",
            reset_marker="reset-global-generation-button",
            reset_confirmation="Reset generation defaults to their built-in values?",
        ),
        action="/settings/global/generation",
        method="post",
        cls="jouzetsu-global-settings-form",
    )


def _spelling_replacements_editor(replacements: list[SpellingReplacement]) -> HTML:
    return Div(
        Span("British spelling replacements", cls="jouzetsu-field-label"),
        Small(
            "One single-word source and replacement per row. Applied only in chats with post-processing enabled.",
            cls="jouzetsu-form-help",
        ),
        Div(
            *(_spelling_replacement_row(replacement) for replacement in replacements),
            cls="jouzetsu-spelling-rows",
            data_spelling_rows="true",
        ),
        Div(
            Button(
                "Add replacement",
                type="button",
                cls="jouzetsu-button jouzetsu-button-muted",
                data_spelling_add="true",
            ),
            Span(
                cls="jouzetsu-spelling-validation",
                data_spelling_validation="true",
                aria_live="polite",
            ),
            cls="jouzetsu-spelling-editor-footer",
        ),
        Textarea(
            _format_spelling_replacements(replacements),
            name="british_spelling_replacements",
            hidden=True,
            data_spelling_serialized="true",
            data_testid="global-british-spelling-replacements",
        ),
        cls="jouzetsu-field jouzetsu-spelling-editor",
        data_spelling_editor="true",
    )


def _spelling_replacement_row(replacement: SpellingReplacement) -> HTML:
    return Div(
        Input(
            value=replacement.source,
            placeholder="color",
            aria_label="Source spelling",
            data_spelling_source="true",
        ),
        Span("→", aria_hidden="true", cls="jouzetsu-spelling-arrow"),
        Input(
            value=replacement.replacement,
            placeholder="colour",
            aria_label="Replacement spelling",
            data_spelling_replacement="true",
        ),
        Button(
            "Remove",
            type="button",
            cls="jouzetsu-spelling-remove",
            data_spelling_remove="true",
        ),
        cls="jouzetsu-spelling-row",
        data_spelling_row="true",
    )


def _format_spelling_replacements(replacements: list[SpellingReplacement]) -> str:
    return "\n".join(
        f"{replacement.source}\t{replacement.replacement}"
        for replacement in replacements
    )
