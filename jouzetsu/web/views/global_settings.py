"""Global application-settings dialog rendering."""

from __future__ import annotations

from ...config import GenerationSettings, SpellingReplacement
from ...state import AppState
from ..html import (
    H2,
    H3,
    HTML,
    Button,
    Dialog,
    Div,
    Form,
    Input,
    Option,
    P,
    Section,
    Select,
    Small,
    Span,
    Textarea,
)
from .context import PageContext
from .controls import checkbox, close_dialog_button, field, post_button
from .feedback import dialog_feedback


def render_global_settings_dialog(
    state: AppState,
    context: PageContext,
    *,
    notice: str = "",
    error: str = "",
) -> HTML:
    """Render application-wide defaults grouped by ownership and persistence route."""

    return Dialog(
        Div(
            H2(
                "Global settings",
                id="global-settings-dialog-title",
                cls="jouzetsu-modal-title",
            ),
            P(
                "Defaults used by every chat without its own override.",
                id="global-settings-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            dialog_feedback(notice=notice, error=error),
            _system_prompt_form(state),
            _model_defaults_form(state),
            _interface_form(state),
            _generation_defaults_form(state),
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
        _submit_actions("Save system prompt", "save-global-prompt-button"),
        action="/settings/global/prompt",
        method="post",
        cls="jouzetsu-global-settings-form",
    )


def _model_defaults_form(state: AppState) -> HTML:
    configured_default: str = state.config.server.default_model
    current_model_key: str = state.current_model_key()
    return Form(
        Section(
            H3("Model defaults", cls="jouzetsu-section-title"),
            Div(
                field(
                    "Default model",
                    Select(
                        Option(
                            "No global default",
                            value="",
                            selected=not configured_default,
                        ),
                        *(
                            Option(
                                state.model_display_name(model_key),
                                value=model_key,
                                selected=model_key == configured_default,
                            )
                            for model_key in _default_model_keys(state)
                        ),
                        name="default_model",
                        data_testid="global-default-model",
                    ),
                ),
                field(
                    "Current model alias",
                    Input(
                        value=state.current_model_alias(),
                        name="model_alias",
                        disabled=not current_model_key,
                        placeholder="No active model",
                        data_testid="global-current-model-alias",
                    ),
                ),
                field(
                    "Auto-unload idle minutes",
                    Input(
                        value=_auto_unload_editor_value(state),
                        name="auto_unload_minutes",
                        type="number",
                        min=1,
                        step=1,
                        placeholder=_auto_unload_placeholder(state),
                        data_testid="global-auto-unload-minutes",
                    ),
                ),
                cls="jouzetsu-form-grid is-two-columns",
            ),
            Small(
                "Leave auto-unload blank to use LM Studio's value for the active model.",
                cls="jouzetsu-form-help",
            ),
            cls="jouzetsu-dialog-section",
        ),
        _submit_actions("Save model defaults", "save-global-model-defaults-button"),
        action="/settings/global/model-defaults",
        method="post",
        cls="jouzetsu-global-settings-form",
    )


def _default_model_keys(state: AppState) -> tuple[str, ...]:
    candidates: tuple[str, ...] = (
        *state.config.server.model_aliases,
        *(model.key for model in state.model_inventory),
        state.config.server.default_model,
    )
    return tuple(dict.fromkeys(key for key in candidates if key))


def _interface_form(state: AppState) -> HTML:
    icon_colors = state.config.ui.icon_colors
    return Form(
        Section(
            H3("Interface", cls="jouzetsu-section-title"),
            checkbox(
                "muted_color_icons",
                checked=state.config.ui.message_action_icon_style == "muted_color",
                label="Muted coloured message action icons",
                marker="global-message-action-icon-style",
            ),
            H3("App icon", cls="jouzetsu-section-title"),
            Div(
                field(
                    "Linework",
                    Input(
                        value=icon_colors.linework_color,
                        name="icon_linework_color",
                        type="color",
                        data_testid="icon-linework-color",
                    ),
                ),
                field(
                    "Accent",
                    Input(
                        value=icon_colors.accent_color,
                        name="icon_accent_color",
                        type="color",
                        data_testid="icon-accent-color",
                    ),
                ),
                field(
                    "Surface",
                    Input(
                        value=icon_colors.surface_color,
                        name="icon_surface_color",
                        type="color",
                        data_testid="icon-surface-color",
                    ),
                ),
                cls="jouzetsu-form-grid is-three-columns",
            ),
            cls="jouzetsu-dialog-section",
        ),
        _submit_actions("Save interface", "save-global-interface-button"),
        action="/settings/global/interface",
        method="post",
        cls="jouzetsu-global-settings-form",
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
        _submit_actions("Save generation defaults", "save-global-generation-button"),
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


def _submit_actions(label: str, marker: str) -> HTML:
    return Div(
        Button(
            label,
            type="submit",
            cls="jouzetsu-button jouzetsu-button-primary",
            data_testid=marker,
        ),
        cls="jouzetsu-dialog-actions",
    )


def _format_spelling_replacements(replacements: list[SpellingReplacement]) -> str:
    return "\n".join(
        f"{replacement.source}\t{replacement.replacement}"
        for replacement in replacements
    )


def _auto_unload_editor_value(state: AppState) -> str:
    configured_value: int | None = state.config.server.auto_unload_minutes
    return str(configured_value) if configured_value is not None else ""


def _auto_unload_placeholder(state: AppState) -> str:
    runtime_value: int | None = state.current_model_auto_unload_minutes()
    return (
        f"LM Studio: {runtime_value} minutes"
        if runtime_value is not None
        else "LM Studio default"
    )
