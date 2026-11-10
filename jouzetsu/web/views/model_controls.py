"""Backend-scoped cached model choices, retaining missing selections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from ...backend_models import BackendConfig
from ...runtime import ModelCapabilities
from ..html import HTML, Button, Div, Input, Small, Span
from .icons import IconName, render_icon


@dataclass(frozen=True, slots=True)
class _Capability:
    """One model flag and its shared textual and visual presentation."""

    label: str
    icon: IconName
    support: Callable[[ModelCapabilities], bool | None]


_CAPABILITIES: Final[tuple[_Capability, ...]] = (
    _Capability("Vision", "vision", lambda model: model.vision),
    _Capability("Tool use", "tool-use", lambda model: model.trained_for_tool_use),
    _Capability("Reasoning", "reasoning", lambda model: model.reasoning),
)


def render_model_capabilities(capabilities: ModelCapabilities) -> HTML | None:
    """Show icons only for capabilities explicitly advertised as supported."""

    badges: list[HTML] = [
        Span(
            render_icon(capability.icon),
            cls=f"jouzetsu-capability-badge is-{capability.icon}",
            role="img",
            aria_label=f"{capability.label} supported",
            title=f"{capability.label} supported",
        )
        for capability in _CAPABILITIES
        if capability.support(capabilities) is True
    ]
    if not badges:
        return None
    return Span(
        *badges,
        cls="jouzetsu-model-capabilities",
        role="group",
        aria_label="Model capabilities",
    )


@dataclass(frozen=True, slots=True)
class ModelChoice:
    key: str
    label: str
    detail: str = ""
    selectable: bool = True
    capabilities: ModelCapabilities | None = None


def model_choices(
    backend: BackendConfig | None, selected_model: str | None = None
) -> tuple[ModelChoice, ...]:
    choices = (
        [
            ModelChoice(
                model.descriptor.key,
                model.alias or model.descriptor.display_name or model.descriptor.key,
                capabilities=model.descriptor.capabilities,
            )
            for model in backend.models
        ]
        if backend
        else []
    )
    if selected_model and all(choice.key != selected_model for choice in choices):
        choices.append(ModelChoice(selected_model, selected_model, "Missing"))
    return tuple(choices)


def _choice_content(choice: ModelChoice) -> HTML:
    return Span(
        Span(choice.label, cls="jouzetsu-model-choice-label"),
        Small(choice.detail, cls="jouzetsu-model-choice-detail")
        if choice.detail
        else None,
        render_model_capabilities(choice.capabilities)
        if choice.capabilities is not None
        else None,
        cls="jouzetsu-model-choice-content",
        data_model_choice_content="true",
    )


def render_model_picker(
    choices: tuple[ModelChoice, ...],
    selected_model: str,
    *,
    marker: str,
    label: str = "Model",
    empty_label: str = "No model selected",
    disabled: bool = False,
    autosave: bool = False,
) -> HTML:
    """Render a labelled model combobox with the same icon rows in its popup."""

    options = (ModelChoice("", empty_label), *choices)
    selected = next(
        (choice for choice in options if choice.key == selected_model), None
    )
    if selected is None:
        raise ValueError("selected model must be included in the model choices")
    label_id = f"{marker}-label"
    list_id = f"{marker}-options"
    return Div(
        Span(label, id=label_id, cls="jouzetsu-field-label"),
        Input(
            type="hidden",
            name="model",
            value=selected_model,
            disabled=disabled,
            data_model_picker_value="true",
            data_live_autosave="true" if autosave else None,
        ),
        Button(
            _choice_content(selected),
            type="button",
            name="model_picker",
            id=marker,
            cls="jouzetsu-model-picker-toggle",
            role="combobox",
            aria_haspopup="listbox",
            aria_expanded="false",
            aria_controls=list_id,
            aria_labelledby=label_id,
            disabled=disabled,
            data_model_picker_toggle="true",
            data_testid=marker,
        ),
        Div(
            *(
                Div(
                    _choice_content(choice),
                    id=f"{marker}-option-{index}",
                    cls="jouzetsu-model-picker-option",
                    role="option",
                    aria_selected="true" if choice.key == selected_model else "false",
                    aria_disabled="true"
                    if disabled
                    or not choice.selectable
                    and choice.key != selected_model
                    else None,
                    title=choice.key or empty_label,
                    data_model_picker_option="true",
                    data_model_value=choice.key,
                    data_model_label=choice.label,
                )
                for index, choice in enumerate(options)
            ),
            id=list_id,
            cls="jouzetsu-model-picker-options",
            role="listbox",
            aria_labelledby=label_id,
            hidden=True,
            data_model_picker_options="true",
        ),
        cls="jouzetsu-field jouzetsu-model-picker",
        data_model_picker="true",
    )
