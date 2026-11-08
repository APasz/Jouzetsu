"""Backend-scoped cached model choices, retaining missing selections."""

from __future__ import annotations

from dataclasses import dataclass

from ...backend_models import BackendConfig
from ..html import HTML, Option


@dataclass(frozen=True, slots=True)
class ModelChoice:
    key: str
    label: str
    detail: str = ""
    selectable: bool = True


def model_choices(
    backend: BackendConfig | None, selected_model: str | None = None
) -> tuple[ModelChoice, ...]:
    choices = (
        [
            ModelChoice(
                model.descriptor.key,
                model.alias or model.descriptor.display_name or model.descriptor.key,
                "Vision"
                if model.descriptor.capabilities.vision is True
                else "Vision unknown"
                if model.descriptor.capabilities.vision is None
                else "Text only",
            )
            for model in backend.models
        ]
        if backend
        else []
    )
    if selected_model and all(choice.key != selected_model for choice in choices):
        choices.append(ModelChoice(selected_model, selected_model, "Missing"))
    return tuple(choices)


def render_model_options(
    choices: tuple[ModelChoice, ...], selected_model: str
) -> tuple[HTML, ...]:
    return tuple(
        Option(
            " · ".join(filter(None, (choice.label, choice.detail))),
            value=choice.key,
            selected=choice.key == selected_model,
            title=choice.key,
            disabled=not choice.selectable and choice.key != selected_model,
        )
        for choice in choices
    )
