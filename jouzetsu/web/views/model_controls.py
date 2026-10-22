"""Shared model identities, summaries, and selector options."""

from __future__ import annotations

from dataclasses import dataclass

from ...runtime import ConnectionState, ModelDescriptor
from ...state import AppState
from ..html import HTML, Option, Span, Summary


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One named model and its current availability in a selector."""

    key: str
    label: str
    detail: str = ""
    selectable: bool = True


def model_summary(
    model: ModelDescriptor,
    connection: ConnectionState,
    context_length: int | None = None,
) -> str:
    """Describe model metadata without presenting cached load state as current."""

    context: int | None = (
        context_length or model.context_length or model.max_context_length
    )
    status: str = (
        ("Loaded" if model.is_loaded else "Unloaded")
        if connection is ConnectionState.ONLINE
        else "Cached"
    )
    return " · ".join(
        part
        for part in (
            model.quantization_name,
            f"{context:,} tokens" if context is not None else "",
            status,
            "Vision" if model.capabilities.vision else "",
        )
        if part
    )


def model_choice(
    state: AppState, model_key: str, *, context_length: int | None = None
) -> ModelChoice:
    """Resolve one model's identity and availability from the shared runtime."""

    model: ModelDescriptor | None = state.model_descriptor(model_key)
    selectable: bool = (
        model is not None or state.model_connection is not ConnectionState.ONLINE
    )
    if model is not None:
        detail: str = model_summary(model, state.model_connection, context_length)
    else:
        detail = "Not checked" if selectable else "Unavailable"
    return ModelChoice(
        model_key, state.model_display_name(model_key), detail, selectable
    )


def model_choices(state: AppState, selected_model: str = "") -> tuple[ModelChoice, ...]:
    """List available models and retained selections, preserving selected instance IDs."""

    selected_key: str = state.resolve_model_key(selected_model)
    retained_keys: set[str] = {
        selected_key,
        state.resolve_model_key(state.active_chat.model),
        state.resolve_model_key(state.config.server.default_model),
    }
    return tuple(
        model_choice(state, selected_model if key == selected_key else key)
        for key in state.model_keys
        if key in retained_keys or state.model_descriptor(key) is not None
    )


def render_model_options(
    choices: tuple[ModelChoice, ...], selected_model: str
) -> tuple[HTML, ...]:
    """Retain unavailable selections while preventing new stale selections."""

    return tuple(
        Option(
            " · ".join(part for part in (choice.label, choice.detail) if part),
            value=choice.key,
            selected=choice.key == selected_model,
            disabled=not choice.selectable and choice.key != selected_model,
            title=choice.key,
        )
        for choice in choices
    )


def render_model_summary(
    state: AppState, model_key: str, *, context_length: int | None = None
) -> HTML:
    """Keep the model name, identifier, and useful metadata visible when collapsed."""

    choice: ModelChoice = model_choice(state, model_key, context_length=context_length)
    return Summary(
        Span(choice.label, cls="jouzetsu-model-name"),
        Span(model_key, cls="jouzetsu-model-key jouzetsu-mono")
        if choice.label != model_key
        else None,
        Span(choice.detail, cls="jouzetsu-model-meta"),
        cls="jouzetsu-model-summary",
    )
