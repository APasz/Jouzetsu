"""Connection tabs, compact model cards, and model configuration dialogs."""

from __future__ import annotations

from hashlib import sha256
from typing import Final, Literal

from ...backend_codec import backend_fingerprint
from ...backend_models import (
    BackendConfig,
    BackendKind,
    LMStudioTransport,
    ModelRef,
    RegisteredModel,
)
from ...backend_protocol import BackendReachability, ModelAvailability
from ...runtime import REASONING_OPTIONS
from ...sampling import SamplingParameters
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
    Option,
    P,
    Section,
    Select,
    Small,
    Span,
    Summary,
)
from .controls import checkbox, close_dialog_button, field
from .feedback import dialog_feedback
from .icons import render_icon
from .model_controls import (
    model_choices,
    model_label,
    render_model_availability,
    render_model_capabilities,
    render_model_picker,
)
from .tabs import (
    DialogTab,
    TabIndicator,
    render_dialog_tab_list,
    render_dialog_tab_panel,
)

_DIALOG_ID: Final[str] = "backends-dialog"
_NEW_TAB: Final[DialogTab] = DialogTab("new", "+ New Backend")
type BackendAction = Literal["test", "remove", "load", "unload", "default-model"]
_ACTION_FEEDBACK: Final[dict[BackendAction, tuple[str, str]]] = {
    "test": ("Refreshing…", ""),
    "remove": ("Removing…", "Backend removed"),
    "load": ("Loading…", "Model loaded"),
    "unload": ("Unloading…", "Model unloaded"),
    "default-model": ("Updating…", "Default model updated"),
}
_BACKEND_TYPES: Final[dict[BackendKind, tuple[str, str]]] = {
    BackendKind.LM_STUDIO: ("LM Studio", "http://localhost:1234/v1"),
    BackendKind.OLLAMA: ("Ollama", "http://127.0.0.1:11434"),
    BackendKind.OPENAI_COMPATIBLE: (
        "OpenAI-compatible API",
        "http://localhost:8080/v1",
    ),
}
_STATUS_INDICATORS: Final[dict[BackendReachability, TabIndicator]] = {
    BackendReachability.ONLINE: "success",
    BackendReachability.OFFLINE: "error",
    BackendReachability.UNKNOWN: "warning",
    BackendReachability.STALE: "warning",
}


def _action(
    label: str,
    backend: BackendConfig,
    action: BackendAction,
    *,
    fingerprint: str,
    confirm: str = "",
    model_id: str | None = None,
    marker: str = "",
) -> HTML:
    busy_label, notice = _ACTION_FEEDBACK[action]
    icon_only = action == "test"
    return Form(
        _snapshot(fingerprint),
        Input(type="hidden", name="model_id", value=model_id) if model_id is not None else None,
        Button(
            render_icon("refresh") if icon_only else label,
            type="submit",
            cls="jouzetsu-button jouzetsu-backend-refresh" if icon_only else "jouzetsu-button",
            aria_label=label if icon_only else None,
            title=label if icon_only else None,
            data_confirm=confirm or None,
            data_busy_label=busy_label if not icon_only else None,
        ),
        action=f"/settings/backends/{backend.id}/{action}",
        method="post",
        data_live_submit="true",
        cls="jouzetsu-inline-form",
        id=marker or None,
        data_live_notice=notice or None,
        data_live_discard_edits=f"backend-{backend.id}" if action == "remove" else None,
    )


def _snapshot(fingerprint: str) -> HTML:
    return Input(type="hidden", name="backend_fingerprint", value=fingerprint)


def _backend_tab(backend: BackendConfig, health: BackendReachability, summary: str) -> DialogTab:
    return DialogTab(
        backend.id,
        backend.name,
        detail=summary,
        indicator=_STATUS_INDICATORS[health] if backend.enabled else "neutral",
        indicator_subdued=backend.enabled and health is BackendReachability.STALE,
    )


def render_backends_dialog(state: AppState, *, active_tab: str = "", notice: str = "", error: str = "") -> HTML:
    """Give each configured connection its own tab, independent of backend kind."""

    configuration = state.config.backends
    backend_tabs = tuple(
        _backend_tab(backend, state.inference.health(backend.id).state, state.inference.connection_summary(backend.id))
        for backend in configuration.backends
    )
    tabs = (*backend_tabs, _NEW_TAB)
    if active_tab not in {tab.name for tab in tabs}:
        active_tab = tabs[0].name
    return Dialog(
        Div(
            Div(
                Div(
                    H2("Backends", id=f"{_DIALOG_ID}-title", cls="jouzetsu-modal-title"),
                    close_dialog_button(_DIALOG_ID),
                    cls="jouzetsu-dialog-heading",
                ),
                render_dialog_tab_list(
                    _DIALOG_ID,
                    tabs,
                    active_tab,
                    label="Backend connections",
                    pinned_tab=_NEW_TAB.name,
                ),
                dialog_feedback(notice=notice, error=error, always_render=True),
                cls="jouzetsu-dialog-header",
            ),
            Div(
                P(
                    "Manage connections and their models.",
                    id=f"{_DIALOG_ID}-description",
                    cls="jouzetsu-dialog-description",
                ),
                Form(
                    Input(
                        type="hidden",
                        name="backend_fingerprint",
                        value=backend_fingerprint(configuration),
                    ),
                    field(
                        "Default backend for new chats",
                        Select(
                            Option(
                                "None",
                                value="",
                                selected=configuration.default_backend_id is None,
                            ),
                            *(
                                Option(
                                    backend.name,
                                    value=backend.id,
                                    selected=backend.id == configuration.default_backend_id,
                                )
                                for backend in configuration.backends
                            ),
                            name="backend_id",
                        ),
                    ),
                    Button(
                        "Save default",
                        type="submit",
                        cls="jouzetsu-button",
                        data_busy_label="Saving…",
                    ),
                    action="/settings/backends/default",
                    method="post",
                    data_live_submit="true",
                    cls="jouzetsu-backend-default-form",
                    data_live_notice="Default backend saved",
                ),
                *(
                    render_dialog_tab_panel(_DIALOG_ID, tab, active_tab, _backend_panel(state, backend))
                    for tab, backend in zip(backend_tabs, configuration.backends, strict=True)
                ),
                render_dialog_tab_panel(_DIALOG_ID, _NEW_TAB, active_tab, _new_backend_form()),
                cls="jouzetsu-dialog-body",
            ),
            cls="jouzetsu-modal-card",
        ),
        id=_DIALOG_ID,
        cls="jouzetsu-dialog jouzetsu-backends-dialog",
        aria_labelledby=f"{_DIALOG_ID}-title",
        aria_describedby=f"{_DIALOG_ID}-description",
    )


def _new_backend_form() -> HTML:
    _, default_example = _BACKEND_TYPES[BackendKind.LM_STUDIO]
    return Div(
        H3("New backend", cls="jouzetsu-section-title"),
        P("Add a named connection to LM Studio, Ollama, or a compatible API."),
        Form(
            field("Name", Input(name="name", required=True)),
            field(
                "Backend type",
                Select(
                    *(
                        Option(
                            label,
                            value=kind.value,
                            data_base_url_example=example,
                        )
                        for kind, (label, example) in _BACKEND_TYPES.items()
                    ),
                    name="kind",
                    data_backend_kind="true",
                ),
            ),
            field(
                "API base URL",
                Input(
                    name="base_url",
                    type="url",
                    required=True,
                    placeholder=default_example,
                    data_backend_base_url="true",
                ),
            ),
            Small("Example: ", Span(default_example, data_backend_url_example="true")),
            Button("Add backend", type="submit", cls="jouzetsu-button"),
            action="/settings/backends/new",
            method="post",
            data_testid="add-backend-form",
            cls="jouzetsu-backend-form",
        ),
        cls="jouzetsu-dialog-section",
    )


def _backend_panel(state: AppState, backend: BackendConfig) -> HTML:
    fingerprint = backend_fingerprint(backend)
    health = state.inference.health(backend.id)
    discovery_error = state.inference.discovery_error(backend.id)
    observed = {model.key: model for model in state.inference.inventory(backend.id)}
    availability = {
        model.descriptor.key: state.inference.model_availability(ModelRef(backend.id, model.descriptor.key))
        for model in backend.models
    }
    discovery = state.inference.discovery(backend.id)
    summary = state.inference.connection_summary(backend.id)
    inventory_warning = bool(discovery and discovery.supported and not discovery.models) or (
        backend.defaults.model_id is not None
        and availability.get(backend.defaults.model_id) is ModelAvailability.UNAVAILABLE
    )

    def instances(model: RegisteredModel) -> tuple[str, ...] | None:
        descriptor = observed.get(model.descriptor.key)
        return (
            descriptor.loaded_instance_ids
            if backend.enabled and health.state is BackendReachability.ONLINE and descriptor is not None
            else None
        )

    return Div(
        Div(
            Div(
                H3(backend.name, cls="jouzetsu-section-title"),
                Small(backend.base_url, cls="jouzetsu-backend-url"),
            ),
            Span(
                render_icon("warning" if inventory_warning else "activity"),
                role="img",
                title=summary,
                aria_label=summary,
                cls="jouzetsu-backend-inventory is-warning" if inventory_warning else "jouzetsu-backend-inventory",
                data_backend_inventory="true",
            ),
            _action("Refresh models", backend, "test", fingerprint=fingerprint),
            cls="jouzetsu-backend-heading",
        ),
        P(str(discovery_error), role="status") if discovery_error else None,
        _backend_settings(state, backend, fingerprint=fingerprint),
        H3(f"Models · {len(backend.models)}", cls="jouzetsu-section-title"),
        Div(
            *(
                _model_card(
                    backend,
                    model,
                    instances(model),
                    fingerprint=fingerprint,
                    availability=availability[model.descriptor.key],
                )
                for model in backend.models
            ),
            cls="jouzetsu-model-grid",
        )
        if backend.models
        else P("No models yet. Refresh models or register a model manually."),
        Details(
            Summary("Register model manually"),
            Form(
                _snapshot(fingerprint),
                field("Model identifier", Input(name="model_id", required=True)),
                field("Alias", Input(name="alias")),
                Button(
                    "Register model",
                    type="submit",
                    cls="jouzetsu-button",
                    data_busy_label="Registering…",
                ),
                action=f"/settings/backends/{backend.id}/register-model",
                method="post",
                data_live_submit="true",
                cls="jouzetsu-backend-form",
                data_live_notice="Model registered",
            ),
            id=f"backend-register-{backend.id}",
            cls="jouzetsu-backend-details",
        ),
        *(
            _model_dialog(
                backend,
                model,
                instances(model),
                fingerprint=fingerprint,
                availability=availability[model.descriptor.key],
            )
            for model in backend.models
        ),
        id=f"backend-{backend.id}",
        cls="jouzetsu-backend-panel",
    )


def _idle_unload_control(backend: BackendConfig) -> HTML | None:
    if backend.model_management is not None:
        name = "idle_unload_minutes"
        minutes = backend.model_management.idle_unload_minutes
    elif backend.kind is BackendKind.OLLAMA:
        name = "ollama_idle_unload_minutes"
        minutes = backend.ollama_idle_unload_minutes
    else:
        return None
    return Div(
        field(
            "Idle unload minutes (blank disables)",
            Input(name=name, value=minutes or "", type="number", min=1),
        ),
        Small("Applies on the next load or generation; Ollama manages the idle timer.")
        if backend.kind is BackendKind.OLLAMA
        else None,
        cls="jouzetsu-backend-idle-control",
    )


def _backend_settings(state: AppState, backend: BackendConfig, *, fingerprint: str) -> HTML:
    return Details(
        Summary("Backend settings"),
        Form(
            _snapshot(fingerprint),
            Section(
                H3("Connection", cls="jouzetsu-section-title"),
                Div(
                    field("Name", Input(name="name", value=backend.name, required=True)),
                    field(
                        "API base URL",
                        Input(
                            name="base_url",
                            value=backend.base_url,
                            type="url",
                            required=True,
                        ),
                    ),
                    field(
                        "Concurrency limit",
                        Input(
                            name="concurrency_limit",
                            value=backend.concurrency_limit,
                            type="number",
                            min=1,
                        ),
                    ),
                    _idle_unload_control(backend),
                    field(
                        "Chat API",
                        Select(
                            Option(
                                "Chat Completions",
                                value=LMStudioTransport.CHAT_COMPLETIONS.value,
                                selected=backend.lmstudio_transport is LMStudioTransport.CHAT_COMPLETIONS,
                            ),
                            Option(
                                "Responses (experimental)",
                                value=LMStudioTransport.RESPONSES.value,
                                selected=backend.lmstudio_transport is LMStudioTransport.RESPONSES,
                            ),
                            name="lmstudio_transport",
                            data_backend_transport="true",
                            data_responses_value=LMStudioTransport.RESPONSES.value,
                        ),
                    )
                    if backend.kind is BackendKind.LM_STUDIO
                    else None,
                    checkbox(
                        "enabled",
                        checked=backend.enabled,
                        label="Enabled",
                        marker=f"backend-enabled-{backend.id}",
                    ),
                    cls="jouzetsu-form-grid is-two-columns",
                ),
                P(
                    "Responses requires LM Studio to confirm truncation is disabled.",
                    data_backend_responses_help="true",
                    hidden=backend.lmstudio_transport is not LMStudioTransport.RESPONSES,
                    cls="jouzetsu-form-help",
                )
                if backend.kind is BackendKind.LM_STUDIO
                else None,
                cls="jouzetsu-backend-settings-group",
            ),
            Section(
                H3("Chat defaults", cls="jouzetsu-section-title"),
                P(
                    "Used when a chat is created or its profile is reset.",
                    cls="jouzetsu-form-help",
                ),
                render_model_picker(
                    model_choices(backend, backend.defaults.model_id, availability=state.inference.model_availability),
                    backend.defaults.model_id or "",
                    marker=f"backend-model-{backend.id}",
                    label="Default model",
                    empty_label="None",
                ),
                Div(
                    *(
                        field(
                            label,
                            Input(
                                name=name,
                                value=getattr(backend.defaults.parameters, name),
                                type="number",
                                min=minimum,
                                max=maximum,
                                step=step,
                            ),
                        )
                        for name, label, minimum, maximum, step in (
                            (
                                "temperature",
                                "Temperature",
                                SamplingParameters.MIN_TEMPERATURE,
                                SamplingParameters.MAX_TEMPERATURE,
                                "0.05",
                            ),
                            (
                                "top_p",
                                "Top P",
                                SamplingParameters.MIN_TOP_P,
                                SamplingParameters.MAX_TOP_P,
                                "0.05",
                            ),
                            (
                                "max_tokens",
                                "Maximum output tokens",
                                SamplingParameters.MIN_MAX_TOKENS,
                                None,
                                "1",
                            ),
                        )
                    ),
                    cls="jouzetsu-form-grid",
                ),
                cls="jouzetsu-backend-settings-group",
            ),
            Div(
                Button(
                    "Save backend",
                    type="submit",
                    cls="jouzetsu-button jouzetsu-button-primary",
                    data_busy_label="Saving…",
                ),
                cls="jouzetsu-dialog-actions",
            ),
            action=f"/settings/backends/{backend.id}/save",
            method="post",
            data_live_submit="true",
            data_live_notice="Backend saved",
            cls="jouzetsu-backend-form",
        ),
        Div(
            _action(
                "Remove backend",
                backend,
                "remove",
                fingerprint=fingerprint,
                confirm="Remove this backend? Chat selections and profiles will be retained.",
            ),
            cls="jouzetsu-dialog-actions",
        ),
        id=f"backend-settings-{backend.id}",
        cls="jouzetsu-backend-details",
    )


def _model_dialog_id(backend: BackendConfig, model: RegisteredModel) -> str:
    """Keep arbitrary model identifiers out of DOM IDs and CSS selectors."""

    model_key = sha256(model.descriptor.key.encode()).hexdigest()[:16]
    return f"model-dialog-{backend.id}-{model_key}"


def _model_metadata(model: RegisteredModel) -> HTML | None:
    descriptor = model.descriptor
    values = [value for value in (descriptor.params_string, descriptor.quantization_name) if value]
    if descriptor.size_bytes is not None:
        values.append(f"{descriptor.size_bytes / 1024**3:.1f} GiB")
    context = descriptor.max_context_length or descriptor.context_length
    if context is not None:
        values.append(f"{context:,} context")
    return Small(" · ".join(values), cls="jouzetsu-model-metadata") if values else None


def _model_load_state(instances: tuple[str, ...] | None) -> HTML:
    return Small(
        "Load state unknown" if instances is None else "Loaded" if instances else "Unloaded",
        cls="jouzetsu-model-load-state is-loaded" if instances else "jouzetsu-model-load-state",
    )


def _model_card(
    backend: BackendConfig,
    model: RegisteredModel,
    instances: tuple[str, ...] | None,
    *,
    fingerprint: str,
    availability: ModelAvailability,
) -> HTML:
    model_id = model.descriptor.key
    dialog_id = _model_dialog_id(backend, model)
    return Div(
        Button(
            Span(
                Span(model_label(model)),
                render_model_availability(availability),
                cls="jouzetsu-model-card-name",
            ),
            _model_metadata(model),
            render_model_capabilities(model.descriptor.capabilities),
            Span(
                Small("Default model", cls="jouzetsu-model-default"),
                cls="jouzetsu-model-card-state",
            )
            if backend.defaults.model_id == model_id
            else None,
            type="button",
            id=f"{dialog_id}-open",
            cls="jouzetsu-model-card-open",
            data_dialog_open=dialog_id,
            aria_label=f"Configure {model_label(model)}",
            aria_haspopup="dialog",
        ),
        _action(
            "Unload",
            backend,
            "unload",
            fingerprint=fingerprint,
            model_id=model_id,
            marker=f"{dialog_id}-card-unload",
        )
        if backend.kind.supports_model_management and instances
        else None,
        cls="jouzetsu-model-card",
        data_model_card=model_id,
    )


def _model_dialog(
    backend: BackendConfig,
    model: RegisteredModel,
    instances: tuple[str, ...] | None,
    *,
    fingerprint: str,
    availability: ModelAvailability,
) -> HTML:
    dialog_id = _model_dialog_id(backend, model)
    return Dialog(
        Div(
            Div(
                Div(
                    H2(
                        model_label(model),
                        id=f"{dialog_id}-title",
                        cls="jouzetsu-modal-title",
                    ),
                    close_dialog_button(dialog_id),
                    cls="jouzetsu-dialog-heading",
                ),
                dialog_feedback(always_render=True),
                cls="jouzetsu-dialog-header",
            ),
            Div(
                P(model.descriptor.key, cls="jouzetsu-model-identifier"),
                render_model_availability(availability),
                _model_metadata(model),
                render_model_capabilities(model.descriptor.capabilities),
                _model_load_state(instances) if backend.kind.supports_model_management else None,
                Small("Loaded instances: " + ", ".join(instances)) if instances else None,
                Div(
                    Small("Default model", cls="jouzetsu-model-default")
                    if backend.defaults.model_id == model.descriptor.key
                    else _action(
                        "Set as default",
                        backend,
                        "default-model",
                        fingerprint=fingerprint,
                        model_id=model.descriptor.key,
                    ),
                    _action(
                        "Load",
                        backend,
                        "load",
                        fingerprint=fingerprint,
                        model_id=model.descriptor.key,
                    )
                    if backend.kind.supports_model_management and availability is not ModelAvailability.UNAVAILABLE
                    else None,
                    _action(
                        "Unload",
                        backend,
                        "unload",
                        fingerprint=fingerprint,
                        model_id=model.descriptor.key,
                    )
                    if backend.kind.supports_model_management
                    else None,
                    cls="jouzetsu-dialog-actions",
                    data_live_group="true",
                ),
                _model_editor(backend, model, dialog_id, fingerprint=fingerprint),
                cls="jouzetsu-dialog-body",
            ),
            cls="jouzetsu-modal-card",
        ),
        id=dialog_id,
        cls="jouzetsu-dialog jouzetsu-model-dialog",
        aria_labelledby=f"{dialog_id}-title",
        data_model_dialog="true",
    )


def _model_editor(
    backend: BackendConfig,
    model: RegisteredModel,
    dialog_id: str,
    *,
    fingerprint: str,
) -> HTML:
    return Form(
        _snapshot(fingerprint),
        Input(type="hidden", name="model_id", value=model.descriptor.key),
        field("Alias", Input(name="alias", value=model.alias)),
        field(
            "Default reasoning",
            Select(
                Option(
                    "Automatic",
                    value="",
                    selected=model.reasoning_default is None,
                ),
                Option(
                    f"{model.reasoning_default.title()} (unavailable)",
                    value=model.reasoning_default,
                    selected=True,
                )
                if model.reasoning_default is not None and model.reasoning_default not in model.reasoning_options
                else None,
                *(
                    Option(
                        option.title(),
                        value=option,
                        selected=option == model.reasoning_default,
                    )
                    for option in model.reasoning_options
                ),
                name="reasoning_default",
            ),
        )
        if backend.kind.supports_reasoning_controls
        else None,
        P(
            "The previous reasoning default is unavailable; using Automatic.",
            role="status",
        )
        if model.reasoning_default_unavailable
        or (model.reasoning_default is not None and model.reasoning_default not in model.reasoning_options)
        else None,
        Div(
            Input(type="hidden", name="reasoning_override", value="true"),
            Small("Manually configured reasoning options"),
            *(
                checkbox(
                    name=f"reasoning_option_{option}",
                    checked=option in model.reasoning_options_override,
                    label=option.title(),
                    marker=f"{dialog_id}-reasoning-option-{option}",
                )
                for option in REASONING_OPTIONS
            ),
        )
        if backend.kind.supports_reasoning_controls
        and (not model.descriptor.capabilities.reasoning_options or model.reasoning_options_override)
        else None,
        Div(
            Button(
                "Cancel",
                type="button",
                cls="jouzetsu-button jouzetsu-button-muted",
                data_dialog_close=dialog_id,
            ),
            Button(
                "Save model",
                type="submit",
                cls="jouzetsu-button jouzetsu-button-primary",
                data_busy_label="Saving…",
            ),
            cls="jouzetsu-dialog-actions",
        ),
        action=f"/settings/backends/{backend.id}/alias",
        method="post",
        data_live_submit="true",
        data_backend_model_settings="true",
        data_reset_on_close="true",
        data_live_close_dialog=dialog_id,
        data_live_notice="Model saved",
    )
