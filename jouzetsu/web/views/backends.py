"""Backend configuration using the existing compact settings controls."""

from __future__ import annotations

from ...backend_codec import backend_fingerprint
from ...backend_models import BackendConfig, BackendKind
from ...backend_protocol import BackendReachability
from ...state import AppState
from ..html import (
    H3,
    HTML,
    Button,
    Details,
    Div,
    Form,
    Input,
    Option,
    P,
    Select,
    Small,
    Summary,
)
from .controls import checkbox, field


def _action(
    label: str, backend: BackendConfig, action: str, *, confirm: str = ""
) -> HTML:
    return Form(
        _snapshot(backend),
        Button(
            label, type="submit", cls="jouzetsu-button", data_confirm=confirm or None
        ),
        action=f"/settings/backends/{backend.id}/{action}",
        method="post",
        data_live_submit="true",
    )


def _snapshot(backend: BackendConfig) -> HTML:
    return Input(
        type="hidden", name="backend_fingerprint", value=backend_fingerprint(backend)
    )


def render_backends(state: AppState) -> HTML:
    configuration = state.config.backends
    return Div(
        P(
            "Backend defaults are copied when a chat is created or its profile is reset."
        ),
        Form(
            Input(
                type="hidden",
                name="backend_fingerprint",
                value=backend_fingerprint(configuration),
            ),
            field(
                "Default backend",
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
            Button("Save default", type="submit", cls="jouzetsu-button"),
            action="/settings/backends/default",
            method="post",
            data_live_submit="true",
        ),
        *(_backend_editor(state, backend) for backend in configuration.backends),
        H3("Add backend", cls="jouzetsu-section-title"),
        Form(
            field("Name", Input(name="name", required=True)),
            field(
                "Kind",
                Select(
                    *(
                        Option(
                            "LM Studio"
                            if kind is BackendKind.LM_STUDIO
                            else "Chat Completions",
                            value=kind.value,
                        )
                        for kind in BackendKind
                    ),
                    name="kind",
                ),
            ),
            field(
                "API base URL",
                Input(
                    name="base_url",
                    type="url",
                    required=True,
                    placeholder="http://localhost:1234/v1",
                ),
            ),
            Button("Add backend", type="submit", cls="jouzetsu-button"),
            action="/settings/backends/new",
            method="post",
            data_testid="add-backend-form",
        ),
    )


def _backend_editor(state: AppState, backend: BackendConfig) -> HTML:
    health = state.inference.health(backend.id)
    discovery_error = state.inference.discovery_error(backend.id)
    management = backend.model_management
    observed = {model.key: model for model in state.inference.inventory(backend.id)}
    return Details(
        Summary(backend.name, " · ", health.state.value),
        Small(backend.id, cls="jouzetsu-mono"),
        P(str(discovery_error), role="status") if discovery_error else None,
        Form(
            _snapshot(backend),
            field("Name", Input(name="name", value=backend.name, required=True)),
            field(
                "API base URL",
                Input(
                    name="base_url", value=backend.base_url, type="url", required=True
                ),
            ),
            checkbox(
                "enabled",
                checked=backend.enabled,
                label="Enabled",
                marker=f"backend-enabled-{backend.id}",
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
            field(
                "Default model",
                Select(
                    Option(
                        "None", value="", selected=backend.defaults.model_id is None
                    ),
                    *(
                        Option(
                            model.alias or model.descriptor.display_name,
                            value=model.descriptor.key,
                            selected=model.descriptor.key == backend.defaults.model_id,
                        )
                        for model in backend.models
                    ),
                    Option(
                        f"Missing model: {backend.defaults.model_id}",
                        value=backend.defaults.model_id,
                        selected=True,
                    )
                    if backend.defaults.model_id is not None
                    and all(
                        model.descriptor.key != backend.defaults.model_id
                        for model in backend.models
                    )
                    else None,
                    name="model",
                ),
            ),
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
                    ("temperature", "Temperature", 0, 2, "0.05"),
                    ("top_p", "Top P", 0, 1, "0.05"),
                    ("max_tokens", "Maximum output tokens", 1, None, "1"),
                )
            ),
            Div(
                checkbox(
                    "automatic_loading",
                    checked=management.automatic_loading,
                    label="Load selected model automatically",
                    marker=f"auto-load-{backend.id}",
                ),
                field(
                    "Idle unload minutes (blank disables)",
                    Input(
                        name="idle_unload_minutes",
                        value=management.idle_unload_minutes or "",
                        type="number",
                        min=1,
                    ),
                ),
            )
            if management
            else P(
                "Model loading and unloading are not supported by this backend kind."
            ),
            Button("Save backend", type="submit", cls="jouzetsu-button"),
            action=f"/settings/backends/{backend.id}/save",
            method="post",
            data_live_submit="true",
        ),
        Div(
            _action("Test / refresh models", backend, "test"),
            _action(
                "Remove backend",
                backend,
                "remove",
                confirm="Remove this backend? Chat selections and profiles will be retained.",
            ),
            cls="jouzetsu-dialog-actions",
        ),
        H3("Models", cls="jouzetsu-section-title"),
        *(
            _model_editor(
                backend,
                model.descriptor.key,
                model.alias,
                model.descriptor.capabilities.vision,
                observed[model.descriptor.key].loaded_instance_ids
                if health.state is BackendReachability.ONLINE
                and model.descriptor.key in observed
                else None,
            )
            for model in backend.models
        ),
        H3("Register model manually", cls="jouzetsu-section-title"),
        Form(
            _snapshot(backend),
            field("Model identifier", Input(name="model_id", required=True)),
            field("Alias", Input(name="alias")),
            Button("Register model", type="submit", cls="jouzetsu-button"),
            action=f"/settings/backends/{backend.id}/register-model",
            method="post",
            data_live_submit="true",
        ),
        id=f"backend-{backend.id}",
        cls="jouzetsu-dialog-section",
    )


def _model_editor(
    backend: BackendConfig,
    model_id: str,
    alias: str,
    vision: bool | None,
    instances: tuple[str, ...] | None,
) -> HTML:
    return Form(
        _snapshot(backend),
        Small(model_id),
        Input(type="hidden", name="model_id", value=model_id),
        field("Alias", Input(name="alias", value=alias)),
        Small(
            "Vision: "
            + (
                "unknown"
                if vision is None
                else "supported"
                if vision
                else "unsupported"
            )
        ),
        Small(
            "Load state unknown"
            if instances is None
            else "Loaded: " + ", ".join(instances)
            if instances
            else "Unloaded"
        )
        if backend.model_management is not None
        else None,
        Button("Save alias", type="submit", cls="jouzetsu-button"),
        *(
            Button(
                label,
                type="submit",
                formaction=f"/settings/backends/{backend.id}/{action}",
                cls="jouzetsu-button",
            )
            for label, action in (("Load", "load"), ("Unload", "unload"))
            if backend.model_management is not None
        ),
        action=f"/settings/backends/{backend.id}/alias",
        method="post",
        data_live_submit="true",
    )
