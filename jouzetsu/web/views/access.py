"""Access-pending and local access-management views."""

from __future__ import annotations

from pathlib import Path

from ...access import DEVICE_ID_MAX_AGE_SECONDS, AccessDecision
from ...config import DeviceAccessSettings
from ...log_files import discover_log_files, read_log_tail
from ...state import AppState
from ..html import (
    H1,
    H2,
    H3,
    HTML,
    A,
    Button,
    Details,
    Dialog,
    Div,
    Form,
    Input,
    P,
    Pre,
    Small,
    Span,
    Summary,
)
from .context import PageContext
from .controls import checkbox, close_dialog_button, csrf_context, field, post_button
from .feedback import dialog_feedback


def render_locked_access_page(
    context: PageContext,
    *,
    bootstrap_device: bool,
    approval_phrase_enabled: bool,
) -> HTML:
    """Render the deliberately data-free page shown to unapproved browsers."""

    decision: AccessDecision = context.decision
    current_device: HTML | None = (
        Div(
            Small("Browser device ID"),
            Span(
                decision.device_id,
                cls="jouzetsu-access-device-id",
                data_testid="access-device-id",
            ),
        )
        if decision.device_id
        else None
    )
    approve_current_device: HTML | None = (
        post_button(
            "Approve this browser",
            action="/access/current-device/approve",
            marker="approve-current-device-button",
            classes="jouzetsu-button jouzetsu-button-primary",
        )
        if decision.can_manage_access and decision.device_id
        else None
    )

    return Div(
        csrf_context(context),
        Div(
            Span("🔒", cls="jouzetsu-access-icon", aria_hidden="true"),
            H1("Access pending", cls="jouzetsu-access-title"),
            P(_locked_page_message(decision), cls="jouzetsu-access-message"),
            current_device,
            _pending_device_form(context, approval_phrase_enabled)
            if decision.device_id
            else None,
            approve_current_device,
            A(
                "Refresh",
                href="/",
                cls="jouzetsu-button jouzetsu-button-muted",
                data_testid="access-refresh-button",
            ),
            cls="jouzetsu-access-panel",
        ),
        cls="jouzetsu-access-page",
        data_testid="access-locked-page",
        data_device_cookie_bootstrap="true" if bootstrap_device else "",
        data_device_cookie_max_age=DEVICE_ID_MAX_AGE_SECONDS,
    )


def _locked_page_message(decision: AccessDecision) -> str:
    if decision.reason == "missing_device_id":
        return "Setting up this browser for approval…"
    if decision.can_manage_access:
        return "This browser is not approved for chats. Approve it below, then refresh."
    return "Waiting for approval from the Jouzetsu host."


def _pending_device_form(context: PageContext, approval_phrase_enabled: bool) -> HTML:
    phrase_control: HTML = (
        field(
            "Approval phrase",
            Input(
                name="approval_phrase",
                type="password",
                autocomplete="off",
                data_testid="access-approval-phrase-input",
            ),
        )
        if approval_phrase_enabled
        else Small(
            "The host can approve this browser from localhost.",
            cls="jouzetsu-form-help",
        )
    )
    return Form(
        field(
            "Browser name",
            Input(
                value=context.client_label,
                name="label",
                autocomplete="off",
                data_testid="access-current-device-label-input",
            ),
        ),
        phrase_control,
        Button(
            "Save browser details",
            type="submit",
            cls="jouzetsu-button",
            data_testid="save-pending-device-button",
        ),
        action="/access/current-device/pending",
        method="post",
        cls="jouzetsu-pending-device-form",
    )


def render_access_dialog(
    state: AppState,
    context: PageContext,
    *,
    notice: str = "",
    error: str = "",
) -> HTML:
    """Render localhost-only device approval, policy, and log controls."""

    return Dialog(
        Div(
            H2("Access & Logs", id="access-dialog-title", cls="jouzetsu-modal-title"),
            P(
                "Control browser approval, network access policy, and Jouzetsu logs.",
                id="access-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            dialog_feedback(notice=notice, error=error),
            _access_dialog_tabs(),
            _access_panel(state, context),
            _logs_panel(state),
            Div(
                close_dialog_button(
                    "access-dialog", marker="access-settings-close-button"
                ),
                cls="jouzetsu-dialog-actions",
            ),
            cls="jouzetsu-modal-card jouzetsu-global-settings-card",
        ),
        id="access-dialog",
        cls="jouzetsu-dialog",
        aria_labelledby="access-dialog-title",
        aria_describedby="access-dialog-description",
    )


def _access_dialog_tabs() -> HTML:
    return Div(
        Button(
            "Access",
            type="button",
            id="access-dialog-access-tab",
            role="tab",
            aria_controls="access-dialog-access-panel",
            aria_selected=True,
            tabindex=0,
            cls="jouzetsu-tab",
            data_tab_target="access-dialog-access-panel",
            data_testid="access-dialog-access-tab",
        ),
        Button(
            "Logs",
            type="button",
            id="access-dialog-logs-tab",
            role="tab",
            aria_controls="access-dialog-logs-panel",
            aria_selected=False,
            tabindex=-1,
            cls="jouzetsu-tab",
            data_tab_target="access-dialog-logs-panel",
            data_testid="access-dialog-logs-tab",
        ),
        role="tablist",
        aria_label="Access and logs",
        cls="jouzetsu-tab-list",
        data_tab_list="true",
    )


def _access_panel(state: AppState, context: PageContext) -> HTML:
    access = state.config.access
    return Div(
        Form(
            checkbox(
                "default_private",
                checked=access.default_private,
                label="Default private",
                marker="access-default-private",
            ),
            checkbox(
                "allow_localhost_without_approval",
                checked=access.allow_localhost_without_approval,
                label="Allow localhost without approval",
                marker="access-localhost-bypass",
            ),
            checkbox(
                "global_settings_for_approved",
                checked=access.global_settings_for_approved,
                label="Approved devices may use global settings",
                marker="access-global-settings-approved",
            ),
            checkbox(
                "allow_network_device_reassociation",
                checked=access.allow_network_device_reassociation,
                label="Allow network device reassociation",
                marker="access-network-reassociation",
            ),
            Div(
                Button(
                    "Save access defaults",
                    type="submit",
                    cls="jouzetsu-button jouzetsu-button-primary",
                ),
                cls="jouzetsu-dialog-actions",
            ),
            action="/access/defaults",
            method="post",
        ),
        H3("Known devices", cls="jouzetsu-section-title"),
        Div(
            *_device_rows(state, context),
            cls="jouzetsu-device-list",
            data_testid="access-device-list",
        ),
        id="access-dialog-access-panel",
        role="tabpanel",
        aria_labelledby="access-dialog-access-tab",
        cls="jouzetsu-tab-panel",
        data_tab_panel="access-dialog-access-panel",
    )


def _device_rows(state: AppState, context: PageContext) -> tuple[HTML, ...]:
    sorted_devices: list[tuple[str, DeviceAccessSettings]] = sorted(
        state.config.access.devices.items(),
        key=lambda item: item[1].last_seen_at or item[1].first_seen_at,
        reverse=True,
    )
    # The current localhost browser remains the recovery path if policy changes.
    return tuple(
        _device_row(device_id, device)
        for device_id, device in sorted_devices
        if device_id != context.decision.device_id
    )


def _device_row(device_id: str, device: DeviceAccessSettings) -> HTML:
    return Details(
        Summary(
            Span(device.label or device_id, data_live_device_label="true"),
            Span(
                " · allowed" if device.access_allowed else " · pending",
                data_live_device_access_status="true",
            ),
        ),
        Div(
            P(f"IP: {device.last_ip or 'Unknown'}"),
            P(f"Hostname: {device.hostname or 'Unknown'}"),
            P(device_id, cls="jouzetsu-mono", data_testid=f"device-access-{device_id}"),
            Form(
                checkbox(
                    "access_allowed",
                    checked=device.access_allowed,
                    label="Allowed",
                    marker=f"device-access-toggle-{device_id}",
                    live_autosave=True,
                ),
                action=f"/access/devices/{device_id}",
                method="post",
                data_live_submit="true",
                data_live_notice="Device access saved",
                data_live_device_access="true",
            ),
            Form(
                Input(
                    value=device.label,
                    name="label",
                    aria_label="Device label",
                    data_live_autosave="true",
                ),
                action=f"/access/devices/{device_id}/label",
                method="post",
                data_live_submit="true",
                data_live_notice="Device label saved",
                data_live_device_fallback=device_id,
            ),
            cls="jouzetsu-device-details",
        ),
        cls="jouzetsu-device-card",
    )


def _logs_panel(state: AppState) -> HTML:
    return Div(
        _log_viewer(state),
        id="access-dialog-logs-panel",
        role="tabpanel",
        aria_labelledby="access-dialog-logs-tab",
        hidden=True,
        cls="jouzetsu-tab-panel",
        data_tab_panel="access-dialog-logs-panel",
    )


def _log_viewer(state: AppState) -> HTML:
    log_files: tuple[Path, ...] = tuple(
        discover_log_files(state.config.log_directory)
    )
    if not log_files:
        return P(
            "No log files found.",
            cls="jouzetsu-empty-state",
            data_testid="access-log-empty-state",
        )

    views: list[HTML] = []
    tabs: list[HTML] = []
    for index, path in enumerate(log_files):
        tab_id: str = f"access-log-tab-{index}"
        panel_id: str = f"access-log-panel-{index}"
        tabs.append(_log_tab(path, tab_id, panel_id, selected=index == 0))
        views.append(_log_panel(path, tab_id, panel_id, selected=index == 0))
    return Div(
        Div(
            *tabs,
            role="tablist",
            aria_label="Log files",
            cls="jouzetsu-tab-list jouzetsu-log-tab-list",
            data_tab_list="true",
        ),
        Div(*views, cls="jouzetsu-log-tab-panels"),
        cls="jouzetsu-log-viewer",
        data_testid="access-log-viewer",
    )


def _log_tab(path: Path, tab_id: str, panel_id: str, *, selected: bool) -> HTML:
    return Button(
        path.name,
        type="button",
        id=tab_id,
        role="tab",
        aria_controls=panel_id,
        aria_selected=selected,
        tabindex=0 if selected else -1,
        cls="jouzetsu-tab jouzetsu-log-tab",
        data_tab_target=panel_id,
        data_testid=tab_id,
    )


def _log_panel(path: Path, tab_id: str, panel_id: str, *, selected: bool) -> HTML:
    try:
        contents: str = read_log_tail(path)
    except OSError as exc:
        contents = f"Unable to read {path.name}: {exc}"
    return Div(
        Pre(contents, cls="jouzetsu-log-output"),
        id=panel_id,
        role="tabpanel",
        aria_labelledby=tab_id,
        hidden=not selected,
        cls="jouzetsu-tab-panel jouzetsu-log-panel",
        data_tab_panel=panel_id,
        data_testid=panel_id,
    )
