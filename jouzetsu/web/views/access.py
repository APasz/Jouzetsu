"""Access-pending and local access-management views."""

from __future__ import annotations

from pathlib import Path
from typing import Final

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
from .controls import (
    checkbox,
    close_dialog_button,
    csrf_context,
    field,
    form_submit_actions,
    post_button,
)
from .feedback import dialog_feedback
from .tabs import DialogTab, render_dialog_tab_list, render_dialog_tab_panel

_ACCESS_TAB: Final[DialogTab] = DialogTab("access", "Access")
_LOGS_TAB: Final[DialogTab] = DialogTab("logs", "Logs")
_ACCESS_DIALOG_TABS: Final[tuple[DialogTab, ...]] = (_ACCESS_TAB, _LOGS_TAB)


def render_locked_access_page(
    context: PageContext,
    *,
    bootstrap_device: bool,
    approval_phrase_enabled: bool,
    preview: bool = False,
) -> HTML:
    """Render the page shown to unapproved browsers or localhost previewers."""

    decision: AccessDecision = context.decision
    show_device_controls: bool = not preview and bool(decision.device_id)
    current_device: HTML | None = (
        Div(
            Small("Browser device ID"),
            Span(
                decision.device_id,
                cls="jouzetsu-access-device-id",
                data_testid="access-device-id",
            ),
        )
        if show_device_controls
        else None
    )
    approve_current_device: HTML | None = (
        post_button(
            "Approve this browser",
            action="/access/current-device/approve",
            marker="approve-current-device-button",
            classes="jouzetsu-button jouzetsu-button-primary",
        )
        if not preview and decision.can_manage_access and decision.device_id
        else None
    )

    return Div(
        csrf_context(context),
        Div(
            Span("🔒", cls="jouzetsu-access-icon", aria_hidden="true"),
            H1("Access pending", cls="jouzetsu-access-title"),
            P(
                _locked_page_message(decision, preview=preview),
                cls="jouzetsu-access-message",
            ),
            current_device,
            _pending_device_form(context, approval_phrase_enabled)
            if show_device_controls
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


def _locked_page_message(decision: AccessDecision, *, preview: bool = False) -> str:
    if preview:
        return "This is the page shown to browsers awaiting approval."
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
                _access_denied_page_button() if context.decision.is_localhost else None,
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


def _access_denied_page_button() -> HTML:
    """Link localhost administrators to the unapproved-browser page preview."""

    return A(
        "Open access denied page",
        href="/access/denied",
        target="_blank",
        rel="noopener",
        cls="jouzetsu-button",
        data_testid="access-denied-page-button",
    )


def _access_dialog_tabs() -> HTML:
    return render_dialog_tab_list(
        "access-dialog",
        _ACCESS_DIALOG_TABS,
        _ACCESS_TAB.name,
        label="Access and logs",
    )


def _access_panel(state: AppState, context: PageContext) -> HTML:
    access = state.config.access
    return render_dialog_tab_panel(
        "access-dialog",
        _ACCESS_TAB,
        _ACCESS_TAB.name,
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
            form_submit_actions(
                "Save access defaults",
                "save-access-defaults-button",
                reset_action="/access/defaults/reset",
                reset_marker="reset-access-defaults-button",
                reset_confirmation="Reset access defaults to their built-in values?",
            ),
            action="/access/defaults",
            method="post",
        ),
        H3("Known devices", cls="jouzetsu-section-title"),
        Div(
            *_device_sections(state, context),
            cls="jouzetsu-device-list",
            data_testid="access-device-list",
        ),
    )


def _device_sections(state: AppState, context: PageContext) -> tuple[HTML, ...]:
    sorted_devices: list[tuple[str, DeviceAccessSettings]] = sorted(
        state.config.access.devices.items(),
        key=lambda item: item[1].last_seen_at or item[1].first_seen_at,
        reverse=True,
    )
    current_device_id: str = context.decision.device_id
    current_device: DeviceAccessSettings | None = state.config.access.devices.get(
        current_device_id
    )
    other_devices: tuple[tuple[str, DeviceAccessSettings], ...] = tuple(
        (device_id, device)
        for device_id, device in sorted_devices
        if device_id != current_device_id
    )
    pending_devices: tuple[tuple[str, DeviceAccessSettings], ...] = tuple(
        (device_id, device)
        for device_id, device in other_devices
        if not device.access_allowed
    )
    approved_devices: tuple[tuple[str, DeviceAccessSettings], ...] = tuple(
        (device_id, device)
        for device_id, device in other_devices
        if device.access_allowed
    )
    sections: list[HTML] = []
    if current_device is not None:
        sections.append(
            Div(
                H3("This browser", cls="jouzetsu-device-section-title"),
                _device_row(
                    current_device_id,
                    current_device,
                    is_current_browser=True,
                ),
                cls="jouzetsu-device-section is-current",
                data_testid="access-current-device",
            )
        )
    if pending_devices:
        sections.append(
            _device_section(
                "Needs review",
                pending_devices,
                data_testid="access-pending-devices",
            )
        )
    if approved_devices:
        sections.append(
            _device_section(
                "Approved",
                approved_devices,
                data_testid="access-approved-devices",
            )
        )
    if sections:
        return tuple(sections)
    return (
        P(
            "No browser devices have requested access yet.",
            cls="jouzetsu-form-help",
            data_testid="access-device-empty-state",
        ),
    )


def _device_section(
    title: str,
    devices: tuple[tuple[str, DeviceAccessSettings], ...],
    *,
    data_testid: str,
) -> HTML:
    """Render one status group while preserving the last-active sort order."""

    return Div(
        Div(
            H3(title, cls="jouzetsu-device-section-title"),
            Small(f"{len(devices)} device" + ("s" if len(devices) != 1 else "")),
            cls="jouzetsu-device-section-header",
        ),
        *(_device_row(device_id, device) for device_id, device in devices),
        cls="jouzetsu-device-section",
        data_testid=data_testid,
    )


def _device_row(
    device_id: str,
    device: DeviceAccessSettings,
    *,
    is_current_browser: bool = False,
) -> HTML:
    """Render one device with immediately visible security state and identity."""

    status: str = "Allowed" if device.access_allowed else "Pending"
    state_class: str = "is-allowed" if device.access_allowed else "is-pending"
    card_classes: str = " ".join(
        (
            "jouzetsu-device-card",
            state_class,
            "is-current" if is_current_browser else "",
        )
    ).strip()
    return Details(
        Summary(
            Div(
                Div(
                    Span(
                        device.label or device_id,
                        cls="jouzetsu-device-name",
                        data_testid=f"device-access-{device_id}",
                    ),
                    Div(
                        Span(status, cls=f"jouzetsu-device-status {state_class}"),
                        Span(
                            "This browser",
                            cls="jouzetsu-device-current-label",
                        )
                        if is_current_browser
                        else None,
                        cls="jouzetsu-device-state",
                    ),
                    cls="jouzetsu-device-heading",
                ),
                Span(
                    _device_network_summary(device),
                    cls="jouzetsu-device-network",
                ),
                Small(
                    _device_last_active_summary(device),
                    cls="jouzetsu-device-last-active",
                ),
                cls="jouzetsu-device-identity",
            ),
            cls="jouzetsu-device-card-summary",
            data_testid=f"device-card-toggle-{device_id}",
        ),
        _device_details(
            device_id,
            device,
            is_current_browser=is_current_browser,
        ),
        cls=card_classes,
    )


def _device_network_summary(device: DeviceAccessSettings) -> str:
    """Summarise the network identity without exposing empty placeholder fields."""

    details: list[str] = [
        detail for detail in (device.hostname, device.last_ip) if detail
    ]
    return " · ".join(details) if details else "Network details unavailable"


def _device_last_active_summary(device: DeviceAccessSettings) -> str:
    """Return a useful activity label for records that lack recent timestamps."""

    timestamp: str = device.last_seen_at or device.first_seen_at
    return f"Last active {timestamp}" if timestamp else "Activity time unknown"


def _device_access_action(device_id: str, *, access_allowed: bool) -> HTML:
    """Render the one intentional access-state transition for a non-local device."""

    next_access_allowed: bool = not access_allowed
    return Form(
        Input(
            type="hidden",
            name="access_allowed",
            value="true" if next_access_allowed else "false",
        ),
        Button(
            "Approve" if next_access_allowed else "Revoke access",
            type="submit",
            cls=(
                "jouzetsu-button jouzetsu-button-primary"
                if next_access_allowed
                else "jouzetsu-button jouzetsu-button-danger"
            ),
            data_confirm=(
                "Revoke this device's access? It will need approval to return."
                if not next_access_allowed
                else None
            ),
            data_testid=(
                f"approve-device-{device_id}"
                if next_access_allowed
                else f"revoke-device-{device_id}"
            ),
        ),
        action=f"/access/devices/{device_id}",
        method="post",
        cls="jouzetsu-inline-form",
    )


def _device_details(
    device_id: str,
    device: DeviceAccessSettings,
    *,
    is_current_browser: bool,
) -> HTML:
    """Render the expanded identity and access controls for one device card."""

    actions: HTML | None = _device_actions(
        device_id,
        access_allowed=device.access_allowed,
        is_current_browser=is_current_browser,
    )
    return Div(
        actions,
        _device_metadata(device_id, device),
        Form(
            field(
                "Device name",
                Input(
                    value=device.label,
                    name="label",
                    autocomplete="off",
                    data_testid=f"device-label-{device_id}",
                ),
            ),
            Button(
                "Save name",
                type="submit",
                cls="jouzetsu-button",
                data_testid=f"save-device-label-{device_id}",
            ),
            action=f"/access/devices/{device_id}/label",
            method="post",
            cls="jouzetsu-device-form",
        ),
        cls="jouzetsu-device-card-details",
    )


def _device_actions(
    device_id: str,
    *,
    access_allowed: bool,
    is_current_browser: bool,
) -> HTML | None:
    """Render non-local device actions, including pending-request dismissal."""

    if is_current_browser:
        return None
    actions: list[HTML] = [
        _device_access_action(device_id, access_allowed=access_allowed)
    ]
    if not access_allowed:
        actions.append(_forget_pending_device_action(device_id))
    return Div(*actions, cls="jouzetsu-device-card-actions")


def _forget_pending_device_action(device_id: str) -> HTML:
    """Render the confirmed discard action beside a pending-device approval."""

    return Form(
        Button(
            "Forget",
            type="submit",
            cls="jouzetsu-button jouzetsu-button-danger",
            data_confirm=(
                "Forget this pending device? Its access request will be removed."
            ),
            data_testid=f"forget-device-{device_id}",
        ),
        action=f"/access/devices/{device_id}/forget",
        method="post",
        cls="jouzetsu-inline-form",
    )


def _device_metadata(device_id: str, device: DeviceAccessSettings) -> HTML:
    """Render the complete stored identity data in a compact, scannable grid."""

    return Div(
        _device_metadata_item("Hostname", device.hostname or "Unknown"),
        _device_metadata_item("IP address", device.last_ip or "Unknown"),
        _device_metadata_item("First seen", device.first_seen_at or "Unknown"),
        _device_metadata_item("Last active", device.last_seen_at or "Unknown"),
        _device_metadata_item("Device ID", device_id, is_monospace=True),
        cls="jouzetsu-device-metadata",
    )


def _device_metadata_item(
    label: str, value: str, *, is_monospace: bool = False
) -> HTML:
    """Render one labelled device identity value."""

    return Div(
        Span(label, cls="jouzetsu-device-metadata-label"),
        Span(
            value,
            cls=(
                "jouzetsu-device-metadata-value jouzetsu-mono"
                if is_monospace
                else "jouzetsu-device-metadata-value"
            ),
        ),
        cls="jouzetsu-device-metadata-item",
    )


def _logs_panel(state: AppState) -> HTML:
    return render_dialog_tab_panel(
        "access-dialog",
        _LOGS_TAB,
        _ACCESS_TAB.name,
        _log_viewer(state),
    )


def _log_viewer(state: AppState) -> HTML:
    log_files: tuple[Path, ...] = tuple(discover_log_files(state.config.log_directory))
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
        aria_selected="true" if selected else "false",
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
