"""Default-private browser/device access helpers."""

from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import Logger
from typing import Final, Literal

from .config import AppConfig, DeviceAccessSettings

log: Logger = logging.getLogger(__name__)

DEVICE_ID_COOKIE_NAME: str = "jouzetsu_device_id"
DEVICE_ID_MAX_AGE_SECONDS: int = 60 * 60 * 24 * 365
_DEVICE_ID_PATTERN = re.compile(r"^dvc_[A-Za-z0-9_-]{8,160}$")
_LOCALHOST_LABELS: frozenset[str] = frozenset({"127.0.0.1", "0.0.0.0", "localhost"})
_DEVICE_ACTIVITY_PERSIST_INTERVAL: Final[timedelta] = timedelta(minutes=2)

AccessReason = Literal[
    "public", "localhost", "approved_device", "pending_device", "missing_device_id"
]


@dataclass(frozen=True)
class AccessDecision:
    """Result of evaluating one browser request against access config."""

    access_allowed: bool
    can_use_global_settings: bool
    can_manage_access: bool
    is_localhost: bool
    device_id: str
    reason: AccessReason


def utc_timestamp() -> str:
    """Return the compact UTC timestamp stored in config.json."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalise_device_id(raw: object) -> str:
    """Return a validated opaque browser/device id, or an empty string."""
    if not isinstance(raw, str):
        return ""
    device_id: str = raw.strip()
    return device_id if _DEVICE_ID_PATTERN.fullmatch(device_id) else ""


def is_device_access_approval_phrase(raw: object, *, required_phrase: str) -> bool:
    """Return whether a user-entered phrase grants access to the current device."""
    return bool(required_phrase) and raw == required_phrase


def is_localhost_ip(raw_ip: str) -> bool:
    """Return whether a client IP is loopback localhost."""
    if raw_ip in {"localhost", "ip6-localhost"}:
        return True
    try:
        return ipaddress.ip_address(raw_ip).is_loopback
    except ValueError:
        return False


def is_replaceable_localhost_label(label: str) -> bool:
    """Return whether a generic localhost label can be replaced by the host name."""
    return label.strip().casefold() in _LOCALHOST_LABELS


def _normalised_label(label: str) -> str:
    return label.strip().casefold()


def _label_conflict(config: AppConfig, *, device_id: str, label: str) -> str:
    normalised_label: str = _normalised_label(label)
    if not normalised_label:
        return ""
    for other_device_id, device in config.access.devices.items():
        if other_device_id == device_id:
            continue
        if _normalised_label(device.label) == normalised_label:
            return other_device_id
    return ""


def _normalised_hostname(hostname: str) -> str:
    return hostname.strip().casefold()


def _network_identity_device_id(
    config: AppConfig,
    *,
    device_id: str,
    last_ip: str,
    hostname: str,
) -> str:
    clean_last_ip: str = last_ip.strip()
    clean_hostname: str = _normalised_hostname(hostname)
    if not clean_last_ip or not clean_hostname:
        return ""

    matching_device_ids: list[str] = [
        other_device_id
        for other_device_id, device in config.access.devices.items()
        if other_device_id != device_id
        and device.last_ip.strip() == clean_last_ip
        and _normalised_hostname(device.hostname) == clean_hostname
    ]
    if len(matching_device_ids) > 1:
        matching_device_id_list: str = ", ".join(matching_device_ids)
        raise ValueError(
            f"multiple device records share hostname {hostname!r} and last_ip {last_ip!r}: {matching_device_id_list}"
        )
    return matching_device_ids[0] if matching_device_ids else ""


def _known_device_for_registration(
    config: AppConfig,
    *,
    device_id: str,
    last_ip: str,
    hostname: str,
) -> tuple[str, DeviceAccessSettings | None, bool]:
    device: DeviceAccessSettings | None = config.access.devices.get(device_id)
    if device is not None:
        return device_id, device, False

    if not config.access.allow_network_device_reassociation:
        return device_id, None, False

    matching_device_id: str = _network_identity_device_id(
        config,
        device_id=device_id,
        last_ip=last_ip,
        hostname=hostname,
    )
    if not matching_device_id:
        return device_id, None, False

    matched_device: DeviceAccessSettings = config.access.devices.pop(matching_device_id)
    config.access.devices[device_id] = matched_device
    log.info(
        "re-associated access device device_id=%s previous_device_id=%s",
        device_id,
        matching_device_id,
    )
    return device_id, matched_device, True


def access_decision(
    config: AppConfig, *, client_ip: str, raw_device_id: object
) -> AccessDecision:
    """Evaluate access for a request without mutating config."""
    device_id: str = normalise_device_id(raw_device_id)
    is_localhost: bool = is_localhost_ip(client_ip)
    can_manage_access: bool = is_localhost
    can_use_localhost_settings: bool = can_manage_access

    if not config.access.default_private:
        return AccessDecision(
            access_allowed=True,
            can_use_global_settings=can_use_localhost_settings,
            can_manage_access=can_manage_access,
            is_localhost=is_localhost,
            device_id=device_id,
            reason="public",
        )

    if is_localhost and config.access.allow_localhost_without_approval:
        return AccessDecision(
            access_allowed=True,
            can_use_global_settings=True,
            can_manage_access=can_manage_access,
            is_localhost=True,
            device_id=device_id,
            reason="localhost",
        )

    if not device_id:
        return AccessDecision(
            access_allowed=False,
            can_use_global_settings=can_use_localhost_settings,
            can_manage_access=can_manage_access,
            is_localhost=is_localhost,
            device_id="",
            reason="missing_device_id",
        )

    device: DeviceAccessSettings | None = config.access.devices.get(device_id)
    is_approved_device: bool = device is not None and device.access_allowed
    return AccessDecision(
        access_allowed=is_approved_device,
        can_use_global_settings=can_use_localhost_settings
        or (is_approved_device and config.access.global_settings_for_approved),
        can_manage_access=can_manage_access,
        is_localhost=is_localhost,
        device_id=device_id,
        reason="approved_device" if is_approved_device else "pending_device",
    )


def register_seen_device(
    config: AppConfig,
    *,
    device_id: str,
    label: str,
    last_ip: str = "",
    hostname: str = "",
) -> bool:
    """Record a new device or refresh a known device's activity."""

    return _record_seen_device(
        config,
        device_id=device_id,
        label=label,
        last_ip=last_ip,
        hostname=hostname,
        register_missing=True,
    )


def refresh_seen_device(
    config: AppConfig,
    *,
    device_id: str,
    label: str,
    last_ip: str = "",
    hostname: str = "",
    hostname_observed: bool = False,
) -> bool:
    """Refresh one existing device without recreating a forgotten request."""

    return _record_seen_device(
        config,
        device_id=device_id,
        label=label,
        last_ip=last_ip,
        hostname=hostname,
        hostname_observed=hostname_observed,
        register_missing=False,
    )


def _record_seen_device(
    config: AppConfig,
    *,
    device_id: str,
    label: str,
    last_ip: str,
    hostname: str,
    register_missing: bool,
    hostname_observed: bool = False,
) -> bool:
    """Apply one browser observation, throttling activity-only config writes."""

    normalised_device_id: str = normalise_device_id(device_id)
    if not normalised_device_id:
        return False

    now: str = utc_timestamp()
    clean_label: str = label.strip()
    clean_last_ip: str = last_ip.strip()
    clean_hostname: str = hostname.strip()
    if register_missing:
        canonical_device_id, device, changed = _known_device_for_registration(
            config,
            device_id=normalised_device_id,
            last_ip=clean_last_ip,
            hostname=clean_hostname,
        )
    else:
        canonical_device_id = normalised_device_id
        device = config.access.devices.get(canonical_device_id)
        changed = False
        if device is None:
            return False
    should_apply_label: bool = device is None or (
        bool(clean_label)
        and device.label != clean_label
        and (not device.label or is_replaceable_localhost_label(device.label))
    )
    if should_apply_label and _label_conflict(
        config, device_id=canonical_device_id, label=clean_label
    ):
        log.warning(
            "device label %r already belongs to another device; keeping existing label",
            clean_label,
        )
        clean_label = ""
    if device is None:
        config.access.devices[canonical_device_id] = DeviceAccessSettings(
            access_allowed=False,
            label=clean_label,
            last_ip=clean_last_ip,
            hostname=clean_hostname,
            first_seen_at=now,
            last_seen_at=now,
        )
        log.info(
            "registered pending access device device_id=%s label=%s",
            canonical_device_id,
            clean_label,
        )
        return True

    if should_apply_label and clean_label:
        device.label = clean_label
        changed = True
    if clean_last_ip and device.last_ip != clean_last_ip:
        device.last_ip = clean_last_ip
        changed = True
    if (clean_hostname or hostname_observed) and device.hostname != clean_hostname:
        device.hostname = clean_hostname
        changed = True
    if changed or _is_device_activity_update_due(device.last_seen_at, now=now):
        device.last_seen_at = now
        changed = True
    return changed


def _is_device_activity_update_due(last_seen_at: str, *, now: str) -> bool:
    """Return whether a browser observation merits a persisted activity update."""

    previous: datetime | None = _parse_utc_timestamp(last_seen_at)
    observed: datetime | None = _parse_utc_timestamp(now)
    if previous is None or observed is None:
        return True
    return observed - previous >= _DEVICE_ACTIVITY_PERSIST_INTERVAL


def _parse_utc_timestamp(value: str) -> datetime | None:
    """Parse a stored ISO timestamp, rejecting legacy naive values."""

    try:
        timestamp: datetime = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp.astimezone(UTC)


def _require_device(
    config: AppConfig, *, device_id: str
) -> tuple[str, DeviceAccessSettings]:
    normalised_device_id: str = normalise_device_id(device_id)
    if not normalised_device_id:
        raise ValueError("invalid device id")
    device: DeviceAccessSettings | None = config.access.devices.get(
        normalised_device_id
    )
    if device is None:
        raise KeyError(f"unknown device id: {normalised_device_id}")
    return normalised_device_id, device


def set_device_access(
    config: AppConfig, *, device_id: str, access_allowed: bool
) -> None:
    """Update one known device's allow flag."""
    _, device = _require_device(config, device_id=device_id)
    device.access_allowed = access_allowed


def forget_pending_device(config: AppConfig, *, device_id: str) -> None:
    """Remove one pending device record so a later request starts fresh."""

    normalised_device_id, device = _require_device(config, device_id=device_id)
    if device.access_allowed:
        raise ValueError("approved devices must have access revoked instead")
    del config.access.devices[normalised_device_id]


def set_device_label(config: AppConfig, *, device_id: str, label: str) -> None:
    """Update one known device's human-readable label."""
    normalised_device_id, device = _require_device(config, device_id=device_id)
    clean_label: str = label.strip()
    conflicting_device_id: str = _label_conflict(
        config, device_id=normalised_device_id, label=clean_label
    )
    if conflicting_device_id:
        raise ValueError("device label already in use")
    device.label = clean_label
