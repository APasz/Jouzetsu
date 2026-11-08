"""Default-private browser/device access helpers."""

from __future__ import annotations

import ipaddress
import logging
import re
import secrets
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
    return (
        bool(required_phrase)
        and isinstance(raw, str)
        and secrets.compare_digest(raw.encode("utf-8"), required_phrase.encode("utf-8"))
    )


def is_localhost_ip(raw_ip: str) -> bool:
    """Return whether a client IP is loopback localhost."""
    if raw_ip in {"localhost", "ip6-localhost"}:
        return True
    try:
        address = ipaddress.ip_address(raw_ip)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        return address.is_loopback
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


def access_decision(
    config: AppConfig, *, client_ip: str, raw_device_id: object
) -> AccessDecision:
    """Evaluate access for a request without mutating config."""
    device_id: str = normalise_device_id(raw_device_id)
    is_localhost: bool = is_localhost_ip(client_ip)
    can_manage_access: bool = is_localhost
    can_use_localhost_settings: bool = can_manage_access

    if is_localhost:
        return AccessDecision(True, True, True, True, device_id, "localhost")

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
        can_use_global_settings=can_use_localhost_settings or is_approved_device,
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
    canonical_device_id = normalised_device_id
    device = config.access.devices.get(canonical_device_id)
    changed = False
    if device is None and not register_missing:
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
            "device label already belongs to another device; keeping existing label",
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
            "registered access device device_id=%s",
            canonical_device_id,
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
