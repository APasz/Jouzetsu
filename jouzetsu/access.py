"""Default-private browser/device access helpers."""

from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import Logger
from typing import Literal

from .config import AppConfig, DeviceAccessSettings

log: Logger = logging.getLogger(__name__)

DEVICE_ID_COOKIE_NAME: str = "jouzetsu_device_id"
DEVICE_ID_MAX_AGE_SECONDS: int = 60 * 60 * 24 * 365
_DEVICE_ID_PATTERN = re.compile(r"^dvc_[A-Za-z0-9_-]{8,160}$")
_LOCALHOST_LABELS: frozenset[str] = frozenset({"127.0.0.1", "0.0.0.0", "localhost"})

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
    """Record or refresh a pending/known device. Returns whether config changed."""
    normalised_device_id: str = normalise_device_id(device_id)
    if not normalised_device_id:
        return False

    now: str = utc_timestamp()
    clean_label: str = label.strip()
    clean_last_ip: str = last_ip.strip()
    clean_hostname: str = hostname.strip()
    canonical_device_id, device, changed = _known_device_for_registration(
        config,
        device_id=normalised_device_id,
        last_ip=clean_last_ip,
        hostname=clean_hostname,
    )
    if _label_conflict(config, device_id=normalised_device_id, label=clean_label):
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

    should_update_label: bool = bool(clean_label) and (
        not device.label or is_replaceable_localhost_label(device.label)
    )
    if should_update_label:
        device.label = clean_label
        changed = True
    if clean_last_ip and device.last_ip != clean_last_ip:
        device.last_ip = clean_last_ip
        changed = True
    if clean_hostname and device.hostname != clean_hostname:
        device.hostname = clean_hostname
        changed = True
    if device.last_seen_at != now:
        device.last_seen_at = now
        changed = True
    return changed


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
