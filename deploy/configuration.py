"""Public deployment defaults and explicitly local machine configuration."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from ipaddress import IPv4Network
from pathlib import Path
from string import Template
from typing import Final, cast

PROJECT_ROOT: Final[Path] = Path("/srv/projects/jouzetsu")
LOCAL_CONFIGURATION: Final[Path] = Path(__file__).with_name("local.toml")


@dataclass(frozen=True, slots=True)
class DeploymentSettings:
    """Machine-specific values kept outside version control."""

    ssh_host: str
    admin_user: str
    lan_interface: str
    lan_subnet: IPv4Network

    def __post_init__(self) -> None:
        for name, value, pattern in (
            ("ssh_host", self.ssh_host, r"[A-Za-z0-9_][A-Za-z0-9_.-]*"),
            ("admin_user", self.admin_user, r"[a-z_][a-z0-9_-]*[$]?"),
            ("lan_interface", self.lan_interface, r"[A-Za-z0-9_][A-Za-z0-9_.:-]*"),
        ):
            if re.fullmatch(pattern, value) is None:
                raise ValueError(f"invalid deployment {name}")


def load_deployment_settings(path: Path = LOCAL_CONFIGURATION) -> DeploymentSettings:
    """Require a local configuration rather than guessing a destination or network."""

    try:
        values = cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise FileNotFoundError("create deploy/local.toml from deploy/local.example.toml before deploying") from exc
    required = {"ssh_host", "admin_user", "lan_interface", "lan_subnet"}
    if set(values) != required:
        raise ValueError("deployment configuration must contain exactly: " + ", ".join(sorted(required)))

    def text(name: str) -> str:
        value = values[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"deployment {name} must be a non-empty string")
        return value

    return DeploymentSettings(
        ssh_host=text("ssh_host"),
        admin_user=text("admin_user"),
        lan_interface=text("lan_interface"),
        lan_subnet=IPv4Network(text("lan_subnet")),
    )


def render_firewall(template: str, settings: DeploymentSettings) -> str:
    """Substitute validated interface and subnet values into the public firewall template."""

    return Template(template).substitute(lan_interface=settings.lan_interface, lan_subnet=str(settings.lan_subnet))
