from __future__ import annotations

from dataclasses import replace
from ipaddress import IPv4Network
from pathlib import Path

import pytest

from deploy.configuration import DeploymentSettings, load_deployment_settings, render_firewall

EXAMPLE = Path(__file__).resolve().parents[1] / "deploy" / "local.example.toml"


def test_documented_configuration_loads_and_renders_the_firewall() -> None:
    settings = load_deployment_settings(EXAMPLE)
    assert settings == DeploymentSettings("app-host.example", "operator", "eth0", IPv4Network("192.0.2.0/24"))
    template = EXAMPLE.with_name("jouzetsu.nft.in").read_text()
    rendered = render_firewall(template, replace(settings, lan_interface="eth1", lan_subnet=IPv4Network("198.51.100.0/24")))
    assert 'iifname "eth1" ip saddr 198.51.100.0/24 tcp dport 6900 accept' in rendered
    assert "${" not in rendered


def test_missing_configuration_requires_explicit_local_setup(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="create deploy/local.toml"):
        _ = load_deployment_settings(tmp_path / "absent.toml")


@pytest.mark.parametrize("original, invalid", [
    ('ssh_host = "app-host.example"', 'ssh_host = "--bad-option"'),
    ('admin_user = "operator"', 'admin_user = "invalid/account"'),
    ('lan_interface = "eth0"', 'lan_interface = "invalid interface"'),
    ('lan_subnet = "192.0.2.0/24"', 'lan_subnet = "192.0.2.1/24"'),
    ('lan_subnet = "192.0.2.0/24"', 'lan_subnet = "::/0"'),
    ('ssh_host = "app-host.example"', 'ssh_host = true'),
    ('admin_user = "operator"', 'admin_user = ""'),
    ('lan_interface = "eth0"', ''),
])
def test_invalid_or_missing_local_values_fail_before_deployment(tmp_path: Path, original: str, invalid: str) -> None:
    path = tmp_path / "local.toml"
    _ = path.write_text(EXAMPLE.read_text().replace(original, invalid))
    with pytest.raises(ValueError):
        _ = load_deployment_settings(path)


def test_unknown_configuration_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "local.toml"
    _ = path.write_text(EXAMPLE.read_text() + '\nunexpected = "value"\n')
    with pytest.raises(ValueError, match="exactly"):
        _ = load_deployment_settings(path)
