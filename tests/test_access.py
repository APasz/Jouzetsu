from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from jouzetsu.access import (
    access_decision,
    is_device_access_approval_phrase,
    register_seen_device,
    set_device_access,
    set_device_label,
)
from jouzetsu.config import AppConfig, AppPaths, ConfigStore, DeviceAccessSettings


class AccessTests(unittest.TestCase):
    def test_default_private_allows_localhost_without_device_id(self) -> None:
        config = AppConfig()

        decision = access_decision(config, client_ip="127.0.0.1", raw_device_id=None)

        self.assertTrue(decision.access_allowed)
        self.assertTrue(decision.can_manage_access)
        self.assertEqual(decision.reason, "localhost")

    def test_default_private_blocks_unknown_remote_until_device_cookie_exists(self) -> None:
        config = AppConfig()

        decision = access_decision(config, client_ip="192.168.1.20", raw_device_id=None)

        self.assertFalse(decision.access_allowed)
        self.assertFalse(decision.can_manage_access)
        self.assertEqual(decision.reason, "missing_device_id")

    def test_device_access_approval_phrase_requires_exact_match(self) -> None:
        self.assertTrue(is_device_access_approval_phrase("configured-phrase", required_phrase="configured-phrase"))
        self.assertFalse(is_device_access_approval_phrase(" configured-phrase ", required_phrase="configured-phrase"))
        self.assertFalse(is_device_access_approval_phrase("CONFIGURED-PHRASE", required_phrase="configured-phrase"))
        self.assertFalse(is_device_access_approval_phrase(None, required_phrase="configured-phrase"))
        self.assertFalse(is_device_access_approval_phrase("", required_phrase=""))

    def test_remote_device_requires_explicit_allow(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()

        changed = register_seen_device(config, device_id=device_id, label="192.168.1.20")
        pending = access_decision(config, client_ip="192.168.1.20", raw_device_id=device_id)
        set_device_access(config, device_id=device_id, access_allowed=True)
        approved = access_decision(config, client_ip="192.168.1.20", raw_device_id=device_id)

        self.assertTrue(changed)
        self.assertFalse(pending.access_allowed)
        self.assertEqual(pending.reason, "pending_device")
        self.assertFalse(pending.can_use_global_settings)
        self.assertTrue(approved.access_allowed)
        self.assertEqual(approved.reason, "approved_device")
        self.assertFalse(approved.can_use_global_settings)

    def test_approved_remote_device_can_use_global_settings_when_enabled(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        config.access.global_settings_for_approved = True
        config.access.devices[device_id] = DeviceAccessSettings(access_allowed=True)

        decision = access_decision(config, client_ip="192.168.1.20", raw_device_id=device_id)

        self.assertTrue(decision.access_allowed)
        self.assertTrue(decision.can_use_global_settings)
        self.assertFalse(decision.can_manage_access)

    def test_device_label_can_be_updated(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        _ = register_seen_device(config, device_id=device_id, label="192.168.1.20")

        set_device_label(config, device_id=device_id, label="Phone")

        self.assertEqual(config.access.devices[device_id].label, "Phone")

    def test_seen_device_records_network_details(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()

        changed = register_seen_device(
            config,
            device_id=device_id,
            label="192.168.1.20",
            last_ip="192.168.1.20",
            hostname="phone.local",
        )

        self.assertTrue(changed)
        self.assertEqual(config.access.devices[device_id].last_ip, "192.168.1.20")
        self.assertEqual(config.access.devices[device_id].hostname, "phone.local")

    def test_network_reassociation_is_disabled_by_default(self) -> None:
        old_device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        new_device_id = "dvc_abcdef12-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        config.access.devices[old_device_id] = DeviceAccessSettings(
            access_allowed=True,
            last_ip="192.168.1.20",
            hostname="phone.local",
        )

        _ = register_seen_device(
            config,
            device_id=new_device_id,
            label="192.168.1.20",
            last_ip="192.168.1.20",
            hostname="phone.local",
        )

        self.assertIn(old_device_id, config.access.devices)
        self.assertIn(new_device_id, config.access.devices)
        self.assertFalse(config.access.devices[new_device_id].access_allowed)

    def test_new_device_id_with_same_hostname_and_last_ip_reuses_existing_device_when_enabled(self) -> None:
        old_device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        new_device_id = "dvc_abcdef12-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        config.access.allow_network_device_reassociation = True
        config.access.devices[old_device_id] = DeviceAccessSettings(
            access_allowed=True,
            label="Phone",
            last_ip="192.168.1.20",
            hostname="phone.local",
            first_seen_at="2026-07-12T00:00:00Z",
            last_seen_at="2026-07-12T00:05:00Z",
        )

        changed = register_seen_device(
            config,
            device_id=new_device_id,
            label="192.168.1.20",
            last_ip="192.168.1.20",
            hostname="PHONE.LOCAL",
        )
        decision = access_decision(config, client_ip="192.168.1.20", raw_device_id=new_device_id)

        self.assertTrue(changed)
        self.assertNotIn(old_device_id, config.access.devices)
        self.assertIn(new_device_id, config.access.devices)
        self.assertTrue(config.access.devices[new_device_id].access_allowed)
        self.assertEqual(config.access.devices[new_device_id].label, "Phone")
        self.assertEqual(config.access.devices[new_device_id].first_seen_at, "2026-07-12T00:00:00Z")
        self.assertEqual(config.access.devices[new_device_id].last_ip, "192.168.1.20")
        self.assertEqual(config.access.devices[new_device_id].hostname, "PHONE.LOCAL")
        self.assertTrue(decision.access_allowed)
        self.assertEqual(decision.reason, "approved_device")

    def test_new_device_id_needs_hostname_and_last_ip_to_reuse_existing_device(self) -> None:
        old_device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        new_device_id = "dvc_abcdef12-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        config.access.devices[old_device_id] = DeviceAccessSettings(
            access_allowed=True,
            label="Phone",
            last_ip="192.168.1.20",
            hostname="phone.local",
        )

        changed = register_seen_device(
            config,
            device_id=new_device_id,
            label="192.168.1.20",
            last_ip="192.168.1.21",
            hostname="phone.local",
        )

        self.assertTrue(changed)
        self.assertIn(old_device_id, config.access.devices)
        self.assertIn(new_device_id, config.access.devices)
        self.assertFalse(config.access.devices[new_device_id].access_allowed)

    def test_device_label_conflicts_are_rejected(self) -> None:
        first_device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        second_device_id = "dvc_abcdef12-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        _ = register_seen_device(config, device_id=first_device_id, label="Phone")
        _ = register_seen_device(config, device_id=second_device_id, label="Tablet")

        with self.assertRaisesRegex(ValueError, "device label already in use"):
            set_device_label(config, device_id=second_device_id, label=" phone ")

        self.assertEqual(config.access.devices[second_device_id].label, "Tablet")

    def test_localhost_generic_label_is_replaced_by_hostname_label(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        _ = register_seen_device(config, device_id=device_id, label="127.0.0.1")

        changed = register_seen_device(config, device_id=device_id, label="workstation")

        self.assertTrue(changed)
        self.assertEqual(config.access.devices[device_id].label, "workstation")

    def test_access_config_round_trips_device_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
            store = ConfigStore.for_config_file(path)
            config = AppConfig(paths=AppPaths.for_config_file(path))
            config.access.devices[device_id] = DeviceAccessSettings(
                access_allowed=True,
                label="192.168.1.20",
                last_ip="192.168.1.20",
                hostname="phone.local",
                first_seen_at="2026-07-12T00:00:00Z",
                last_seen_at="2026-07-12T00:05:00Z",
            )
            store.save(config)

            reloaded = ConfigStore.for_config_file(path).load()
            saved = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))

            self.assertTrue(reloaded.access.default_private)
            self.assertTrue(reloaded.access.allow_localhost_without_approval)
            self.assertFalse(reloaded.access.global_settings_for_approved)
            self.assertEqual(reloaded.access.approval_phrase, "")
            self.assertTrue(reloaded.access.devices[device_id].access_allowed)
            self.assertEqual(reloaded.access.devices[device_id].last_ip, "192.168.1.20")
            self.assertEqual(reloaded.access.devices[device_id].hostname, "phone.local")
            access: dict[str, object] = cast(dict[str, object], saved["access"])
            devices: dict[str, dict[str, object]] = cast(dict[str, dict[str, object]], access["devices"])
            self.assertFalse(access["global_settings_for_approved"])
            self.assertEqual(access["approval_phrase"], "")
            self.assertEqual(devices[device_id]["label"], "192.168.1.20")
            self.assertEqual(devices[device_id]["last_ip"], "192.168.1.20")
            self.assertEqual(devices[device_id]["hostname"], "phone.local")
