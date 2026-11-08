from __future__ import annotations

import unittest
from unittest.mock import patch

from jouzetsu.access import (
    access_decision,
    forget_pending_device,
    is_device_access_approval_phrase,
    refresh_seen_device,
    register_seen_device,
    set_device_access,
    set_device_label,
)
from jouzetsu.config import AppConfig, DeviceAccessSettings


class AccessTests(unittest.TestCase):
    def test_default_private_allows_localhost_without_device_id(self) -> None:
        config = AppConfig()

        decision = access_decision(config, client_ip="127.0.0.1", raw_device_id=None)

        self.assertTrue(decision.access_allowed)
        self.assertTrue(decision.can_manage_access)
        self.assertEqual(decision.reason, "localhost")

    def test_default_private_blocks_unknown_remote_until_device_cookie_exists(
        self,
    ) -> None:
        config = AppConfig()

        decision = access_decision(config, client_ip="192.168.1.20", raw_device_id=None)

        self.assertFalse(decision.access_allowed)
        self.assertFalse(decision.can_manage_access)
        self.assertEqual(decision.reason, "missing_device_id")

    def test_device_access_approval_phrase_requires_exact_match(self) -> None:
        self.assertTrue(
            is_device_access_approval_phrase(
                "configured-phrase", required_phrase="configured-phrase"
            )
        )
        self.assertFalse(
            is_device_access_approval_phrase(
                " configured-phrase ", required_phrase="configured-phrase"
            )
        )
        self.assertFalse(
            is_device_access_approval_phrase(
                "CONFIGURED-PHRASE", required_phrase="configured-phrase"
            )
        )
        self.assertFalse(
            is_device_access_approval_phrase(None, required_phrase="configured-phrase")
        )
        self.assertFalse(is_device_access_approval_phrase("", required_phrase=""))

    def test_remote_device_requires_explicit_allow(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()

        changed = register_seen_device(
            config, device_id=device_id, label="192.168.1.20"
        )
        pending = access_decision(
            config, client_ip="192.168.1.20", raw_device_id=device_id
        )
        set_device_access(config, device_id=device_id, access_allowed=True)
        approved = access_decision(
            config, client_ip="192.168.1.20", raw_device_id=device_id
        )

        self.assertTrue(changed)
        self.assertFalse(pending.access_allowed)
        self.assertEqual(pending.reason, "pending_device")
        self.assertFalse(pending.can_use_global_settings)
        self.assertTrue(approved.access_allowed)
        self.assertEqual(approved.reason, "approved_device")
        self.assertTrue(approved.can_use_global_settings)

    def test_device_label_can_be_updated(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        _ = register_seen_device(config, device_id=device_id, label="192.168.1.20")

        set_device_label(config, device_id=device_id, label="Phone")

        self.assertEqual(config.access.devices[device_id].label, "Phone")

    def test_pending_device_can_be_forgotten_but_approved_device_cannot(self) -> None:
        pending_device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        approved_device_id = "dvc_abcdef12-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        config.access.devices[pending_device_id] = DeviceAccessSettings()
        config.access.devices[approved_device_id] = DeviceAccessSettings(
            access_allowed=True
        )

        forget_pending_device(config, device_id=pending_device_id)

        self.assertNotIn(pending_device_id, config.access.devices)
        with self.assertRaisesRegex(
            ValueError, "approved devices must have access revoked instead"
        ):
            forget_pending_device(config, device_id=approved_device_id)
        self.assertIn(approved_device_id, config.access.devices)

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

    def test_device_activity_refresh_is_throttled_but_network_changes_persist(
        self,
    ) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        config.access.devices[device_id] = DeviceAccessSettings(
            label="Phone",
            last_ip="192.168.1.20",
            hostname="phone.local",
            first_seen_at="2026-08-26T12:00:00Z",
            last_seen_at="2026-08-26T12:00:00Z",
        )

        with patch(
            "jouzetsu.access.utc_timestamp", return_value="2026-08-26T12:01:00Z"
        ):
            changed = refresh_seen_device(
                config,
                device_id=device_id,
                label="Phone",
                last_ip="192.168.1.20",
            )

        self.assertFalse(changed)
        self.assertEqual(
            config.access.devices[device_id].last_seen_at, "2026-08-26T12:00:00Z"
        )

        with patch(
            "jouzetsu.access.utc_timestamp", return_value="2026-08-26T12:02:00Z"
        ):
            changed = refresh_seen_device(
                config,
                device_id=device_id,
                label="Phone",
                last_ip="192.168.1.20",
            )

        self.assertTrue(changed)
        self.assertEqual(
            config.access.devices[device_id].last_seen_at, "2026-08-26T12:02:00Z"
        )

        with patch(
            "jouzetsu.access.utc_timestamp", return_value="2026-08-26T12:02:30Z"
        ):
            changed = refresh_seen_device(
                config,
                device_id=device_id,
                label="Phone",
                last_ip="192.168.1.21",
                hostname="new-phone.local",
            )

        self.assertTrue(changed)
        self.assertEqual(config.access.devices[device_id].last_ip, "192.168.1.21")
        self.assertEqual(config.access.devices[device_id].hostname, "new-phone.local")
        self.assertEqual(
            config.access.devices[device_id].last_seen_at, "2026-08-26T12:02:30Z"
        )

        with patch(
            "jouzetsu.access.utc_timestamp", return_value="2026-08-26T12:03:00Z"
        ):
            changed = refresh_seen_device(
                config,
                device_id=device_id,
                label="Phone",
                last_ip="192.168.1.22",
                hostname_observed=True,
            )

        self.assertTrue(changed)
        self.assertEqual(config.access.devices[device_id].last_ip, "192.168.1.22")
        self.assertEqual(config.access.devices[device_id].hostname, "")

    def test_activity_refresh_does_not_recreate_a_forgotten_device(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()

        changed = refresh_seen_device(
            config,
            device_id=device_id,
            label="Phone",
            last_ip="192.168.1.20",
        )

        self.assertFalse(changed)
        self.assertNotIn(device_id, config.access.devices)

    def test_activity_refresh_does_not_rewrite_an_unchanged_generic_label(self) -> None:
        device_id = "dvc_12345678-abcd-4000-abcd-123456789abc"
        config = AppConfig()
        config.access.devices[device_id] = DeviceAccessSettings(
            label="localhost",
            last_ip="127.0.0.1",
            hostname="localhost",
            first_seen_at="2026-08-26T12:00:00Z",
            last_seen_at="2026-08-26T12:00:00Z",
        )

        with patch(
            "jouzetsu.access.utc_timestamp", return_value="2026-08-26T12:01:00Z"
        ):
            changed = refresh_seen_device(
                config,
                device_id=device_id,
                label="localhost",
                last_ip="127.0.0.1",
                hostname="localhost",
                hostname_observed=True,
            )

        self.assertFalse(changed)
        self.assertEqual(
            config.access.devices[device_id].last_seen_at, "2026-08-26T12:00:00Z"
        )

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

    def test_new_device_id_needs_hostname_and_last_ip_to_reuse_existing_device(
        self,
    ) -> None:
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


def test_ipv4_mapped_loopback_is_local_but_mapped_remote_is_not() -> None:
    config = AppConfig()
    local = access_decision(config, client_ip="::ffff:127.0.0.1", raw_device_id=None)
    remote = access_decision(config, client_ip="::ffff:192.0.2.1", raw_device_id=None)
    assert local.access_allowed and local.can_manage_access
    assert not remote.access_allowed and not remote.can_manage_access
