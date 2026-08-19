from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from jouzetsu.config import (
    APP_HOME_ENVIRONMENT_VARIABLE,
    DEFAULT_APP_HOME,
    DEFAULT_CONFIG_JSON,
    AppConfig,
    AppConfigJson,
    IconColorSettings,
    SpellingReplacement,
    ThemeSettings,
    load_config,
)


class LoadConfigTests(unittest.TestCase):
    def test_custom_data_directory_keeps_private_preset_packs_beside_characters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_directory: Path = Path(tmp_dir) / "data"
            config: AppConfig = AppConfig(data_dir=data_directory)

            config.ensure_directories()

            self.assertEqual(config.characters_directory, data_directory / "characters")
            self.assertEqual(config.character_presets_directory, data_directory / "character-presets")
            self.assertTrue(config.character_presets_directory.is_dir())

    def test_load_config_writes_defaults_and_applies_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"

            with patch.dict(os.environ, {"JOUZETSU_HOST": "0.0.0.0", "JOUZETSU_PORT": "9090"}, clear=False):
                config: AppConfig = load_config(path)

            self.assertTrue(path.exists())
            self.assertEqual(config.ui.host, "0.0.0.0")
            self.assertEqual(config.ui.port, 9090)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), DEFAULT_CONFIG_JSON)

    def test_explicit_app_home_contains_all_default_mutable_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home: Path = Path(tmp_dir) / "Jouzetsu"

            config: AppConfig = load_config(app_home=home)

            self.assertEqual(config.config_file, home / "config.json")
            self.assertEqual(config.data_dir, home / "data")
            self.assertEqual(config.chats_file, home / "data" / "chats.json")
            self.assertEqual(config.characters_directory, home / "data" / "characters")
            self.assertEqual(config.character_presets_directory, home / "data" / "character-presets")
            self.assertEqual(config.logging.directory, home / "data" / "logs")
            self.assertTrue(config.config_file.is_file())
            self.assertTrue(config.data_dir.is_dir())

    def test_environment_app_home_is_resolved_when_config_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home: Path = Path(tmp_dir) / "Jouzetsu"

            with patch.dict(os.environ, {APP_HOME_ENVIRONMENT_VARIABLE: str(home)}, clear=False):
                config: AppConfig = load_config()

            self.assertEqual(config.config_file, home / "config.json")
            self.assertEqual(config.data_dir, home / "data")

    def test_source_checkout_remains_the_default_app_home(self) -> None:
        self.assertEqual(DEFAULT_APP_HOME, Path(__file__).resolve().parents[1])

    def test_host_stats_visibility_settings_load_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps(
                    {
                        "host_stats": {
                            "system": False,
                            "cpu": True,
                            "gpu": False,
                            "network": True,
                            "gpus": {"0000:0a:00.0": {"visible": True, "label": "Radeon"}},
                            "interfaces": {
                                "enp5s0": {"visible": True, "label": "Ethernet"},
                                "wlan0": {"visible": False, "label": "Wi-Fi"},
                            },
                            "activity_start_color": "#0088ff",
                            "activity_end_color": "#ff0000",
                        }
                    }
                ),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertFalse(config.host_stats.system)
            self.assertTrue(config.host_stats.cpu)
            self.assertFalse(config.host_stats.gpu)
            self.assertTrue(config.host_stats.network)
            self.assertTrue(config.host_stats.gpus["0000:0a:00.0"].visible)
            self.assertEqual(config.host_stats.gpus["0000:0a:00.0"].label, "Radeon")
            self.assertTrue(config.host_stats.interfaces["enp5s0"].visible)
            self.assertEqual(config.host_stats.interfaces["enp5s0"].label, "Ethernet")
            self.assertFalse(config.host_stats.interfaces["wlan0"].visible)
            self.assertEqual(config.host_stats.interfaces["wlan0"].label, "Wi-Fi")
            self.assertEqual(config.host_stats.activity_start_color, "#0088ff")
            self.assertEqual(config.host_stats.activity_end_color, "#ff0000")

    def test_access_approval_phrase_loads_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            _ = path.write_text(json.dumps({"access": {"approval_phrase": "custom-phrase"}}), encoding="utf-8")

            config: AppConfig = load_config(path)

            self.assertEqual(config.access.approval_phrase, "custom-phrase")

    def test_network_device_reassociation_loads_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps({"access": {"allow_network_device_reassociation": True}}),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertTrue(config.access.allow_network_device_reassociation)

    def test_load_config_coerces_valid_scalars_and_falls_back_for_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps(
                    {
                        "server": {"base_url": 123, "api_key": "token"},
                        "generation": {"temperature": "0.5", "max_tokens": "512"},
                        "ui": {"host": 7, "port": "18080", "dark_mode": "yes", "auto_open_browser": False},
                    }
                ),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(config.server.base_url, DEFAULT_CONFIG_JSON["server"]["base_url"])
            self.assertEqual(config.server.api_key, "token")
            self.assertIsNone(config.server.auto_unload_minutes)
            self.assertEqual(config.generation.temperature, 0.5)
            self.assertEqual(config.generation.max_tokens, 512)
            self.assertFalse(config.generation.british_english)
            self.assertGreater(len(config.generation.british_spelling_replacements), 20)
            self.assertEqual(config.ui.host, DEFAULT_CONFIG_JSON["ui"]["host"])
            self.assertEqual(config.ui.port, 18080)
            self.assertEqual(config.ui.dark_mode, DEFAULT_CONFIG_JSON["ui"]["dark_mode"])
            self.assertFalse(config.ui.auto_open_browser)
            self.assertEqual(config.ui.active_chat_id, DEFAULT_CONFIG_JSON["ui"]["active_chat_id"])
            self.assertEqual(config.ui.message_action_icon_style, DEFAULT_CONFIG_JSON["ui"]["message_action_icon_style"])
            self.assertTrue(config.logging.enabled)
            self.assertEqual(config.logging.directory, path.parent / "data" / "logs")

    def test_invalid_port_override_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"

            with patch.dict(os.environ, {"JOUZETSU_PORT": "not-a-port"}, clear=False):
                with self.assertRaisesRegex(ValueError, "JOUZETSU_PORT must be an integer"):
                    _ = load_config(path)

    def test_config_save_writes_back_to_loaded_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "custom-config.json"
            _ = path.write_text(json.dumps(DEFAULT_CONFIG_JSON), encoding="utf-8")

            config: AppConfig = load_config(path)
            config.generation.system_prompt = "Custom prompt"
            config.generation.continuity_review = False
            config.generation.british_english = True
            config.generation.british_spelling_replacements = [SpellingReplacement("mom", "mum")]
            config.save()

            saved = cast(AppConfigJson, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(saved["generation"]["system_prompt"], "Custom prompt")
            self.assertFalse(saved["generation"]["continuity_review"])
            self.assertTrue(saved["generation"]["british_english"])
            self.assertEqual(saved["generation"]["british_spelling_replacements"], [{"source": "mom", "replacement": "mum"}])

    def test_british_english_generation_setting_loads_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps({"generation": {"british_english": True}}),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertTrue(config.generation.british_english)

    def test_continuity_review_generation_setting_loads_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps({"generation": {"continuity_review": False}}),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertFalse(config.generation.continuity_review)

    def test_british_spelling_replacements_load_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps(
                    {
                        "generation": {
                            "british_spelling_replacements": [
                                {"source": " Color ", "replacement": " Colour "},
                                {"source": "color", "replacement": "duplicate"},
                                {"source": "", "replacement": "blank"},
                                {"source": "favorite", "replacement": "favourite"},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(
                config.generation.british_spelling_replacements,
                [
                    SpellingReplacement("Color", "Colour"),
                    SpellingReplacement("favorite", "favourite"),
                ],
            )

    def test_model_aliases_are_validated_and_normalised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps(
                    {
                        "server": {
                            "model_aliases": {
                                " model-key ": " Friendly Name ",
                                "blank-alias": "   ",
                                "invalid-alias": 7,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(config.server.model_aliases, {"model-key": "Friendly Name"})

    def test_auto_unload_minutes_is_loaded_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps({"server": {"auto_unload_minutes": "15"}}),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(config.server.auto_unload_minutes, 15)

    def test_invalid_validated_sections_fall_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps(
                    {
                        "server": {"base_url": "http://example.test/v1", "auto_unload_minutes": 0},
                        "generation": {"temperature": "nan", "top_p": 1.5, "max_tokens": 0},
                    }
                ),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(config.server, AppConfig().server)
            self.assertEqual(config.generation, AppConfig().generation)

    def test_logging_settings_are_loaded_and_resolve_relative_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps({"logging": {"enabled": False, "directory": "custom-logs"}}),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertFalse(config.logging.enabled)
            self.assertEqual(config.logging.directory, path.parent / "custom-logs")

    def test_starter_prompts_preserve_distinct_labels_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps(
                    {
                        "ui": {
                            "starter_prompts": [
                                {"label": "Brief me", "content": "Give me a concise briefing about: "},
                                {"label": "Continue answer", "content": "Continue exactly where you stopped."},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(
                [(prompt.label, prompt.content) for prompt in config.ui.starter_prompts],
                [
                    ("Brief me", "Give me a concise briefing about: "),
                    ("Continue answer", "Continue exactly where you stopped."),
                ],
            )
            config.save()
            saved = cast(AppConfigJson, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(saved["ui"]["starter_prompts"][0]["label"], "Brief me")
            self.assertEqual(saved["ui"]["starter_prompts"][0]["content"], "Give me a concise briefing about: ")

    def test_message_action_icon_style_loads_and_saves_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps({"ui": {"message_action_icon_style": "muted_color"}}),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(config.ui.message_action_icon_style, "muted_color")
            config.save()
            saved = cast(AppConfigJson, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(saved["ui"]["message_action_icon_style"], "muted_color")

    def test_icon_colors_load_save_and_fall_back_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps(
                    {
                        "ui": {
                            "icon_colors": {
                                "linework_color": "#112233",
                                "accent_color": "not-a-colour",
                                "surface_color": "#AABBCC",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(config.ui.icon_colors.linework_color, "#112233")
            self.assertEqual(config.ui.icon_colors.accent_color, IconColorSettings().accent_color)
            self.assertEqual(config.ui.icon_colors.surface_color, "#aabbcc")
            config.save()
            saved = cast(AppConfigJson, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(saved["ui"]["icon_colors"]["linework_color"], "#112233")

    def test_theme_palette_loads_safely_and_persists_all_semantic_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps(
                    {
                        "theme": {
                            "primary": "#112233",
                            "text": "invalid",
                            "action_merge_muted": "#AABBCC",
                        }
                    }
                ),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(config.theme.primary, "#112233")
            self.assertEqual(config.theme.text, ThemeSettings().text)
            self.assertEqual(config.theme.action_merge_muted, "#aabbcc")
            config.save()
            saved = cast(AppConfigJson, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(saved["theme"], config.theme.to_dict())

    def test_invalid_message_action_icon_style_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "config.json"
            _ = path.write_text(
                json.dumps({"ui": {"message_action_icon_style": "rainbow"}}),
                encoding="utf-8",
            )

            config: AppConfig = load_config(path)

            self.assertEqual(config.ui.message_action_icon_style, AppConfig().ui.message_action_icon_style)
