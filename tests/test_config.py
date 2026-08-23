from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from jouzetsu.config import (
    APP_HOME_ENVIRONMENT_VARIABLE,
    AppPaths,
    ConfigStore,
    ConfigValidationError,
    EnvironmentOverrides,
    SpellingReplacement,
    ThemeSettings,
    default_config,
    encode_config,
    resolve_app_home,
)


class ConfigStoreTests(unittest.TestCase):
    def test_missing_config_creates_complete_indented_default_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir) / "Jouzetsu"

            config = ConfigStore.for_home(home).load()

            self.assertEqual(config.paths, AppPaths.for_home(home))
            self.assertEqual(config.chats_file, home / "data" / "chats.json")
            self.assertEqual(config.characters_directory, home / "data" / "characters")
            self.assertEqual(
                config.character_presets_directory, home / "data" / "character-presets"
            )
            self.assertEqual(config.log_directory, home / "data" / "logs")
            content = config.config_file.read_text(encoding="utf-8")
            self.assertIn('\n    "version": 1,', content)
            self.assertEqual(json.loads(content), encode_config(config))

    def test_custom_config_file_name_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "custom-settings.json"

            config = ConfigStore.for_config_file(config_file).load()

            self.assertEqual(config.config_file, config_file)
            self.assertTrue(config_file.is_file())
            self.assertFalse((config_file.parent / "config.json").exists())

    def test_packaged_defaults_are_used_but_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ConfigStore.for_home(Path(tmp_dir)).load()
            document = json.loads(config.config_file.read_text(encoding="utf-8"))

            self.assertGreater(len(config.generation.british_spelling_replacements), 20)
            self.assertEqual(config.theme, ThemeSettings())
            self.assertNotIn("british_spelling_replacements", document["generation"])
            self.assertNotIn("theme", document)

    def test_store_persists_custom_packaged_defaults_only_when_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ConfigStore.for_home(Path(tmp_dir))
            config = store.load()
            config.generation.british_spelling_replacements = [
                SpellingReplacement("color", "colour")
            ]
            theme_values = config.theme.values()
            theme_values["primary"] = "#112233"
            config.theme = ThemeSettings(**theme_values)

            store.save(config)

            saved = json.loads(config.config_file.read_text(encoding="utf-8"))
            self.assertEqual(
                saved["generation"]["british_spelling_replacements"],
                [{"source": "color", "replacement": "colour"}],
            )
            self.assertEqual(saved["theme"]["primary"], "#112233")
            self.assertEqual(len(saved["theme"]), len(config.theme.values()))

    def test_environment_overrides_are_never_written_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            persisted = ConfigStore.for_home(home).load()
            persisted.ui.host = "127.0.0.1"
            persisted.ui.port = 8081
            ConfigStore.for_home(home).save(persisted)

            store = ConfigStore(
                paths=AppPaths.for_home(home),
                overrides=EnvironmentOverrides(host="0.0.0.0", port=9090),
            )
            effective = store.load()
            self.assertEqual((effective.ui.host, effective.ui.port), ("0.0.0.0", 9090))

            effective.ui.dark_mode = False
            store.save(effective)

            reloaded = ConfigStore.for_home(home).load()
            self.assertEqual((reloaded.ui.host, reloaded.ui.port), ("127.0.0.1", 8081))
            self.assertFalse(reloaded.ui.dark_mode)

    def test_save_before_load_does_not_write_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = AppPaths.for_home(Path(tmp_dir))
            persisted = ConfigStore.for_home(paths.home).load()
            persisted.ui.host = "127.0.0.1"
            persisted.ui.port = 8081
            ConfigStore.for_home(paths.home).save(persisted)

            overrides = EnvironmentOverrides(host="0.0.0.0", port=9090)
            store = ConfigStore(paths=paths, overrides=overrides)
            effective = overrides.apply(default_config(paths))
            effective.ui.dark_mode = False
            store.save(effective)

            reloaded = ConfigStore.for_home(paths.home).load()
            self.assertEqual((reloaded.ui.host, reloaded.ui.port), ("127.0.0.1", 8081))
            self.assertFalse(reloaded.ui.dark_mode)

    def test_config_rejects_missing_sections_and_unknown_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = AppPaths.for_home(Path(tmp_dir))
            document = encode_config(default_config(paths))
            del document["access"]
            document["surprise"] = True
            paths.config_file.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaises(ConfigValidationError) as context:
                ConfigStore.for_home(paths.home).load()

            messages = str(context.exception)
            self.assertIn("config.access: is required", messages)
            self.assertIn("config.surprise: is not a recognised setting", messages)

    def test_config_rejects_invalid_values_without_coercion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = AppPaths.for_home(Path(tmp_dir))
            document = encode_config(default_config(paths))
            cast(dict[str, object], document["ui"])["port"] = "8080"
            cast(dict[str, object], document["generation"])["temperature"] = 3.0
            paths.config_file.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaises(ConfigValidationError) as context:
                ConfigStore.for_home(paths.home).load()

            messages = str(context.exception)
            self.assertIn("config.ui.port: must be an integer", messages)
            self.assertIn("temperature must be between 0 and 2", messages)

    def test_config_rejects_a_log_directory_that_contains_application_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = AppPaths.for_home(Path(tmp_dir))
            document = encode_config(default_config(paths))
            logging = cast(dict[str, object], document["logging"])
            logging["directory"] = ""
            paths.config_file.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                ConfigValidationError, "must not contain application data"
            ):
                ConfigStore.for_home(paths.home).load()

    def test_config_rejects_invalid_model_aliases_without_auto_unload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = AppPaths.for_home(Path(tmp_dir))
            document = encode_config(default_config(paths))
            server = cast(dict[str, object], document["server"])
            server["model_aliases"] = {"": "Demo model"}
            paths.config_file.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                ConfigValidationError, "model aliases must use non-empty"
            ):
                ConfigStore.for_home(paths.home).load()

    def test_config_requires_current_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = AppPaths.for_home(Path(tmp_dir))
            document = encode_config(default_config(paths))
            document["version"] = 0
            paths.config_file.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                ConfigValidationError, "config.version: must be 1"
            ):
                ConfigStore.for_home(paths.home).load()

    def test_config_store_rejects_a_config_for_another_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = ConfigStore.for_home(root / "first")
            second = ConfigStore.for_home(root / "second")

            with self.assertRaisesRegex(ValueError, "different path set"):
                first.save(second.load())


class AppHomeTests(unittest.TestCase):
    def test_explicit_home_takes_precedence_over_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            explicit = root / "explicit"
            configured = root / "configured"
            with patch.dict(
                "os.environ", {APP_HOME_ENVIRONMENT_VARIABLE: str(configured)}
            ):
                self.assertEqual(resolve_app_home(explicit), explicit.resolve())

    def test_blank_environment_home_is_rejected(self) -> None:
        with (
            patch.dict("os.environ", {APP_HOME_ENVIRONMENT_VARIABLE: "   "}),
            self.assertRaisesRegex(ValueError, APP_HOME_ENVIRONMENT_VARIABLE),
        ):
            resolve_app_home()

    def test_invalid_environment_port_is_rejected(self) -> None:
        with (
            patch.dict("os.environ", {"JOUZETSU_PORT": "0"}),
            self.assertRaisesRegex(ValueError, "between 1 and 65535"),
        ):
            EnvironmentOverrides.from_environment()
