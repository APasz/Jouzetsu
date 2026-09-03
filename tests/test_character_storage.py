from __future__ import annotations

import json
import tempfile
from pathlib import Path

from jouzetsu.character_storage import CharacterStorage
from jouzetsu.models import Character, CharacterField, CharacterName


def test_character_storage_persists_each_profile_in_its_own_file() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory) / "characters"
        storage = CharacterStorage(directory)
        first = Character(
            name_parts=CharacterName("Mira"),
            fields=[CharacterField(label="Role", value="Pilot")],
        )
        second = Character(
            name_parts=CharacterName("Sol"),
            fields=[CharacterField(label="Species", value="Human")],
        )

        storage.save(first)
        storage.save(second)
        loaded = storage.load_all()

        assert storage.path_for(first.id).is_file()
        assert storage.path_for(second.id).is_file()
        assert {character.id for character in loaded} == {first.id, second.id}
        saved = json.loads(storage.path_for(first.id).read_text(encoding="utf-8"))
        assert saved["schema_version"] == 1
        assert saved["is_draft"] is False


def test_character_storage_loads_an_unversioned_document_and_round_trips_it_as_v1() -> (
    None
):
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory) / "characters"
        storage = CharacterStorage(directory)
        legacy = Character(name_parts=CharacterName("Mira"))
        legacy_document: dict[str, object] = dict(legacy.to_dict())
        _ = legacy_document.pop("schema_version")
        _ = legacy_document.pop("is_draft")
        _ = storage.path_for(legacy.id).write_text(
            json.dumps(legacy_document), encoding="utf-8"
        )

        [loaded] = storage.load_all()
        storage.save(loaded)

        saved = json.loads(storage.path_for(legacy.id).read_text(encoding="utf-8"))
        assert loaded == legacy
        assert saved["schema_version"] == 1
        assert saved["is_draft"] is False


def test_character_storage_leaves_a_future_schema_document_intact() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory) / "characters"
        storage = CharacterStorage(directory)
        character = Character(name_parts=CharacterName("Mira"))
        future_document: dict[str, object] = dict(character.to_dict())
        future_document["schema_version"] = 2
        path = storage.path_for(character.id)
        _ = path.write_text(json.dumps(future_document), encoding="utf-8")

        assert storage.load_all() == []
        assert json.loads(path.read_text(encoding="utf-8")) == future_document
        assert list(directory.glob(f"{path.name}.corrupt-*")) == []


def test_character_storage_quarantines_only_the_invalid_document() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory) / "characters"
        storage = CharacterStorage(directory)
        valid = Character(name_parts=CharacterName("Mira"))
        storage.save(valid)
        damaged: Path = directory / "bad.json"
        _ = damaged.write_text("{", encoding="utf-8")

        loaded = storage.load_all()

        assert [character.id for character in loaded] == [valid.id]
        assert not damaged.exists()
        assert len(list(directory.glob("bad.json.corrupt-*"))) == 1
