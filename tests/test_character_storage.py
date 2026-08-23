from __future__ import annotations

import tempfile
from pathlib import Path

from jouzetsu.character_storage import CharacterStorage
from jouzetsu.models import Character, CharacterField


def test_character_storage_persists_each_profile_in_its_own_file() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory) / "characters"
        storage = CharacterStorage(directory)
        first = Character(
            name="Mira", fields=[CharacterField(label="Role", value="Pilot")]
        )
        second = Character(
            name="Sol", fields=[CharacterField(label="Species", value="Human")]
        )

        storage.save(first)
        storage.save(second)
        loaded = storage.load_all()

        assert storage.path_for(first.id).is_file()
        assert storage.path_for(second.id).is_file()
        assert {character.id for character in loaded} == {first.id, second.id}


def test_character_storage_quarantines_only_the_invalid_document() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory: Path = Path(temporary_directory) / "characters"
        storage = CharacterStorage(directory)
        valid = Character(name="Mira")
        storage.save(valid)
        damaged: Path = directory / "bad.json"
        _ = damaged.write_text("{", encoding="utf-8")

        loaded = storage.load_all()

        assert [character.id for character in loaded] == [valid.id]
        assert not damaged.exists()
        assert len(list(directory.glob("bad.json.corrupt-*"))) == 1
