"""Individual JSON-document persistence for reusable character profiles."""

from __future__ import annotations

import json
import logging
from logging import Logger
from pathlib import Path
from typing import cast

from .atomic_write import (
    DurableWriteGuard,
    atomic_write_text,
    ensure_directory,
    remove_path,
)
from .models import Character, FutureSchemaVersionError

log: Logger = logging.getLogger(__name__)


class CharacterStorage:
    """Persist each character independently so one damaged profile cannot affect others."""

    def __init__(self, directory: Path) -> None:
        self.directory: Path = Path(directory)
        self._write_guard: DurableWriteGuard = DurableWriteGuard()
        self._isolated_ids: set[str] = set()
        ensure_directory(self.directory)

    def load_all(self) -> list[Character]:
        """Load valid profiles, leaving forward-incompatible documents untouched."""

        characters: list[Character] = []
        self._isolated_ids.clear()
        for path in sorted(self.directory.glob("*.json")):
            character: Character | None = self._load_one(path)
            if character is not None:
                characters.append(character)
        log.info(
            "loaded character store directory=%s character_count=%d",
            self.directory,
            len(characters),
        )
        return characters

    @property
    def isolated_count(self) -> int:
        """Number of unreadable documents excluded from the loaded library."""
        return len(self._isolated_ids)

    def save(self, character: Character) -> None:
        """Atomically replace the one document for ``character``."""

        data: str = json.dumps(character.to_dict(), ensure_ascii=False, indent=4) + "\n"
        path: Path = self.path_for(character.id)
        if character.id in self._isolated_ids:
            raise ValueError("cannot overwrite an isolated character document")
        with self._write_guard.protect(path):
            atomic_write_text(path, data)
        log.info(
            "saved character path=%s character_id=%s revision=%d",
            path,
            character.id,
            character.revision,
        )

    def delete(self, character_id: str) -> None:
        """Remove the known profile document after its id was resolved in application state."""

        path: Path = self.path_for(character_id)
        with self._write_guard.protect(path):
            remove_path(path)
        log.info("deleted character path=%s character_id=%s", path, character_id)

    def path_for(self, character_id: str) -> Path:
        """Return an in-directory path for one generated document identifier."""

        if not character_id or not character_id.isalnum():
            raise ValueError("character id must be alphanumeric")
        return self.directory / f"{character_id}.json"

    def _load_one(self, path: Path) -> Character | None:
        try:
            raw: object = cast(object, json.loads(path.read_text(encoding="utf-8")))
            if not isinstance(raw, dict):
                raise TypeError("root must be an object")
            mapping: dict[object, object] = cast(dict[object, object], raw)
            character: Character = Character.from_dict(
                {key: value for key, value in mapping.items() if isinstance(key, str)}
            )
            if path.stem != character.id:
                raise ValueError("document id does not match its file name")
            return character
        except FutureSchemaVersionError as exc:
            self._isolated_ids.add(path.stem)
            log.warning(
                "skipped future character document path=%s reason=%s", path, exc
            )
            return None
        except Exception as exc:  # noqa: BLE001
            self._isolated_ids.add(path.stem)
            log.error(
                "isolated invalid character document path=%s error_type=%s",
                path,
                type(exc).__name__,
            )
            return None
