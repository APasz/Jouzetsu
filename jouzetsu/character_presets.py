"""Validated declarative data for the character profile editor."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypedDict, cast

log: logging.Logger = logging.getLogger(__name__)
type CharacterFieldKind = Literal["short_text", "long_text"]
type CharacterPresetLayer = Literal["base", "extra"]
_CHARACTER_FIELD_KINDS: Final[frozenset[CharacterFieldKind]] = frozenset(
    {"short_text", "long_text"}
)
_CHARACTER_PRESET_LAYERS: Final[frozenset[CharacterPresetLayer]] = frozenset(
    {"base", "extra"}
)
_PACK_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9][a-z0-9-]{0,63}"
)
_BUILTIN_PACK_PATH: Final[Path] = (
    Path(__file__).with_name("character_preset_data") / "builtin.json"
)
_BUILTIN_CHARACTER_NAMES_PATH: Final[Path] = (
    Path(__file__).with_name("character_preset_data") / "names.json"
)


class CharacterPresetFieldClientValue(TypedDict):
    """The narrow JSON boundary consumed by the browser's field-row builder."""

    label: str
    kind: CharacterFieldKind
    value: str


@dataclass(frozen=True, slots=True)
class CharacterPresetField:
    """One editable field supplied by a profile template."""

    label: str
    kind: CharacterFieldKind = "short_text"
    value: str = ""

    def to_client_value(self) -> CharacterPresetFieldClientValue:
        """Return the compact shape consumed by the browser's field-row builder."""

        return {"label": self.label, "kind": self.kind, "value": self.value}


@dataclass(frozen=True, slots=True)
class CharacterFieldPreset:
    """A named collection of compatible profile fields from one preset pack."""

    id: str
    label: str
    description: str
    layer: CharacterPresetLayer
    fields: tuple[CharacterPresetField, ...]

    def fields_json(self) -> str:
        """Encode fields once for the static browser editor without duplicating the catalogue."""

        return json.dumps(
            [field.to_client_value() for field in self.fields], separators=(",", ":")
        )


@dataclass(frozen=True, slots=True)
class CharacterPresetPack:
    """One independently maintained collection of character field presets."""

    id: str
    label: str
    presets: tuple[CharacterFieldPreset, ...]


@dataclass(frozen=True, slots=True)
class CharacterPresetLoadIssue:
    """An actionable reason a private preset pack was not included in the catalogue."""

    path: Path
    message: str


@dataclass(frozen=True, slots=True)
class CharacterNameSuggestions:
    """The independently selectable name parts used by the onboarding randomiser."""

    given_names: tuple[str, ...]
    family_names: tuple[str, ...]

    def to_client_json(self) -> str:
        """Encode the compact, validated shape consumed by the browser editor."""

        return json.dumps(
            {
                "given_names": self.given_names,
                "family_names": self.family_names,
            },
            separators=(",", ":"),
        )

    def with_additions(
        self, additions: CharacterNameSuggestions
    ) -> CharacterNameSuggestions:
        """Append distinct user-provided suggestions without weighting duplicates."""

        return CharacterNameSuggestions(
            given_names=_merged_character_name_suggestions(
                self.given_names, additions.given_names
            ),
            family_names=_merged_character_name_suggestions(
                self.family_names, additions.family_names
            ),
        )


@dataclass(frozen=True, slots=True)
class _CharacterNameSuggestionDocument:
    """One validated custom-name document and its built-in-pool policy."""

    suggestions: CharacterNameSuggestions
    disable_vanilla: bool


@dataclass(frozen=True, slots=True)
class CharacterNameSuggestionLoadIssue:
    """An actionable reason optional user-supplied names were not included."""

    path: Path
    message: str


@dataclass(frozen=True, slots=True)
class CharacterNameSuggestionsCatalog:
    """Built-in name suggestions plus one optional user-supplied JSON file."""

    suggestions: CharacterNameSuggestions
    load_issue: CharacterNameSuggestionLoadIssue | None = None

    @classmethod
    def load(cls, custom_path: Path) -> CharacterNameSuggestionsCatalog:
        """Load custom names without preventing the built-in randomiser from working."""

        try:
            document: _CharacterNameSuggestionDocument = (
                _load_character_name_suggestion_document(
                    custom_path, allow_empty_pools=True
                )
            )
            additions: CharacterNameSuggestions = document.suggestions
            if not additions.given_names and not additions.family_names:
                raise ValueError(
                    f"{custom_path.name} must define at least one name suggestion"
                )
            if document.disable_vanilla:
                if not additions.given_names or not additions.family_names:
                    raise ValueError(
                        f"{custom_path.name} must define non-empty given_names and "
                        "family_names when disable_vanilla is true"
                    )
                return cls(additions)
        except FileNotFoundError:
            return cls(BUILTIN_CHARACTER_NAME_SUGGESTIONS)
        except (OSError, TypeError, ValueError) as exc:
            issue: CharacterNameSuggestionLoadIssue = CharacterNameSuggestionLoadIssue(
                path=custom_path, message=str(exc)
            )
            log.warning(
                "ignored character name suggestions path=%s reason=%s",
                custom_path,
                issue.message,
            )
            return cls(BUILTIN_CHARACTER_NAME_SUGGESTIONS, load_issue=issue)
        return cls(BUILTIN_CHARACTER_NAME_SUGGESTIONS.with_additions(additions))


@dataclass(frozen=True, slots=True)
class CharacterPresetCatalog:
    """One validated, collision-free catalogue used by character routes and views."""

    base_presets: tuple[CharacterFieldPreset, ...]
    extra_presets: tuple[CharacterFieldPreset, ...]
    load_issues: tuple[CharacterPresetLoadIssue, ...]
    _presets_by_id: Mapping[str, CharacterFieldPreset]

    @classmethod
    def from_packs(
        cls,
        packs: tuple[CharacterPresetPack, ...],
        *,
        load_issues: tuple[CharacterPresetLoadIssue, ...] = (),
    ) -> CharacterPresetCatalog:
        """Build a catalogue, rejecting ambiguous pack and preset identifiers."""

        known_pack_ids: set[str] = set()
        presets_by_id: dict[str, CharacterFieldPreset] = {}
        base_presets: list[CharacterFieldPreset] = []
        extra_presets: list[CharacterFieldPreset] = []
        for pack in packs:
            if pack.id in known_pack_ids:
                raise ValueError(f"duplicate character preset pack id {pack.id!r}")
            known_pack_ids.add(pack.id)
            for preset in pack.presets:
                if preset.id in presets_by_id:
                    raise ValueError(f"duplicate character preset id {preset.id!r}")
                presets_by_id[preset.id] = preset
                if preset.layer == "base":
                    base_presets.append(preset)
                elif preset.layer == "extra":
                    extra_presets.append(preset)
                else:
                    raise ValueError(
                        f"character preset {preset.id!r} has an unknown layer"
                    )
        return cls(
            base_presets=tuple(base_presets),
            extra_presets=tuple(extra_presets),
            load_issues=load_issues,
            _presets_by_id=MappingProxyType(presets_by_id),
        )

    @classmethod
    def load(cls, private_directory: Path) -> CharacterPresetCatalog:
        """Load release-safe built-ins and every valid private JSON pack in ``private_directory``."""

        builtin_pack: CharacterPresetPack = _load_preset_pack(_BUILTIN_PACK_PATH)
        packs: list[CharacterPresetPack] = [builtin_pack]
        issues: list[CharacterPresetLoadIssue] = []
        known_pack_ids: set[str] = {builtin_pack.id}
        known_preset_ids: set[str] = {preset.id for preset in builtin_pack.presets}

        if not private_directory.exists():
            return cls.from_packs(tuple(packs))
        if not private_directory.is_dir():
            raise ValueError(
                f"character preset directory is not a directory: {private_directory}"
            )

        for path in sorted(
            private_directory.glob("*.json"),
            key=lambda candidate: candidate.name.casefold(),
        ):
            try:
                pack: CharacterPresetPack = _load_preset_pack(path)
                _validate_pack_is_unique(pack, known_pack_ids, known_preset_ids)
            except (OSError, TypeError, ValueError) as exc:
                issue: CharacterPresetLoadIssue = CharacterPresetLoadIssue(
                    path=path, message=str(exc)
                )
                issues.append(issue)
                log.warning(
                    "ignored character preset pack path=%s reason=%s",
                    path,
                    issue.message,
                )
                continue
            packs.append(pack)
            known_pack_ids.add(pack.id)
            known_preset_ids.update(preset.id for preset in pack.presets)

        return cls.from_packs(tuple(packs), load_issues=tuple(issues))

    def selected(
        self, base_id: str, extra_ids: list[str]
    ) -> tuple[CharacterFieldPreset, ...]:
        """Resolve one optional base and a unique ordered set of extras from form values."""

        selected: list[CharacterFieldPreset] = []
        normalized_base_id: str = base_id.strip()
        if normalized_base_id:
            base_preset: CharacterFieldPreset = self._preset_with_layer(
                normalized_base_id, "base"
            )
            selected.append(base_preset)

        seen_extra_ids: set[str] = set()
        for raw_extra_id in extra_ids:
            extra_id: str = raw_extra_id.strip()
            if not extra_id or extra_id in seen_extra_ids:
                continue
            extra_preset: CharacterFieldPreset = self._preset_with_layer(
                extra_id, "extra"
            )
            selected.append(extra_preset)
            seen_extra_ids.add(extra_id)
        return tuple(selected)

    def _preset_with_layer(
        self, preset_id: str, layer: CharacterPresetLayer
    ) -> CharacterFieldPreset:
        """Return one known preset only when it belongs to the requested editor layer."""

        try:
            preset: CharacterFieldPreset = self._presets_by_id[preset_id]
        except KeyError as exc:
            raise ValueError(f"unknown character {layer} preset") from exc
        if preset.layer != layer:
            article: str = "an" if layer == "extra" else "a"
            raise ValueError(
                f"character preset {preset_id!r} is not {article} {layer} preset"
            )
        return preset


def _load_preset_pack(path: Path) -> CharacterPresetPack:
    """Read and validate one JSON pack, reporting its exact source path on failure."""

    try:
        raw: object = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc.msg}") from exc
    return _parse_preset_pack(raw, source_name=path.name)


def _parse_preset_pack(raw: object, *, source_name: str) -> CharacterPresetPack:
    """Validate a JSON document without allowing untyped data past this boundary."""

    pack: dict[str, object] = _json_object(raw, path=source_name)
    _reject_unexpected_keys(
        pack, {"schema_version", "id", "label", "presets"}, path=source_name
    )
    _require_schema_version(pack, path=source_name)

    pack_id: str = _preset_identifier(pack.get("id"), path=f"{source_name}.id")
    pack_label: str = _required_text(pack.get("label"), path=f"{source_name}.label")
    raw_presets: object = pack.get("presets")
    if not isinstance(raw_presets, list):
        raise TypeError(f"{source_name}.presets must be an array")
    if not raw_presets:
        raise ValueError(f"{source_name}.presets must be a non-empty array")
    preset_values: list[object] = cast(list[object], raw_presets)

    presets: list[CharacterFieldPreset] = []
    local_ids: set[str] = set()
    for index, raw_preset in enumerate(preset_values):
        preset_path: str = f"{source_name}.presets[{index}]"
        preset: dict[str, object] = _json_object(raw_preset, path=preset_path)
        _reject_unexpected_keys(
            preset,
            {"id", "label", "description", "layer", "fields"},
            path=preset_path,
        )
        local_id: str = _preset_identifier(preset.get("id"), path=f"{preset_path}.id")
        if local_id in local_ids:
            raise ValueError(
                f"{preset_path}.id duplicates preset id {local_id!r} in this pack"
            )
        local_ids.add(local_id)
        layer: CharacterPresetLayer = _preset_layer(
            preset.get("layer"), path=f"{preset_path}.layer"
        )
        raw_fields: object = preset.get("fields")
        if not isinstance(raw_fields, list):
            raise TypeError(f"{preset_path}.fields must be an array")
        if not raw_fields:
            raise ValueError(f"{preset_path}.fields must be a non-empty array")
        field_values: list[object] = cast(list[object], raw_fields)
        fields: tuple[CharacterPresetField, ...] = _parse_preset_fields(
            field_values, path=preset_path
        )
        presets.append(
            CharacterFieldPreset(
                id=f"{pack_id}:{local_id}",
                label=_required_text(preset.get("label"), path=f"{preset_path}.label"),
                description=_required_text(
                    preset.get("description"), path=f"{preset_path}.description"
                ),
                layer=layer,
                fields=fields,
            )
        )
    return CharacterPresetPack(id=pack_id, label=pack_label, presets=tuple(presets))


def load_character_name_suggestions(path: Path) -> CharacterNameSuggestions:
    """Load distinct, non-empty JSON pools for given and family-name suggestions."""

    return _load_character_name_suggestion_document(
        path, allow_empty_pools=False
    ).suggestions


def _load_character_name_suggestion_document(
    path: Path, *, allow_empty_pools: bool
) -> _CharacterNameSuggestionDocument:
    """Read one name-suggestion document with the caller's empty-pool policy."""

    try:
        raw: object = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc.msg}") from exc
    document: dict[str, object] = _json_object(raw, path=path.name)
    _reject_unexpected_keys(
        document,
        {"schema_version", "given_names", "family_names", "disable_vanilla"},
        path=path.name,
    )
    _require_schema_version(document, path=path.name)
    return _CharacterNameSuggestionDocument(
        suggestions=CharacterNameSuggestions(
            given_names=_character_name_suggestions(
                document.get("given_names"),
                path=f"{path.name}.given_names",
                description="given name",
                allow_empty=allow_empty_pools,
            ),
            family_names=_character_name_suggestions(
                document.get("family_names"),
                path=f"{path.name}.family_names",
                description="family name",
                allow_empty=allow_empty_pools,
            ),
        ),
        disable_vanilla=_boolean(
            document.get("disable_vanilla", False),
            path=f"{path.name}.disable_vanilla",
        ),
    )


def _character_name_suggestions(
    raw_names: object, *, path: str, description: str, allow_empty: bool
) -> tuple[str, ...]:
    """Validate one distinct name pool from a JSON document."""

    if not isinstance(raw_names, list):
        raise TypeError(f"{path} must be an array")
    if not raw_names:
        if allow_empty:
            return ()
        raise ValueError(f"{path} must be a non-empty array")
    name_values: list[object] = cast(list[object], raw_names)
    names: list[str] = []
    known_names: set[str] = set()
    for index, raw_name in enumerate(name_values):
        name_path: str = f"{path}[{index}]"
        name: str = _required_text(raw_name, path=name_path)
        name_key: str = name.casefold()
        if name_key in known_names:
            raise ValueError(f"{name_path} duplicates {description} {name!r}")
        names.append(name)
        known_names.add(name_key)
    return tuple(names)


def _merged_character_name_suggestions(
    base: tuple[str, ...], additions: tuple[str, ...]
) -> tuple[str, ...]:
    """Append names that do not already exist, case-insensitively, in the base pool."""

    merged: list[str] = list(base)
    known_names: set[str] = {name.casefold() for name in base}
    for name in additions:
        if name.casefold() in known_names:
            continue
        merged.append(name)
        known_names.add(name.casefold())
    return tuple(merged)


def _parse_preset_fields(
    raw_fields: list[object], *, path: str
) -> tuple[CharacterPresetField, ...]:
    """Parse one non-empty field list and reject duplicate profile labels."""

    fields: list[CharacterPresetField] = []
    labels: set[str] = set()
    for index, raw_field in enumerate(raw_fields):
        field_path: str = f"{path}.fields[{index}]"
        field: dict[str, object] = _json_object(raw_field, path=field_path)
        _reject_unexpected_keys(field, {"label", "kind", "value"}, path=field_path)
        label: str = _required_text(field.get("label"), path=f"{field_path}.label")
        label_key: str = label.casefold()
        if label_key in labels:
            raise ValueError(f"{field_path}.label duplicates {label!r} in this preset")
        labels.add(label_key)
        fields.append(
            CharacterPresetField(
                label=label,
                kind=_field_kind(
                    field.get("kind", "short_text"), path=f"{field_path}.kind"
                ),
                value=_optional_text(
                    field.get("value", ""), path=f"{field_path}.value"
                ),
            )
        )
    return tuple(fields)


def _json_object(raw: object, *, path: str) -> dict[str, object]:
    """Return a JSON-object boundary with string keys, or a focused validation error."""

    if not isinstance(raw, dict):
        raise TypeError(f"{path} must be an object")
    object_value: dict[object, object] = cast(dict[object, object], raw)
    if any(not isinstance(key, str) for key in object_value):
        raise TypeError(f"{path} contains a non-string key")
    return cast(dict[str, object], object_value)


def _reject_unexpected_keys(
    values: Mapping[str, object], allowed: set[str], *, path: str
) -> None:
    """Reject misspelled schema keys rather than silently dropping user input."""

    unexpected: list[str] = sorted(set(values).difference(allowed))
    if unexpected:
        raise ValueError(f"{path} contains unsupported key {unexpected[0]!r}")


def _require_schema_version(values: Mapping[str, object], *, path: str) -> None:
    """Require the one supported version for bundled character-data documents."""

    version: object = values.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(f"{path}.schema_version must be an integer")
    if version != 1:
        raise ValueError(f"{path}.schema_version must be 1")


def _boolean(raw: object, *, path: str) -> bool:
    """Return a JSON boolean without accepting integer lookalikes."""

    if not isinstance(raw, bool):
        raise TypeError(f"{path} must be a boolean")
    return raw


def _preset_identifier(raw: object, *, path: str) -> str:
    """Validate a namespace-safe machine identifier for a pack or local preset."""

    value: str = _required_text(raw, path=path)
    if not _PACK_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{path} must use lowercase letters, digits, and hyphens")
    return value


def _required_text(raw: object, *, path: str) -> str:
    """Return a trimmed non-empty text value."""

    if not isinstance(raw, str):
        raise TypeError(f"{path} must be text")
    value: str = raw.strip()
    if not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _optional_text(raw: object, *, path: str) -> str:
    """Return a text value while allowing the intentionally empty field default."""

    if not isinstance(raw, str):
        raise TypeError(f"{path} must be text")
    return raw


def _field_kind(raw: object, *, path: str) -> CharacterFieldKind:
    """Validate a supported profile input style."""

    if not isinstance(raw, str):
        raise TypeError(f"{path} must be text")
    if raw not in _CHARACTER_FIELD_KINDS:
        raise ValueError(f"{path} must be 'short_text' or 'long_text'")
    return raw


def _preset_layer(raw: object, *, path: str) -> CharacterPresetLayer:
    """Validate a supported preset layer."""

    if not isinstance(raw, str):
        raise TypeError(f"{path} must be text")
    if raw not in _CHARACTER_PRESET_LAYERS:
        raise ValueError(f"{path} must be 'base' or 'extra'")
    return raw


def _validate_pack_is_unique(
    pack: CharacterPresetPack,
    known_pack_ids: set[str],
    known_preset_ids: set[str],
) -> None:
    """Reject a whole user pack when it conflicts with an already loaded pack."""

    if pack.id in known_pack_ids:
        raise ValueError(f"duplicate character preset pack id {pack.id!r}")
    for preset in pack.presets:
        if preset.id in known_preset_ids:
            raise ValueError(f"duplicate character preset id {preset.id!r}")


def _load_builtin_character_preset_catalog() -> CharacterPresetCatalog:
    """Load the release-safe pack eagerly so a broken install fails at startup."""

    return CharacterPresetCatalog.from_packs((_load_preset_pack(_BUILTIN_PACK_PATH),))


BUILTIN_CHARACTER_PRESET_CATALOG: Final[CharacterPresetCatalog] = (
    _load_builtin_character_preset_catalog()
)
BUILTIN_CHARACTER_NAME_SUGGESTIONS: Final[CharacterNameSuggestions] = (
    load_character_name_suggestions(_BUILTIN_CHARACTER_NAMES_PATH)
)
BUILTIN_CHARACTER_NAME_SUGGESTIONS_JSON: Final[str] = (
    BUILTIN_CHARACTER_NAME_SUGGESTIONS.to_client_json()
)
BASE_CHARACTER_PRESETS: Final[tuple[CharacterFieldPreset, ...]] = (
    BUILTIN_CHARACTER_PRESET_CATALOG.base_presets
)
EXTRA_CHARACTER_PRESETS: Final[tuple[CharacterFieldPreset, ...]] = (
    BUILTIN_CHARACTER_PRESET_CATALOG.extra_presets
)


def selected_character_presets(
    base_id: str, extra_ids: list[str]
) -> tuple[CharacterFieldPreset, ...]:
    """Resolve built-in presets for compatibility with callers that do not load private packs."""

    return BUILTIN_CHARACTER_PRESET_CATALOG.selected(base_id, extra_ids)
