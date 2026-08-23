from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from jouzetsu.character_presets import (
    BASE_CHARACTER_PRESETS,
    EXTRA_CHARACTER_PRESETS,
    CharacterPresetCatalog,
)


def _pack_document(
    *, pack_id: str, preset_id: str, layer: str = "extra"
) -> dict[str, object]:
    """Return one minimal valid pack document for filesystem-loader tests."""

    return {
        "schema_version": 1,
        "id": pack_id,
        "label": f"{pack_id} pack",
        "presets": [
            {
                "id": preset_id,
                "label": "Custom details",
                "description": "Fields supplied by a private pack.",
                "layer": layer,
                "fields": [
                    {"label": "Private note", "kind": "long_text"},
                    {"label": "Rating", "value": "Unrated"},
                ],
            }
        ],
    }


def _write_pack(directory: Path, name: str, document: dict[str, object]) -> Path:
    """Write one JSON test pack and return its path."""

    path: Path = directory / name
    _ = path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_builtin_presets_are_loaded_from_the_release_safe_pack() -> None:
    assert {preset.layer for preset in BASE_CHARACTER_PRESETS} == {"base"}
    assert {preset.layer for preset in EXTRA_CHARACTER_PRESETS} == {"extra"}
    assert all(
        preset.id.startswith("builtin:")
        for preset in (*BASE_CHARACTER_PRESETS, *EXTRA_CHARACTER_PRESETS)
    )


def test_catalog_loads_private_packs_in_filename_order_and_namespaces_ids(
    tmp_path: Path,
) -> None:
    _ = _write_pack(
        tmp_path, "zeta.json", _pack_document(pack_id="zeta", preset_id="extra")
    )
    _ = _write_pack(
        tmp_path,
        "alpha.json",
        _pack_document(pack_id="alpha", preset_id="base", layer="base"),
    )

    catalog: CharacterPresetCatalog = CharacterPresetCatalog.load(tmp_path)

    assert [preset.id for preset in catalog.base_presets][-1] == "alpha:base"
    assert [preset.id for preset in catalog.extra_presets][-1] == "zeta:extra"
    selected = catalog.selected("alpha:base", ["zeta:extra", "zeta:extra"])
    assert [preset.id for preset in selected] == ["alpha:base", "zeta:extra"]
    assert selected[1].fields[0].label == "Private note"


def test_catalog_keeps_valid_packs_when_one_private_document_is_invalid(
    tmp_path: Path,
) -> None:
    _ = _write_pack(
        tmp_path, "valid.json", _pack_document(pack_id="valid", preset_id="detail")
    )
    invalid_path: Path = tmp_path / "invalid.json"
    _ = invalid_path.write_text("{ invalid json", encoding="utf-8")

    catalog: CharacterPresetCatalog = CharacterPresetCatalog.load(tmp_path)

    assert [preset.id for preset in catalog.selected("", ["valid:detail"])] == [
        "valid:detail"
    ]
    assert len(catalog.load_issues) == 1
    assert catalog.load_issues[0].path == invalid_path
    assert "not valid JSON" in catalog.load_issues[0].message


def test_catalog_keeps_valid_packs_when_one_private_document_has_invalid_types(
    tmp_path: Path,
) -> None:
    _ = _write_pack(
        tmp_path, "valid.json", _pack_document(pack_id="valid", preset_id="detail")
    )
    invalid_document: dict[str, object] = _pack_document(
        pack_id="invalid", preset_id="detail"
    )
    invalid_document["label"] = 1
    invalid_path: Path = _write_pack(tmp_path, "invalid.json", invalid_document)

    catalog: CharacterPresetCatalog = CharacterPresetCatalog.load(tmp_path)

    assert [preset.id for preset in catalog.selected("", ["valid:detail"])] == [
        "valid:detail"
    ]
    assert len(catalog.load_issues) == 1
    assert catalog.load_issues[0].path == invalid_path
    assert "must be text" in catalog.load_issues[0].message


def test_catalog_rejects_an_invalid_custom_pack_without_registering_its_presets(
    tmp_path: Path,
) -> None:
    document: dict[str, object] = _pack_document(pack_id="invalid", preset_id="detail")
    presets: list[object] = cast(list[object], document["presets"])
    preset: dict[str, object] = cast(dict[str, object], presets[0])
    fields: list[object] = cast(list[object], preset["fields"])
    fields.append({"label": "private note"})
    _ = _write_pack(tmp_path, "invalid.json", document)

    catalog: CharacterPresetCatalog = CharacterPresetCatalog.load(tmp_path)

    with pytest.raises(ValueError, match="unknown character extra preset"):
        _ = catalog.selected("", ["invalid:detail"])
    assert len(catalog.load_issues) == 1
    assert "duplicates" in catalog.load_issues[0].message


def test_catalog_rejects_base_presets_submitted_as_extras(tmp_path: Path) -> None:
    catalog: CharacterPresetCatalog = CharacterPresetCatalog.load(tmp_path)

    with pytest.raises(ValueError, match="is not an extra preset"):
        _ = catalog.selected("", ["builtin:humanoid"])
