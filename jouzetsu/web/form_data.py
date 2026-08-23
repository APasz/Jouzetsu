"""Typed parsing and validation for browser form submissions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from starlette.datastructures import FormData
from starlette.exceptions import HTTPException
from starlette.requests import Request

from ..character_presets import CharacterFieldKind, CharacterFieldPreset
from ..config import (
    GenerationSettings,
    IconColorSettings,
    ServerSettings,
    SpellingReplacement,
)
from ..models import CharacterField
from ..state import AppState


@dataclass(frozen=True, slots=True)
class FormValues:
    """Explicit string and checkbox reads from one already-parsed HTML form."""

    values: FormData

    def text(self, name: str, *, default: str = "") -> str:
        """Return one text value, rejecting uploaded files for scalar fields."""

        value: object = self.values.get(name, default)
        if not isinstance(value, str):
            raise TypeError(f"{name} must be text")
        return value

    def required_text(self, name: str) -> str:
        """Return a non-empty identifier-like field."""

        value: str = self.text(name).strip()
        if not value:
            raise ValueError(f"{name} is required")
        return value

    def texts(self, name: str) -> list[str]:
        """Return all values for a repeated scalar field, rejecting uploads."""

        raw_values: list[object] = list(self.values.getlist(name))
        if not all(isinstance(value, str) for value in raw_values):
            raise TypeError(f"{name} must contain only text values")
        return [cast(str, value) for value in raw_values]

    def flag(self, name: str) -> bool:
        """Parse an ordinary HTML checkbox without framework coercion."""

        value: str = self.text(name).strip().casefold()
        if value in {"", "0", "false", "off"}:
            return False
        if value in {"1", "true", "on"}:
            return True
        raise ValueError(f"{name} must be a checkbox value")


async def form_values(request: Request) -> FormValues:
    """Parse one request body at an explicit, typed route boundary."""

    try:
        return FormValues(await request.form())
    except Exception as exc:
        raise HTTPException(400, "invalid form body") from exc


def character_fields_from_form(form: FormValues) -> list[CharacterField]:
    """Build the profile's schema and values from repeated editor controls."""

    ids: list[str] = form.texts("field_id")
    labels: list[str] = form.texts("field_label")
    kinds: list[str] = form.texts("field_kind")
    values: list[str] = form.texts("field_value")
    if not (len(ids) == len(labels) == len(kinds) == len(values)):
        raise ValueError(
            "character fields must include matching ids, labels, styles, and values"
        )

    fields: list[CharacterField] = []
    for field_id, label, kind, value in zip(ids, labels, kinds, values, strict=True):
        trimmed_label: str = label.strip()
        if not trimmed_label and not value.strip():
            continue
        if not trimmed_label:
            raise ValueError("each character field needs a label")
        if kind not in {"short_text", "long_text"}:
            raise ValueError("character field style must be short text or long text")
        field_kind: CharacterFieldKind = cast(CharacterFieldKind, kind)
        fields.append(
            CharacterField(
                id=field_id, label=trimmed_label, kind=field_kind, value=value
            )
            if field_id
            else CharacterField(label=trimmed_label, kind=field_kind, value=value)
        )
    return fields


def missing_preset_fields(
    fields: list[CharacterField],
    presets: tuple[CharacterFieldPreset, ...],
) -> list[CharacterField]:
    """Create only template fields that do not already exist in the submitted profile."""

    labels: set[str] = {field.label.casefold() for field in fields}
    additions: list[CharacterField] = []
    for preset in presets:
        for preset_field in preset.fields:
            label_key: str = preset_field.label.casefold()
            if label_key in labels:
                continue
            additions.append(
                CharacterField(
                    label=preset_field.label,
                    kind=preset_field.kind,
                    value=preset_field.value,
                )
            )
            labels.add(label_key)
    return additions


def require_float(raw: str, *, field_name: str) -> float:
    """Parse one finite numeric form field with an explicit error message."""

    try:
        value: float = float(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def require_integer(raw: str, *, field_name: str) -> int:
    """Parse an integer-only form field without accepting decimal input silently."""

    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def optional_float(raw: str, *, field_name: str) -> float | None:
    """Parse a nullable finite floating-point form field."""

    return None if not raw.strip() else require_float(raw, field_name=field_name)


def optional_integer(raw: str, *, field_name: str) -> int | None:
    """Parse a nullable integer form field."""

    return None if not raw.strip() else require_integer(raw, field_name=field_name)


def updated_generation_settings(
    state: AppState, form: FormValues, *, system_prompt: str
) -> GenerationSettings:
    """Construct the complete generation settings model from one form section."""

    current: GenerationSettings = state.config.generation
    return GenerationSettings(
        temperature=require_float(form.text("temperature"), field_name="temperature"),
        top_p=require_float(form.text("top_p"), field_name="top_p"),
        max_tokens=require_integer(form.text("max_tokens"), field_name="max_tokens"),
        system_prompt=system_prompt.strip(),
        continuity_review=form.flag("continuity_review"),
        british_english=current.british_english,
        british_spelling_replacements=_parse_spelling_replacements(
            form.text("british_spelling_replacements")
        ),
    )


def updated_server_settings(
    state: AppState,
    *,
    default_model: str,
    alias: str,
    auto_unload_minutes: str,
) -> ServerSettings:
    """Construct complete server settings from global-form deltas."""

    current: ServerSettings = state.config.server
    aliases: dict[str, str] = dict(current.model_aliases)
    model_key: str = state.current_model_key()
    clean_alias: str = alias.strip()
    if model_key and clean_alias:
        aliases[model_key] = clean_alias
    elif model_key:
        _ = aliases.pop(model_key, None)
    settings: ServerSettings = ServerSettings(
        base_url=current.base_url,
        api_key=current.api_key,
        default_model=default_model.strip(),
        model_aliases=aliases,
        auto_unload_minutes=optional_integer(
            auto_unload_minutes, field_name="auto_unload_minutes"
        ),
    )
    settings.validate()
    return settings


def updated_icon_color_settings(state: AppState, form: FormValues) -> IconColorSettings:
    """Construct the complete, validated icon palette from interface controls."""

    current: IconColorSettings = state.config.ui.icon_colors
    colors: IconColorSettings = IconColorSettings(
        linework_color=form.text(
            "icon_linework_color", default=current.linework_color
        ).strip(),
        accent_color=form.text(
            "icon_accent_color", default=current.accent_color
        ).strip(),
        surface_color=form.text(
            "icon_surface_color", default=current.surface_color
        ).strip(),
    )
    colors.validate()
    return colors


def _parse_spelling_replacements(raw: str) -> list[SpellingReplacement]:
    """Parse and validate the two-column British-spelling editor once."""

    replacements: list[SpellingReplacement] = []
    seen_sources: set[str] = set()
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line: str = raw_line.strip()
        if not line:
            continue
        columns: list[str] = line.split()
        if len(columns) != 2:
            raise ValueError(
                f"British spelling replacement line {line_number} must have two columns"
            )
        replacement: SpellingReplacement = SpellingReplacement(
            source=columns[0], replacement=columns[1]
        )
        replacement.validate()
        source_key: str = replacement.source.casefold()
        if source_key in seen_sources:
            raise ValueError(
                f"duplicate British spelling replacement source: {replacement.source}"
            )
        seen_sources.add(source_key)
        replacements.append(replacement)
    return replacements
