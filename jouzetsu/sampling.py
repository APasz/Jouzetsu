"""Backend-neutral sampling values and their supported wire parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, cast

type ParameterName = Literal["temperature", "top_p", "max_tokens"]


@dataclass(frozen=True, slots=True)
class SamplingParameters:
    """Concrete values snapshotted into defaults, profiles, and requests."""

    MIN_TEMPERATURE: ClassVar[float] = 0.0
    MAX_TEMPERATURE: ClassVar[float] = 2.0
    MIN_TOP_P: ClassVar[float] = 0.0
    MAX_TOP_P: ClassVar[float] = 1.0
    MIN_MAX_TOKENS: ClassVar[int] = 1

    temperature: float = 0.75
    top_p: float = 0.95
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        validate_sampling(self.temperature, self.top_p, self.max_tokens)

    def copy_compatible_from(
        self, source: SamplingParameters, support: ParameterSupport
    ) -> SamplingParameters:
        """Copy values known to be supported by the destination backend."""

        return SamplingParameters(
            temperature=source.temperature
            if support.temperature is True
            else self.temperature,
            top_p=source.top_p if support.top_p is True else self.top_p,
            max_tokens=source.max_tokens
            if support.max_tokens is True
            else self.max_tokens,
        )


def validate_sampling(temperature: float, top_p: float, max_tokens: int) -> None:
    """Apply the same validation to current settings and backend profiles."""

    for name, value, minimum, maximum in (
        (
            "temperature",
            temperature,
            SamplingParameters.MIN_TEMPERATURE,
            SamplingParameters.MAX_TEMPERATURE,
        ),
        ("top_p", top_p, SamplingParameters.MIN_TOP_P, SamplingParameters.MAX_TOP_P),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(cast(object, value), int | float)
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise ValueError(f"{name} must be a finite number")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    if type(max_tokens) is not int or max_tokens < SamplingParameters.MIN_MAX_TOKENS:
        raise ValueError(
            f"max_tokens must be an integer of at least {SamplingParameters.MIN_MAX_TOKENS}"
        )


@dataclass(frozen=True, slots=True)
class ParameterSupport:
    """True/False are known support; None means support has not been established."""

    temperature: bool | None = None
    top_p: bool | None = None
    max_tokens: bool | None = None

    def __post_init__(self) -> None:
        if any(
            value is not None and type(value) is not bool
            for value in (self.temperature, self.top_p, self.max_tokens)
        ):
            raise TypeError("parameter support must be boolean or null")

    def wire_values(
        self, parameters: SamplingParameters
    ) -> dict[ParameterName, float | int]:
        """Emit only established parameters, retaining all profile values separately."""

        values: dict[ParameterName, float | int] = {}
        if self.temperature is True:
            values["temperature"] = parameters.temperature
        if self.top_p is True:
            values["top_p"] = parameters.top_p
        if self.max_tokens is True:
            values["max_tokens"] = parameters.max_tokens
        return values

    def restrict_by(self, model: ParameterSupport) -> ParameterSupport:
        """Model restrictions override protocol support; unknown adds no promise."""

        return ParameterSupport(
            temperature=self.temperature if model.temperature is not False else False,
            top_p=self.top_p if model.top_p is not False else False,
            max_tokens=self.max_tokens if model.max_tokens is not False else False,
        )


DEFAULT_SAMPLING: Final[SamplingParameters] = SamplingParameters()
UNKNOWN_PARAMETERS: Final[ParameterSupport] = ParameterSupport()
CHAT_COMPLETIONS_PARAMETERS: Final[ParameterSupport] = ParameterSupport(
    temperature=True, top_p=True, max_tokens=True
)
