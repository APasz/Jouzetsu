"""CSP-compatible style rules for the Host Stats activity meters."""

from __future__ import annotations

from typing import Final


_METER_CLASS_PREFIX: Final[str] = "jouzetsu-host-stat-meter-"
_METER_PERCENT_MINIMUM: Final[int] = 0
_METER_PERCENT_MAXIMUM: Final[int] = 100


def host_stat_meter_class(activity_percent: float) -> str:
    """Return the stylesheet class for a bounded whole-percent activity reading."""

    percentage: int = round(min(max(activity_percent, _METER_PERCENT_MINIMUM), _METER_PERCENT_MAXIMUM))
    return f"{_METER_CLASS_PREFIX}{percentage}"


def host_stat_meter_rules(start_color: str, end_color: str) -> str:
    """Build CSP-safe CSS rules for every supported Host Stats meter percentage."""

    return "\n".join(
        _host_stat_meter_rule(percentage, start_color, end_color)
        for percentage in range(_METER_PERCENT_MINIMUM, _METER_PERCENT_MAXIMUM + 1)
    )


def _host_stat_meter_rule(percentage: int, start_color: str, end_color: str) -> str:
    end_at_activity: str = _activity_color(start_color, end_color, percentage)
    return (
        f".{_METER_CLASS_PREFIX}{percentage} {{ width: {percentage}%; "
        f"background: linear-gradient(90deg, {start_color}, {end_at_activity}); }}"
    )


def _activity_color(start_color: str, end_color: str, activity_percent: int) -> str:
    start_red, start_green, start_blue = _hex_rgb(start_color)
    end_red, end_green, end_blue = _hex_rgb(end_color)
    ratio: float = activity_percent / _METER_PERCENT_MAXIMUM
    return "#{:02x}{:02x}{:02x}".format(
        round(start_red + (end_red - start_red) * ratio),
        round(start_green + (end_green - start_green) * ratio),
        round(start_blue + (end_blue - start_blue) * ratio),
    )


def _hex_rgb(color: str) -> tuple[int, int, int]:
    """Parse a validated six-digit CSS hexadecimal colour."""

    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
