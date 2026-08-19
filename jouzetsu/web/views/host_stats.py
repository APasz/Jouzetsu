"""Pure renderers for the host diagnostics dialog's live telemetry content."""

from __future__ import annotations

import re

from fastcore.xml import FT  # pyright: ignore[reportMissingTypeStubs]

from ...config import HostStatsDeviceSettings, HostStatsSettings
from ...host_stats import (
    GpuStats,
    HostStatsSnapshot,
    NetworkStats,
    is_gpu_enabled,
    is_network_interface_enabled,
)
from ..host_stats_styles import host_stat_meter_class
from ..html import H3, Details, Div, P, Section, Span, Summary

type HTML = FT


def render_content(snapshot: HostStatsSnapshot, settings: HostStatsSettings) -> HTML:
    """Render the independently refreshable body of the host diagnostics dialog."""

    sections: list[HTML] = []
    if settings.system:
        sections.append(
            _expandable_section(
                "System",
                _row(
                    "System RAM",
                    _used_of_total_text(
                        snapshot.memory_used_bytes, snapshot.memory_total_bytes
                    ),
                    activity_percent=snapshot.recent_memory_utilization_percent,
                ),
                _row("Operating system", snapshot.operating_system),
                _row("Uptime", _duration_text(snapshot.uptime_seconds)),
                _row(
                    "Swap",
                    _used_of_total_text(
                        snapshot.swap_used_bytes, snapshot.swap_total_bytes
                    ),
                    activity_percent=_used_percentage(
                        snapshot.swap_used_bytes, snapshot.swap_total_bytes
                    ),
                ),
            )
        )
    if settings.cpu:
        sections.append(
            _expandable_section(
                "CPU",
                _row(
                    "Current load",
                    _percent_text(snapshot.cpu_utilization_percent),
                    activity_percent=snapshot.recent_cpu_utilization_percent,
                ),
                _row(
                    "1-minute load average",
                    _load_average_text(snapshot),
                    activity_percent=snapshot.recent_load_utilization_percent,
                ),
                _row(
                    "10-minute average load",
                    _percent_text(snapshot.average_cpu_utilization_percent),
                    activity_percent=snapshot.average_cpu_utilization_percent,
                ),
                _row(
                    "Peaking core load",
                    _percent_text(snapshot.peak_core_utilization_percent),
                    activity_percent=snapshot.recent_peak_core_utilization_percent,
                ),
                _row(
                    "Temperature", _temperature_text(snapshot.cpu_temperature_celsius)
                ),
                _row("Average clock", _clock_text(snapshot.cpu_clock_mhz)),
                _row("Package power", _watts_text(snapshot.cpu_package_power_watts)),
                _row(
                    "Thermal throttle events",
                    _optional_int(snapshot.cpu_throttle_events),
                ),
            )
        )
    if settings.gpu:
        _append_gpu_sections(sections, snapshot, settings)
    if settings.network:
        visible_networks: list[NetworkStats] = [
            network
            for network in snapshot.networks
            if is_network_interface_enabled(network, settings.interfaces)
        ]
        if visible_networks:
            sections.append(_network_section(visible_networks, settings.interfaces))
    if not sections:
        sections.append(
            P("No host-stat sections are enabled.", cls="jouzetsu-empty-state")
        )
    return Div(
        *sections,
        id="host-stats-content",
        cls="jouzetsu-stats-sections",
        data_sampled_at=f"{snapshot.sampled_at:.6f}",
        data_testid="host-stats-list",
    )


def _append_gpu_sections(
    sections: list[HTML],
    snapshot: HostStatsSnapshot,
    settings: HostStatsSettings,
) -> None:
    visible_gpus: list[tuple[int, GpuStats]] = [
        (index, gpu)
        for index, gpu in enumerate(snapshot.gpus)
        if is_gpu_enabled(gpu, settings.gpus)
    ]
    if not visible_gpus:
        sections.append(
            _section(
                "GPU",
                P("No enabled GPU telemetry is available.", cls="jouzetsu-form-help"),
            )
        )
        return
    for index, gpu in visible_gpus:
        gpu_settings: HostStatsDeviceSettings | None = settings.gpus.get(gpu.id)
        gpu_label: str = gpu_settings.label if gpu_settings is not None else gpu.name
        sections.append(
            _expandable_section(
                f"GPU {index}: {gpu_label or gpu.name}",
                _row(
                    "Current load",
                    _percent_text(gpu.utilization_percent),
                    activity_percent=_metric_at(
                        snapshot.recent_gpu_utilization_percent, index
                    ),
                ),
                _row(
                    "10-minute average load",
                    _percent_text(
                        _metric_at(snapshot.average_gpu_utilization_percent, index)
                    ),
                    activity_percent=_metric_at(
                        snapshot.average_gpu_utilization_percent, index
                    ),
                ),
                _row(
                    "Video RAM",
                    _used_of_total_text(gpu.memory_used_bytes, gpu.memory_total_bytes),
                    activity_percent=_used_percentage(
                        gpu.memory_used_bytes, gpu.memory_total_bytes
                    ),
                ),
                _row(
                    "VRAM pressure",
                    _percent_text(
                        _used_percentage(gpu.memory_used_bytes, gpu.memory_total_bytes)
                    ),
                    activity_percent=_used_percentage(
                        gpu.memory_used_bytes, gpu.memory_total_bytes
                    ),
                ),
                _row("Temperature", _temperature_text(gpu.temperature_celsius)),
                _row(
                    "Junction temperature",
                    _temperature_text(gpu.junction_temperature_celsius),
                ),
                _row(
                    "VRAM temperature",
                    _temperature_text(gpu.memory_temperature_celsius),
                ),
                _row("VRM temperature", _temperature_text(gpu.vrm_temperature_celsius)),
                _row(
                    "Power draw", _power_draw_text(gpu.power_watts, gpu.power_cap_watts)
                ),
                _row("Core clock", _clock_text(gpu.clock_mhz)),
                _row("Memory clock", _clock_text(gpu.memory_clock_mhz)),
                _row("Performance level", gpu.performance_level or "Unavailable"),
                _row("DPM state", gpu.dpm_state or "Unavailable"),
                _row("Encoder load", _percent_text(gpu.encoder_utilization_percent)),
                _row("Decoder load", _percent_text(gpu.decoder_utilization_percent)),
            )
        )


def _section(title: str, *rows: HTML) -> HTML:
    return Section(
        H3(title, cls="jouzetsu-section-title"), *rows, cls="jouzetsu-stats-section"
    )


def _expandable_section(title: str, summary: HTML, *details: HTML) -> HTML:
    return Section(
        H3(title, cls="jouzetsu-section-title"),
        Details(
            Summary(summary, cls="jouzetsu-host-stat-summary"),
            Div(*details, cls="jouzetsu-host-stat-details"),
        ),
        cls="jouzetsu-stats-section",
    )


def _network_section(
    networks: list[NetworkStats],
    interface_settings: dict[str, HostStatsDeviceSettings],
) -> HTML:
    interfaces: list[HTML] = []
    for network in networks:
        settings: HostStatsDeviceSettings | None = interface_settings.get(network.name)
        label: str = (
            settings.label if settings is not None and settings.label else network.name
        )
        interfaces.append(
            Details(
                Summary(
                    _row(
                        label,
                        _network_throughput_text(network),
                        activity_percent=_network_utilization_percent(
                            network,
                            (network.received_bytes_per_second or 0.0)
                            + (network.transmitted_bytes_per_second or 0.0),
                        ),
                    ),
                    cls="jouzetsu-host-stat-summary",
                ),
                Div(
                    _row(
                        "Receive",
                        _rate_with_average_text(
                            network.received_bytes_per_second,
                            network.average_received_bytes_per_second_5s,
                        ),
                        activity_percent=_network_utilization_percent(
                            network, network.received_bytes_per_second
                        ),
                    ),
                    _row(
                        "Transmit",
                        _rate_with_average_text(
                            network.transmitted_bytes_per_second,
                            network.average_transmitted_bytes_per_second_5s,
                        ),
                        activity_percent=_network_utilization_percent(
                            network, network.transmitted_bytes_per_second
                        ),
                    ),
                    _row(
                        "Link speed",
                        f"{network.link_speed_megabits_per_second:,} Mbps"
                        if network.link_speed_megabits_per_second is not None
                        else "Unavailable",
                    ),
                    cls="jouzetsu-host-stat-details",
                ),
                cls="jouzetsu-network-interface",
            )
        )
    return _section("Network", *interfaces)


def _row(label: str, value: str, *, activity_percent: float | None = None) -> HTML:
    meter: HTML | None = None
    classes: str = "jouzetsu-host-stat-row"
    if activity_percent is not None:
        clamped_activity_percent: float = _clamp_percent(activity_percent)
        meter = Span(
            cls=f"jouzetsu-host-stat-meter {host_stat_meter_class(clamped_activity_percent)}",
            role="progressbar",
            aria_label=f"{label} activity",
            aria_valuemin="0",
            aria_valuemax="100",
            aria_valuenow=f"{clamped_activity_percent:.2f}",
        )
        classes += " is-metered"
    return Span(
        meter,
        Span(
            Span(label, cls="jouzetsu-host-stat-label"),
            Span(*_value_parts(value), cls="jouzetsu-host-stat-value"),
            cls="jouzetsu-host-stat-copy",
        ),
        cls=classes,
        data_host_stat_row=label,
    )


def _optional_int(value: int | None) -> str:
    return str(value) if value is not None else "Unknown"


def _temperature_text(value: int | None) -> str:
    return f"{value} °C" if value is not None else "Unavailable"


def _clock_text(value: int | None) -> str:
    return f"{value:,} MHz" if value is not None else "Unavailable"


def _watts_text(value: float | None) -> str:
    return f"{value:.1f} W" if value is not None else "Unavailable"


def _power_draw_text(draw: float | None, cap: float | None) -> str:
    if draw is None:
        return "Unavailable"
    return (
        f"{draw:.1f} W / cap {cap:.1f} W"
        if cap is not None and cap > 0.0
        else f"{draw:.1f} W"
    )


def _bytes_per_second_text(value: float | None) -> str:
    if value is None:
        return "Sampling…"
    units: tuple[str, ...] = ("B", "KiB", "MiB", "GiB", "TiB")
    amount: float = value
    for unit in units[:-1]:
        if amount < 1024.0:
            return _fixed_rate_text(amount, unit)
        amount /= 1024.0
    return _fixed_rate_text(amount, units[-1])


def _network_throughput_text(network: NetworkStats) -> str:
    return f"↓ {_bytes_per_second_text(network.received_bytes_per_second)} · ↑ {_bytes_per_second_text(network.transmitted_bytes_per_second)}"


def _rate_with_average_text(current: float | None, average: float | None) -> str:
    return f"{_bytes_per_second_text(current)} · {_bytes_per_second_text(average)}"


def _fixed_rate_text(amount: float, unit: str) -> str:
    number: str = f"{amount:6.1f}".replace(" ", "\u00a0")
    padded_unit: str = f"{unit:<3}".replace(" ", "\u00a0")
    return f"{number}\u00a0{padded_unit}/s"


def _network_utilization_percent(
    network: NetworkStats, rate: float | None
) -> float | None:
    if rate is None or network.link_speed_megabits_per_second is None:
        return None
    return 100.0 * rate * 8.0 / (network.link_speed_megabits_per_second * 1_000_000)


def _percent_text(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "Sampling…"


def _metric_at(values: tuple[float | None, ...], index: int) -> float | None:
    return values[index] if 0 <= index < len(values) else None


def _clamp_percent(value: float) -> float:
    return min(max(value, 0.0), 100.0)


_NUMBER_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:%|[A-Za-z]+)?"
)


def _value_parts(value: str) -> tuple[HTML, ...]:
    parts: list[HTML] = []
    previous_end: int = 0
    for match in _NUMBER_PATTERN.finditer(value):
        if match.start() > previous_end:
            parts.append(Span(value[previous_end : match.start()]))
        parts.append(Span(match.group(), cls="jouzetsu-digital-value"))
        previous_end = match.end()
    if previous_end < len(value):
        parts.append(Span(value[previous_end:]))
    if not parts:
        parts.append(Span(value))
    return tuple(parts)


def _load_average_text(snapshot: HostStatsSnapshot) -> str:
    if snapshot.load_average_1m is None:
        return "Unavailable"
    normalised: float = 100.0 * snapshot.load_average_1m / snapshot.logical_cpu_count
    return f"{snapshot.load_average_1m:.2f} ({normalised:.0f}%)"


def _used_of_total_text(used: int | None, total: int | None) -> str:
    if used is None or total is None:
        return "Unavailable"
    percentage: float = 100.0 * used / total if total else 0.0
    return f"{_bytes_text(used)} / {_bytes_text(total)} ({percentage:.0f}%)"


def _used_percentage(used: int | None, total: int | None) -> float | None:
    if used is None or total is None or total == 0:
        return None
    return 100.0 * used / total


def _bytes_text(value: int) -> str:
    units: tuple[str, ...] = ("B", "KiB", "MiB", "GiB", "TiB")
    amount: float = float(value)
    for unit in units[:-1]:
        if amount < 1024.0:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} {units[-1]}"


def _duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "Unknown"
    total_seconds: int = max(0, int(seconds))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, _ = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m"
