from __future__ import annotations

import asyncio
import platform
from dataclasses import replace

from pytest import MonkeyPatch, approx

from jouzetsu import host_stats
from jouzetsu.config import HostStatsDeviceSettings
from jouzetsu.host_stats import GpuStats, HostStatsMonitor, is_gpu_enabled
from jouzetsu.web.host_stats_styles import host_stat_meter_class, host_stat_meter_rules


def test_host_stats_snapshot_includes_memory_and_cpu_metadata() -> None:
    snapshot = HostStatsMonitor().snapshot()

    assert snapshot.hostname
    assert snapshot.operating_system
    assert snapshot.logical_cpu_count >= 1
    assert snapshot.memory_total_bytes is None or snapshot.memory_total_bytes > 0
    assert snapshot.memory_used_bytes is None or snapshot.memory_used_bytes >= 0


def test_host_stats_async_snapshot_uses_the_application_worker_pool() -> None:
    async def scenario() -> None:
        snapshot = await HostStatsMonitor().sample_async()

        assert snapshot.hostname

    asyncio.run(scenario())


def test_friendly_operating_system_uses_system_and_release(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "10")

    assert host_stats._friendly_operating_system() == "Windows 10"  # pyright: ignore[reportPrivateUsage]


def test_host_stat_meter_rules_cover_bounded_activity_values() -> None:
    rules: str = host_stat_meter_rules("#123456", "#fedcba")

    assert host_stat_meter_class(-1.0) == "jouzetsu-host-stat-meter-0"
    assert host_stat_meter_class(100.1) == "jouzetsu-host-stat-meter-100"
    assert (
        ".jouzetsu-host-stat-meter-0 { width: 0%; background: linear-gradient(90deg, #123456, #123456); }"
        in rules
    )
    assert (
        ".jouzetsu-host-stat-meter-100 { width: 100%; background: linear-gradient(90deg, #123456, #fedcba); }"
        in rules
    )


def test_host_stats_retains_a_rolling_cpu_average() -> None:
    monitor = HostStatsMonitor()
    _ = monitor.snapshot()
    snapshot = monitor.snapshot()

    assert snapshot.average_window_seconds >= 0.0
    assert (
        snapshot.average_cpu_utilization_percent is None
        or 0.0 <= snapshot.average_cpu_utilization_percent <= 100.0
    )


def test_host_stats_reuses_slow_metrics_between_fast_cpu_samples(
    monkeypatch: MonkeyPatch,
) -> None:
    timestamps = iter((0.0, 0.1, 1.1))
    monitor = HostStatsMonitor(clock=lambda: next(timestamps))
    memory_reads: int = 0

    def read_meminfo() -> dict[str, int]:
        nonlocal memory_reads
        memory_reads += 1
        return {"MemTotal": 8 * 1024**3, "MemAvailable": 4 * 1024**3}

    monkeypatch.setattr(host_stats, "_read_meminfo", read_meminfo)

    _ = monitor.snapshot()
    _ = monitor.snapshot()
    _ = monitor.snapshot()

    assert memory_reads == 2


def test_time_weighted_activity_average_is_not_biased_by_fast_sampling() -> None:
    base = HostStatsMonitor().snapshot()
    samples = (
        replace(base, sampled_at=0.0, cpu_utilization_percent=0.0),
        replace(base, sampled_at=9.0, cpu_utilization_percent=100.0),
        replace(base, sampled_at=9.1, cpu_utilization_percent=0.0),
        replace(base, sampled_at=10.0, cpu_utilization_percent=0.0),
    )

    ten_second_average = host_stats._time_weighted_average(  # pyright: ignore[reportPrivateUsage]
        samples,
        lambda sample: sample.cpu_utilization_percent,
    )
    recent_samples = (
        replace(base, sampled_at=9.0, cpu_utilization_percent=0.0),
        replace(base, sampled_at=9.1, cpu_utilization_percent=100.0),
        replace(base, sampled_at=10.0, cpu_utilization_percent=0.0),
    )
    recent_average = host_stats._time_weighted_average(  # pyright: ignore[reportPrivateUsage]
        recent_samples,
        lambda sample: sample.cpu_utilization_percent,
        window_seconds=1.0,
    )

    assert ten_second_average == approx(90.0)
    assert recent_average == approx(10.0)


def test_latest_host_stats_snapshot_reuses_the_monitor_cache() -> None:
    monitor = HostStatsMonitor()
    sampled = monitor.snapshot()
    latest = monitor.latest_snapshot()

    assert latest.sampled_at == sampled.sampled_at


def test_integrated_gpu_is_hidden_unless_explicitly_enabled() -> None:
    integrated_gpu = GpuStats(
        id="0000:0a:00.0",
        name="AMD GPU",
        utilization_percent=10.0,
        memory_used_bytes=128 * 1024**2,
        memory_total_bytes=1024 * 1024**2,
        temperature_celsius=None,
        power_watts=None,
        clock_mhz=None,
        encoder_utilization_percent=None,
        decoder_utilization_percent=None,
        junction_temperature_celsius=None,
        memory_temperature_celsius=None,
        vrm_temperature_celsius=None,
        power_cap_watts=None,
        memory_clock_mhz=None,
        performance_level=None,
        dpm_state=None,
    )

    assert not is_gpu_enabled(integrated_gpu, {})
    assert is_gpu_enabled(
        integrated_gpu, {integrated_gpu.id: HostStatsDeviceSettings(visible=True)}
    )
    assert not is_gpu_enabled(
        integrated_gpu, {integrated_gpu.id: HostStatsDeviceSettings(visible=False)}
    )
