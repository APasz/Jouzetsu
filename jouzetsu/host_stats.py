"""Application-wide host resource sampling for the diagnostics UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .async_workers import run_in_worker

log: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .config import HostStatsDeviceSettings

_SAMPLE_INTERVAL_SECONDS: float = 5.0
_AVERAGE_WINDOW_SECONDS: float = 10 * 60
_SLOW_METRIC_REFRESH_INTERVAL_SECONDS: float = 1.0
_NVIDIA_SMI_TIMEOUT_SECONDS: float = 2.0
_GPU_REFRESH_INTERVAL_SECONDS: float = 1.0
_ROCM_SMI_TIMEOUT_SECONDS: float = 2.0
_ROCM_SMI_STANDARD_PATH: Path = Path("/opt/rocm/bin/rocm-smi")
_INTEGRATED_GPU_MEMORY_BYTES: int = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class GpuStats:
    """Current metrics for one detected GPU."""

    id: str
    name: str
    utilization_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    temperature_celsius: int | None
    power_watts: float | None
    clock_mhz: int | None
    encoder_utilization_percent: float | None
    decoder_utilization_percent: float | None
    junction_temperature_celsius: int | None
    memory_temperature_celsius: int | None
    vrm_temperature_celsius: int | None
    power_cap_watts: float | None
    memory_clock_mhz: int | None
    performance_level: str | None
    dpm_state: str | None


@dataclass(frozen=True, slots=True)
class NetworkStats:
    """Current transfer rates for one network interface."""

    name: str
    received_bytes_per_second: float | None
    transmitted_bytes_per_second: float | None
    link_speed_megabits_per_second: int | None
    average_received_bytes_per_second_5s: float | None = None
    average_transmitted_bytes_per_second_5s: float | None = None


@dataclass(frozen=True, slots=True)
class HostStatsSnapshot:
    """One point-in-time host resource reading with rolling-average fields."""

    sampled_at: float
    hostname: str
    operating_system: str
    uptime_seconds: float | None
    logical_cpu_count: int
    load_average_1m: float | None
    cpu_temperature_celsius: int | None
    cpu_clock_mhz: int | None
    cpu_package_power_watts: float | None
    cpu_throttle_events: int | None
    cpu_utilization_percent: float | None
    peak_core_utilization_percent: float | None
    average_cpu_utilization_percent: float | None
    recent_cpu_utilization_percent: float | None
    recent_peak_core_utilization_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    recent_memory_utilization_percent: float | None
    recent_load_utilization_percent: float | None
    swap_used_bytes: int | None
    swap_total_bytes: int | None
    memory_pressure_percent: float | None
    oom_kill_count: int | None
    gpus: tuple[GpuStats, ...]
    networks: tuple[NetworkStats, ...]
    average_gpu_utilization_percent: tuple[float | None, ...]
    recent_gpu_utilization_percent: tuple[float | None, ...]
    average_window_seconds: float


@dataclass(frozen=True, slots=True)
class _CpuTimes:
    total: int
    idle: int


@dataclass(frozen=True, slots=True)
class _SlowHostMetrics:
    """Host readings that do not need to be collected with each CPU sample."""

    hostname: str
    operating_system: str
    uptime_seconds: float | None
    logical_cpu_count: int
    load_average_1m: float | None
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    swap_total_bytes: int | None
    swap_free_bytes: int | None
    memory_pressure_percent: float | None
    oom_kill_count: int | None
    cpu_temperature_celsius: int | None
    cpu_clock_mhz: int | None
    cpu_throttle_events: int | None


class HostStatsMonitor:
    """Collect host metrics and retain time-weighted activity over the latest ten minutes."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock: Callable[[], float] = clock
        self._previous_cpu_times: tuple[_CpuTimes, ...] | None = None
        self._samples: deque[HostStatsSnapshot] = deque()
        self._latest_snapshot: HostStatsSnapshot | None = None
        self._task: asyncio.Task[None] | None = None
        self._sampling_lock: threading.Lock = threading.Lock()
        self._last_gpu_sample_at: float = 0.0
        self._last_gpus: tuple[GpuStats, ...] = ()
        self._last_slow_metrics_at: float = 0.0
        self._last_slow_metrics: _SlowHostMetrics | None = None
        self._last_package_energy_uj: int | None = None
        self._last_package_energy_at: float | None = None
        self._previous_network_counters: dict[str, tuple[int, int, float]] = {}

    async def start(self) -> None:
        """Begin background sampling once for the process."""
        if self._task is None:
            _ = await self.sample_async()
            self._task = asyncio.create_task(self._sample_forever())
            log.info(
                "started host statistics monitor sample_interval_seconds=%d",
                _SAMPLE_INTERVAL_SECONDS,
            )

    async def stop(self) -> None:
        """Stop background sampling without losing the most recent reading."""
        if self._task is None:
            return
        _ = self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        log.info("stopped host statistics monitor")

    def snapshot(self) -> HostStatsSnapshot:
        """Take and return a fresh host reading."""
        return self.sample()

    def latest_snapshot(self) -> HostStatsSnapshot:
        """Return the latest reading, taking one bootstrap sample only when needed."""

        snapshot: HostStatsSnapshot | None = self._latest_snapshot
        return snapshot if snapshot is not None else self.sample()

    def sample(self) -> HostStatsSnapshot:
        """Collect one synchronous reading; suitable for a short UI callback."""

        with self._sampling_lock:
            return self._sample()

    async def sample_async(self) -> HostStatsSnapshot:
        """Collect a fresh reading without blocking the application's event loop."""

        return await run_in_worker(self.sample)

    def _sample(self) -> HostStatsSnapshot:
        """Collect one reading while ``_sampling_lock`` protects sampler state."""

        now: float = self._clock()
        cpu_times: tuple[_CpuTimes, ...] = _read_cpu_times()
        cpu_utilization, peak_core_utilization = _cpu_utilization(
            self._previous_cpu_times, cpu_times
        )
        self._previous_cpu_times = cpu_times
        slow_metrics: _SlowHostMetrics = self._slow_metrics_for_sample(now)
        snapshot: HostStatsSnapshot = HostStatsSnapshot(
            sampled_at=now,
            hostname=slow_metrics.hostname,
            operating_system=slow_metrics.operating_system,
            uptime_seconds=slow_metrics.uptime_seconds,
            logical_cpu_count=slow_metrics.logical_cpu_count,
            load_average_1m=slow_metrics.load_average_1m,
            cpu_temperature_celsius=slow_metrics.cpu_temperature_celsius,
            cpu_clock_mhz=slow_metrics.cpu_clock_mhz,
            cpu_package_power_watts=self._package_power_watts(now),
            cpu_throttle_events=slow_metrics.cpu_throttle_events,
            cpu_utilization_percent=cpu_utilization,
            peak_core_utilization_percent=peak_core_utilization,
            average_cpu_utilization_percent=None,
            recent_cpu_utilization_percent=None,
            recent_peak_core_utilization_percent=None,
            memory_used_bytes=(
                slow_metrics.memory_total_bytes - slow_metrics.memory_available_bytes
            )
            if slow_metrics.memory_total_bytes is not None
            and slow_metrics.memory_available_bytes is not None
            else None,
            memory_total_bytes=slow_metrics.memory_total_bytes,
            recent_memory_utilization_percent=None,
            recent_load_utilization_percent=None,
            swap_used_bytes=(
                slow_metrics.swap_total_bytes - slow_metrics.swap_free_bytes
            )
            if slow_metrics.swap_total_bytes is not None
            and slow_metrics.swap_free_bytes is not None
            else None,
            swap_total_bytes=slow_metrics.swap_total_bytes,
            memory_pressure_percent=slow_metrics.memory_pressure_percent,
            oom_kill_count=slow_metrics.oom_kill_count,
            gpus=self._gpus_for_sample(now),
            networks=self._networks_for_sample(now),
            average_gpu_utilization_percent=(),
            recent_gpu_utilization_percent=(),
            average_window_seconds=0.0,
        )
        self._samples.append(snapshot)
        cutoff: float = now - _AVERAGE_WINDOW_SECONDS
        while self._samples and self._samples[0].sampled_at < cutoff:
            _ = self._samples.popleft()
        latest: HostStatsSnapshot = self._with_averages(snapshot)
        self._latest_snapshot = latest
        return latest

    async def _sample_forever(self) -> None:
        while True:
            await asyncio.sleep(_SAMPLE_INTERVAL_SECONDS)
            try:
                _ = await self.sample_async()
            except Exception:
                log.exception("host statistics sample failed")

    def _gpus_for_sample(self, now: float) -> tuple[GpuStats, ...]:
        if (
            now - self._last_gpu_sample_at >= _GPU_REFRESH_INTERVAL_SECONDS
            or not self._last_gpus
        ):
            self._last_gpus = _read_gpus()
            self._last_gpu_sample_at = now
        return self._last_gpus

    def _slow_metrics_for_sample(self, now: float) -> _SlowHostMetrics:
        """Reuse slow system readings while CPU activity samples arrive every 100 ms."""

        cached: _SlowHostMetrics | None = self._last_slow_metrics
        if (
            cached is not None
            and now - self._last_slow_metrics_at < _SLOW_METRIC_REFRESH_INTERVAL_SECONDS
        ):
            return cached
        memory = _read_meminfo()
        metrics = _SlowHostMetrics(
            hostname=socket.gethostname().strip() or "Unknown host",
            operating_system=_friendly_operating_system(),
            uptime_seconds=_read_uptime_seconds(),
            logical_cpu_count=os.cpu_count() or 1,
            load_average_1m=_read_load_average_1m(),
            memory_total_bytes=memory.get("MemTotal"),
            memory_available_bytes=memory.get("MemAvailable"),
            swap_total_bytes=memory.get("SwapTotal"),
            swap_free_bytes=memory.get("SwapFree"),
            memory_pressure_percent=_read_memory_pressure_percent(),
            oom_kill_count=_read_oom_kill_count(),
            cpu_temperature_celsius=_read_cpu_temperature_celsius(),
            cpu_clock_mhz=_read_cpu_clock_mhz(),
            cpu_throttle_events=_read_cpu_throttle_events(),
        )
        self._last_slow_metrics = metrics
        self._last_slow_metrics_at = now
        return metrics

    def _package_power_watts(self, now: float) -> float | None:
        energy_uj: int | None = _read_package_energy_uj()
        previous_energy: int | None = self._last_package_energy_uj
        previous_at: float | None = self._last_package_energy_at
        self._last_package_energy_uj = energy_uj
        self._last_package_energy_at = now if energy_uj is not None else None
        if (
            energy_uj is None
            or previous_energy is None
            or previous_at is None
            or energy_uj < previous_energy
        ):
            return None
        elapsed: float = now - previous_at
        return (
            (energy_uj - previous_energy) / elapsed / 1_000_000
            if elapsed > 0.0
            else None
        )

    def _networks_for_sample(self, now: float) -> tuple[NetworkStats, ...]:
        current: dict[str, tuple[int, int]] = _read_network_counters()
        networks: list[NetworkStats] = []
        for name, (received, transmitted) in current.items():
            previous: tuple[int, int, float] | None = (
                self._previous_network_counters.get(name)
            )
            received_rate: float | None = None
            transmitted_rate: float | None = None
            if previous is not None:
                previous_received, previous_transmitted, previous_at = previous
                elapsed: float = now - previous_at
                if (
                    elapsed > 0.0
                    and received >= previous_received
                    and transmitted >= previous_transmitted
                ):
                    received_rate = (received - previous_received) / elapsed
                    transmitted_rate = (transmitted - previous_transmitted) / elapsed
            networks.append(
                NetworkStats(
                    name,
                    received_rate,
                    transmitted_rate,
                    _read_network_link_speed_megabits_per_second(name),
                )
            )
        self._previous_network_counters = {
            name: (received, transmitted, now)
            for name, (received, transmitted) in current.items()
        }
        return tuple(networks)

    def _with_averages(self, snapshot: HostStatsSnapshot) -> HostStatsSnapshot:
        samples: tuple[HostStatsSnapshot, ...] = tuple(self._samples)
        window_seconds: float = max(snapshot.sampled_at - samples[0].sampled_at, 0.0)
        cpu_average: float | None = _time_weighted_average(
            samples, lambda sample: sample.cpu_utilization_percent
        )
        gpu_averages: tuple[float | None, ...] = tuple(
            _time_weighted_average(
                samples, lambda sample, index=index: _gpu_utilization(sample, index)
            )
            for index in range(len(snapshot.gpus))
        )
        recent_gpu_averages: tuple[float | None, ...] = tuple(
            _time_weighted_average(
                samples,
                lambda sample, index=index: _gpu_utilization(sample, index),
                window_seconds=1.0,
            )
            for index in range(len(snapshot.gpus))
        )
        networks: tuple[NetworkStats, ...] = tuple(
            replace(
                network,
                average_received_bytes_per_second_5s=_time_weighted_average(
                    samples,
                    lambda sample, name=network.name: _network_rate(
                        sample, name, received=True
                    ),
                    window_seconds=5.0,
                ),
                average_transmitted_bytes_per_second_5s=_time_weighted_average(
                    samples,
                    lambda sample, name=network.name: _network_rate(
                        sample, name, received=False
                    ),
                    window_seconds=5.0,
                ),
            )
            for network in snapshot.networks
        )
        return replace(
            snapshot,
            average_cpu_utilization_percent=cpu_average,
            recent_cpu_utilization_percent=_time_weighted_average(
                samples,
                lambda sample: sample.cpu_utilization_percent,
                window_seconds=1.0,
            ),
            recent_peak_core_utilization_percent=_time_weighted_average(
                samples,
                lambda sample: sample.peak_core_utilization_percent,
                window_seconds=1.0,
            ),
            recent_memory_utilization_percent=_time_weighted_average(
                samples,
                _memory_utilization_percent,
                window_seconds=1.0,
            ),
            recent_load_utilization_percent=_time_weighted_average(
                samples,
                _load_utilization_percent,
                window_seconds=1.0,
            ),
            average_gpu_utilization_percent=gpu_averages,
            recent_gpu_utilization_percent=recent_gpu_averages,
            networks=networks,
            average_window_seconds=window_seconds,
        )


def _friendly_operating_system() -> str:
    """Return a concise operating system name suitable for diagnostics."""

    system: str = platform.system().strip()
    release: str = platform.release().strip()
    if system == "Darwin":
        system = "macOS"
        release = platform.mac_ver()[0].strip() or release
    return (
        " ".join(part for part in (system, release) if part)
        or "Unknown operating system"
    )


def _read_cpu_times() -> tuple[_CpuTimes, ...]:
    try:
        lines: list[str] = Path("/proc/stat").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    times: list[_CpuTimes] = []
    for line in lines:
        fields: list[str] = line.split()
        if not fields or not fields[0].startswith("cpu"):
            continue
        if not fields[0][3:].isdigit() and fields[0] != "cpu":
            continue
        values: list[int] = [int(value) for value in fields[1:]]
        if len(values) < 4:
            continue
        total: int = sum(values)
        idle: int = values[3] + (values[4] if len(values) > 4 else 0)
        times.append(_CpuTimes(total=total, idle=idle))
    return tuple(times)


def _memory_utilization_percent(snapshot: HostStatsSnapshot) -> float | None:
    if (
        snapshot.memory_used_bytes is None
        or snapshot.memory_total_bytes is None
        or snapshot.memory_total_bytes == 0
    ):
        return None
    return 100.0 * snapshot.memory_used_bytes / snapshot.memory_total_bytes


def _load_utilization_percent(snapshot: HostStatsSnapshot) -> float | None:
    if snapshot.load_average_1m is None:
        return None
    return 100.0 * snapshot.load_average_1m / snapshot.logical_cpu_count


def _cpu_utilization(
    previous: tuple[_CpuTimes, ...] | None, current: tuple[_CpuTimes, ...]
) -> tuple[float | None, float | None]:
    if previous is None or len(previous) != len(current) or not current:
        return None, None
    usages: list[float] = []
    for old, new in zip(previous, current, strict=True):
        total_delta: int = new.total - old.total
        idle_delta: int = new.idle - old.idle
        if total_delta > 0:
            usages.append(100.0 * (total_delta - idle_delta) / total_delta)
    if not usages:
        return None, None
    aggregate: float | None = usages[0]
    core_usages: list[float] = usages[1:]
    return aggregate, max(core_usages, default=aggregate)


def _read_load_average_1m() -> float | None:
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def _read_uptime_seconds() -> float | None:
    try:
        return float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except OSError, ValueError, IndexError:
        return None


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines: list[str] = (
            Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        )
    except OSError:
        return {}
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        fields: list[str] = raw_value.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) * 1024
    return values


def _read_memory_pressure_percent() -> float | None:
    for line in _read_text(Path("/proc/pressure/memory")).splitlines():
        if line.startswith("some "):
            for field in line.split()[1:]:
                if field.startswith("avg10="):
                    return _optional_float(field.removeprefix("avg10="))
    return None


def _read_oom_kill_count() -> int | None:
    for line in _read_text(Path("/proc/vmstat")).splitlines():
        key, _, value = line.partition(" ")
        if key == "oom_kill":
            return _optional_int(value.strip())
    return None


def _read_cpu_temperature_celsius() -> int | None:
    temperatures: list[int] = []
    for hwmon_path in Path("/sys/class/hwmon").glob("hwmon*"):
        sensor_name: str = _read_text(hwmon_path / "name")
        if sensor_name not in {"coretemp", "k10temp", "zenpower"}:
            continue
        for path in hwmon_path.glob("temp*_input"):
            value = _optional_int(_read_text(path))
            if value is not None and value > 0:
                temperatures.append(value // 1000)
    return max(temperatures, default=None)


def _read_cpu_clock_mhz() -> int | None:
    frequencies: list[int] = []
    for path in Path("/sys/devices/system/cpu").glob(
        "cpu[0-9]*/cpufreq/scaling_cur_freq"
    ):
        value: int | None = _optional_int(_read_text(path))
        if value is not None:
            frequencies.append(value // 1000)
    return round(sum(frequencies) / len(frequencies)) if frequencies else None


def _read_cpu_throttle_events() -> int | None:
    counts: list[int] = []
    for path in Path("/sys/devices/system/cpu").glob(
        "cpu[0-9]*/thermal_throttle/*_throttle_count"
    ):
        value: int | None = _optional_int(_read_text(path))
        if value is not None:
            counts.append(value)
    return sum(counts) if counts else None


def _read_package_energy_uj() -> int | None:
    values: list[int] = []
    for path in Path("/sys/class/powercap").glob("*-rapl:*/energy_uj"):
        value: int | None = _optional_int(_read_text(path))
        if value is not None:
            values.append(value)
    return sum(values) if values else None


def _read_network_counters() -> dict[str, tuple[int, int]]:
    counters: dict[str, tuple[int, int]] = {}
    for line in _read_text(Path("/proc/net/dev")).splitlines()[2:]:
        name, separator, values = line.partition(":")
        fields: list[str] = values.split()
        if not separator or len(fields) < 16:
            continue
        received: int | None = _optional_int(fields[0])
        transmitted: int | None = _optional_int(fields[8])
        if received is not None and transmitted is not None:
            counters[name.strip()] = (received, transmitted)
    return counters


def _read_network_link_speed_megabits_per_second(name: str) -> int | None:
    speed: int | None = _optional_int(
        _read_text(Path("/sys/class/net") / name / "speed")
    )
    return speed if speed is not None and speed > 0 else None


def _read_gpus() -> tuple[GpuStats, ...]:
    """Return NVIDIA telemetry when available, otherwise AMDGPU kernel telemetry."""
    nvidia_gpus: tuple[GpuStats, ...] = _read_nvidia_gpus()
    return nvidia_gpus if nvidia_gpus else _read_amdgpu_gpus()


def _read_nvidia_gpus() -> tuple[GpuStats, ...]:
    executable: str | None = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [
                executable,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,utilization.encoder,utilization.decoder,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except OSError, subprocess.SubprocessError:
        return ()
    return tuple(
        gpu
        for line in result.stdout.splitlines()
        if (gpu := _parse_nvidia_gpu(line)) is not None
    )


def _read_amdgpu_gpus() -> tuple[GpuStats, ...]:
    """Read AMDGPU's sysfs counters without depending on ROCm utilities."""
    gpus: list[GpuStats] = []
    rocm_cards: dict[str, dict[str, str]] = _read_rocm_smi_cards()
    for card_path in Path("/sys/class/drm").glob("card[0-9]*"):
        device_path: Path = card_path / "device"
        if _read_text(device_path / "vendor") != "0x1002":
            continue
        name: str = _read_amdgpu_name(device_path, card_path.name)
        rocm: dict[str, str] = rocm_cards.get(card_path.name, {})
        gpus.append(
            GpuStats(
                id=_read_amdgpu_id(device_path, card_path.name),
                name=name,
                utilization_percent=_optional_float(
                    _read_text(device_path / "gpu_busy_percent")
                ),
                memory_used_bytes=_optional_int(
                    _read_text(device_path / "mem_info_vram_used")
                ),
                memory_total_bytes=_optional_int(
                    _read_text(device_path / "mem_info_vram_total")
                ),
                temperature_celsius=_rocm_temperature(rocm, "edge")
                or _read_amdgpu_temperature_celsius(device_path),
                power_watts=_rocm_power_draw_watts(rocm)
                or _read_amdgpu_power_watts(device_path),
                clock_mhz=_rocm_clock_mhz(rocm, "sclk")
                or _read_amdgpu_clock_mhz(device_path),
                encoder_utilization_percent=None,
                decoder_utilization_percent=None,
                junction_temperature_celsius=_rocm_temperature(rocm, "junction"),
                memory_temperature_celsius=_rocm_temperature(rocm, "memory"),
                vrm_temperature_celsius=_rocm_temperature(rocm, "vrm"),
                power_cap_watts=_rocm_power_cap_watts(rocm),
                memory_clock_mhz=_rocm_clock_mhz(rocm, "mclk"),
                performance_level=_rocm_value(rocm, "Performance Level"),
                dpm_state=_rocm_dpm_state(rocm),
            )
        )
    return tuple(gpus)


def _read_amdgpu_name(device_path: Path, fallback: str) -> str:
    identifier: str = _read_amdgpu_id(device_path, fallback)
    return _read_pci_gpu_name(identifier) or f"AMD GPU ({identifier})"


@cache
def _read_pci_gpu_name(identifier: str) -> str:
    """Return the PCI database's friendly GPU name when lspci is available."""

    executable: str | None = shutil.which("lspci")
    if executable is None:
        return ""
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [executable, "-s", identifier],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except OSError, subprocess.SubprocessError:
        return ""
    _, separator, name = result.stdout.strip().partition(": ")
    return name.strip() if separator else ""


def _read_rocm_smi_cards() -> dict[str, dict[str, str]]:
    """Return optional AMD GPU telemetry exposed by ROCm SMI's JSON interface."""

    executable: str | None = (
        str(_ROCM_SMI_STANDARD_PATH)
        if _ROCM_SMI_STANDARD_PATH.is_file()
        else shutil.which("rocm-smi")
    )
    if executable is None:
        return {}
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [
                executable,
                "--showtemp",
                "--showpower",
                "--showclocks",
                "--showmaxpower",
                "--showperflevel",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_ROCM_SMI_TIMEOUT_SECONDS,
        )
        raw_cards: object = cast(object, json.loads(result.stdout))
    except OSError, subprocess.SubprocessError, json.JSONDecodeError:
        return {}
    if not isinstance(raw_cards, dict):
        return {}
    cards: dict[str, dict[str, str]] = {}
    raw_card_mapping: dict[object, object] = cast(dict[object, object], raw_cards)
    for name, raw_values in raw_card_mapping.items():
        if not isinstance(name, str) or not isinstance(raw_values, dict):
            continue
        values: dict[str, str] = {
            key: value
            for key, value in cast(dict[object, object], raw_values).items()
            if isinstance(key, str) and isinstance(value, str)
        }
        cards[name] = values
    return cards


def _rocm_value(values: dict[str, str], fragment: str) -> str | None:
    fragment_lower: str = fragment.casefold()
    return next(
        (value for key, value in values.items() if fragment_lower in key.casefold()),
        None,
    )


def _rocm_temperature(values: dict[str, str], sensor: str) -> int | None:
    return _rocm_number(values, f"Sensor {sensor}")


def _rocm_power_draw_watts(values: dict[str, str]) -> float | None:
    for key, value in values.items():
        key_lower: str = key.casefold()
        if "graphics package power" in key_lower and "max" not in key_lower:
            return _optional_float(value)
    return None


def _rocm_power_cap_watts(values: dict[str, str]) -> float | None:
    return _rocm_number(values, "Max Graphics Package Power")


def _rocm_clock_mhz(values: dict[str, str], clock: str) -> int | None:
    return _rocm_number(values, f"{clock} clock speed")


def _rocm_dpm_state(values: dict[str, str]) -> str | None:
    sclk: str | None = _rocm_value(values, "sclk clock level")
    mclk: str | None = _rocm_value(values, "mclk clock level")
    if sclk is None and mclk is None:
        return None
    parts: list[str] = []
    if sclk is not None:
        parts.append(f"SCLK {sclk}")
    if mclk is not None:
        parts.append(f"MCLK {mclk}")
    return " · ".join(parts)


def _rocm_number(values: dict[str, str], fragment: str) -> int | None:
    value: str | None = _rocm_value(values, fragment)
    if value is None:
        return None
    match: re.Match[str] | None = re.search(r"-?\d+(?:\.\d+)?", value)
    return round(float(match.group())) if match is not None else None


def _read_amdgpu_id(device_path: Path, fallback: str) -> str:
    for line in _read_text(device_path / "uevent").splitlines():
        if line.startswith("PCI_SLOT_NAME="):
            return line.removeprefix("PCI_SLOT_NAME=")
    return fallback


def _read_amdgpu_temperature_celsius(device_path: Path) -> int | None:
    values: list[int] = [
        value
        for path in (device_path / "hwmon").glob("hwmon*/temp*_input")
        if (value := _optional_int(_read_text(path))) is not None
    ]
    return max((value // 1000 for value in values), default=None)


def _read_amdgpu_power_watts(device_path: Path) -> float | None:
    for path in (device_path / "hwmon").glob("hwmon*/power*_average"):
        value: int | None = _optional_int(_read_text(path))
        if value is not None:
            return value / 1_000_000
    return None


def _read_amdgpu_clock_mhz(device_path: Path) -> int | None:
    lines: list[str] = _read_text(device_path / "pp_dpm_sclk").splitlines()
    for line in lines:
        if "*" not in line:
            continue
        match: re.Match[str] | None = re.search(r"(\d+)Mhz", line, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _parse_nvidia_gpu(line: str) -> GpuStats | None:
    fields: list[str] = [field.strip() for field in line.split(",")]
    if len(fields) != 10:
        return None
    return GpuStats(
        id=fields[9],
        name=fields[0],
        utilization_percent=_optional_float(fields[1]),
        memory_used_bytes=_mebibytes_to_bytes(_optional_int(fields[2])),
        memory_total_bytes=_mebibytes_to_bytes(_optional_int(fields[3])),
        temperature_celsius=_optional_int(fields[4]),
        power_watts=_optional_float(fields[5]),
        clock_mhz=_optional_int(fields[6]),
        encoder_utilization_percent=_optional_float(fields[7]),
        decoder_utilization_percent=_optional_float(fields[8]),
        junction_temperature_celsius=None,
        memory_temperature_celsius=None,
        vrm_temperature_celsius=None,
        power_cap_watts=None,
        memory_clock_mhz=None,
        performance_level=None,
        dpm_state=None,
    )


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _mebibytes_to_bytes(value: int | None) -> int | None:
    return value * 1024 * 1024 if value is not None else None


def is_gpu_enabled(
    gpu: GpuStats, configured: dict[str, HostStatsDeviceSettings]
) -> bool:
    """Return an explicit setting, or hide likely integrated GPUs by default."""
    settings: HostStatsDeviceSettings | None = configured.get(gpu.id)
    if settings is not None:
        return settings.visible
    return (
        gpu.memory_total_bytes is None
        or gpu.memory_total_bytes >= _INTEGRATED_GPU_MEMORY_BYTES
    )


def is_network_interface_enabled(
    network: NetworkStats, configured: dict[str, HostStatsDeviceSettings]
) -> bool:
    """Return an interface override, otherwise hide only the loopback interface."""

    settings: HostStatsDeviceSettings | None = configured.get(network.name)
    return settings.visible if settings is not None else network.name != "lo"


def _gpu_utilization(snapshot: HostStatsSnapshot, index: int) -> float | None:
    """Return one GPU utilization reading when that GPU still occupies its slot."""

    return (
        snapshot.gpus[index].utilization_percent
        if 0 <= index < len(snapshot.gpus)
        else None
    )


def _network_rate(
    snapshot: HostStatsSnapshot, name: str, *, received: bool
) -> float | None:
    """Return one interface's receive or transmit rate when it remains present."""

    network: NetworkStats | None = next(
        (item for item in snapshot.networks if item.name == name), None
    )
    if network is None:
        return None
    return (
        network.received_bytes_per_second
        if received
        else network.transmitted_bytes_per_second
    )


def _time_weighted_average(
    samples: tuple[HostStatsSnapshot, ...],
    value_for_sample: Callable[[HostStatsSnapshot], float | None],
    *,
    window_seconds: float | None = None,
) -> float | None:
    """Average readings by elapsed time so fast UI sampling cannot bias the result."""

    if not samples:
        return None
    latest: HostStatsSnapshot = samples[-1]
    window_start: float = (
        latest.sampled_at - window_seconds
        if window_seconds is not None
        else samples[0].sampled_at
    )
    weighted_total: float = 0.0
    measured_seconds: float = 0.0
    for previous, current in pairwise(samples):
        interval_start: float = max(previous.sampled_at, window_start)
        interval_seconds: float = current.sampled_at - interval_start
        value: float | None = value_for_sample(current)
        if interval_seconds > 0.0 and value is not None:
            weighted_total += value * interval_seconds
            measured_seconds += interval_seconds
    if measured_seconds > 0.0:
        return weighted_total / measured_seconds
    return value_for_sample(latest)
