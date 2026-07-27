"""Run-local, confidentiality-safe performance observation."""

from __future__ import annotations

import math
import platform
import resource
import statistics
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TimingCategory = Literal["success", "failure", "instrumentation_failure"]
TimingObserver = Callable[[str, str, float, TimingCategory], None]
TimingLifecycleLabel = Literal["cold", "warm", "warm_up", "unknown"]
TimingLifecycleState = Literal[
    "not_invoked",
    "initializing",
    "successfully_invoked",
    "failed_before_completion",
]


@dataclass(frozen=True)
class TimingEvent:
    """A deliberately text-free timing event."""

    phase: str
    safe_id: str
    duration_seconds: float
    category: TimingCategory


class TimingCollector:
    """Thread-safe instance collector; no application-global state is used."""

    def __init__(self) -> None:
        self._events: list[TimingEvent] = []
        self._lock = threading.Lock()
        self._instrumentation_failures = 0

    @property
    def events(self) -> tuple[TimingEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def instrumentation_failures(self) -> int:
        with self._lock:
            return self._instrumentation_failures

    def observe(
        self,
        phase: str,
        safe_id: str,
        duration_seconds: float,
        category: TimingCategory,
    ) -> None:
        """Store an already completed phase event."""
        event = TimingEvent(
            phase=phase,
            safe_id=safe_id,
            duration_seconds=max(0.0, duration_seconds),
            category=category,
        )
        with self._lock:
            self._events.append(event)

    def observer_failed(self) -> None:
        """Record an observer failure independently from service outcomes."""
        with self._lock:
            self._instrumentation_failures += 1


class TimingLifecycle:
    """Instance-local stage lifecycle used to derive cold and warm labels."""

    def __init__(self) -> None:
        self._states: dict[str, TimingLifecycleState] = {}
        self._lock = threading.Lock()

    def begin(self, stage: str) -> TimingLifecycleLabel:
        """Start one stage invocation without assuming another stage is warm."""
        with self._lock:
            state = self._states.get(stage, "not_invoked")
            if state == "successfully_invoked":
                return "warm"
            self._states[stage] = "initializing"
            return "cold"

    def finish(self, stage: str, *, succeeded: bool) -> None:
        """Advance only successful stages to warm lifecycle state."""
        with self._lock:
            self._states[stage] = (
                "successfully_invoked" if succeeded else "failed_before_completion"
            )

    def state(self, stage: str) -> TimingLifecycleState:
        """Return the current state for deterministic tests and reporting."""
        with self._lock:
            return self._states.get(stage, "not_invoked")


def notify_observer(
    observer: TimingObserver | None,
    collector: TimingCollector | None,
    *,
    phase: str,
    safe_id: str,
    duration_seconds: float,
    category: Literal["success", "failure"],
) -> None:
    """Notify after a phase; observer exceptions never alter the phase result."""
    if collector is not None:
        collector.observe(phase, safe_id, duration_seconds, category)
    if observer is None:
        return
    try:
        observer(phase, safe_id, duration_seconds, category)
    except Exception:
        if collector is not None:
            collector.observer_failed()


class PerformanceSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_kind: Literal["fake", "real"]
    temperature: Literal["cold", "warm"]
    repetition: int = Field(ge=1)
    duration_seconds: float = Field(ge=0.0)


class PerformanceAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_kind: Literal["fake", "real"]
    cold_seconds: float | None
    cold_exclusion_reason: str | None = None
    warm_seconds: list[float]


class PhaseTimingSample(BaseModel):
    """One text-free measured service phase."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    provider_kind: Literal["fake", "real"]
    measurement_category: TimingLifecycleLabel
    state_established: bool = True
    lifecycle_exclusion_reason: Literal["cold_state_not_established"] | None = None
    success_category: Literal["success", "failure"] = "success"
    repetition: int = Field(ge=1, le=100)
    duration_seconds: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> PhaseTimingSample:
        if self.state_established:
            if (
                self.measurement_category == "unknown"
                or self.lifecycle_exclusion_reason is not None
            ):
                raise ValueError("timing_lifecycle_state_inconsistent")
        elif (
            self.measurement_category != "unknown"
            or self.lifecycle_exclusion_reason != "cold_state_not_established"
        ):
            raise ValueError("timing_lifecycle_state_inconsistent")
        return self

    @classmethod
    def unknown_state(
        cls,
        *,
        phase: str,
        provider_kind: Literal["fake", "real"],
        repetition: int,
        duration_seconds: float,
        success_category: Literal["success", "failure"] = "success",
    ) -> PhaseTimingSample:
        """Create an explicit unavailable lifecycle observation."""
        return cls(
            phase=phase,
            provider_kind=provider_kind,
            measurement_category="unknown",
            state_established=False,
            lifecycle_exclusion_reason="cold_state_not_established",
            success_category=success_category,
            repetition=repetition,
            duration_seconds=duration_seconds,
        )


class PhaseTimingAggregate(BaseModel):
    """Deterministic nearest-rank timing summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: str
    provider_kind: Literal["fake", "real"]
    measurement_category: TimingLifecycleLabel
    state_established: bool
    lifecycle_exclusion_reason: Literal["cold_state_not_established"] | None = None
    success_category: Literal["success", "failure"]
    count: int = Field(ge=1)
    minimum_seconds: float = Field(ge=0.0)
    maximum_seconds: float = Field(ge=0.0)
    median_seconds: float = Field(ge=0.0)
    p90_seconds: float | None = Field(default=None, ge=0.0)
    p90_exclusion_reason: str | None = None
    p95_seconds: float | None = Field(default=None, ge=0.0)
    p95_exclusion_reason: str | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> PhaseTimingAggregate:
        if self.state_established:
            if (
                self.measurement_category == "unknown"
                or self.lifecycle_exclusion_reason is not None
            ):
                raise ValueError("timing_lifecycle_state_inconsistent")
        elif (
            self.measurement_category != "unknown"
            or self.lifecycle_exclusion_reason != "cold_state_not_established"
        ):
            raise ValueError("timing_lifecycle_state_inconsistent")
        return self


class MemoryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    process_high_water_rss_bytes: int | None = Field(default=None, ge=0)
    exclusion_reason: str | None = None


def aggregate_performance(
    samples: list[PerformanceSample],
    *,
    cold_state_available: bool,
) -> list[PerformanceAggregate]:
    """Keep fake/real aggregates separate and represent unsupported cold runs."""
    aggregates: list[PerformanceAggregate] = []
    for provider_kind in ("fake", "real"):
        selected = [
            sample for sample in samples if sample.provider_kind == provider_kind
        ]
        if not selected:
            continue
        cold = [sample for sample in selected if sample.temperature == "cold"]
        if len(cold) > 1:
            raise ValueError("only one measured cold call is permitted")
        warm = sorted(
            (sample for sample in selected if sample.temperature == "warm"),
            key=lambda item: item.repetition,
        )
        aggregates.append(
            PerformanceAggregate(
                provider_kind=provider_kind,
                cold_seconds=(
                    cold[0].duration_seconds if cold and cold_state_available else None
                ),
                cold_exclusion_reason=(
                    None if cold and cold_state_available else "cold_state_unavailable"
                ),
                warm_seconds=[sample.duration_seconds for sample in warm],
            )
        )
    return aggregates


def aggregate_phase_timings(
    samples: list[PhaseTimingSample],
) -> list[PhaseTimingAggregate]:
    """Aggregate with the deterministic nearest-rank percentile definition."""
    grouped: dict[tuple[str, str, str, str, bool, str | None], list[float]] = {}
    for sample in samples:
        identity = (
            sample.phase,
            sample.provider_kind,
            sample.measurement_category,
            sample.success_category,
            sample.state_established,
            sample.lifecycle_exclusion_reason,
        )
        grouped.setdefault(identity, []).append(sample.duration_seconds)
    aggregates: list[PhaseTimingAggregate] = []
    for (
        phase,
        provider_kind,
        category,
        success_category,
        state_established,
        lifecycle_exclusion_reason,
    ), values in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(values)
        p90 = _nearest_rank(ordered, 0.90) if len(ordered) >= 10 else None
        p95 = _nearest_rank(ordered, 0.95) if len(ordered) >= 20 else None
        aggregates.append(
            PhaseTimingAggregate(
                phase=phase,
                provider_kind=provider_kind,  # type: ignore[arg-type]
                measurement_category=category,  # type: ignore[arg-type]
                state_established=state_established,
                lifecycle_exclusion_reason=lifecycle_exclusion_reason,  # type: ignore[arg-type]
                success_category=success_category,  # type: ignore[arg-type]
                count=len(ordered),
                minimum_seconds=ordered[0],
                maximum_seconds=ordered[-1],
                median_seconds=statistics.median(ordered),
                p90_seconds=p90,
                p90_exclusion_reason=(
                    None if p90 is not None else "insufficient_samples_p90"
                ),
                p95_seconds=p95,
                p95_exclusion_reason=(
                    None if p95 is not None else "insufficient_samples_p95"
                ),
            )
        )
    return aggregates


def process_memory_observation() -> MemoryObservation:
    """Return process high-water RSS with platform-correct units."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return MemoryObservation(exclusion_reason="memory_observation_unavailable")
    system = platform.system()
    if system == "Darwin":
        value = int(rss)
    elif system in {"Linux", "FreeBSD"}:
        value = int(rss * 1024)
    else:
        return MemoryObservation(exclusion_reason="memory_units_unsupported")
    return MemoryObservation(process_high_water_rss_bytes=max(0, value))


def _nearest_rank(values: list[float], percentile: float) -> float:
    rank = max(1, math.ceil(percentile * len(values)))
    return values[rank - 1]
