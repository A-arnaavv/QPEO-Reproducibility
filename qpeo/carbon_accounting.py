"""Optional, time-aligned carbon accounting for the QPEO scheduling simulator.

This module intentionally DOES NOT modify the locked v1 experimental outputs.
Forecast rows are interpreted as piecewise-constant regional grid-carbon intensity
in gCO2e/kWh; times and power are in hours and kW, giving carbon mass in gCO2e.

The default 'strict' coverage policy rejects any workload interval that extends
past the forecast. 'repeat' explicitly replays the finite forecast periodically,
which is a counterfactual sensitivity assumption, not a claim of observed data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import math

import numpy as np


CoveragePolicy = Literal["strict", "repeat"]


@dataclass(frozen=True)
class CarbonTimeSeries:
    intensities_g_per_kwh: np.ndarray  # (time_steps, resources)
    step_hours: float  # duration of one forecast row, e.g. 0.5 h
    forecast_start_hours: float = 0.0  # relative to first selected job arrival
    coverage_policy: CoveragePolicy = "strict"

    def validate(self, n_resources: int) -> None:
        ci = np.asarray(self.intensities_g_per_kwh, dtype=float)
        if ci.ndim != 2 or ci.shape[1] != n_resources or ci.shape[0] < 1:
            raise ValueError("carbon forecast must be a nonempty (time_steps, resources) matrix")
        if not np.all(np.isfinite(ci)) or np.any(ci < 0):
            raise ValueError("carbon forecast intensities must be finite and nonnegative")
        if not math.isfinite(self.step_hours) or self.step_hours <= 0:
            raise ValueError("carbon forecast step_hours must be finite and positive")
        if not math.isfinite(self.forecast_start_hours):
            raise ValueError("forecast_start_hours must be finite")
        if self.coverage_policy not in ("strict", "repeat"):
            raise ValueError("coverage_policy must be 'strict' or 'repeat'")

    @property
    def coverage_hours(self) -> float:
        return len(self.intensities_g_per_kwh) * self.step_hours


@dataclass(frozen=True)
class CarbonAccountingResult:
    energy_kwh_by_resource: np.ndarray
    carbon_g_by_resource: np.ndarray
    total_energy_kwh: float
    total_carbon_g: float
    energy_weighted_intensity_g_per_kwh: float
    coverage_policy: CoveragePolicy


def integrate_time_aligned_carbon(
    intervals_by_resource: Sequence[Sequence[tuple[float, float, float]]],
    *,
    gpu_capacities: Sequence[float],
    idle_power_kw: Sequence[float],
    dynamic_power_kw: Sequence[float],
    power_exponent: float,
    forecast: CarbonTimeSeries,
) -> CarbonAccountingResult:
    """Integrate GPU energy and grid-carbon exposure along the simulated clock.

    Each interval is (start_hour, end_hour, allocated_GPU_demand) on a resource.
    Energy is evaluated on each constant-utilisation event segment, and carbon
    mass is integrated over all forecast slices intersected by that segment.
    A resource consumes idle model power only while it has active GPU demand,
    consistent with the existing simulator's GPU-side power convention.
    """
    n = len(intervals_by_resource)
    forecast.validate(n)
    cap = np.asarray(gpu_capacities, dtype=float)
    idle = np.asarray(idle_power_kw, dtype=float)
    dynamic = np.asarray(dynamic_power_kw, dtype=float)
    if any(a.shape != (n,) for a in (cap, idle, dynamic)):
        raise ValueError("capacity and power arrays must match the number of resources")
    if not all(np.all(np.isfinite(a)) for a in (cap, idle, dynamic)):
        raise ValueError("capacity and power values must be finite")
    if np.any(cap <= 0) or np.any(idle < 0) or np.any(dynamic < 0):
        raise ValueError("capacities must be positive and power values nonnegative")
    if not math.isfinite(power_exponent) or power_exponent <= 0:
        raise ValueError("power_exponent must be finite and positive")

    intensity = np.asarray(forecast.intensities_g_per_kwh, dtype=float)
    step = float(forecast.step_hours)
    period = float(forecast.coverage_hours)
    origin = float(forecast.forecast_start_hours)
    prefix = np.vstack((np.zeros((1, n)), np.cumsum(intensity * step, axis=0)))

    def carbon_area(time_hour: float, resource: int) -> float:
        """Integral of the forecast intensity from origin to time_hour, in g h/kWh."""
        x = time_hour - origin
        tolerance = 1e-10 * max(1.0, period)
        if x < -tolerance:
            raise ValueError("workload starts before the carbon forecast window")
        if forecast.coverage_policy == "strict":
            if x > period + tolerance:
                raise ValueError(
                    "carbon forecast does not cover the workload execution horizon; "
                    "extend the forecast or explicitly select repeat for a sensitivity test"
                )
            x = min(max(x, 0.0), period)
            if x >= period:
                return float(prefix[-1, resource])
            idx = min(int(x / step), len(intensity) - 1)
            return float(prefix[idx, resource] + (x - idx * step) * intensity[idx, resource])
        # Explicit periodic extrapolation: distinct from measured/forecast coverage.
        x = max(x, 0.0)
        cycle, phase = divmod(x, period)
        idx = min(int(phase / step), len(intensity) - 1)
        return float(cycle * prefix[-1, resource]
                     + prefix[idx, resource]
                     + (phase - idx * step) * intensity[idx, resource])

    energy = np.zeros(n, dtype=float)
    carbon = np.zeros(n, dtype=float)
    epsilon = 1e-12  # hours; agree with simulator's event roundoff tolerance
    for j, intervals in enumerate(intervals_by_resource):
        events: list[tuple[float, float]] = []
        for start, end, demand in intervals:
            s, e, g = float(start), float(end), float(demand)
            if not all(math.isfinite(v) for v in (s, e, g)):
                raise ValueError("schedule intervals must be finite")
            if e < s or g < 0:
                raise ValueError("schedule intervals must have end >= start and demand >= 0")
            if e - s <= epsilon or g == 0:
                continue
            if g > cap[j] + 1e-9:
                raise ValueError("an individual workload exceeds GPU capacity")
            events.extend(((s, g), (e, -g)))
        if not events:
            continue
        events.sort(key=lambda event: event[0])
        last_t = events[0][0]
        demand_now = 0.0
        for t, delta in events:
            duration = t - last_t
            if duration > epsilon and demand_now > 1e-15:
                util = demand_now / cap[j]
                if util > 1.0 + 1e-9:
                    raise ValueError("concurrent GPU utilisation exceeds resource capacity")
                util = min(max(util, 0.0), 1.0)
                p_kw = idle[j] + dynamic[j] * util**power_exponent
                energy[j] += p_kw * duration
                carbon[j] += p_kw * (carbon_area(t, j) - carbon_area(last_t, j))
            demand_now += delta
            if abs(demand_now) <= 1e-12:
                demand_now = 0.0
            if demand_now < -1e-9:
                raise ValueError("invalid negative running GPU demand")
            last_t = t
        if abs(demand_now) > 1e-9:
            raise ValueError("GPU scheduling event stream is unbalanced")

    total_energy = float(np.sum(energy))
    total_carbon = float(np.sum(carbon))
    if total_energy <= 0:
        raise ValueError("energy-weighted carbon intensity is undefined for zero-energy schedules")
    if total_carbon < -1e-7:
        raise ArithmeticError("integrated carbon mass is negative")
    total_carbon = max(0.0, total_carbon)
    return CarbonAccountingResult(
        energy_kwh_by_resource=energy,
        carbon_g_by_resource=carbon,
        total_energy_kwh=total_energy,
        total_carbon_g=total_carbon,
        energy_weighted_intensity_g_per_kwh=total_carbon / total_energy,
        coverage_policy=forecast.coverage_policy,
    )


def account_simulation_result(problem, assignment, simulator_inputs, simulation_result,
                              forecast: CarbonTimeSeries) -> CarbonAccountingResult:
    """Compute time-aligned carbon for one finished, fixed-assignment simulation.

    This read-only adapter uses the simulator's per-workload start/completion
    times. It verifies that its integrated energy equals the simulator's own
    energy output before returning carbon mass and energy-weighted intensity.
    It can be used in an opt-in new objective evaluation without mutating v1.
    """
    m, n = len(problem.workloads), len(problem.resources)
    X = np.asarray(assignment)
    if X.shape != (m, n) or not np.all((X == 0) | (X == 1)) or not np.all(X.sum(axis=1) == 1):
        raise ValueError("assignment must be a binary one-hot workload/resource matrix")
    starts = np.asarray(simulation_result.start_times, float)
    finishes = np.asarray(simulation_result.completion_times, float)
    if starts.shape != (m,) or finishes.shape != (m,):
        raise ValueError("simulation times must have one entry per workload")
    intervals: list[list[tuple[float, float, float]]] = [[] for _ in range(n)]
    for i, workload in enumerate(problem.workloads):
        j = int(np.argmax(X[i]))
        intervals[j].append((float(starts[i]), float(finishes[i]),
                             float(workload.gpu_demand)))
    power = simulator_inputs.power_model
    carbon = integrate_time_aligned_carbon(
        intervals, gpu_capacities=[r.gpu_capacity for r in problem.resources],
        idle_power_kw=power.idle_power_kw,
        dynamic_power_kw=power.dynamic_power_kw,
        power_exponent=power.power_exponent,
        forecast=forecast,
    )
    original = np.asarray(simulation_result.energy_kwh_by_resource, float)
    if original.shape != (n,) or not np.allclose(
        carbon.energy_kwh_by_resource, original, rtol=1e-9, atol=1e-9
    ):
        raise RuntimeError("carbon accounting energy does not match simulator energy")
    return carbon
