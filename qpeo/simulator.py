from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .carbon import CarbonReplay
from .evaluator import check_constraints
from .models import EvaluationState, ProblemInstance


BillingMode = Literal["active_resource_hours", "allocated_gpu_hours"]

# Numerical tolerance for event timestamps. 1e-12 h is 3.6 nanoseconds;
# sub-tolerance overlaps can arise when back-to-back jobs have separately
# rounded floating-point start/end times. Ignore only these zero-duration
# numerical segments, never intervals long enough to consume capacity.
TIME_ROUNDOFF_EPS_HOURS = 1e-12

# Per-algorithm deterministic schedule memoization. Dynamic carbon changes objective
# evaluation and feasibility, but not service times, arrivals, capacities or the
# resulting schedule for an already-decoded assignment.
_SIMULATION_CACHE: dict[tuple, "SimulationResult"] = {}
_SIMULATION_CACHE_HITS = 0
_SIMULATION_CACHE_MISSES = 0


def clear_simulation_cache() -> None:
    global _SIMULATION_CACHE_HITS, _SIMULATION_CACHE_MISSES
    _SIMULATION_CACHE.clear()
    _SIMULATION_CACHE_HITS = 0
    _SIMULATION_CACHE_MISSES = 0


def simulation_cache_stats() -> dict[str, int]:
    return {
        "hits": int(_SIMULATION_CACHE_HITS),
        "misses": int(_SIMULATION_CACHE_MISSES),
        "entries": int(len(_SIMULATION_CACHE)),
    }


def _simulation_cache_key(
    problem: ProblemInstance,
    X: np.ndarray,
    inputs: "SimulatorInputs",
) -> tuple:
    static_resource_signature = tuple(
        (
            float(r.gpu_capacity),
            None if r.cpu_capacity is None else float(r.cpu_capacity),
            float(r.memory_capacity),
        )
        for r in problem.resources
    )
    return (
        id(inputs),
        id(problem.workloads),
        static_resource_signature,
        np.asarray(X, dtype=np.int8).tobytes(),
    )


@dataclass(frozen=True)
class PowerModel:
    """
    Explicit reconstructed resource power model.

    Power while a GPU resource is active is
        P(u) = idle_power_kw + dynamic_power_kw * u**power_exponent,
    where u is instantaneous GPU utilisation in [0, 1].

    This model is a reconstruction assumption, not something recovered from the
    lost implementation. All values must therefore be reported with experiments.
    """
    idle_power_kw: np.ndarray
    dynamic_power_kw: np.ndarray
    power_exponent: float = 1.0

    def validate(self, N: int) -> None:
        idle = np.asarray(self.idle_power_kw, dtype=float)
        dyn = np.asarray(self.dynamic_power_kw, dtype=float)
        if idle.shape != (N,) or dyn.shape != (N,):
            raise ValueError(f"Power arrays must have shape {(N,)}.")
        if np.any(idle < 0) or np.any(dyn < 0):
            raise ValueError("Power values must be non-negative.")
        if self.power_exponent <= 0:
            raise ValueError("power_exponent must be > 0.")


@dataclass(frozen=True)
class SimulatorInputs:
    """
    Inputs that must come from trace preprocessing / hardware characterization.

    service_time_hours[i,j] is the execution duration of workload i on resource j.
    The simulator does not infer this matrix from arbitrary hidden constants.
    """
    service_time_hours: np.ndarray
    power_model: PowerModel
    billing_mode: BillingMode
    carbon_replay: CarbonReplay | None = None

    def validate(self, M: int, N: int) -> None:
        service = np.asarray(self.service_time_hours, dtype=float)
        if service.shape != (M, N):
            raise ValueError(f"service_time_hours must have shape {(M, N)}.")
        if np.any(~np.isfinite(service)) or np.any(service <= 0):
            raise ValueError("All service times must be finite and > 0.")
        self.power_model.validate(N)
        if self.billing_mode not in ("active_resource_hours", "allocated_gpu_hours"):
            raise ValueError("Unsupported billing_mode.")


@dataclass(frozen=True)
class SimulationResult:
    state: EvaluationState
    start_times: np.ndarray
    completion_times: np.ndarray
    resource_active_hours: np.ndarray
    allocated_gpu_hours: np.ndarray
    energy_kwh_by_resource: np.ndarray
    time_average_gpu_utilization: np.ndarray
    carbon_emissions_g_by_resource: np.ndarray | None = None


def _integrate_resource_energy(
    intervals: list[tuple[float, float, float]],
    *,
    gpu_capacity: float,
    idle_power_kw: float,
    dynamic_power_kw: float,
    power_exponent: float,
) -> tuple[float, float, float]:
    """Return (energy_kwh, active_hours, allocated_gpu_hours)."""
    if not intervals:
        return 0.0, 0.0, 0.0
    if gpu_capacity <= 0:
        raise ValueError("gpu_capacity must be > 0.")

    events: list[tuple[float, int, float]] = []
    allocated_gpu_hours = 0.0
    for start, end, demand in intervals:
        if end < start:
            raise ValueError("Interval end precedes start.")
        if demand < 0:
            raise ValueError("GPU demand cannot be negative.")
        events.append((start, 1, demand))
        events.append((end, 0, -demand))
        allocated_gpu_hours += (demand / gpu_capacity) * (end - start)
    events.sort(key=lambda e: (e[0], e[1]))

    energy = 0.0
    active_hours = 0.0
    demand_now = 0.0
    prev_t = events[0][0]

    for t, _, delta in events:
        dt = t - prev_t
        if dt < -1e-12:
            raise RuntimeError("Events are not time ordered.")
        if dt > TIME_ROUNDOFF_EPS_HOURS and demand_now > 1e-15:
            util = demand_now / gpu_capacity
            if util > 1.0 + 1e-9:
                raise ValueError(
                    f"Instantaneous GPU utilisation exceeds capacity: {util:.6f}."
                )
            util = min(max(util, 0.0), 1.0)
            power_kw = idle_power_kw + dynamic_power_kw * (util ** power_exponent)
            energy += power_kw * dt
            active_hours += dt
        demand_now += delta
        if demand_now < 0 and abs(demand_now) < 1e-12:
            demand_now = 0.0
        prev_t = t

    return float(energy), float(active_hours), float(allocated_gpu_hours)



def _integrate_resource_carbon(
    intervals: list[tuple[float, float, float]],
    *,
    carbon_replay: CarbonReplay,
    resource_index: int,
    gpu_capacity: float,
    idle_power_kw: float,
    dynamic_power_kw: float,
    power_exponent: float,
) -> float:
    """Integrate active GPU power against the forecast timeline."""
    if not intervals:
        return 0.0

    events = []
    for start, end, demand in intervals:
        events.append((float(start), 1, float(demand)))
        events.append((float(end), 0, -float(demand)))

    events.sort(key=lambda event: (event[0], event[1]))

    previous = events[0][0]
    active_gpu = 0.0
    emissions_g = 0.0

    for time, _, delta in events:
        duration = time - previous

        if duration > TIME_ROUNDOFF_EPS_HOURS and active_gpu > 1e-15:
            utilization = active_gpu / gpu_capacity

            if utilization > 1.0 + 1e-9:
                raise ValueError(
                    "Instantaneous GPU utilization exceeds capacity."
                )

            utilization = min(max(utilization, 0.0), 1.0)
            power_kw = (
                idle_power_kw
                + dynamic_power_kw * utilization ** power_exponent
            )

            integrated_ci = carbon_replay.integrate_intensity(
                previous, time, resource_index
            )
            emissions_g += power_kw * integrated_ci

        active_gpu += delta
        if active_gpu < 0 and abs(active_gpu) < 1e-12:
            active_gpu = 0.0
        previous = time

    return float(emissions_g)


def _window_capacity_ok(
    intervals: list[tuple[float, float, float, float, float]],
    start: float,
    end: float,
    gpu_demand: float,
    cpu_demand: float,
    memory_demand: float,
    gpu_capacity: float,
    cpu_capacity: float,
    memory_capacity: float,
) -> bool:
    boundaries = {start, end}

    for a, b, _, _, _ in intervals:
        if b > start and a < end:
            boundaries.add(max(start, a))
            boundaries.add(min(end, b))

    ordered = sorted(boundaries)

    for left, right in zip(ordered[:-1], ordered[1:]):
        if right - left <= TIME_ROUNDOFF_EPS_HOURS:
            continue

        mid = (left + right) / 2
        gpu = gpu_demand
        cpu = cpu_demand
        memory = memory_demand

        for a, b, g, c, m in intervals:
            if a <= mid < b:
                gpu += g
                cpu += c
                memory += m

        if (
            gpu > gpu_capacity + 1e-12
            or cpu > cpu_capacity + 1e-12
            or memory > memory_capacity + 1e-12
        ):
            return False

    return True


def _earliest_feasible_start(
    intervals: list[tuple[float, float, float, float, float]],
    arrival: float,
    duration: float,
    gpu_demand: float,
    cpu_demand: float,
    memory_demand: float,
    gpu_capacity: float,
    cpu_capacity: float,
    memory_capacity: float,
) -> float:
    if (
        gpu_demand > gpu_capacity + 1e-12
        or cpu_demand > cpu_capacity + 1e-12
        or memory_demand > memory_capacity + 1e-12
    ):
        raise ValueError(
            "Workload does not individually fit assigned resource."
        )

    candidate = float(arrival)

    for _ in range(len(intervals) + 2):
        end = candidate + duration

        if _window_capacity_ok(
            intervals,
            candidate,
            end,
            gpu_demand,
            cpu_demand,
            memory_demand,
            gpu_capacity,
            cpu_capacity,
            memory_capacity,
        ):
            return candidate

        future_ends = sorted({
            b for a, b, _, _, _ in intervals
            if b > candidate + 1e-12
        })

        if not future_ends:
            break

        candidate = future_ends[0]

    raise RuntimeError(
        "Could not find a temporally feasible start time."
    )


def simulate_fixed_assignment(
    problem: ProblemInstance,
    X: np.ndarray,
    inputs: SimulatorInputs,
    *,
    require_formal_feasibility: bool = True,
    carbon_replay: CarbonReplay | None = None,
) -> SimulationResult:
    """Simulate a fixed resource assignment with deterministic temporal capacity.

    Reconstruction convention S1-v10
    --------------------------------
    X selects one resource per workload. Workloads are processed in nondecreasing
    arrival order (original index breaks ties). On the assigned resource, each job
    starts at the earliest time >= arrival for which GPU, CPU, and memory capacities are
    available for its full service interval. This allows sequential and partially
    overlapping trace jobs while enforcing capacity at every simulated time.
    """
    M, N = len(problem.workloads), len(problem.resources)
    X = np.asarray(X)
    if X.shape != (M, N):
        raise ValueError(f"X must have shape {(M, N)}.")
    inputs.validate(M, N)

    if carbon_replay is None:
        carbon_replay = inputs.carbon_replay

    if carbon_replay is not None:
        carbon_replay.validate()
        expected_names = tuple(r.name for r in problem.resources)
        if carbon_replay.slot_names != expected_names:
            raise ValueError(
                "Carbon-replay resource order does not match the simulator."
            )

    report = check_constraints(problem, X)
    if require_formal_feasibility and not report.feasible:
        raise ValueError("Infeasible assignment: " + " | ".join(report.violations))
    if not report.binary_assignment_ok or not report.exactly_one_assignment_ok:
        raise ValueError("Simulator requires a binary, exactly-one assignment matrix.")

    # Carbon feasibility is checked above on every call. The deterministic temporal
    # schedule itself is carbon-independent, so an identical assignment can reuse it.
    global _SIMULATION_CACHE_HITS, _SIMULATION_CACHE_MISSES
    cache_key = _simulation_cache_key(problem, X, inputs) + (
        (id(carbon_replay),) if carbon_replay is not None
        else (None,)
    )
    cached = _SIMULATION_CACHE.get(cache_key)
    if cached is not None:
        _SIMULATION_CACHE_HITS += 1
        return cached
    _SIMULATION_CACHE_MISSES += 1

    assigned = np.argmax(X, axis=1)
    service = np.asarray(inputs.service_time_hours, dtype=float)
    arrivals = np.asarray([w.arrival_time for w in problem.workloads], dtype=float)
    starts = np.zeros(M, dtype=float)
    completion = np.zeros(M, dtype=float)
    scheduled: list[list[tuple[float,float,float,float,float]]] = [[] for _ in range(N)]

    for i in sorted(range(M), key=lambda k: (float(arrivals[k]), k)):
        j=int(assigned[i]); w=problem.workloads[i]; r=problem.resources[j]
        duration=float(service[i,j])
        cpu_limit = (
            float("inf")
            if r.cpu_capacity is None
            else float(r.cpu_capacity)
        )
        st = _earliest_feasible_start(
            scheduled[j],
            float(arrivals[i]),
            duration,
            float(w.gpu_demand),
            float(w.cpu_demand),
            float(w.memory_demand),
            float(r.gpu_capacity),
            cpu_limit,
            float(r.memory_capacity),
        )
        en=st+duration
        starts[i]=st; completion[i]=en
        scheduled[j].append((st,en,float(w.gpu_demand),float(w.cpu_demand),float(w.memory_demand)))

    idle = np.asarray(inputs.power_model.idle_power_kw, dtype=float)
    dyn = np.asarray(inputs.power_model.dynamic_power_kw, dtype=float)
    energy = np.zeros(N, dtype=float)
    active = np.zeros(N, dtype=float)
    allocated = np.zeros(N, dtype=float)
    emissions = (
        np.zeros(N, dtype=float)
        if carbon_replay is not None
        else None
    )

    for j, resource in enumerate(problem.resources):
        gpu_intervals=[(a,b,g) for a,b,g,_,_ in scheduled[j]]
        energy[j], active[j], allocated[j] = _integrate_resource_energy(
            gpu_intervals, gpu_capacity=float(resource.gpu_capacity),
            idle_power_kw=float(idle[j]), dynamic_power_kw=float(dyn[j]),
            power_exponent=float(inputs.power_model.power_exponent),
        )
        if carbon_replay is not None:
            emissions[j] = _integrate_resource_carbon(
                gpu_intervals,
                carbon_replay=carbon_replay,
                resource_index=j,
                gpu_capacity=float(resource.gpu_capacity),
                idle_power_kw=float(idle[j]),
                dynamic_power_kw=float(dyn[j]),
                power_exponent=float(inputs.power_model.power_exponent),
            )

    avg_util = np.divide(allocated,active,out=np.zeros_like(allocated),where=active>0)
    occupied = active if inputs.billing_mode == "active_resource_hours" else allocated
    state = EvaluationState(
        completion_times=completion,
        energy_kwh_by_resource=energy,
        occupied_gpu_hours=occupied,
        allocated_gpu_fraction_hours=allocated,
        carbon_emissions_g_by_resource=emissions,
    )
    result = SimulationResult(
        state=state,start_times=starts,completion_times=completion,
        resource_active_hours=active,allocated_gpu_hours=allocated,
        energy_kwh_by_resource=energy,time_average_gpu_utilization=avg_util,
        carbon_emissions_g_by_resource=emissions,
    )
    _SIMULATION_CACHE[cache_key] = result
    return result

