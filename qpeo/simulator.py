from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .evaluator import check_constraints
from .models import EvaluationState, ProblemInstance


BillingMode = Literal["active_resource_hours", "allocated_gpu_hours"]


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
        if dt > 0 and demand_now > 1e-15:
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



def _window_capacity_ok(
    intervals: list[tuple[float, float, float, float]],
    start: float,
    end: float,
    gpu_demand: float,
    memory_demand: float,
    gpu_capacity: float,
    memory_capacity: float,
) -> bool:
    boundaries={start,end}
    for a,b,_,_ in intervals:
        if b>start and a<end:
            boundaries.add(max(start,a)); boundaries.add(min(end,b))
    ordered=sorted(boundaries)
    for left,right in zip(ordered[:-1],ordered[1:]):
        if right-left<=1e-15:
            continue
        mid=(left+right)/2
        gpu=gpu_demand; mem=memory_demand
        for a,b,g,m in intervals:
            if a<=mid<b:
                gpu+=g; mem+=m
        if gpu>gpu_capacity+1e-12 or mem>memory_capacity+1e-12:
            return False
    return True


def _earliest_feasible_start(
    intervals: list[tuple[float, float, float, float]],
    arrival: float,
    duration: float,
    gpu_demand: float,
    memory_demand: float,
    gpu_capacity: float,
    memory_capacity: float,
) -> float:
    if gpu_demand>gpu_capacity+1e-12 or memory_demand>memory_capacity+1e-12:
        raise ValueError("Workload does not individually fit assigned resource.")
    candidate=float(arrival)
    for _ in range(len(intervals)+2):
        end=candidate+duration
        if _window_capacity_ok(intervals,candidate,end,gpu_demand,memory_demand,gpu_capacity,memory_capacity):
            return candidate
        future_ends=sorted({b for a,b,_,_ in intervals if b>candidate+1e-12})
        if not future_ends:
            break
        candidate=future_ends[0]
    raise RuntimeError("Could not find a temporally feasible start time.")

def simulate_fixed_assignment(
    problem: ProblemInstance,
    X: np.ndarray,
    inputs: SimulatorInputs,
    *,
    require_formal_feasibility: bool = True,
) -> SimulationResult:
    """Simulate a fixed resource assignment with deterministic temporal capacity.

    Reconstruction convention S1-v10
    --------------------------------
    X selects one resource per workload. Workloads are processed in nondecreasing
    arrival order (original index breaks ties). On the assigned resource, each job
    starts at the earliest time >= arrival for which both GPU and memory capacity are
    available for its full service interval. This allows sequential and partially
    overlapping trace jobs while enforcing capacity at every simulated time.
    """
    M, N = len(problem.workloads), len(problem.resources)
    X = np.asarray(X)
    if X.shape != (M, N):
        raise ValueError(f"X must have shape {(M, N)}.")
    inputs.validate(M, N)

    report = check_constraints(problem, X)
    if require_formal_feasibility and not report.feasible:
        raise ValueError("Infeasible assignment: " + " | ".join(report.violations))
    if not report.binary_assignment_ok or not report.exactly_one_assignment_ok:
        raise ValueError("Simulator requires a binary, exactly-one assignment matrix.")

    assigned = np.argmax(X, axis=1)
    service = np.asarray(inputs.service_time_hours, dtype=float)
    arrivals = np.asarray([w.arrival_time for w in problem.workloads], dtype=float)
    starts = np.zeros(M, dtype=float)
    completion = np.zeros(M, dtype=float)
    scheduled: list[list[tuple[float,float,float,float]]] = [[] for _ in range(N)]

    for i in sorted(range(M), key=lambda k: (float(arrivals[k]), k)):
        j=int(assigned[i]); w=problem.workloads[i]; r=problem.resources[j]
        duration=float(service[i,j])
        st=_earliest_feasible_start(
            scheduled[j], float(arrivals[i]), duration, float(w.gpu_demand),
            float(w.memory_demand), float(r.gpu_capacity), float(r.memory_capacity),
        )
        en=st+duration
        starts[i]=st; completion[i]=en
        scheduled[j].append((st,en,float(w.gpu_demand),float(w.memory_demand)))

    idle = np.asarray(inputs.power_model.idle_power_kw, dtype=float)
    dyn = np.asarray(inputs.power_model.dynamic_power_kw, dtype=float)
    energy = np.zeros(N, dtype=float)
    active = np.zeros(N, dtype=float)
    allocated = np.zeros(N, dtype=float)

    for j, resource in enumerate(problem.resources):
        gpu_intervals=[(a,b,g) for a,b,g,_ in scheduled[j]]
        energy[j], active[j], allocated[j] = _integrate_resource_energy(
            gpu_intervals, gpu_capacity=float(resource.gpu_capacity),
            idle_power_kw=float(idle[j]), dynamic_power_kw=float(dyn[j]),
            power_exponent=float(inputs.power_model.power_exponent),
        )

    avg_util = np.divide(allocated,active,out=np.zeros_like(allocated),where=active>0)
    occupied = active if inputs.billing_mode == "active_resource_hours" else allocated
    state = EvaluationState(
        completion_times=completion,
        energy_kwh_by_resource=energy,
        occupied_gpu_hours=occupied,
        allocated_gpu_fraction_hours=allocated,
    )
    return SimulationResult(
        state=state,start_times=starts,completion_times=completion,
        resource_active_hours=active,allocated_gpu_hours=allocated,
        energy_kwh_by_resource=energy,time_average_gpu_utilization=avg_util,
    )

