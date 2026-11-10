from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence
import numpy as np

class CarbonClass(str, Enum):
    CRITICAL = "critical"
    DEFERRED = "deferred"
    FLEXIBLE = "flexible"

@dataclass(frozen=True)
class Workload:
    name: str
    cpu_demand: float
    gpu_demand: float
    memory_demand: float
    bandwidth_demand: float
    arrival_time: float
    deadline: float
    carbon_class: CarbonClass

@dataclass(frozen=True)
class Resource:
    name: str
    gpu_capacity: float
    memory_capacity: float
    carbon_intensity: float
    cost_per_gpu_hour: float
    cpu_capacity: float | None = None
    bandwidth_capacity: float | None = None

@dataclass(frozen=True)
class ProblemInstance:
    workloads: Sequence[Workload]
    resources: Sequence[Resource]
    carbon_thresholds: Mapping[CarbonClass, float]
    reference_cost: float
    def __post_init__(self):
        if not self.workloads or not self.resources:
            raise ValueError("At least one workload and one resource are required.")
        if self.reference_cost <= 0:
            raise ValueError("reference_cost must be > 0.")
        missing=set(CarbonClass)-set(self.carbon_thresholds)
        if missing:
            raise ValueError(f"Missing carbon thresholds: {[x.value for x in missing]}")

@dataclass(frozen=True)
class EvaluationState:
    completion_times: np.ndarray
    energy_kwh_by_resource: np.ndarray
    occupied_gpu_hours: np.ndarray
    allocated_gpu_fraction_hours: np.ndarray
    def validate(self,M:int,N:int):
        if np.asarray(self.completion_times).shape!=(M,): raise ValueError(f"completion_times must have shape {(M,)}")
        if np.asarray(self.energy_kwh_by_resource).shape!=(N,): raise ValueError(f"energy_kwh_by_resource must have shape {(N,)}")
        if np.asarray(self.occupied_gpu_hours).shape!=(N,): raise ValueError(f"occupied_gpu_hours must have shape {(N,)}")
        if np.asarray(self.allocated_gpu_fraction_hours).shape!=(N,): raise ValueError(f"allocated_gpu_fraction_hours must have shape {(N,)}")
        if np.any(np.asarray(self.energy_kwh_by_resource)<0): raise ValueError("Energy cannot be negative.")
        if np.any(np.asarray(self.occupied_gpu_hours)<0): raise ValueError("GPU-hours cannot be negative.")
        if np.any(np.asarray(self.allocated_gpu_fraction_hours)<0): raise ValueError("Allocated GPU fraction-hours cannot be negative.")

@dataclass(frozen=True)
class ObjectiveVector:
    f1_negative_gpu_util: float
    f2_energy_kwh: float
    f3_energy_weighted_carbon_intensity: float
    f4_makespan: float
    f5_sla_violation_percent: float
    f6_normalized_cost: float
    def as_array(self):
        return np.asarray([self.f1_negative_gpu_util,self.f2_energy_kwh,self.f3_energy_weighted_carbon_intensity,
                           self.f4_makespan,self.f5_sla_violation_percent,self.f6_normalized_cost],dtype=float)

@dataclass(frozen=True)
class ConstraintReport:
    feasible: bool
    binary_assignment_ok: bool
    exactly_one_assignment_ok: bool
    gpu_capacity_ok: bool
    cpu_capacity_ok: bool
    memory_capacity_ok: bool
    carbon_threshold_ok: bool
    violations: tuple[str,...]
