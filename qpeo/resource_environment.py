from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .models import CarbonClass, ProblemInstance, Resource, Workload
from .simulator import PowerModel


@dataclass(frozen=True)
class HardwareClassSpec:
    key: str
    display_name: str
    gpu_memory_gb: float
    max_gpu_power_kw: float
    fp32_tflops: float
    source: str


@dataclass(frozen=True)
class ResourceSlotSpec:
    name: str
    layer: str
    hardware_class: str
    gpu_capacity: float
    cpu_capacity: float | None
    host_memory_gb: float
    cost_index_primary: float


@dataclass(frozen=True)
class ResourceEnvironment:
    version: str
    baseline_fp32_tflops: float
    reference_cost: float
    hardware_classes: Mapping[str, HardwareClassSpec]
    slots: tuple[ResourceSlotSpec, ...]
    primary_idle_fraction: float
    primary_power_exponent: float
    sensitivity_idle_fractions: tuple[float, ...]

    def validate(self) -> None:
        if self.baseline_fp32_tflops <= 0:
            raise ValueError("baseline_fp32_tflops must be > 0")
        if self.reference_cost <= 0:
            raise ValueError("reference cost must be > 0")
        names = [s.name for s in self.slots]
        if len(names) != len(set(names)):
            raise ValueError("resource slot names must be unique")
        for slot in self.slots:
            if slot.hardware_class not in self.hardware_classes:
                raise ValueError(f"Unknown hardware class {slot.hardware_class!r}")
            if slot.gpu_capacity <= 0 or slot.host_memory_gb <= 0 or slot.cost_index_primary <= 0:
                raise ValueError("resource capacities, host memory, and primary costs must be > 0")
        if not (0 <= self.primary_idle_fraction < 1):
            raise ValueError("primary idle fraction must lie in [0,1)")
        if self.primary_power_exponent <= 0:
            raise ValueError("primary power exponent must be > 0")
        if any((x < 0 or x >= 1) for x in self.sensitivity_idle_fractions):
            raise ValueError("idle-fraction sensitivity values must lie in [0,1)")

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(s.layer for s in self.slots)

    def speed_factors(self) -> np.ndarray:
        return np.asarray([
            self.hardware_classes[s.hardware_class].fp32_tflops / self.baseline_fp32_tflops
            for s in self.slots
        ], dtype=float)

    def max_power_kw(self) -> np.ndarray:
        return np.asarray([
            self.hardware_classes[s.hardware_class].max_gpu_power_kw
            for s in self.slots
        ], dtype=float)

    def cost_indices(self, profile: str = "primary_tier_index") -> np.ndarray:
        if profile == "primary_tier_index":
            return np.asarray([s.cost_index_primary for s in self.slots], dtype=float)
        if profile == "performance_proportional":
            return self.speed_factors()
        if profile == "power_proportional":
            return self.max_power_kw() / 0.300
        raise ValueError(f"Unknown cost profile: {profile}")

    def power_model(self, *, idle_fraction: float | None = None) -> PowerModel:
        f = self.primary_idle_fraction if idle_fraction is None else float(idle_fraction)
        if not (0 <= f < 1):
            raise ValueError("idle_fraction must lie in [0,1)")
        pmax = self.max_power_kw()
        return PowerModel(
            idle_power_kw=f * pmax,
            dynamic_power_kw=(1.0 - f) * pmax,
            power_exponent=self.primary_power_exponent,
        )

    def build_resources(self, carbon_intensity_by_slot: Mapping[str, float] | Sequence[float], *, cost_profile: str = "primary_tier_index") -> list[Resource]:
        if isinstance(carbon_intensity_by_slot, Mapping):
            missing = [s.name for s in self.slots if s.name not in carbon_intensity_by_slot]
            if missing:
                raise ValueError(f"Missing carbon intensities for slots: {missing}")
            ci = np.asarray([float(carbon_intensity_by_slot[s.name]) for s in self.slots], dtype=float)
        else:
            ci = np.asarray(carbon_intensity_by_slot, dtype=float)
            if ci.shape != (len(self.slots),):
                raise ValueError(f"carbon_intensity_by_slot must have shape {(len(self.slots),)}")
        if np.any(~np.isfinite(ci)) or np.any(ci < 0):
            raise ValueError("carbon intensities must be finite and non-negative")
        costs = self.cost_indices(cost_profile)
        resources = []
        for idx, slot in enumerate(self.slots):
            hw = self.hardware_classes[slot.hardware_class]
            resources.append(Resource(
                name=slot.name,
                gpu_capacity=float(slot.gpu_capacity),
                memory_capacity=float(slot.host_memory_gb),
                carbon_intensity=float(ci[idx]),
                cost_per_gpu_hour=float(costs[idx]),
                cpu_capacity=None if slot.cpu_capacity is None else float(slot.cpu_capacity),
                bandwidth_capacity=None,
            ))
        return resources

    def build_problem(self, workloads: Sequence[Workload], carbon_intensity_by_slot: Mapping[str, float] | Sequence[float], carbon_thresholds: Mapping[CarbonClass, float], *, cost_profile: str = "primary_tier_index") -> ProblemInstance:
        return ProblemInstance(
            workloads=list(workloads),
            resources=self.build_resources(carbon_intensity_by_slot, cost_profile=cost_profile),
            carbon_thresholds=carbon_thresholds,
            reference_cost=self.reference_cost,
        )


def load_resource_environment(path: str | Path) -> ResourceEnvironment:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    hw = {
        key: HardwareClassSpec(
            key=key,
            display_name=str(value["display_name"]),
            gpu_memory_gb=float(value["gpu_memory_gb"]),
            max_gpu_power_kw=float(value["max_gpu_power_kw"]),
            fp32_tflops=float(value["fp32_tflops"]),
            source=str(value["source"]),
        )
        for key, value in raw["hardware_classes"].items()
    }
    slots = tuple(ResourceSlotSpec(
        name=str(x["name"]),
        layer=str(x["layer"]),
        hardware_class=str(x["hardware_class"]),
        gpu_capacity=float(x["gpu_capacity"]),
        cpu_capacity=None if x.get("cpu_capacity") is None else float(x["cpu_capacity"]),
        host_memory_gb=float(x["host_memory_gb"]),
        cost_index_primary=float(x["cost_index_primary"]),
    ) for x in raw["resource_slots"])
    power = raw["power_model"]
    env = ResourceEnvironment(
        version=str(raw["environment_version"]),
        baseline_fp32_tflops=float(raw["baseline_fp32_tflops"]),
        reference_cost=float(raw["reference_cost_units_per_gpu_hour"]),
        hardware_classes=hw,
        slots=slots,
        primary_idle_fraction=float(power["primary"]["idle_fraction_of_max"]),
        primary_power_exponent=float(power["primary"]["power_exponent"]),
        sensitivity_idle_fractions=tuple(float(x) for x in power["sensitivity_idle_fractions"]),
    )
    env.validate()
    return env
