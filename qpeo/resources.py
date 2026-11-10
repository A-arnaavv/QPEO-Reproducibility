from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .models import Resource
from .simulator import PowerModel


@dataclass(frozen=True)
class HardwareTemplate:
    key: str
    vendor_model: str
    memory_gib: float
    max_power_w: float
    fp32_tflops: float
    layer: str
    source_kind: str = "vendor-published"

    def validate(self) -> None:
        if self.memory_gib <= 0 or self.max_power_w <= 0 or self.fp32_tflops <= 0:
            raise ValueError("Hardware memory, maximum power, and FP32 throughput must be positive.")
        if self.layer not in {"cloud", "edge", "serverless"}:
            raise ValueError(f"Unsupported layer: {self.layer}")


@dataclass(frozen=True)
class ResourcePoolSpec:
    counts: Mapping[str, int]
    idle_fraction_of_max: float
    cost_per_gpu_hour: Mapping[str, float]
    speed_factor: Mapping[str, float]

    def validate(self, templates: Mapping[str, HardwareTemplate]) -> None:
        if not (0 < self.idle_fraction_of_max < 1):
            raise ValueError("idle_fraction_of_max must lie in (0,1).")
        for key, count in self.counts.items():
            if key not in templates:
                raise ValueError(f"Unknown hardware template: {key}")
            if int(count) < 0:
                raise ValueError("Resource counts must be non-negative.")
            if key not in self.cost_per_gpu_hour:
                raise ValueError(f"Missing cost_per_gpu_hour for {key}")
            if key not in self.speed_factor:
                raise ValueError(f"Missing speed_factor for {key}")
            if self.cost_per_gpu_hour[key] <= 0:
                raise ValueError(f"Cost must be > 0 for {key}")
            if self.speed_factor[key] <= 0:
                raise ValueError(f"Speed factor must be > 0 for {key}")


def load_resource_config(path: str | Path) -> tuple[dict[str, HardwareTemplate], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    templates: dict[str, HardwareTemplate] = {}
    for key, raw in data["hardware_templates"].items():
        t = HardwareTemplate(
            key=key,
            vendor_model=str(raw["vendor_model"]),
            memory_gib=float(raw["memory_gib"]),
            max_power_w=float(raw["max_power_w"]),
            fp32_tflops=float(raw["fp32_tflops"]),
            layer=str(raw["layer"]),
            source_kind=str(raw.get("source_kind", "vendor-published")),
        )
        t.validate()
        templates[key] = t
    return templates, data


def build_resource_pool(
    templates: Mapping[str, HardwareTemplate],
    spec: ResourcePoolSpec,
    *,
    carbon_intensity_by_layer: Mapping[str, float],
) -> tuple[list[Resource], PowerModel, np.ndarray, list[str]]:
    spec.validate(templates)
    resources, idle_kw, dynamic_kw, speed, template_keys = [], [], [], [], []

    for key, count in spec.counts.items():
        t = templates[key]
        if t.layer not in carbon_intensity_by_layer:
            raise ValueError(f"Missing carbon intensity for layer {t.layer}")
        ci = float(carbon_intensity_by_layer[t.layer])
        if ci < 0:
            raise ValueError("Carbon intensity cannot be negative.")
        pmax_kw = t.max_power_w / 1000.0
        idle = spec.idle_fraction_of_max * pmax_kw
        dynamic = pmax_kw - idle

        for idx in range(int(count)):
            resources.append(
                Resource(
                    name=f"{t.layer}-{key}-{idx:02d}",
                    gpu_capacity=1.0,
                    memory_capacity=t.memory_gib,
                    carbon_intensity=ci,
                    cost_per_gpu_hour=float(spec.cost_per_gpu_hour[key]),
                )
            )
            idle_kw.append(idle)
            dynamic_kw.append(dynamic)
            speed.append(float(spec.speed_factor[key]))
            template_keys.append(key)

    if not resources:
        raise ValueError("Resource pool cannot be empty.")

    return (
        resources,
        PowerModel(
            idle_power_kw=np.asarray(idle_kw, dtype=float),
            dynamic_power_kw=np.asarray(dynamic_kw, dtype=float),
            power_exponent=1.0,
        ),
        np.asarray(speed, dtype=float),
        template_keys,
    )


def fp32_normalized_speed_factors(
    templates: Mapping[str, HardwareTemplate],
    *,
    baseline_key: str = "a5000",
) -> dict[str, float]:
    """Vendor-derived relative compute factors from published FP32 throughput."""
    if baseline_key not in templates:
        raise ValueError(f"Unknown baseline template: {baseline_key}")
    base = float(templates[baseline_key].fp32_tflops)
    return {key: float(t.fp32_tflops / base) for key, t in templates.items()}
