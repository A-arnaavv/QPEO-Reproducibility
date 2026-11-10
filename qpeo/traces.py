from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .models import CarbonClass, Workload


@dataclass(frozen=True)
class RawTraceTask:
    task_id: str
    arrival_hours: float
    runtime_hours: float
    cpu_demand: float | None = None
    gpu_demand: float | None = None
    memory_demand: float | None = None
    bandwidth_demand: float | None = None
    qos: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceMaterializationPolicy:
    fallback_cpu_demand: float | None
    fallback_gpu_demand: float | None
    fallback_memory_demand: float | None
    fallback_bandwidth_demand: float | None
    deadline_slack_multiplier: float
    default_carbon_class: CarbonClass
    qos_to_carbon_class: Mapping[str, CarbonClass] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.deadline_slack_multiplier <= 0:
            raise ValueError("deadline_slack_multiplier must be > 0.")


def _required_or_fallback(value: float | None, fallback: float | None, name: str) -> float:
    if value is not None and np.isfinite(value):
        return float(value)
    if fallback is None:
        raise ValueError(
            f"Trace does not provide {name}; the materialization policy must supply an explicit fallback."
        )
    return float(fallback)


def materialize_workloads(
    tasks: Sequence[RawTraceTask],
    policy: TraceMaterializationPolicy,
) -> list[Workload]:
    workloads: list[Workload] = []
    for task in tasks:
        cpu = _required_or_fallback(task.cpu_demand, policy.fallback_cpu_demand, "CPU demand")
        gpu = _required_or_fallback(task.gpu_demand, policy.fallback_gpu_demand, "GPU demand")
        mem = _required_or_fallback(task.memory_demand, policy.fallback_memory_demand, "memory demand")
        bw = _required_or_fallback(task.bandwidth_demand, policy.fallback_bandwidth_demand, "bandwidth demand")
        cclass = policy.qos_to_carbon_class.get(task.qos or "", policy.default_carbon_class)
        deadline = task.arrival_hours + policy.deadline_slack_multiplier * task.runtime_hours
        workloads.append(
            Workload(
                name=task.task_id,
                cpu_demand=cpu,
                gpu_demand=gpu,
                memory_demand=mem,
                bandwidth_demand=bw,
                arrival_time=float(task.arrival_hours),
                deadline=float(deadline),
                carbon_class=cclass,
            )
        )
    return workloads


def service_time_matrix_from_observed_runtime(
    tasks: Sequence[RawTraceTask],
    resource_speed_factors: Sequence[float],
) -> np.ndarray:
    speed = np.asarray(resource_speed_factors, dtype=float)
    if speed.ndim != 1 or np.any(~np.isfinite(speed)) or np.any(speed <= 0):
        raise ValueError("resource_speed_factors must be a positive 1-D array.")
    base = np.asarray([t.runtime_hours for t in tasks], dtype=float)
    if np.any(~np.isfinite(base)) or np.any(base <= 0):
        raise ValueError("All trace runtimes must be finite and > 0.")
    return base[:, None] / speed[None, :]


def load_alibaba_gpu_v2023_pods_csv(
    path: str | Path,
    *,
    gpu_only: bool = True,
    limit: int | None = None,
) -> list[RawTraceTask]:
    out: list[RawTraceTask] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"name", "cpu_milli", "memory_mib", "num_gpu", "gpu_milli", "qos", "creation_time", "deletion_time", "scheduled_time"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Alibaba CSV missing columns: {sorted(missing)}")
        first_creation: float | None = None
        for row in reader:
            num_gpu = float(row["num_gpu"] or 0)
            gpu_milli = float(row["gpu_milli"] or 0)
            if gpu_only and num_gpu <= 0 and gpu_milli <= 0:
                continue
            creation = float(row["creation_time"])
            scheduled = float(row["scheduled_time"])
            deletion = float(row["deletion_time"])
            runtime_s = deletion - scheduled
            if runtime_s <= 0:
                continue
            if first_creation is None:
                first_creation = creation
            if num_gpu == 1 and gpu_milli > 0:
                gpu = gpu_milli / 1000.0
            else:
                gpu = num_gpu
            out.append(RawTraceTask(
                task_id=row["name"],
                arrival_hours=(creation - first_creation) / 3600.0,
                runtime_hours=runtime_s / 3600.0,
                cpu_demand=float(row["cpu_milli"] or 0) / 1000.0,
                gpu_demand=float(gpu),
                memory_demand=float(row["memory_mib"] or 0) / 1024.0,
                bandwidth_demand=None,
                qos=row.get("qos") or None,
                metadata={"source": "alibaba-gpu-v2023"},
            ))
            if limit is not None and len(out) >= limit:
                break
    return out


def load_azure_functions_2021_csv(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[RawTraceTask]:
    out: list[RawTraceTask] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"app", "func", "end_timestamp", "duration"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Azure Functions CSV missing columns: {sorted(missing)}")
        first_start: float | None = None
        for idx, row in enumerate(reader):
            end_s = float(row["end_timestamp"])
            duration_s = float(row["duration"])
            if duration_s <= 0:
                continue
            start_s = end_s - duration_s
            if first_start is None:
                first_start = start_s
            out.append(RawTraceTask(
                task_id=f"{row['app']}:{row['func']}:{idx}",
                arrival_hours=(start_s - first_start) / 3600.0,
                runtime_hours=duration_s / 3600.0,
                cpu_demand=None,
                gpu_demand=None,
                memory_demand=None,
                bandwidth_demand=None,
                qos=None,
                metadata={"source": "azure-functions-2021", "app": row["app"], "func": row["func"]},
            ))
            if limit is not None and len(out) >= limit:
                break
    return out


def load_canonical_trace_csv(path: str | Path, *, limit: int | None = None) -> list[RawTraceTask]:
    out: list[RawTraceTask] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"task_id", "arrival_hours", "runtime_hours"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Canonical trace CSV missing columns: {sorted(missing)}")

        def opt(row, key):
            v = row.get(key)
            return None if v in (None, "", "NA", "nan") else float(v)

        for row in reader:
            out.append(RawTraceTask(
                task_id=row["task_id"],
                arrival_hours=float(row["arrival_hours"]),
                runtime_hours=float(row["runtime_hours"]),
                cpu_demand=opt(row, "cpu_demand"),
                gpu_demand=opt(row, "gpu_demand"),
                memory_demand=opt(row, "memory_demand"),
                bandwidth_demand=opt(row, "bandwidth_demand"),
                qos=row.get("qos") or None,
                metadata={"source": row.get("source", "canonical")},
            ))
            if limit is not None and len(out) >= limit:
                break
    return out
