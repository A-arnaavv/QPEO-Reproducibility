from __future__ import annotations

import csv
import json
from datetime import datetime
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
                metadata={"source": "alibaba-gpu-v2023", "queue_hours": str((scheduled - creation) / 3600.0)},
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


def _parse_datetime(text: str) -> datetime:
    value = text.strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def load_helios_cluster_csv(path: str | Path, *, completed_only: bool = True, limit_jobs: int | None = None) -> list[RawTraceTask]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"job_id","gpu_num","cpu_num","state","submit_time","duration"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Helios CSV missing columns: {sorted(missing)}")
        for row in reader:
            if completed_only and row["state"].upper() != "COMPLETED":
                continue
            gpu = float(row["gpu_num"] or 0)
            runtime_s = float(row["duration"] or 0)
            if gpu <= 0 or runtime_s <= 0:
                continue
            submit = _parse_datetime(row["submit_time"])
            rows.append((submit, RawTraceTask(
                task_id=str(row["job_id"]), arrival_hours=0.0, runtime_hours=runtime_s/3600.0,
                cpu_demand=float(row["cpu_num"] or 0), gpu_demand=gpu,
                memory_demand=None, bandwidth_demand=None, qos=row.get("vc") or None,
                metadata={"source":"helios","vc":row.get("vc",""),"user":row.get("user",""),"queue_hours":str(float(row.get("queue") or 0)/3600.0)},
            )))
            if limit_jobs is not None and len(rows) >= limit_jobs:
                break
    rows.sort(key=lambda x:x[0])
    if not rows: return []
    t0 = rows[0][0]
    return [RawTraceTask(t.task_id,(dt-t0).total_seconds()/3600.0,t.runtime_hours,t.cpu_demand,t.gpu_demand,t.memory_demand,t.bandwidth_demand,t.qos,t.metadata) for dt,t in rows]


def load_acme_job_csv(path: str | Path, *, completed_only: bool = True, limit_jobs: int | None = None) -> list[RawTraceTask]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        needed={"job_id","gpu_num","cpu_num","state","submit_time","duration"}
        missing=needed-set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Acme CSV missing columns: {sorted(missing)}")
        for row in reader:
            if completed_only and row["state"].upper()!="COMPLETED":
                continue
            gpu=float(row["gpu_num"] or 0); runtime_s=float(row["duration"] or 0)
            if gpu<=0 or runtime_s<=0:
                continue
            submit=_parse_datetime(row["submit_time"])
            mem=None
            if "mem_per_pod_GB" in row and row.get("mem_per_pod_GB") not in (None,"","NA","nan"):
                mem=float(row["mem_per_pod_GB"])
            rows.append((submit, RawTraceTask(
                task_id=str(row["job_id"]), arrival_hours=0.0, runtime_hours=runtime_s/3600.0,
                cpu_demand=float(row["cpu_num"] or 0), gpu_demand=gpu, memory_demand=mem,
                bandwidth_demand=None, qos=row.get("type") or None,
                metadata={"source":"acme","type":row.get("type",""),"user":row.get("user",""),"node_num":row.get("node_num",""),"queue_hours":str(float(row.get("queue") or 0)/3600.0)},
            )))
            if limit_jobs is not None and len(rows)>=limit_jobs:
                break
    rows.sort(key=lambda x:x[0])
    if not rows: return []
    t0=rows[0][0]
    return [RawTraceTask(t.task_id,(dt-t0).total_seconds()/3600.0,t.runtime_hours,t.cpu_demand,t.gpu_demand,t.memory_demand,t.bandwidth_demand,t.qos,t.metadata) for dt,t in rows]


def _read_philly_objects(path: Path):
    files = sorted(p for p in (path.rglob("*") if path.is_dir() else [path]) if p.is_file())
    for file in files:
        text=file.read_text(encoding="utf-8").strip()
        if not text: continue
        if text.startswith("["):
            for obj in json.loads(text): yield obj
        else:
            for line in text.splitlines():
                line=line.strip()
                if line: yield json.loads(line)


def load_philly_job_log(path: str | Path, *, passed_only: bool = True, limit_jobs: int | None = None) -> list[RawTraceTask]:
    rows=[]
    for obj in _read_philly_objects(Path(path)):
        if passed_only and str(obj.get("status","")).lower()!="pass":
            continue
        submitted=obj.get("submitted_time")
        if not submitted: continue
        valid=[]
        for a in obj.get("attempts") or []:
            if not a.get("start_time") or not a.get("end_time"): continue
            start=_parse_datetime(a["start_time"]); end=_parse_datetime(a["end_time"])
            if end<=start: continue
            gpu_count=sum(len(d.get("gpus") or []) for d in (a.get("detail") or []))
            if gpu_count>0: valid.append((start,end,gpu_count))
        if not valid: continue
        start,end,gpu_count=valid[-1]
        submit=_parse_datetime(submitted)
        rows.append((submit, RawTraceTask(
            task_id=str(obj.get("jobid","")), arrival_hours=0.0,
            runtime_hours=(end-start).total_seconds()/3600.0,
            cpu_demand=None, gpu_demand=float(gpu_count), memory_demand=None,
            bandwidth_demand=None, qos=obj.get("vc") or None,
            metadata={"source":"philly","vc":obj.get("vc",""),"user":obj.get("user",""),"queue_hours":str(max(0.0,(start-submit).total_seconds()/3600.0))},
        )))
        if limit_jobs is not None and len(rows)>=limit_jobs: break
    rows.sort(key=lambda x:x[0])
    if not rows: return []
    t0=rows[0][0]
    return [RawTraceTask(t.task_id,(dt-t0).total_seconds()/3600.0,t.runtime_hours,t.cpu_demand,t.gpu_demand,t.memory_demand,t.bandwidth_demand,t.qos,t.metadata) for dt,t in rows]


def shard_multi_gpu_tasks(tasks: Sequence[RawTraceTask], *, max_scheduling_units: int) -> tuple[list[RawTraceTask], np.ndarray]:
    if max_scheduling_units < 1:
        raise ValueError("max_scheduling_units must be positive.")
    units=[]; groups=[]
    for task in tasks:
        if task.gpu_demand is None or task.gpu_demand <= 0:
            continue
        g=float(task.gpu_demand)
        if g <= 1:
            n=1; shard_gpu=g
        else:
            rounded=int(round(g))
            if not np.isclose(g,rounded):
                raise ValueError(f"Non-integer multi-GPU demand >1 for {task.task_id}: {g}")
            n=rounded; shard_gpu=1.0
        for k in range(n):
            if len(units)>=max_scheduling_units: break
            split=lambda v: None if v is None else float(v)/n
            md=dict(task.metadata)
            md.update({"original_task_id":task.task_id,"shard_index":str(k),"shard_count":str(n)})
            units.append(RawTraceTask(
                task_id=task.task_id if n==1 else f"{task.task_id}::shard{k+1}of{n}",
                arrival_hours=task.arrival_hours, runtime_hours=task.runtime_hours,
                cpu_demand=split(task.cpu_demand), gpu_demand=shard_gpu,
                memory_demand=split(task.memory_demand), bandwidth_demand=split(task.bandwidth_demand),
                qos=task.qos, metadata=md,
            ))
            groups.append(task.task_id)
        if len(units)>=max_scheduling_units: break
    M=len(units)
    dep=np.zeros((M,M),dtype=float)
    for i in range(M):
        for j in range(M):
            if i!=j and groups[i]==groups[j]:
                dep[i,j]=1.0
    return units,dep


@dataclass(frozen=True)
class NativeTraceMaterializationPolicy:
    memory_imputation_gib_per_gpu: float
    cpu_imputation_per_gpu: float
    default_bandwidth_demand: float
    default_carbon_class: CarbonClass

    def validate(self) -> None:
        if self.memory_imputation_gib_per_gpu <= 0:
            raise ValueError("memory_imputation_gib_per_gpu must be > 0.")
        if self.cpu_imputation_per_gpu <= 0:
            raise ValueError("cpu_imputation_per_gpu must be > 0.")
        if self.default_bandwidth_demand < 0:
            raise ValueError("default_bandwidth_demand must be >= 0.")


def observed_queue_hours(task: RawTraceTask) -> float:
    raw = task.metadata.get("queue_hours") if task.metadata else None
    if raw in (None, ""):
        raise ValueError(f"Task {task.task_id} lacks observed queue-time metadata.")
    q = float(raw)
    if q < 0:
        raise ValueError("Observed queue time cannot be negative.")
    return q


def trace_derived_deadline(task: RawTraceTask) -> float:
    return float(task.arrival_hours + observed_queue_hours(task) + task.runtime_hours)


def pooled_native_per_gpu_median(tasks: Sequence[RawTraceTask], *, field: str) -> float:
    if field not in {"memory_demand","cpu_demand"}:
        raise ValueError("field must be memory_demand or cpu_demand")
    vals=[]
    for task in tasks:
        demand=getattr(task,field)
        gpu=task.gpu_demand
        if demand is None or gpu is None or gpu <= 0:
            continue
        value=float(demand)/float(gpu)
        if np.isfinite(value) and value>0:
            vals.append(value)
    if not vals:
        raise ValueError(f"No observed positive values available for {field}.")
    return float(np.median(np.asarray(vals,dtype=float)))


def materialize_gpu_native_workloads(tasks: Sequence[RawTraceTask], policy: NativeTraceMaterializationPolicy) -> list[Workload]:
    policy.validate()
    out=[]
    for task in tasks:
        gpu=float(task.gpu_demand) if task.gpu_demand is not None else None
        if gpu is None or gpu <= 0 or gpu > 1+1e-12:
            raise ValueError("materialize_gpu_native_workloads expects already-sharded tasks with 0<gpu<=1.")
        cpu=float(task.cpu_demand) if task.cpu_demand is not None else policy.cpu_imputation_per_gpu*gpu
        mem=float(task.memory_demand) if task.memory_demand is not None else policy.memory_imputation_gib_per_gpu*gpu
        bw=float(task.bandwidth_demand) if task.bandwidth_demand is not None else policy.default_bandwidth_demand
        out.append(Workload(
            name=task.task_id, cpu_demand=cpu, gpu_demand=gpu, memory_demand=mem,
            bandwidth_demand=bw, arrival_time=float(task.arrival_hours),
            deadline=trace_derived_deadline(task), carbon_class=policy.default_carbon_class,
        ))
    return out
