from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from .carbon import CarbonReplay
from .models import CarbonClass, ProblemInstance, Workload
from .optimizer import QPEOInputs
from .resource_environment import ResourceEnvironment
from .simulator import SimulatorInputs


@dataclass(frozen=True)
class ScenarioBundle:
    dataset_id: str
    task_ids: tuple[str, ...]
    workloads: tuple[Workload, ...]
    base_runtime_hours: np.ndarray
    service_time_hours: np.ndarray
    problem: ProblemInstance
    qpeo_inputs: QPEOInputs


def _qos_class(value: str, mapping: dict[str, str]) -> CarbonClass:
    raw=mapping.get(value)
    if raw is None:
        raise ValueError(f"No carbon-class mapping for QoS {value!r}")
    return CarbonClass(raw)


def _load_alibaba_snapshot(path: str | Path) -> list[dict[str,str]]:
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    needed={"name","cpu_milli","memory_mib","num_gpu","gpu_milli","qos","pod_phase","creation_time","deletion_time","scheduled_time"}
    if not rows:
        raise ValueError("Alibaba snapshot is empty")
    missing=needed-set(rows[0])
    if missing:
        raise ValueError(f"Alibaba snapshot missing fields: {sorted(missing)}")
    rows.sort(key=lambda r:(float(r["creation_time"]),r["name"]))
    return rows


def build_alibaba_gpu_v2023_scenario(
    snapshot_path: str | Path,
    trace_config_path: str | Path,
    resource_environment: ResourceEnvironment,
    carbon_replay: CarbonReplay,
    *,
    size: int = 64,
) -> ScenarioBundle:
    cfg=json.loads(Path(trace_config_path).read_text(encoding="utf-8"))
    if cfg.get("dataset_id")!="alibaba_gpu_v2023":
        raise ValueError("Unexpected trace config dataset_id")
    rows=_load_alibaba_snapshot(snapshot_path)
    if len(rows)!=int(cfg["eligible_count"]):
        raise ValueError("Alibaba eligible snapshot row count does not match locked config")
    key=str(int(size))
    window=cfg["sampling"]["windows"].get(key)
    if window is None:
        raise ValueError(f"No locked Alibaba window for size={size}")
    start,end=int(window["start_index"]),int(window["end_index"])
    selected=rows[start:end+1]
    if len(selected)!=size:
        raise ValueError("Locked Alibaba window has wrong size")
    if selected[0]["name"]!=window["first_task"] or selected[-1]["name"]!=window["last_task"]:
        raise ValueError("Alibaba snapshot does not match locked window task IDs")

    base_creation=float(selected[0]["creation_time"])
    qos_map=cfg["workload_mapping"]["carbon_class_by_qos"]
    workloads=[]; runtimes=[]; ids=[]
    for row in selected:
        creation=float(row["creation_time"]); scheduled=float(row["scheduled_time"]); deletion=float(row["deletion_time"])
        arrival=(creation-base_creation)/3600.0
        runtime=(deletion-scheduled)/3600.0
        e2e=(deletion-creation)/3600.0
        if runtime<=0 or e2e<=0:
            raise ValueError("Locked Alibaba snapshot contains non-positive duration")
        workload=Workload(
            name=row["name"],
            cpu_demand=float(row["cpu_milli"])/1000.0,
            gpu_demand=float(row["gpu_milli"])/1000.0,
            memory_demand=float(row["memory_mib"])/1024.0,
            bandwidth_demand=0.0,
            arrival_time=arrival,
            deadline=arrival+e2e,
            carbon_class=_qos_class(row["qos"],qos_map),
        )
        workloads.append(workload); runtimes.append(runtime); ids.append(row["name"])

    if tuple(slot.name for slot in resource_environment.slots)!=carbon_replay.slot_names:
        raise ValueError("Resource environment and carbon replay slot order differ")
    problem=resource_environment.build_problem(
        workloads, carbon_replay.intensities[0], carbon_replay.thresholds(),
        cost_profile="primary_tier_index",
    )
    speed=resource_environment.speed_factors()
    base_runtime=np.asarray(runtimes,dtype=float)
    service=base_runtime[:,None]/speed[None,:]
    inv=1.0/service
    performance=inv/np.max(inv,axis=1,keepdims=True)
    pmax=resource_environment.max_power_kw()
    thermal=1.0-pmax/np.max(pmax)
    dependency=np.zeros((size,size),dtype=float)
    sim=SimulatorInputs(service,resource_environment.power_model(),"allocated_gpu_hours")
    qinputs=QPEOInputs(sim,performance,thermal,dependency,carbon_replay)
    qinputs.validate(size,len(problem.resources))
    return ScenarioBundle(
        dataset_id="alibaba_gpu_v2023",
        task_ids=tuple(ids),
        workloads=tuple(workloads),
        base_runtime_hours=base_runtime,
        service_time_hours=service,
        problem=problem,
        qpeo_inputs=qinputs,
    )


def build_elasticflow_itp_scenario(
    snapshot_path: str | Path,
    trace_config_path: str | Path,
    resource_environment: ResourceEnvironment,
    carbon_replay: CarbonReplay,
    *,
    size: int = 64,
) -> ScenarioBundle:
    cfg=json.loads(Path(trace_config_path).read_text(encoding="utf-8"))
    if cfg.get("dataset_id")!="microsoft_elasticflow_itp_cluster08":
        raise ValueError("Unexpected ElasticFlow trace config dataset_id")
    with Path(snapshot_path).open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=int(cfg["selection"]["eligible_count"]):
        raise ValueError("ElasticFlow snapshot row count does not match locked config")
    rows.sort(key=lambda r:(float(r["submission_time"]),r["job_id"]))
    window=cfg["selection"]["windows"].get(str(int(size)))
    if window is None:
        raise ValueError(f"No locked ElasticFlow window for size={size}")
    start,end=int(window["start_index"]),int(window["end_index"])
    selected=rows[start:end+1]
    if len(selected)!=size:
        raise ValueError("Locked ElasticFlow window has wrong size")
    if selected[0]["job_id"]!=window["first_task"] or selected[-1]["job_id"]!=window["last_task"]:
        raise ValueError("ElasticFlow snapshot does not match locked window task IDs")

    base_submission=float(selected[0]["submission_time"])
    q33=float(cfg["deadline"]["slack_ratio_tertiles"]["q33"])
    q67=float(cfg["deadline"]["slack_ratio_tertiles"]["q67"])
    workloads=[]; runtimes=[]; ids=[]
    for row in selected:
        submission=float(row["submission_time"])
        duration=float(row["duration"])
        deadline_abs=float(row["deadline"])
        ratio=float(row["deadline_slack_ratio"])
        if duration<=0 or deadline_abs<=submission:
            raise ValueError("Locked ElasticFlow snapshot contains invalid duration/deadline")
        if ratio<=q33:
            cclass=CarbonClass.CRITICAL
        elif ratio<=q67:
            cclass=CarbonClass.DEFERRED
        else:
            cclass=CarbonClass.FLEXIBLE
        arrival=(submission-base_submission)/3600.0
        deadline=(deadline_abs-base_submission)/3600.0
        workloads.append(Workload(
            name=row["job_id"],
            cpu_demand=0.0,
            gpu_demand=1.0,
            memory_demand=0.0,
            bandwidth_demand=0.0,
            arrival_time=arrival,
            deadline=deadline,
            carbon_class=cclass,
        ))
        runtimes.append(duration/3600.0)
        ids.append(row["job_id"])

    if tuple(slot.name for slot in resource_environment.slots)!=carbon_replay.slot_names:
        raise ValueError("Resource environment and carbon replay slot order differ")
    problem=resource_environment.build_problem(
        workloads,carbon_replay.intensities[0],carbon_replay.thresholds(),
        cost_profile="primary_tier_index",
    )
    speed=resource_environment.speed_factors()
    base_runtime=np.asarray(runtimes,dtype=float)
    service=base_runtime[:,None]/speed[None,:]
    inv=1.0/service
    performance=inv/np.max(inv,axis=1,keepdims=True)
    pmax=resource_environment.max_power_kw()
    thermal=1.0-pmax/np.max(pmax)
    dependency=np.zeros((size,size),dtype=float)
    sim=SimulatorInputs(service,resource_environment.power_model(),"allocated_gpu_hours")
    qinputs=QPEOInputs(sim,performance,thermal,dependency,carbon_replay)
    qinputs.validate(size,len(problem.resources))
    return ScenarioBundle(
        dataset_id="microsoft_elasticflow_itp_cluster08",
        task_ids=tuple(ids),
        workloads=tuple(workloads),
        base_runtime_hours=base_runtime,
        service_time_hours=service,
        problem=problem,
        qpeo_inputs=qinputs,
    )
