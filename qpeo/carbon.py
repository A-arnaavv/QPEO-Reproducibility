from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .models import CarbonClass, ProblemInstance, Resource


@dataclass(frozen=True)
class CarbonReplay:
    timestamps_utc: tuple[datetime, ...]
    slot_names: tuple[str, ...]
    intensities: np.ndarray
    source_name: str
    window_start_utc: datetime
    window_end_utc: datetime

    def validate(self) -> None:
        x=np.asarray(self.intensities,dtype=float)
        if x.ndim!=2 or x.shape!=(len(self.timestamps_utc),len(self.slot_names)):
            raise ValueError("intensities shape must match timestamps x slot_names")
        if len(self.timestamps_utc)<2:
            raise ValueError("carbon replay requires at least two time points")
        if any(t.tzinfo is None for t in self.timestamps_utc):
            raise ValueError("all replay timestamps must be timezone-aware")
        if any(self.timestamps_utc[i]>=self.timestamps_utc[i+1] for i in range(len(self.timestamps_utc)-1)):
            raise ValueError("replay timestamps must be strictly increasing")
        if np.any(~np.isfinite(x)) or np.any(x<0):
            raise ValueError("carbon intensities must be finite and non-negative")

    @property
    def rows(self)->int:
        return len(self.timestamps_utc)

    @property
    def resources(self)->int:
        return len(self.slot_names)

    def integrate_intensity(
        self,
        start_hour: float,
        end_hour: float,
        resource_index: int,
    ) -> float:
        """Integrate stepwise carbon intensity over simulated hours.

        The first simulated hour is aligned with the first forecast row.
        Forecast values repeat periodically beyond the locked week.
        The returned quantity has units gCO2e/kWh * hour.
        """
        start = float(start_hour)
        end = float(end_hour)

        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("Simulation times must be finite.")
        if end < start:
            raise ValueError("Execution interval ends before it starts.")
        if not 0 <= resource_index < self.resources:
            raise IndexError("Invalid carbon-replay resource index.")
        if end == start:
            return 0.0

        offsets = np.asarray([
            (stamp - self.timestamps_utc[0]).total_seconds() / 3600.0
            for stamp in self.timestamps_utc
        ])
        differences = np.diff(offsets)
        step = float(differences[0])

        if step <= 0 or not np.allclose(
            differences, step, rtol=0, atol=1e-8
        ):
            raise ValueError(
                "Time-aligned accounting requires a regular forecast grid."
            )

        values = np.asarray(
            self.intensities[:, resource_index], dtype=float
        )

        prefix = np.concatenate((
            [0.0], np.cumsum(values) * step
        ))

        period = self.rows * step

        def antiderivative(hour):
            cycles = np.floor(hour / period)
            remainder = hour - cycles * period

            if remainder >= period:
                cycles += 1
                remainder = 0.0

            index = min(
                int(np.floor(remainder / step)),
                self.rows - 1,
            )

            return float(
                cycles * prefix[-1]
                + prefix[index]
                + (remainder - index * step) * values[index]
            )

        return antiderivative(end) - antiderivative(start)

    def planning_mean(self) -> np.ndarray:
        """Mean regional intensity over the complete frozen forecast."""
        return np.asarray(
            self.intensities, dtype=float
        ).mean(axis=0).copy()

    def at_iteration(self,t:int,max_iterations:int)->np.ndarray:
        if max_iterations<1 or not (0<=t<=max_iterations):
            raise ValueError("require max_iterations>=1 and 0<=t<=max_iterations")
        idx=int(round(t*(self.rows-1)/max_iterations))
        return np.asarray(self.intensities[idx],dtype=float).copy()

    def thresholds(self)->dict[CarbonClass,float]:
        x=np.asarray(self.intensities,dtype=float)
        ordered=np.sort(x,axis=1)
        eps=1e-9
        deferred_index=min(2,ordered.shape[1]-1)
        return {
            CarbonClass.CRITICAL: float(np.max(ordered[:,0])+eps),
            CarbonClass.DEFERRED: float(np.max(ordered[:,deferred_index])+eps),
            CarbonClass.FLEXIBLE: float(np.max(ordered[:,-1])+eps),
        }


def _parse_timestamp(value:str)->datetime:
    s=value.strip()
    if not s:
        raise ValueError("empty timestamp")
    candidates=["%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%d/%m/%Y %H:%M","%d/%m/%Y %H:%M:%S"]
    try:
        dt=datetime.fromisoformat(s.replace("Z","+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in candidates:
        try:
            return datetime.strptime(s,fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {value!r}")


def _fill_missing_column(x:np.ndarray)->np.ndarray:
    y=np.asarray(x,dtype=float).copy(); finite=np.isfinite(y)
    if not np.any(finite):
        raise ValueError("selected carbon region contains no finite observations")
    first=int(np.flatnonzero(finite)[0]); y[:first]=y[first]
    for i in range(first+1,len(y)):
        if not np.isfinite(y[i]):
            y[i]=y[i-1]
    return y


def load_carbon_replay_config(path:str|Path)->dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_neso_regional_replay(csv_path:str|Path,config_path:str|Path,*,expected_slot_names:Sequence[str]|None=None)->CarbonReplay:
    cfg=load_carbon_replay_config(config_path)
    start=_parse_timestamp(cfg["window_utc"]["start"]); end=_parse_timestamp(cfg["window_utc"]["end"])
    mapping=cfg["resource_region_mapping"]; slot_names=tuple(mapping.keys())
    if expected_slot_names is not None and tuple(expected_slot_names)!=slot_names:
        raise ValueError("carbon replay slot order does not match the resource environment")
    columns=[mapping[s]["column"] for s in slot_names]
    rows=[]
    with Path(csv_path).open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f); fields=set(reader.fieldnames or [])
        ts_col=next((c for c in ["Date & Time","Datetime","Date Time","timestamp","Timestamp"] if c in fields),None)
        if ts_col is None:
            raise ValueError(f"Could not find timestamp column in NESO CSV; fields={sorted(fields)}")
        missing=[c for c in columns if c not in fields]
        if missing:
            raise ValueError(f"NESO CSV missing selected regional columns: {missing}")
        for row in reader:
            ts=_parse_timestamp(row[ts_col])
            if ts<start or ts>end:
                continue
            vals=[]
            for c in columns:
                raw=(row.get(c) or "").strip()
                try: vals.append(float(raw))
                except ValueError: vals.append(float("nan"))
            rows.append((ts,vals))
    if not rows:
        raise ValueError("No carbon rows found inside the locked replay window")
    rows.sort(key=lambda z:z[0]); timestamps=tuple(r[0] for r in rows); matrix=np.asarray([r[1] for r in rows],dtype=float)
    for j in range(matrix.shape[1]):
        matrix[:,j]=_fill_missing_column(matrix[:,j])
    replay=CarbonReplay(timestamps,slot_names,matrix,str(cfg["source"]["name"]),start,end); replay.validate(); return replay


def with_carbon_intensities(problem:ProblemInstance,values:Mapping[str,float]|Sequence[float])->ProblemInstance:
    if isinstance(values,Mapping):
        ci=np.asarray([float(values[r.name]) for r in problem.resources],dtype=float)
    else:
        ci=np.asarray(values,dtype=float)
        if ci.shape!=(len(problem.resources),):
            raise ValueError(f"carbon vector must have shape {(len(problem.resources),)}")
    resources=[Resource(
        name=r.name,gpu_capacity=r.gpu_capacity,memory_capacity=r.memory_capacity,
        carbon_intensity=float(ci[i]),cost_per_gpu_hour=r.cost_per_gpu_hour,
        cpu_capacity=r.cpu_capacity,bandwidth_capacity=r.bandwidth_capacity,
    ) for i,r in enumerate(problem.resources)]
    return ProblemInstance(problem.workloads,resources,problem.carbon_thresholds,problem.reference_cost)
