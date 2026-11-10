from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CarbonBenchmarkPolicy:
    provider: str
    api_version: str
    temporal_granularity: str
    emission_factor_type: str
    flow_traced: bool
    disable_caller_lookup: bool
    start_utc: str
    end_utc: str
    layer_zones: Mapping[str, str]

    def validate(self) -> None:
        if self.provider != "Electricity Maps":
            raise ValueError("This reconstruction currently supports Electricity Maps only.")
        if self.api_version != "v4":
            raise ValueError("Expected Electricity Maps API v4.")
        if self.temporal_granularity != "hourly":
            raise ValueError("Primary benchmark must use hourly carbon data.")
        if self.emission_factor_type not in {"lifecycle", "direct"}:
            raise ValueError("Unsupported emission factor type.")
        start = datetime.fromisoformat(self.start_utc.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.end_utc.replace("Z", "+00:00"))
        if end <= start:
            raise ValueError("Carbon benchmark window must be non-empty.")
        if (end-start).total_seconds()/3600 != 168:
            raise ValueError("Primary carbon benchmark window must contain exactly 168 hours.")
        if set(self.layer_zones) != {"cloud", "edge", "serverless"}:
            raise ValueError("layer_zones must define cloud, edge, and serverless.")


@dataclass(frozen=True)
class CarbonSeries:
    zone: str
    datetimes_utc: tuple[str, ...]
    intensity_gco2eq_per_kwh: np.ndarray
    is_estimated: np.ndarray

    def validate(self, expected_hours: int = 168) -> None:
        n = len(self.datetimes_utc)
        if n != expected_hours:
            raise ValueError(f"Expected {expected_hours} hourly points, got {n}.")
        if np.asarray(self.intensity_gco2eq_per_kwh).shape != (n,):
            raise ValueError("Intensity vector shape mismatch.")
        if np.asarray(self.is_estimated).shape != (n,):
            raise ValueError("is_estimated vector shape mismatch.")
        if np.any(~np.isfinite(self.intensity_gco2eq_per_kwh)) or np.any(self.intensity_gco2eq_per_kwh < 0):
            raise ValueError("Invalid carbon intensity values.")


def load_carbon_policy(path: str | Path) -> tuple[CarbonBenchmarkPolicy, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    p = data["primary_carbon_benchmark"]
    policy = CarbonBenchmarkPolicy(
        provider=p["provider"],
        api_version=p["api_version"],
        temporal_granularity=p["temporal_granularity"],
        emission_factor_type=p["emission_factor_type"],
        flow_traced=bool(p["flow_traced"]),
        disable_caller_lookup=bool(p["disable_caller_lookup"]),
        start_utc=p["start_utc"],
        end_utc=p["end_utc"],
        layer_zones=dict(p["layer_zones"]),
    )
    policy.validate()
    return policy, data


def electricity_maps_past_range_query(policy: CarbonBenchmarkPolicy, layer: str) -> dict:
    policy.validate()
    if layer not in policy.layer_zones:
        raise ValueError(f"Unknown layer: {layer}")
    return {
        "zone": policy.layer_zones[layer],
        "start": policy.start_utc,
        "end": policy.end_utc,
        "temporalGranularity": policy.temporal_granularity,
        "emissionFactorType": policy.emission_factor_type,
        "flowTraced": policy.flow_traced,
        "disableCallerLookup": policy.disable_caller_lookup,
    }


def save_carbon_snapshot_csv(path: str | Path, series: CarbonSeries) -> str:
    series.validate()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zone","datetime_utc","carbon_intensity_gco2eq_per_kwh","is_estimated"])
        for dt, ci, est in zip(series.datetimes_utc, series.intensity_gco2eq_per_kwh, series.is_estimated):
            w.writerow([series.zone, dt, f"{float(ci):.10g}", int(bool(est))])
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_carbon_snapshot_csv(path: str | Path, expected_zone: str | None = None) -> CarbonSeries:
    p = Path(path)
    zone = None
    times=[]; vals=[]; est=[]
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            zone = zone or row["zone"]
            if row["zone"] != zone:
                raise ValueError("One snapshot file must contain exactly one zone.")
            times.append(row["datetime_utc"])
            vals.append(float(row["carbon_intensity_gco2eq_per_kwh"]))
            est.append(bool(int(row["is_estimated"])))
    if zone is None:
        raise ValueError("Carbon snapshot is empty.")
    if expected_zone is not None and zone != expected_zone:
        raise ValueError(f"Expected zone {expected_zone}, got {zone}.")
    s = CarbonSeries(zone, tuple(times), np.asarray(vals,float), np.asarray(est,bool))
    s.validate()
    return s


def estimated_fraction(series: CarbonSeries) -> float:
    series.validate()
    return float(np.mean(series.is_estimated))


def layer_intensity_at_hour(layer_series: Mapping[str, CarbonSeries], hour_index: int) -> dict[str,float]:
    if hour_index < 0:
        raise ValueError("hour_index must be non-negative.")
    out={}
    for layer in ("cloud","edge","serverless"):
        s=layer_series[layer]
        s.validate()
        out[layer]=float(s.intensity_gco2eq_per_kwh[hour_index % 168])
    return out
