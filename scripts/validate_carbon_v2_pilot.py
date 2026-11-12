"""Audit a carbon-v2 test shard before it can be used as manuscript evidence.

This is a read-only validator: it does not run any optimizer, edit a result, or
upgrade a pilot to primary. Usage:
    python scripts/validate_carbon_v2_pilot.py \
      results/carbon_v2_pilot/<test-output>.json

It reconstructs the locked trace inputs and checks recorded guide schedules,
all concurrent resource capacities, modeled energy, and time-aligned carbon.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from qpeo.carbon_accounting import CarbonTimeSeries, integrate_time_aligned_carbon
from qpeo.carbon import with_carbon_intensities
from qpeo.simulator import simulate_fixed_assignment, clear_simulation_cache
from scripts.run_final_experiment_shard import build_scenario

EXPECTED = (
    "QPEO", "PSO", "DE", "GWO", "WOA", "ABC", "GA",
    "HS", "FA", "ACO", "FPA", "QIEA", "NSGA-II",
)


def _near(actual: float, expected: float, what: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise AssertionError(f"{what}: actual={actual!r}, expected={expected!r}")


def validate_pilot(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("final_protocol") is not False:
        raise AssertionError("Expected an explicitly test-tagged, non-primary shard")
    version = payload.get("carbon_time_accounting")
    if not isinstance(version, dict) or version.get("version") != "v2":
        raise AssertionError("Not a carbon-v2 output")
    policy = version.get("coverage_policy")
    if policy not in ("strict", "repeat"):
        raise AssertionError("Undefined carbon coverage policy")
    if payload.get("size") not in (32, 64, 128):
        raise AssertionError("Unrecognized workload size")
    algorithms = payload.get("algorithms")
    if not isinstance(algorithms, list) or len(algorithms) != 13:
        raise AssertionError("Expected exactly 13 algorithm records")
    names = [a["algorithm"] for a in algorithms]
    if len(set(names)) != 13 or set(names) != set(EXPECTED):
        raise AssertionError("Missing/duplicated/unknown algorithms")

    bundle, env, replay = build_scenario(payload["scenario"], int(payload["size"]))
    plan = version.get("planning_signal")
    if plan != "fixed full-week regional mean intensity":
        raise AssertionError("Pilot is not using the locked stationary planning signal")
    planning_ci = np.mean(replay.intensities, axis=0)
    np.testing.assert_allclose(
        np.asarray(version["planning_carbon_intensities_g_per_kwh"], dtype=float),
        planning_ci, rtol=1e-12, atol=1e-12,
    )
    planning_problem = with_carbon_intensities(bundle.problem, planning_ci)
    m, n = len(bundle.problem.workloads), len(bundle.problem.resources)
    if int(payload["carbon_rows"]) != replay.rows:
        raise AssertionError("Forecast row count differs from saved pilot")

    timestamps = np.array([t.timestamp() for t in replay.timestamps_utc], dtype=float)
    step_h = float(np.median(np.diff(timestamps)) / 3600.0)
    if not np.allclose(np.diff(timestamps) / 3600.0, step_h, atol=1e-9):
        raise AssertionError("Forecast timestamps are not uniformly spaced")
    forecast = CarbonTimeSeries(
        replay.intensities, step_h, forecast_start_hours=0.0,
        coverage_policy=policy
    )
    inputs = bundle.qpeo_inputs.simulator_inputs
    cap = np.array([r.gpu_capacity for r in bundle.problem.resources], dtype=float)
    cpu_cap = np.array([
        math.inf if r.cpu_capacity is None else r.cpu_capacity
        for r in bundle.problem.resources
    ], dtype=float)
    mem_cap = np.array([r.memory_capacity for r in bundle.problem.resources], dtype=float)
    workloads = bundle.problem.workloads
    gpu = np.array([w.gpu_demand for w in workloads], dtype=float)
    cpu = np.array([w.cpu_demand for w in workloads], dtype=float)
    memory = np.array([w.memory_demand for w in workloads], dtype=float)
    arrivals = np.array([w.arrival_time for w in workloads], dtype=float)
    durations = np.asarray(inputs.service_time_hours, dtype=float)
    findings = []

    for record in algorithms:
        name = record["algorithm"]
        assignment = np.asarray(record["guide_assignment_indices"], dtype=int)
        start = np.asarray(record["guide_start_hours"], dtype=float)
        finish = np.asarray(record["guide_completion_hours"], dtype=float)
        objective = np.asarray(record["guide_objectives"], dtype=float)
        if assignment.shape != (m,) or start.shape != (m,) or finish.shape != (m,):
            raise AssertionError(f"{name}: incomplete assignment/interval export")
        if objective.shape != (6,) or not np.all(np.isfinite(objective)):
            raise AssertionError(f"{name}: invalid objective vector")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(finish)):
            raise AssertionError(f"{name}: invalid schedule times")
        if np.any(assignment < 0) or np.any(assignment >= n):
            raise AssertionError(f"{name}: invalid resource index")
        if np.any(start < arrivals - 1e-9):
            raise AssertionError(f"{name}: job scheduled before its arrival")
        for i, j in enumerate(assignment):
            _near(float(finish[i] - start[i]), float(durations[i, j]),
                  f"{name} job {i}: service-time mismatch")
            if gpu[i] > cap[j] + 1e-9 or cpu[i] > cpu_cap[j] + 1e-9 or memory[i] > mem_cap[j] + 1e-9:
                raise AssertionError(f"{name} job {i}: individual resource-capacity violation")

        X = np.zeros((m, n), dtype=int)
        X[np.arange(m), assignment] = 1
        clear_simulation_cache()
        reconstructed = simulate_fixed_assignment(planning_problem, X, inputs)
        np.testing.assert_allclose(reconstructed.start_times, start, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(reconstructed.completion_times, finish, rtol=1e-9, atol=1e-9)
        intervals_by_resource: list[list[tuple[float, float, float]]] = [[] for _ in range(n)]
        for j in range(n):
            indices = np.flatnonzero(assignment == j)
            boundaries = sorted(set(start[indices].tolist() + finish[indices].tolist()))
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                if right - left <= 1e-12:
                    continue
                center = (left + right) / 2.0
                active = indices[(start[indices] <= center) & (center < finish[indices])]
                if gpu[active].sum() > cap[j] + 1e-9:
                    raise AssertionError(f"{name}: concurrent GPU overflow on resource {j}")
                if cpu[active].sum() > cpu_cap[j] + 1e-9:
                    raise AssertionError(f"{name}: concurrent CPU overflow on resource {j}")
                if memory[active].sum() > mem_cap[j] + 1e-9:
                    raise AssertionError(f"{name}: concurrent RAM overflow on resource {j}")
            intervals_by_resource[j] = [
                (float(start[i]), float(finish[i]), float(gpu[i]))
                for i in indices
            ]

        carbon = integrate_time_aligned_carbon(
            intervals_by_resource,
            gpu_capacities=cap,
            idle_power_kw=inputs.power_model.idle_power_kw,
            dynamic_power_kw=inputs.power_model.dynamic_power_kw,
            power_exponent=inputs.power_model.power_exponent,
            forecast=forecast,
        )
        _near(float(objective[1]), carbon.total_energy_kwh, f"{name}: energy mismatch")
        _near(float(objective[2]), carbon.energy_weighted_intensity_g_per_kwh,
              f"{name}: objective carbon-intensity mismatch")
        _near(float(record["guide_carbon_g"]), carbon.total_carbon_g,
              f"{name}: carbon-mass mismatch")
        _near(float(record["guide_carbon_intensity_g_per_kwh"]),
              carbon.energy_weighted_intensity_g_per_kwh,
              f"{name}: saved intensity mismatch")
        findings.append({
            "algorithm": name,
            "energy_kwh": carbon.total_energy_kwh,
            "carbon_g": carbon.total_carbon_g,
            "intensity_gco2e_per_kwh": carbon.energy_weighted_intensity_g_per_kwh,
            "capacity_check": "passed",
            "energy_carbon_recalculation": "passed",
            "reconstructed_schedule_and_planning_constraints": "passed",
        })

    return {
        "status": "PASSED_TEST_SHARD_NOT_PRIMARY",
        "scenario": payload["scenario"],
        "seed_index": payload["seed_index"],
        "population_size": payload["population_size"],
        "max_iterations": payload["max_iterations"],
        "coverage_policy": policy,
        "workloads": m,
        "resources": n,
        "algorithms_verified": len(findings),
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_file", type=Path)
    parser.add_argument("--report-file", type=Path, default=None)
    args = parser.parse_args()
    report = validate_pilot(args.shard_file)
    value = json.dumps(report, indent=2)
    if args.report_file is None:
        print(value)
    else:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(value + "\n", encoding="utf-8")
        print(args.report_file)


if __name__ == "__main__":
    main()
