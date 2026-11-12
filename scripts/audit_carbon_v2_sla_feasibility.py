"""Independent, read-only SLA deadline feasibility diagnostic for Carbon-v2.

This is an optimistic per-workload lower bound: each job can start exactly at
arrival, with no competing workload. It is not an optimizer run and does not
prove that all individually feasible jobs can simultaneously finish on time.

Carbon eligibility is evaluated from the frozen full-week regional mean,
matching the candidate Carbon-v2 planning policy. No input is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from qpeo.carbon import with_carbon_intensities
from scripts.run_final_experiment_shard import SCENARIOS, build_scenario


# Tolerance on simulation-clock hours: 1e-8 h = 36 microseconds.
SLA_TIME_TOLERANCE_HOURS = 1e-8


def audit_instance(problem, service_time_hours: np.ndarray, *, tol: float = SLA_TIME_TOLERANCE_HOURS) -> dict:
    """Classify optimistic deadline slack with CPU/GPU/RAM/carbon eligibility."""
    workloads, resources = problem.workloads, problem.resources
    durations = np.asarray(service_time_hours, dtype=float)
    m, n = len(workloads), len(resources)
    if durations.shape != (m, n) or not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("positive finite service times are required for each workload/resource")
    if not (np.isfinite(tol) and tol >= 0):
        raise ValueError("tol must be finite and nonnegative")

    allowed = np.zeros((m, n), dtype=bool)
    for i, w in enumerate(workloads):
        threshold = float(problem.carbon_thresholds[w.carbon_class])
        for j, r in enumerate(resources):
            allowed[i, j] = (
                w.gpu_demand <= r.gpu_capacity + 1e-12
                and w.cpu_demand <= (np.inf if r.cpu_capacity is None else r.cpu_capacity) + 1e-12
                and w.memory_demand <= r.memory_capacity + 1e-12
                and r.carbon_intensity <= threshold + 1e-12
            )
    eligible_resource_counts = allowed.sum(axis=1)
    min_service = np.min(np.where(allowed, durations, np.inf), axis=1)
    arrivals = np.asarray([w.arrival_time for w in workloads], dtype=float)
    deadlines = np.asarray([w.deadline for w in workloads], dtype=float)
    finite_deadline = np.isfinite(deadlines)
    slack = deadlines - arrivals - min_service
    impossible = finite_deadline & (slack < -tol)
    tight = finite_deadline & np.isfinite(slack) & (np.abs(slack) <= tol)
    positive = finite_deadline & (slack > tol)
    n_finite = int(finite_deadline.sum())
    impossible_count = int(impossible.sum())
    by_class = {}
    for label in ("critical", "deferred", "flexible"):
        mask = np.asarray([w.carbon_class.value == label for w in workloads])
        finite_mask = mask & finite_deadline
        by_class[label] = {
            "jobs": int(mask.sum()),
            "finite_deadlines": int(finite_mask.sum()),
            "optimistically_impossible": int((finite_mask & impossible).sum()),
            "zero_slack_jobs": int((finite_mask & tight).sum()),
        }
    finite_slack = slack[finite_deadline & np.isfinite(slack)]
    return {
        "workloads": m,
        "resources": n,
        "finite_deadlines": n_finite,
        "no_deadline_jobs": int((~finite_deadline).sum()),
        "jobs_with_no_eligible_resource": int((eligible_resource_counts == 0).sum()),
        "optimistically_impossible_jobs": impossible_count,
        "optimistic_sla_violation_floor_pct":
            100.0 * impossible_count / n_finite if n_finite else None,
        "zero_slack_jobs": int(tight.sum()),
        "positive_slack_jobs": int(positive.sum()),
        "deadline_slack_hours_median": float(np.median(finite_slack)) if len(finite_slack) else None,
        "deadline_slack_hours_min": float(np.min(finite_slack)) if len(finite_slack) else None,
        "deadline_slack_hours_max": float(np.max(finite_slack)) if len(finite_slack) else None,
        "by_carbon_class": by_class,
    }


def audit_scenario(name: str, *, size: int = 64) -> dict:
    bundle, environment, replay = build_scenario(name, size)
    fixed_ci = np.mean(replay.intensities, axis=0)
    planning_problem = with_carbon_intensities(bundle.problem, fixed_ci)
    report = audit_instance(planning_problem, bundle.service_time_hours)
    report["scenario"] = name
    report["planning_carbon_policy"] = "fixed arithmetic mean of frozen 168-hour regional replay"
    report["carbon_eligibility_thresholds_gco2e_per_kwh"] = {
        k.value: float(v) for k, v in planning_problem.carbon_thresholds.items()
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics/carbon_v2_sla_feasibility.json")
    args = parser.parse_args()
    results = [audit_scenario(name) for name in SCENARIOS]
    payload = {
        "status": "READ_ONLY_OPTIMISTIC_SLA_FEASIBILITY_DIAGNOSTIC_NOT_AN_EXPERIMENT",
        "scope": "locked 64-job scenario snapshots; Carbon-v2 frozen week-mean eligibility",
        "definition": "A job is intrinsically late if arrival + shortest service time on any eligible resource exceeds its deadline by more than 1e-8 hours, even assuming no resource contention.",
        "caveats": [
            "No competition or queuing is assumed, so this is a lower bound on violations, not achievable cohort SLA performance.",
            "Deadline definitions differ between datasets and may be observed end-time or synthetic proxies rather than contractual SLAs.",
            "A zero-slack job can meet the deadline only if assigned a fastest eligible resource and never queued.",
            "No optimizer was run, no original trace or results file was modified.",
        ],
        "scenarios": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
    for r in results:
        print(r["scenario"], f"impossible={r['optimistically_impossible_jobs']}/{r['finite_deadlines']}",
              f"zero_slack={r['zero_slack_jobs']}/{r['finite_deadlines']}")


if __name__ == "__main__":
    main()
