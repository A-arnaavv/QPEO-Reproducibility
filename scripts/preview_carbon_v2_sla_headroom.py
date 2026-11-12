"""Read-only preview of a PROPOSED, separately approved SLA headroom sensitivity.

Does not run optimizers, change original scenario data, or replace the primary
f5 deadline definition. Only reports the effects of candidate synthetic
headroom policies to inform protocol/coauthor review.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from qpeo.carbon import with_carbon_intensities
from qpeo.models import ProblemInstance
from scripts.audit_carbon_v2_sla_feasibility import audit_instance
from scripts.run_final_experiment_shard import SCENARIOS, build_scenario

# Proposed fixed margin grid, subject to coauthor approval before any optimizer run.
HEADROOM_FRACTIONS = (0.25, 0.50, 1.00)


def minimum_eligible_service_hours(problem: ProblemInstance, service_time_hours: np.ndarray) -> np.ndarray:
    """Return the uncontended per-job lower service bound under frozen v2 eligibility."""
    duration = np.asarray(service_time_hours, dtype=float)
    m, n = len(problem.workloads), len(problem.resources)
    if duration.shape != (m, n) or not np.all(np.isfinite(duration)) or np.any(duration <= 0):
        raise ValueError("service_time_hours must have finite positive (workloads,resources) entries")
    eligible = np.zeros((m, n), dtype=bool)
    for i, w in enumerate(problem.workloads):
        for j, r in enumerate(problem.resources):
            eligible[i, j] = (
                w.gpu_demand <= r.gpu_capacity + 1e-12
                and w.cpu_demand <= (np.inf if r.cpu_capacity is None else r.cpu_capacity) + 1e-12
                and w.memory_demand <= r.memory_capacity + 1e-12
                and r.carbon_intensity <= problem.carbon_thresholds[w.carbon_class] + 1e-12
            )
    if np.any(~np.any(eligible, axis=1)):
        indices = np.flatnonzero(~np.any(eligible, axis=1)).tolist()
        raise ValueError(f"No individually eligible resource for workloads: {indices}")
    return np.min(np.where(eligible, duration, np.inf), axis=1)


def make_headroom_variant(problem: ProblemInstance, service_time_hours: np.ndarray,
                          fraction: float) -> tuple[ProblemInstance, dict]:
    """Never tighten a finite deadline or convert an infinite one into a finite SLA."""
    if not np.isfinite(fraction) or fraction <= 0:
        raise ValueError("positive finite headroom fraction required")
    service_min = minimum_eligible_service_hours(problem, service_time_hours)
    originals = np.asarray([w.deadline for w in problem.workloads], dtype=float)
    arrivals = np.asarray([w.arrival_time for w in problem.workloads], dtype=float)
    if not np.all(np.isfinite(arrivals)) or np.any(np.isnan(originals)) or np.any(np.isneginf(originals)):
        raise ValueError("arrivals must be finite and deadlines must be finite or +inf")
    finite = np.isfinite(originals)
    # At least fraction * service_min of uncontended slack for finite-deadline jobs.
    floor = arrivals + (1.0 + fraction) * service_min
    new_deadlines = np.where(finite, np.maximum(originals, floor), np.inf)
    variant = replace(problem, workloads=tuple(
        replace(w, deadline=float(new_deadlines[i]))
        for i, w in enumerate(problem.workloads)
    ))
    delta = new_deadlines[finite] - originals[finite]
    diagnostic = audit_instance(variant, service_time_hours)
    metadata = {
        "headroom_fraction": fraction,
        "finite_deadlines": int(finite.sum()),
        "unchanged_infinite_deadlines": int((~finite).sum()),
        "modified_finite_deadlines": int(np.count_nonzero(delta > 1e-8)),
        "deadline_increase_hours_median": float(np.median(delta)) if delta.size else None,
        "deadline_increase_hours_max": float(np.max(delta)) if delta.size else None,
        "optimistically_impossible_after": diagnostic["optimistically_impossible_jobs"],
        "zero_slack_after": diagnostic["zero_slack_jobs"],
        "optimistic_sla_violation_floor_pct_after": diagnostic["optimistic_sla_violation_floor_pct"],
        "positive_slack_after": diagnostic["positive_slack_jobs"],
        "min_service_time_hours_median": float(np.median(service_min)),
    }
    return variant, metadata


def preview_scenario(name: str) -> dict:
    bundle, _env, replay = build_scenario(name, 64)
    planning = with_carbon_intensities(bundle.problem, np.mean(replay.intensities, axis=0))
    baseline = audit_instance(planning, bundle.service_time_hours)
    variants = []
    for fraction in HEADROOM_FRACTIONS:
        _p, record = make_headroom_variant(planning, bundle.service_time_hours, fraction)
        variants.append(record)
    return {"scenario": name, "baseline": {
        "finite_deadlines": baseline["finite_deadlines"],
        "no_deadline_jobs": baseline["no_deadline_jobs"],
        "optimistically_impossible": baseline["optimistically_impossible_jobs"],
        "zero_slack": baseline["zero_slack_jobs"],
        "positive_slack": baseline["positive_slack_jobs"],
    }, "synthetic_variants": variants}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics/carbon_v2_sla_headroom_preview.json")
    args = parser.parse_args()
    data = {
        "status": "PROPOSED_DIAGNOSTIC_ONLY_NOT_AN_OPTIMIZER_EXPERIMENT",
        "policy": "deadline_new=max(deadline_original, arrival+(1+fraction)*minimum_eligible_service) for finite deadlines; +inf unchanged",
        "fractions": list(HEADROOM_FRACTIONS),
        "eligibility": "frozen forecast-week regional mean with existing carbon class limits, individual GPU/CPU/RAM fits",
        "interpretation": "synthetic, no-tightening headroom floor, not measured or contractual SLA",
        "scientific_gate": "coauthor approval required before running any optimizer with modified deadlines",
        "scenarios": [preview_scenario(s) for s in SCENARIOS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
    for r in data["scenarios"]:
        b = r["baseline"]
        print(r["scenario"], f"baseline impossible={b['optimistically_impossible']}, zero_slack={b['zero_slack']}")
        for a in r["synthetic_variants"]:
            print(f"   fraction={a['headroom_fraction']:.2f}",
                  f"deadlines_changed={a['modified_finite_deadlines']}/{a['finite_deadlines']}",
                  f"impossible={a['optimistically_impossible_after']}",
                  f"zero_slack={a['zero_slack_after']}")


if __name__ == "__main__":
    main()
