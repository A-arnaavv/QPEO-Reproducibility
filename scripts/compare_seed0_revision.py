"""Compare original seed-0 files to full reruns at the canonical batch source.

The comparisons are purely numerical and do not modify raw experimental data.
Runtime differences are excluded because runner contention affects wall time.
The acceptance tolerance is fixed *before* inspecting recheck results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.audit_final_shards import ALGORITHMS, validate_shard
from qpeo.protocol import load_protocol


CANONICAL_SHA = "7d1efcf0980620e66db01b4d7c32d80081038471"
SCENARIOS = (
    "alibaba_gpu_v2023",
    "microsoft_elasticflow_itp_cluster08",
    "alibaba_spot_gpu_2026_a100",
)
RTOL = 1e-9
ATOL = 1e-10


def compare_one(original: dict, rerun: dict) -> dict:
    if original["scenario"] != rerun["scenario"]:
        raise ValueError("Mismatched scenario")
    if original["seed_index"] != 0 or rerun["seed_index"] != 0:
        raise ValueError("Comparison requires seed index zero")
    if original["seed"] != rerun["seed"]:
        raise ValueError("Mismatched seed")
    if rerun.get("git_sha") != CANONICAL_SHA:
        raise ValueError("Rerun source is not the canonical batch commit")
    for key in (
        "dataset_id",
        "size",
        "population_size",
        "max_iterations",
        "resource_environment_version",
        "carbon_source",
        "carbon_rows",
    ):
        if original.get(key) != rerun.get(key):
            raise ValueError(f"Changed environment or source field: {key}")
    old = {a["algorithm"]: a for a in original["algorithms"]}
    new = {a["algorithm"]: a for a in rerun["algorithms"]}
    if set(old) != set(ALGORITHMS) or set(new) != set(ALGORITHMS):
        raise ValueError("Missing or unexpected algorithm")
    comparisons = []
    for name in ALGORITHMS:
        a, b = old[name], new[name]
        o = np.asarray(a["guide_objectives"], dtype=float)
        n = np.asarray(b["guide_objectives"], dtype=float)
        oh = np.asarray(a["guide_objective_history"], dtype=float)
        nh = np.asarray(b["guide_objective_history"], dtype=float)
        if o.shape != (6,) or n.shape != (6,) or oh.shape != (501, 6) or nh.shape != (501, 6):
            raise ValueError(f"{name}: invalid objectives or objective history")
        if not all(np.all(np.isfinite(x)) for x in (o,n,oh,nh)):
            raise ValueError(f"{name}: non-finite objective values")
        same_final = bool(np.allclose(o,n,rtol=RTOL,atol=ATOL))
        same_history = bool(np.allclose(oh,nh,rtol=RTOL,atol=ATOL))
        old_counts = (
            a["simulation_cache_hits"] + a["simulation_cache_misses"]
        )
        new_counts = (
            b["simulation_cache_hits"] + b["simulation_cache_misses"]
        )
        counts_equal = old_counts == new_counts
        archive_equal = a["archive_size"] == b["archive_size"]
        comparisons.append({
            "algorithm": name,
            "final_objectives_equivalent": same_final,
            "all_501_history_rows_equivalent": same_history,
            "max_final_objective_abs_delta": float(np.max(np.abs(o - n))),
            "max_history_abs_delta": float(np.max(np.abs(oh - nh))),
            "simulator_calls_equal": bool(counts_equal),
            "archive_size_equal": bool(archive_equal),
        })
    accepted = all(
        c["final_objectives_equivalent"]
        and c["all_501_history_rows_equivalent"]
        and c["simulator_calls_equal"]
        and c["archive_size_equal"]
        for c in comparisons
    )
    return {
        "scenario": original["scenario"],
        "seed": original["seed"],
        "original_git_sha": original.get("git_sha"),
        "canonical_git_sha": CANONICAL_SHA,
        "rtol": RTOL,
        "atol": ATOL,
        "equivalent": accepted,
        "algorithms": comparisons,
    }


def compare_files(old_dir: Path, recheck_dir: Path, protocol: dict) -> dict:
    original_dir = Path(old_dir)
    canonical_dir = Path(recheck_dir)
    details = []
    for scenario in SCENARIOS:
        name = f"{scenario}__seed00_104729__final.json"
        orig = original_dir / name
        rerun = canonical_dir / name
        if not orig.exists() or not rerun.exists():
            raise FileNotFoundError(f"Missing seed0 parity file: {name}")
        validate_shard(orig, protocol)
        validate_shard(rerun, protocol)
        details.append(compare_one(json.loads(orig.read_text()), json.loads(rerun.read_text())))
    return {
        "comparison_rule": "same objectives and all 501 histories within predeclared rtol and atol; exact simulator-call and archive-size counts",
        "all_three_equivalent": all(x["equivalent"] for x in details),
        "count": len(details),
        "comparisons": details,
        "policy": "When not equivalent, preserve both versions and use canonical rerun for that seed in final analysis. No values are interpolated or guessed.",
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--original-dir", default="results/raw")
    p.add_argument("--canonical-dir", required=True)
    p.add_argument("--output", default="results/reproducibility/seed00_parity.json")
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    protocol=load_protocol(root/"configs/reconstruction_protocol.json")[0]
    raw=json.loads((root/"configs/reconstruction_protocol.json").read_text())["primary"]
    report=compare_files(root/args.original_dir,Path(args.canonical_dir),raw)
    dest=root/args.output
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("SEED0_CANONICAL_PARITY",report["all_three_equivalent"],"saved to",dest)


if __name__ == "__main__":
    main()
