"""Validate locked final-protocol shards and report coverage.

Does not generate, alter, or impute experiment results.
Usage:
    python scripts/audit_final_shards.py
    python scripts/audit_final_shards.py --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCENARIOS = (
    "alibaba_gpu_v2023",
    "microsoft_elasticflow_itp_cluster08",
    "helios_earth",
    "alibaba_spot_gpu_2026_a100",
)
ALGORITHMS = (
    "QPEO", "PSO", "DE", "GWO", "WOA", "ABC", "GA", "HS",
    "FA", "ACO", "FPA", "QIEA", "NSGA-II",
)


def validate_shard(path: Path, primary: dict) -> dict:
    """Validate a single result and return its content-integrity metadata."""
    raw = path.read_bytes()
    d = json.loads(raw)
    if d.get("schema_version") != "qpeo-final-shard-v1":
        raise ValueError("Incorrect result schema version")
    if d.get("final_protocol") is not True:
        raise ValueError("Result is not marked final protocol")
    scenario = d.get("scenario")
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    i = d.get("seed_index")
    seeds = primary["seeds"]
    if not isinstance(i, int) or not 0 <= i < len(seeds):
        raise ValueError("Invalid seed index")
    if d.get("seed") != seeds[i]:
        raise ValueError("Seed does not match locked seed list")
    name = f"{scenario}__seed{i:02d}_{seeds[i]}__final.json"
    if path.name != name:
        raise ValueError("Result filename does not match scenario and seed")
    if d.get("size") != 64:
        raise ValueError("Expected 64 workloads")
    if d.get("population_size") != primary["population_size"]:
        raise ValueError("Population does not match locked protocol")
    T = primary["max_iterations"]
    if d.get("max_iterations") != T:
        raise ValueError("Iterations do not match locked protocol")
    algos = d.get("algorithms")
    if not isinstance(algos, list) or len(algos) != len(ALGORITHMS):
        raise ValueError("Wrong algorithm count")
    if {a.get("algorithm") for a in algos} != set(ALGORITHMS):
        raise ValueError("Wrong algorithm names")
    if len({a["algorithm"] for a in algos}) != len(algos):
        raise ValueError("Duplicate algorithm")
    total_calls = {}
    for a in algos:
        if a.get("seed") != seeds[i]:
            raise ValueError(f"{a['algorithm']}: seed mismatch")
        values = a.get("guide_objectives")
        if not isinstance(values, list) or len(values) != 6:
            raise ValueError(f"{a['algorithm']}: invalid objective vector")
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"{a['algorithm']}: non-finite objective")
        history = a.get("guide_objective_history")
        if not isinstance(history, list) or len(history) != T + 1:
            raise ValueError(f"{a['algorithm']}: invalid objective history length")
        for row in history:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError(f"{a['algorithm']}: invalid objective history shape")
            if not all(math.isfinite(float(v)) for v in row):
                raise ValueError(f"{a['algorithm']}: non-finite objective history")
        elapsed = float(a.get("runtime_seconds", -1))
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"{a['algorithm']}: invalid runtime")
        hits, misses = a.get("simulation_cache_hits"), a.get("simulation_cache_misses")
        if not all(isinstance(x, int) and x >= 0 for x in (hits, misses)):
            raise ValueError(f"{a['algorithm']}: invalid simulation counters")
        total_calls[a["algorithm"]] = hits + misses
    return {
        "scenario": scenario,
        "seed_index": i,
        "seed": seeds[i],
        "filename": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_git_sha": d.get("git_sha"),
        "algorithm_count": len(algos),
        "simulation_evaluation_calls_by_algorithm": total_calls,
    }


def audit(directory: Path, protocol: dict) -> dict:
    primary = protocol["primary"]
    seeds = primary["seeds"]
    if len(seeds) != 30 or len(set(seeds)) != len(seeds):
        raise ValueError("Expected 30 unique locked seeds")
    expected = {(s, i) for s in SCENARIOS for i in range(len(seeds))}
    valid: list[dict] = []
    invalid: list[dict] = []
    seen = set()
    for path in sorted(directory.glob("*__final.json")):
        try:
            rec = validate_shard(path, primary)
            key = (rec["scenario"], rec["seed_index"])
            if key in seen:
                raise ValueError("Duplicate scenario and seed")
            seen.add(key)
            valid.append(rec)
        except Exception as exc:
            invalid.append({"filename": path.name, "error": str(exc)})
    missing = sorted(expected - seen)
    return {
        "protocol_version": protocol.get("protocol_version"),
        "expected_shards": len(expected),
        "valid_shards": len(valid),
        "missing_shards": len(missing),
        "invalid_shards": len(invalid),
        "expected_algorithm_rows": len(expected) * len(ALGORITHMS),
        "validated_algorithm_rows": len(valid) * len(ALGORITHMS),
        "by_scenario": {
            s: {
                "completed_seed_indices": sorted(
                    x["seed_index"] for x in valid if x["scenario"] == s
                ),
                "missing_seed_indices": [i for scenario, i in missing if scenario == s],
            }
            for s in SCENARIOS
        },
        "valid_files": valid,
        "invalid_files": invalid,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results/raw")
    ap.add_argument("--protocol", default="configs/reconstruction_protocol.json")
    ap.add_argument("--output", default="results/final_shard_audit.json")
    ap.add_argument("--strict", action="store_true", help="Fail when incomplete or invalid")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / args.protocol).read_text(encoding="utf-8"))
    summary = audit(root / args.input_dir, protocol)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Validated {summary['valid_shards']}/{summary['expected_shards']} shards, "
        f"{summary['invalid_shards']} invalid, "
        f"{summary['validated_algorithm_rows']} algorithm rows"
    )
    print(out)
    if args.strict and (
        summary["missing_shards"] != 0 or summary["invalid_shards"] != 0
    ):
        raise SystemExit("Final experiment collection is incomplete or invalid")


if __name__ == "__main__":
    main()
