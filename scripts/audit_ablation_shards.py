"""Validate saved QPEO operator-ablation shards without synthesizing results.

This module intentionally does not run optimizers. It checks that the five
single-component ablations were executed on the same locked scenario, seed,
environment and iteration budget as each corresponding primary QPEO shard.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scripts.audit_final_shards import SCENARIOS, validate_shard
from scripts.run_ablation_shard import VARIANTS


def validate_ablation_shard(path: Path, primary: dict) -> dict:
    raw = path.read_bytes()
    d = json.loads(raw)
    if d.get("schema_version") != "qpeo-ablation-shard-v1":
        raise ValueError("Invalid ablation schema version")
    if d.get("final_protocol") is not True:
        raise ValueError("Ablation shard is not a full-protocol run")
    scenario = d.get("scenario")
    seed_index = d.get("seed_index")
    if scenario not in SCENARIOS or type(seed_index) is not int:
        raise ValueError("Invalid ablation scenario or seed index")
    if not 0 <= seed_index < len(primary["seeds"]):
        raise ValueError("Ablation seed index out of bounds")
    seed = primary["seeds"][seed_index]
    if d.get("seed") != seed:
        raise ValueError("Ablation seed does not match locked list")
    expected_name = (
        f"{scenario}__seed{seed_index:02d}_{seed}__ablation__final.json"
    )
    if path.name != expected_name:
        raise ValueError("Ablation filename/seed mismatch")
    if d.get("size") != 64:
        raise ValueError("Ablation size must be 64")
    if d.get("population_size") != primary["population_size"]:
        raise ValueError("Ablation population mismatch")
    T = primary["max_iterations"]
    if d.get("max_iterations") != T:
        raise ValueError("Ablation iteration budget mismatch")
    if not d.get("resource_environment_version") or not d.get("carbon_source"):
        raise ValueError("Ablation environment provenance missing")
    ablations = d.get("variants")
    if not isinstance(ablations, list) or len(ablations) != len(VARIANTS):
        raise ValueError("Expected five ablation variants")
    if {a.get("variant") for a in ablations} != set(VARIANTS):
        raise ValueError("Ablation operator switches do not match protocol")
    if len({a["variant"] for a in ablations}) != len(VARIANTS):
        raise ValueError("Duplicate ablation variant")
    for a in ablations:
        label = a["variant"]
        if a.get("seed") != seed:
            raise ValueError(f"{label}: seed mismatch")
        objectives = a.get("guide_objectives")
        if not isinstance(objectives, list) or len(objectives) != 6:
            raise ValueError(f"{label}: objective vector shape mismatch")
        if not all(math.isfinite(float(v)) for v in objectives):
            raise ValueError(f"{label}: non-finite objective vector")
        history = a.get("guide_objective_history")
        if not isinstance(history, list) or len(history) != T+1:
            raise ValueError(f"{label}: objective history shape mismatch")
        for row in history:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError(f"{label}: bad objective history row")
            if not all(math.isfinite(float(v)) for v in row):
                raise ValueError(f"{label}: non-finite objective history")
        entropy = a.get("entropy_history")
        if not isinstance(entropy, list):
            raise ValueError(f"{label}: missing entropy history")
        entropy_n = 0 if label == "no_entropy_bro" else T
        if len(entropy) != entropy_n:
            raise ValueError(f"{label}: entropy-history length mismatch")
        if not all(math.isfinite(float(v)) for v in entropy):
            raise ValueError(f"{label}: non-finite entropy")
        for k in ("simulation_cache_hits", "simulation_cache_misses"):
            if type(a.get(k)) is not int or a[k] < 0:
                raise ValueError(f"{label}: bad {k}")
        if not math.isfinite(float(a.get("runtime_seconds", float("nan")))):
            raise ValueError(f"{label}: non-finite runtime")
        if float(a["runtime_seconds"]) < 0:
            raise ValueError(f"{label}: negative runtime")
    return {
        "scenario": scenario, "seed_index": seed_index, "seed": seed,
        "filename": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        "variants": list(VARIANTS),
    }


def audit_ablation_directory(
    ablation_dir: Path, primary_dir: Path, primary: dict
) -> dict:
    ablation_dir = Path(ablation_dir)
    primary_dir = Path(primary_dir)
    valid, invalid = [], []
    seen = set()
    for p in sorted(ablation_dir.glob("*__ablation__final.json")):
        try:
            info = validate_ablation_shard(p, primary)
            key = (info["scenario"], info["seed_index"])
            if key in seen:
                raise ValueError("Duplicate scenario-seed ablation shard")
            seen.add(key)
            matching_primary = primary_dir / (
                f"{info['scenario']}__seed{info['seed_index']:02d}_"
                f"{info['seed']}__final.json"
            )
            if not matching_primary.exists():
                raise ValueError("Matching primary QPEO result is not available")
            source = validate_shard(matching_primary, primary)
            if source["scenario"] != info["scenario"] or source["seed"] != info["seed"]:
                raise ValueError("Primary result does not match ablation")
            valid.append(info)
        except Exception as exc:
            invalid.append({"filename": p.name, "error": str(exc)})
    expected = {(scenario, i) for scenario in SCENARIOS for i in range(30)}
    return {
        "expected_shards": len(expected),
        "paired_valid_shards": len(valid),
        "invalid_or_unpaired": invalid,
        "missing_pairs": [
            {"scenario": s, "seed_index": i}
            for s, i in sorted(expected-seen)
        ],
        "validated_shards": valid,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results/ablations/raw")
    ap.add_argument("--primary-dir", default="results/raw")
    ap.add_argument("--output", default="results/ablations/audit.json")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"configs/reconstruction_protocol.json").read_text())
    report = audit_ablation_directory(
        root/args.input_dir, root/args.primary_dir, config["primary"]
    )
    target = root/args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(
        f"ABLATION PAIRS: {report['paired_valid_shards']}/"
        f"{report['expected_shards']} validated, "
        f"{len(report['invalid_or_unpaired'])} invalid"
    )
    if args.strict and (
        report["paired_valid_shards"] != report["expected_shards"]
        or report["invalid_or_unpaired"]
    ):
        raise SystemExit("Incomplete or invalid ablation collection")


if __name__ == "__main__":
    main()
