"""Read-only collection audit and gated analysis for Carbon-v2 candidate full cohort.

This is NOT the v1 analyzer. It never treats Carbon-v2 test-tagged outputs as v1.
The six primary objectives are analyzed only after 4x30 valid, paired shards.
Forecast-based carbon mass is reported as a *secondary descriptive metric*.

Usage:
    python scripts/audit_carbon_v2_campaign.py --input-dir results/carbon_v2_campaign/<gitsha12>
    python scripts/audit_carbon_v2_campaign.py --input-dir ... --analyze

No optimizer runs, no cloud jobs, and no imputation of missing results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

from scripts.analyze_final_results import compute_statistics
from scripts.audit_final_shards import ALGORITHMS, SCENARIOS
from scripts.validate_carbon_v2_pilot import validate_pilot

ROOT = Path(__file__).resolve().parents[1]
TAG = "test-P50-T500-N64-carbon-time-v2-repeat-fixedmean"
SCHEMA = "qpeo-carbon-v2-test-shard-v1"
METRIC_NAMES = (
    "f1_negative_gpu_utilization", "f2_energy_kwh",
    "f3_energy_weighted_carbon_intensity", "f4_makespan",
    "f5_sla_violation_percent", "f6_normalized_cost",
)


def _read_protocol():
    p = json.loads((ROOT / "configs/reconstruction_protocol.json").read_text())
    primary = p["primary"]
    if (len(primary["seeds"]) != 30 or len(set(primary["seeds"])) != 30
            or primary["population_size"] != 50 or primary["max_iterations"] != 500):
        raise ValueError("Unexpected locked v1 primary seed/budget specification")
    return primary


def expected_filename(scenario: str, index: int, seed: int) -> str:
    return f"{scenario}__seed{index:02d}_{seed}__{TAG}.json"


def validate_campaign_shard(path: Path, scenario: str, index: int, seed: int,
                            *, deep: bool = True) -> dict:
    """Check metadata, source consistency and optionally replay every guide schedule."""
    raw = path.read_bytes()
    d = json.loads(raw)
    if path.name != expected_filename(scenario, index, seed):
        raise ValueError("Incorrect scenario/seed/budget/carbon protocol in shard filename")
    if d.get("schema_version") != SCHEMA or d.get("final_protocol") is not False:
        raise ValueError("Wrong schema or v1 final-protocol marker")
    if (d.get("scenario"), d.get("seed_index"), d.get("seed")) != (scenario, index, seed):
        raise ValueError("Scenario or locked seed mismatch")
    if (d.get("size"), d.get("population_size"), d.get("max_iterations")) != (64, 50, 500):
        raise ValueError("Wrong problem/budget dimensions")
    v = d.get("carbon_time_accounting")
    if not isinstance(v, dict) or (v.get("version"), v.get("coverage_policy")) != ("v2", "repeat"):
        raise ValueError("Wrong Carbon-v2 coverage policy")
    if v.get("planning_signal") != "fixed full-week regional mean intensity":
        raise ValueError("Carbon-v2 planning signal mismatch")
    if not isinstance(d.get("algorithms"), list) or len(d["algorithms"]) != 13:
        raise ValueError("Expected all 13 algorithms")
    if set(a.get("algorithm") for a in d["algorithms"]) != set(ALGORITHMS):
        raise ValueError("Missing/duplicated/unknown algorithm")
    if len({a["algorithm"] for a in d["algorithms"]}) != 13:
        raise ValueError("Duplicate algorithm")
    for a in d["algorithms"]:
        if a.get("seed") != seed:
            raise ValueError("Seed differs between methods")
        obj = a.get("guide_objectives")
        if not isinstance(obj, list) or len(obj) != 6 or not all(math.isfinite(float(x)) for x in obj):
            raise ValueError(f"{a['algorithm']}: invalid objective vector")
        if float(obj[1]) <= 0 or float(obj[2]) < 0:
            raise ValueError(f"{a['algorithm']}: nonpositive energy or negative carbon intensity")
        hist = a.get("guide_objective_history")
        if not isinstance(hist, list) or len(hist) != 501:
            raise ValueError(f"{a['algorithm']}: incomplete 501-point objective history")
        if not all(isinstance(row, list) and len(row) == 6 and
                   all(math.isfinite(float(v)) for v in row) for row in hist):
            raise ValueError(f"{a['algorithm']}: invalid history values")
        mass = a.get("guide_carbon_g")
        if mass is None or not math.isfinite(float(mass)) or float(mass) < 0:
            raise ValueError(f"{a['algorithm']}: invalid modeled carbon mass")
        if not math.isclose(float(a.get("guide_carbon_intensity_g_per_kwh", -1)), float(obj[2]),
                            rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"{a['algorithm']}: summary intensity differs from f3")
        if not math.isclose(float(mass), float(obj[1]) * float(obj[2]),
                            rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f"{a['algorithm']}: carbon mass/energy/intensity identity failed")
    prov = d.get("execution_provenance")
    if not isinstance(prov, dict) or prov.get("source_git_sha") != d.get("git_sha"):
        raise ValueError("Missing or contradictory source provenance")
    sha = d.get("git_sha")
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise ValueError("Invalid full source Git SHA")
    if prov.get("source_has_tracked_changes") is not False:
        raise ValueError("Run used a dirty tracked checkout")
    # A successful read-only deep replay is not proof the original run used
    # the predeclared software environment. Check its recorded provenance.
    req = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            req[name.strip()] = version.strip()
    python_ver = prov.get("python_version")
    if not isinstance(python_ver, str) or not python_ver.startswith("3.13."):
        raise ValueError("Full campaign was not executed under Python 3.13.x")
    for pkg in ("numpy", "scipy", "pytest"):
        if prov.get(f"{pkg}_version") != req[pkg]:
            raise ValueError(
                f"Full campaign used an unpinned {pkg} version: "
                f"{prov.get(f'{pkg}_version')!r} instead of {req[pkg]!r}"
            )
    if (path.parent.name != sha[:12]):
        raise ValueError("Shard must live in a source-revision-named directory")
    if deep:
        # Independent simulator replay, CPU/GPU/memory constraints, complete energy
        # and time-integrated carbon recalculation, and source/config hash verification.
        result = validate_pilot(path)
        if result["algorithms_verified"] != 13:
            raise ValueError("Deep replay did not verify all 13 algorithms")
    return {"scenario": scenario, "seed_index": index, "seed": seed,
            "file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
            "git_sha": sha, "modeled_carbon_mass_checked": True,
            "deep_simulator_validation": bool(deep)}


def audit_campaign(input_dir: Path, *, deep: bool = True) -> dict:
    primary = _read_protocol()
    seeds = primary["seeds"]
    expected = {(scenario, idx): expected_filename(scenario, idx, seed)
                for scenario in SCENARIOS for idx, seed in enumerate(seeds)}
    valid, missing, invalid = [], [], []
    expected_names = set(expected.values())
    for (scenario, idx), filename in expected.items():
        path = input_dir / filename
        if not path.is_file():
            missing.append({"scenario": scenario, "seed_index": idx, "expected_file": filename})
            continue
        try:
            valid.append(validate_campaign_shard(path, scenario, idx, seeds[idx], deep=deep))
        except Exception as exc:
            invalid.append({"filename": filename, "reason": str(exc)})
    unexpected = sorted(p.name for p in input_dir.glob("*__test-*.json")
                        if p.name not in expected_names)
    for filename in unexpected:
        invalid.append({"filename": filename, "reason": "Unexpected run protocol in campaign directory"})
    source_shas = sorted(set(x["git_sha"] for x in valid))
    if len(source_shas) > 1:
        invalid.append({"reason": "Mixed source revisions in the same campaign directory",
                        "git_shas": source_shas})
    return {"expected_shards": len(expected), "expected_algorithm_rows": len(expected) * 13,
            "valid_shards": len(valid), "missing_shards": len(missing), "invalid_shards": len(invalid),
            "validated_algorithm_rows": len(valid) * 13, "source_git_shas": source_shas,
            "complete": len(valid) == len(expected) and not missing and not invalid,
            "deep_validation": deep, "valid_files": valid, "missing_files": missing,
            "invalid_files": invalid}


def _write_csv(path: Path, data: list[dict]) -> None:
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def analyze_complete_campaign(input_dir: Path, output_dir: Path, audit: dict) -> dict:
    if not audit.get("complete") or not audit.get("deep_validation"):
        raise ValueError("REFUSING analysis: all 120 shards must pass DEEP validation")
    primary = _read_protocol()
    seeds = primary["seeds"]
    shards = []
    mass_rows = []
    for scenario in SCENARIOS:
        for i, seed in enumerate(seeds):
            shard = json.loads((input_dir / expected_filename(scenario, i, seed)).read_text())
            shards.append(shard)
            for a in shard["algorithms"]:
                mass_rows.append({"scenario": scenario, "seed_index": i, "seed": seed,
                                  "algorithm": a["algorithm"], "modeled_carbon_g": a["guide_carbon_g"]})
    result = compute_statistics(shards, len(seeds))
    for key, rows in result.items():
        _write_csv(output_dir / f"{('descriptive_statistics' if key == 'descriptives' else key)}.csv", rows)
    _write_csv(output_dir / "modeled_carbon_mass_by_run.csv", mass_rows)
    desc_mass = []
    for scenario in SCENARIOS:
        for name in ALGORITHMS:
            vals = np.asarray([float(x["modeled_carbon_g"]) for x in mass_rows
                               if x["scenario"] == scenario and x["algorithm"] == name])
            if len(vals) != 30:
                raise ValueError("Incomplete secondary modeled-carbon-mass samples")
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            hw = float(stats.t.ppf(.975, len(vals)-1) * sd / math.sqrt(len(vals)))
            desc_mass.append({"scenario": scenario, "algorithm": name,
                              "n": len(vals), "mean_gco2e": mean, "sd_gco2e": sd,
                              "ci95_low_gco2e": mean-hw, "ci95_high_gco2e": mean+hw,
                              "median_gco2e": float(np.median(vals))})
    _write_csv(output_dir / "modeled_carbon_mass_descriptive.csv", desc_mass)
    summary = {"source_git_sha": audit["source_git_shas"][0],
               "source_shards": len(shards), "objective_metrics": list(METRIC_NAMES),
               "paired_wilcoxon_count": len(result["paired_wilcoxon"]),
               "friedman_rows": len(result["friedman"]),
               "secondary_metric": "modeled operational GPU carbon mass gCO2e, descriptive only",
               "carbon_scope": "simulated time-aligned NESO forecast with explicit periodic week; NOT observed emissions",
               "no_imputation": True,
               "limitations": ["No measured physical emissions", "One UK forecast week repeated counterfactually",
                               "Host and off-workload idle power not included",
                               "Offline full-forecast knowledge assumption"],
               "audit_sha256": hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()}
    (output_dir / "analysis_manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--analyze", action="store_true", help="Requires complete deep-validated cohort")
    ap.add_argument("--fast-audit", action="store_true", help="Skip schedule replay; cannot analyze")
    args = ap.parse_args()
    out = args.output_dir or args.input_dir.parent / "analysis" / args.input_dir.name
    report = audit_campaign(args.input_dir, deep=not args.fast_audit)
    out.mkdir(parents=True, exist_ok=True)
    (out / "campaign_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"AUDIT: {report['valid_shards']}/{report['expected_shards']} valid; "
          f"{report['missing_shards']} missing; {report['invalid_shards']} invalid")
    print(f"Report: {out / 'campaign_audit.json'}")
    if args.analyze:
        summary = analyze_complete_campaign(args.input_dir, out, report)
        print(f"Complete study analyzed: {summary['source_shards']} shards; "
              f"{summary['paired_wilcoxon_count']} paired tests")
    if report["invalid_shards"]:
        raise SystemExit("Campaign contains invalid shards; audit failed")
    if args.analyze and not report["complete"]:
        raise SystemExit("Campaign is incomplete; no inferential outputs created")


if __name__ == "__main__":
    main()
