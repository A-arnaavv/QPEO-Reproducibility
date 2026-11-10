"""Strict 30-seed QPEO ablation statistics; never infer missing results.

Primary QPEO/FPA rows are joined by exact scenario and seed index with
separately executed five-operator-ablation shards. All six objectives are
minimization objectives. Classical FPA is a secondary non-QPEO reference,
not an ablation. Raw p-values and both Bonferroni families are reported.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
from scipy import stats

from scripts.analyze_final_results import OBJECTIVES, _mean_ci, _wilcoxon_paired
from scripts.audit_ablation_shards import audit_ablation_directory
from scripts.audit_final_shards import SCENARIOS, validate_shard
from scripts.run_ablation_shard import VARIANTS

REF = "full_qpeo"
SECONDARY = "classical_fpa"
COMPARE = (*VARIANTS, SECONDARY)
ALGORITHMS = (REF, *COMPARE)
METRICS = (*OBJECTIVES, "runtime_seconds", "simulator_calls", "archive_size")
FAMILYWISE_ALPHA = .01
COMPARISONS_PER_SCENARIO = len(COMPARE)*len(OBJECTIVES)  # 36
GLOBAL_COMPARISONS = len(SCENARIOS)*COMPARISONS_PER_SCENARIO  # 144


def _metrics(a: dict) -> dict[str,float]:
    obj=a["guide_objectives"]
    return {
        **{name:float(v) for name,v in zip(OBJECTIVES,obj)},
        "runtime_seconds":float(a["runtime_seconds"]),
        "simulator_calls":float(
            a["simulation_cache_hits"]+a["simulation_cache_misses"]
        ),
        "archive_size":float(a["archive_size"]),
    }


def compute_ablation_stats(
    ablation_shards: list[dict],
    primary_shards: list[dict],
    seed_count: int = 30,
) -> dict[str,list[dict]]:
    """Refuse all missing/duplicate seed pairs before hypothesis testing."""
    expected={(s,i) for s in SCENARIOS for i in range(seed_count)}
    ab_idx={(d["scenario"],d["seed_index"]):d for d in ablation_shards}
    pr_idx={(d["scenario"],d["seed_index"]):d for d in primary_shards}
    if (len(ab_idx)!=len(ablation_shards) or len(pr_idx)!=len(primary_shards)
        or set(ab_idx)!=expected or set(pr_idx)!=expected):
        raise ValueError("Complete paired primary and ablation shards are required")
    descriptives, paired = [], []
    for scenario in SCENARIOS:
        observations={label:[] for label in ALGORITHMS}
        for i in range(seed_count):
            reference=pr_idx[scenario,i]
            ablation=ab_idx[scenario,i]
            if reference.get("seed")!=ablation.get("seed"):
                raise ValueError("Unpaired seed values in the two collections")
            primary={a["algorithm"]:a for a in reference["algorithms"]}
            variant={a["variant"]:a for a in ablation["variants"]}
            if not {"QPEO","FPA"}.issubset(primary) or set(variant)!=set(VARIANTS):
                raise ValueError("Missing QPEO, FPA or ablation variant")
            observations[REF].append(_metrics(primary["QPEO"]))
            observations[SECONDARY].append(_metrics(primary["FPA"]))
            for name in VARIANTS:
                observations[name].append(_metrics(variant[name]))
        for metric in METRICS:
            arrays={name:np.asarray([r[metric] for r in observations[name]],dtype=float)
                    for name in ALGORITHMS}
            if any(len(a)!=seed_count or not np.all(np.isfinite(a)) for a in arrays.values()):
                raise ValueError(f"Invalid {metric} data for {scenario}")
            for name, vals in arrays.items():
                mean, sd, low, high=_mean_ci(vals)
                descriptives.append({
                    "scenario":scenario,"metric":metric,"algorithm":name,
                    "n":seed_count,"mean":mean,"sd":sd,
                    "ci95_low":low,"ci95_high":high,
                    "median":float(np.median(vals))
                })
            if metric not in OBJECTIVES:
                continue
            reference=arrays[REF]
            for name in COMPARE:
                other=arrays[name]
                statistic, p, dz=_wilcoxon_paired(reference,other)
                paired.append({
                    "scenario":scenario,"objective":metric,
                    "reference":REF,"comparator":name,
                    "comparator_type":("classical_reference" if name==SECONDARY else "operator_ablation"),
                    "n":seed_count,
                    "mean_qpeo_minus_comparator":float(np.mean(reference-other)),
                    "qpeo_better_seed_count":int(np.sum(reference<other)),
                    "comparator_better_seed_count":int(np.sum(other<reference)),
                    "ties":int(np.sum(reference==other)),
                    "wilcoxon_statistic":statistic,
                    "p_raw":p,
                    "p_bonferroni_36":min(1.0,p*COMPARISONS_PER_SCENARIO),
                    "p_bonferroni_144":min(1.0,p*GLOBAL_COMPARISONS),
                    "cohens_dz":dz,
                })
    return {"descriptive_statistics":descriptives,"paired_wilcoxon":paired}


def _write_csv(path:Path,rows:list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--ablation-dir",default="results/ablations/raw")
    ap.add_argument("--primary-dir",default="results/raw")
    ap.add_argument("--output-dir",default="results/ablations/analysis")
    args=ap.parse_args()
    protocol=json.loads((ROOT/"configs/reconstruction_protocol.json").read_text())
    primary=protocol["primary"]
    ab_dir=ROOT/args.ablation_dir
    pr_dir=ROOT/args.primary_dir
    audit=audit_ablation_directory(ab_dir,pr_dir,primary)
    if audit["paired_valid_shards"]!=120 or audit["invalid_or_unpaired"]:
        raise SystemExit(
            f"REFUSING INCOMPLETE ABLATION ANALYSIS: "
            f"{audit['paired_valid_shards']}/120 paired shards, "
            f"{len(audit['invalid_or_unpaired'])} invalid or unpaired"
        )
    ablations=[]; references=[]
    for scenario in SCENARIOS:
        for i,seed in enumerate(primary["seeds"]):
            a=ab_dir/f"{scenario}__seed{i:02d}_{seed}__ablation__final.json"
            p=pr_dir/f"{scenario}__seed{i:02d}_{seed}__final.json"
            validate_shard(p,primary)
            ablations.append(json.loads(a.read_text()))
            references.append(json.loads(p.read_text()))
    stats_out=compute_ablation_stats(ablations,references,len(primary["seeds"]))
    out=ROOT/args.output_dir
    for name,rows in stats_out.items():
        _write_csv(out/f"{name}.csv",rows)
    provenance={
        "pair_count":len(ablations),"scenario_count":len(SCENARIOS),
        "paired_seeds_per_scenario":len(primary["seeds"]),
        "variants":list(VARIANTS),"reference":REF,
        "classical_reference":SECONDARY,
        "familywise_alpha":FAMILYWISE_ALPHA,
        "wilcoxon_tests_per_scenario":COMPARISONS_PER_SCENARIO,
        "wilcoxon_tests_total":GLOBAL_COMPARISONS,
        "correction_policy":"raw, per-scenario Bonferroni-36, and global Bonferroni-144",
        "objective_direction":"all six minimized",
        "notes":[
            "no_caqc_penalty preserves hard carbon-class feasibility and Omega carbon affinity",
            "classical_fpa is a comparator, not a QPEO component ablation",
            "dynamic carbon means these histories are not stationary-objective convergence traces",
            "full Pareto-front objective vectors are unavailable from primary shards, so hypervolume is not reported",
        ],
        "ablation_sha256":{r["filename"]:r["sha256"] for r in audit["validated_shards"]},
        "primary_sha256":{
            p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in pr_dir.glob("*__final.json")
        }
    }
    (out/"methodology_audit.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(
        "ABLATION STATISTICS COMPLETE:",
        len(ablations),"paired shards,",
        len(stats_out["paired_wilcoxon"]),"paired Wilcoxon rows"
    )


if __name__=="__main__":
    main()
