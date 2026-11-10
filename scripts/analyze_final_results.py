"""Generate final QPEO manuscript statistics from validated, complete raw shards.

This script NEVER fabricates or imputes results. It refuses incomplete collections.

Outputs:
  results/analysis/descriptive_statistics.csv
  results/analysis/paired_wilcoxon.csv
  results/analysis/friedman.csv
  results/analysis/nemenyi.csv
  results/analysis/methodology_audit.json

The primary family-wise comparison is 12 baselines x 6 objectives = 72 paired
Wilcoxon tests *per scenario*. A conservative 288-test threshold across all
four scenarios is also reported; neither correction is hidden.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

from scripts.audit_final_shards import ALGORITHMS, SCENARIOS, audit, validate_shard

OBJECTIVES = (
    "f1_negative_gpu_utilization",
    "f2_energy_kwh",
    "f3_energy_weighted_carbon_intensity",
    "f4_makespan",
    "f5_sla_violation_percent",
    "f6_normalized_cost",
)
METRICS = (*OBJECTIVES, "runtime_seconds", "simulator_calls", "archive_size")
FAMILYWISE_ALPHA = 0.01
SCENARIO_COMPARISONS = 6 * 12
GLOBAL_COMPARISONS = 4 * SCENARIO_COMPARISONS


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean_ci(data: np.ndarray) -> tuple[float, float, float, float]:
    n = len(data)
    mean = float(np.mean(data))
    sd = float(np.std(data, ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return mean, sd, mean - half, mean + half


def _wilcoxon_paired(ref: np.ndarray, other: np.ndarray) -> tuple[float, float, float]:
    diff = ref - other
    if np.allclose(diff, 0.0, atol=1e-15, rtol=0):
        return 0.0, 1.0, 0.0
    w = stats.wilcoxon(ref, other, zero_method="wilcox", alternative="two-sided", method="auto")
    p = float(w.pvalue)
    if not math.isfinite(p):
        raise ValueError("Non-finite Wilcoxon p-value")
    s = float(np.std(diff, ddof=1))
    dz = float(np.mean(diff) / s) if s > 1e-15 else math.copysign(float("inf"),float(np.mean(diff)))
    return float(w.statistic), p, dz


def compute_statistics(shards: list[dict], seed_count: int) -> dict[str, list[dict]]:
    """Calculate paired statistics; all scenario×seed entries must exist."""
    if len(shards) != len(SCENARIOS) * seed_count:
        raise ValueError("Incomplete shard count; refusing significance analysis")
    index = {(d["scenario"], d["seed_index"]): d for d in shards}
    if len(index) != len(shards):
        raise ValueError("Duplicate scenario/seed pair")
    if any((s,i) not in index for s in SCENARIOS for i in range(seed_count)):
        raise ValueError("Missing paired seed")

    desc, pairwise, friedman_rows, nemenyi_rows = [], [], [], []
    for scenario in SCENARIOS:
        metrics = {name: {} for name in ALGORITHMS}
        for seed_index in range(seed_count):
            d = index[(scenario, seed_index)]
            for a in d["algorithms"]:
                name=a["algorithm"]
                v=a["guide_objectives"]
                metrics[name][seed_index] = {
                    **{objective: float(v[k]) for k,objective in enumerate(OBJECTIVES)},
                    "runtime_seconds": float(a["runtime_seconds"]),
                    "simulator_calls": int(a["simulation_cache_hits"]) + int(a["simulation_cache_misses"]),
                    "archive_size": int(a["archive_size"]),
                }
        for name in ALGORITHMS:
            if set(metrics[name]) != set(range(seed_count)):
                raise ValueError(f"{scenario}: incomplete paired values for {name}")

        for metric in METRICS:
            arrays = {
                name: np.asarray([metrics[name][i][metric] for i in range(seed_count)],float)
                for name in ALGORITHMS
            }
            if not all(np.all(np.isfinite(values)) for values in arrays.values()):
                raise ValueError(f"{scenario}: non-finite metric {metric}")
            for name, data in arrays.items():
                mean, sd, low, high = _mean_ci(data)
                desc.append({
                    "scenario": scenario, "metric": metric, "algorithm": name,
                    "n": seed_count, "mean": mean, "sd": sd, "ci95_low": low,
                    "ci95_high": high, "median": float(np.median(data)),
                })
            # Inferential tests are limited to the six primary objective metrics.
            # Runtime and NFE are descriptive until a testing family is declared.
            if metric not in OBJECTIVES:
                continue

            ref = arrays["QPEO"]
            for name in ALGORITHMS:
                if name == "QPEO":
                    continue
                other=arrays[name]
                stat,p,dz=_wilcoxon_paired(ref,other)
                pairwise.append({
                    "scenario":scenario,"objective":metric,"baseline":name,
                    "n":seed_count, "mean_qpeo_minus_baseline":float(np.mean(ref-other)),
                    "qpeo_better_seed_count":int(np.sum(ref<other)),
                    "baseline_better_seed_count":int(np.sum(other<ref)),
                    "ties":int(np.sum(ref==other)), "wilcoxon_statistic":stat,
                    "p_raw":p,
                    "p_bonferroni_72":min(1.0,p*SCENARIO_COMPARISONS),
                    "p_bonferroni_288":min(1.0,p*GLOBAL_COMPARISONS),
                    "cohens_dz":dz,
                })

            matrix = np.column_stack([arrays[name] for name in ALGORITHMS])
            ranks=np.stack([stats.rankdata(row, method="average") for row in matrix])
            average_ranks=np.mean(ranks,axis=0)
            # scipy returns nan when all algorithm columns are identical.
            if np.all(np.ptp(matrix, axis=1) == 0):
                f_stat, f_p = 0.0, 1.0
            else:
                res = stats.friedmanchisquare(*(arrays[name] for name in ALGORITHMS))
                f_stat, f_p = float(res.statistic), float(res.pvalue)
                if not math.isfinite(f_p):
                    raise ValueError(f"Friedman returned non-finite p for {scenario}:{metric}")
            for k,name in enumerate(ALGORITHMS):
                friedman_rows.append({
                    "scenario":scenario,"objective":metric,"algorithm":name,
                    "n":seed_count,"friedman_statistic":f_stat,
                    "friedman_p":f_p, "average_rank":float(average_ranks[k])
                })
            se=math.sqrt(len(ALGORITHMS)*(len(ALGORITHMS)+1)/(6*seed_count))
            for i,a in enumerate(ALGORITHMS):
                for j in range(i+1,len(ALGORITHMS)):
                    b=ALGORITHMS[j]
                    diff=abs(float(average_ranks[i]-average_ranks[j]))
                    q=diff/se
                    p=float(stats.studentized_range.sf(q*math.sqrt(2),len(ALGORITHMS),np.inf))
                    nemenyi_rows.append({
                        "scenario":scenario,"objective":metric,
                        "algorithm_a":a,"algorithm_b":b,
                        "rank_difference":diff,"q_statistic":q,"p_value":p,
                    })
    return {
        "descriptives":desc, "paired_wilcoxon":pairwise,
        "friedman":friedman_rows, "nemenyi":nemenyi_rows,
    }


def main() -> None:
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",default="results/raw")
    ap.add_argument("--output-dir",default="results/analysis")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    protocol=json.loads((root/"configs/reconstruction_protocol.json").read_text())
    indir=root/args.input_dir
    audit_report=audit(indir,protocol)
    if audit_report["missing_shards"] or audit_report["invalid_shards"]:
        raise SystemExit(
            f"REFUSING INCOMPLETE ANALYSIS: {audit_report['valid_shards']}/"
            f"{audit_report['expected_shards']} valid; "
            f"{audit_report['invalid_shards']} invalid."
        )
    primary=protocol["primary"]
    raw=[]
    for path in sorted(indir.glob("*__final.json")):
        validate_shard(path,primary)
        raw.append(json.loads(path.read_text()))
    results=compute_statistics(raw,len(primary["seeds"]))
    out=root/args.output_dir
    for key,rows in results.items():
        _save_csv(out/(("descriptive_statistics" if key=="descriptives" else key)+".csv"),
                  rows,list(rows[0]) if rows else [])
    checksums={x["filename"]:x["sha256"] for x in audit_report["valid_files"]}
    report={
        "source_files":len(raw),
        "scenario_count":len(SCENARIOS),
        "seeds_per_scenario":len(primary["seeds"]),
        "algorithms":len(ALGORITHMS),
        "objective_metrics":list(OBJECTIVES),
        "paired_tests_per_scenario":SCENARIO_COMPARISONS,
        "paired_tests_across_scenarios":GLOBAL_COMPARISONS,
        "familywise_alpha":FAMILYWISE_ALPHA,
        "bonferroni_raw_alpha_per_scenario":FAMILYWISE_ALPHA/SCENARIO_COMPARISONS,
        "bonferroni_raw_alpha_across_scenarios":FAMILYWISE_ALPHA/GLOBAL_COMPARISONS,
        "sampling_protocol":"paired 30-seed comparisons; minimization objectives",
        "evaluation_budget_caveat":"Population and iterations matched, simulator calls can differ. See descriptives.",
        "convergence_status":"Raw objective histories are saved for all algorithms, but f3 and carbon-dependent feasibility are evaluated against a changing NESO carbon replay. A conventional stationary time-to-90-percent convergence claim is not reported from these histories without a fixed-carbon control or a predeclared time-dependent comparator.",
        "hypervolume_status":"Not computed: shard files contain archive sizes but not full per-run Pareto-front objective sets. Hypervolume cannot be reconstructed legitimately from archive sizes.",
        "validated_source_sha256":checksums,
    }
    (out/"methodology_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print("FINAL STATISTICS COMPLETE:",len(raw),"validated shards")
    for k,rows in results.items():print(k,len(rows))


if __name__=="__main__":
    main()
