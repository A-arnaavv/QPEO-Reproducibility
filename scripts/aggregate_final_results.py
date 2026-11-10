from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ALGORITHMS=("QPEO","PSO","DE","GWO","WOA","ABC","GA","HS","FA","ACO","FPA","QIEA","NSGA-II")
SCENARIOS=("alibaba_gpu_v2023","microsoft_elasticflow_itp_cluster08","helios_earth","alibaba_spot_gpu_2026_a100")
OBJECTIVES=("f1_gpu_util_min","f2_energy_kwh","f3_carbon_intensity","f4_makespan","f5_sla_violation_pct","f6_normalized_cost")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",default="results/raw")
    ap.add_argument("--output",default="results/final_summary_long.csv")
    ap.add_argument("--require-complete",action="store_true")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    indir=root/args.input_dir
    files=sorted(indir.glob("*__final.json"))
    rows=[]; seen=set()
    for p in files:
        data=json.loads(p.read_text(encoding="utf-8"))
        if not data.get("final_protocol"):
            continue
        scenario=data["scenario"]; seed_index=int(data["seed_index"]); seed=int(data["seed"])
        algs=data["algorithms"]
        names=[a["algorithm"] for a in algs]
        if set(names)!=set(ALGORITHMS) or len(names)!=len(ALGORITHMS):
            raise RuntimeError(f"{p}: algorithm set mismatch")
        for a in algs:
            key=(scenario,seed_index,a["algorithm"])
            if key in seen: raise RuntimeError(f"duplicate shard row: {key}")
            seen.add(key)
            obj=a["guide_objectives"]
            row={"scenario":scenario,"seed_index":seed_index,"seed":seed,"algorithm":a["algorithm"],
                 "runtime_seconds":a["runtime_seconds"],"archive_size":a["archive_size"]}
            row.update({name:obj[i] for i,name in enumerate(OBJECTIVES)})
            rows.append(row)
    expected={(s,i,a) for s in SCENARIOS for i in range(30) for a in ALGORITHMS}
    missing=expected-seen
    extra=seen-expected
    print(f"final files={len(files)} rows={len(rows)} missing={len(missing)} extra={len(extra)}")
    if args.require_complete and (missing or extra):
        sample=sorted(missing)[:10]
        raise SystemExit(f"final run set is incomplete; first missing entries: {sample}")
    out=root/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    fields=["scenario","seed_index","seed","algorithm","runtime_seconds","archive_size",*OBJECTIVES]
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(out)


if __name__=="__main__":
    main()
