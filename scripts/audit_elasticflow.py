#!/usr/bin/env python3
import csv, glob, json, os, statistics
from pathlib import Path

def read_csvs(root):
    out={}
    for p in sorted(Path(root).rglob("*.csv")):
        with p.open(newline="",encoding="utf-8-sig") as f:
            rows=list(csv.DictReader(f))
        out[p.name]=rows
    return out

raw=read_csvs("audit/raw")
exp=read_csvs("audit/exp")
print("RAW_FILES",sorted((k,len(v)) for k,v in raw.items()))
print("EXP_FILES",sorted((k,len(v)) for k,v in exp.items()))

summary=[]
for name, rows in raw.items():
    single=[r for r in rows if float(r.get("num_gpu") or 0)==1 and float(r.get("duration") or 0)>0]
    durs=sorted(float(r["duration"]) for r in single)
    subs=sorted(float(r["submission_time"]) for r in single)
    exp_rows=exp.get(name,[])
    exp_map={r["job_id"]:r for r in exp_rows if r.get("job_id")}
    matched=[]
    ratios=[]
    for r in single:
        e=exp_map.get(r["job_id"])
        if e is None: continue
        matched.append(r["job_id"])
        try:
            slack=float(e["deadline"])-float(e["submission_time"])
            dur=float(r["duration"])
            if dur>0: ratios.append(slack/dur)
        except Exception:
            pass
    item={
        "file":name,
        "raw_rows":len(rows),
        "single_gpu_positive_duration":len(single),
        "matched_to_experiment":len(matched),
        "duration_median_s":statistics.median(durs) if durs else None,
        "duration_p90_s":durs[int(.9*(len(durs)-1))] if durs else None,
        "submission_span_s":(max(subs)-min(subs)) if subs else None,
        "deadline_slack_ratio_median":statistics.median(ratios) if ratios else None,
        "deadline_slack_ratio_p10":sorted(ratios)[int(.1*(len(ratios)-1))] if ratios else None,
        "deadline_slack_ratio_p90":sorted(ratios)[int(.9*(len(ratios)-1))] if ratios else None,
    }
    summary.append(item)

print("SUMMARY_JSON")
print(json.dumps(summary,indent=2))
if summary:
    largest=max(summary,key=lambda x:(x["single_gpu_positive_duration"],x["file"]))
    print("LARGEST_SINGLE_GPU_CLUSTER",json.dumps(largest,sort_keys=True))

# print schema/sample for each archive
if raw:
    n=sorted(raw)[0]
    print("RAW_SCHEMA",n,list(raw[n][0].keys()) if raw[n] else [])
    print("RAW_SAMPLE",json.dumps(raw[n][0] if raw[n] else {},sort_keys=True))
if exp:
    n=sorted(exp)[0]
    print("EXP_SCHEMA",n,list(exp[n][0].keys()) if exp[n] else [])
    print("EXP_SAMPLE",json.dumps(exp[n][0] if exp[n] else {},sort_keys=True))
