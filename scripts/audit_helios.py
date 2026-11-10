#!/usr/bin/env python3
import csv, json, statistics
from pathlib import Path

files=sorted(Path("audit/helios").rglob("*.csv"))
print("FILES",[(str(p),p.stat().st_size) for p in files])

summary=[]
for p in files:
    with p.open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if not rows: continue
    fields=list(rows[0])
    print("SCHEMA",p.name,fields)
    completed=[]
    for r in rows:
        try:
            gpu=float(r.get("gpu_num") or 0)
            dur=float(r.get("duration") or 0)
        except Exception:
            continue
        state=(r.get("state") or "").strip().upper()
        if gpu==1 and dur>0 and state=="COMPLETED":
            completed.append(r)
    durs=sorted(float(r["duration"]) for r in completed)
    queues=sorted(float(r.get("queue") or 0) for r in completed)
    cpus=sorted(float(r.get("cpu_num") or 0) for r in completed)
    submits=[r.get("submit_time") for r in completed]
    item={
        "file":p.name,
        "rows":len(rows),
        "completed_single_gpu":len(completed),
        "duration_median_s":statistics.median(durs) if durs else None,
        "duration_p90_s":durs[int(.9*(len(durs)-1))] if durs else None,
        "queue_median_s":statistics.median(queues) if queues else None,
        "cpu_median":statistics.median(cpus) if cpus else None,
        "cpu_p90":cpus[int(.9*(len(cpus)-1))] if cpus else None,
        "first_submit":min(submits) if submits else None,
        "last_submit":max(submits) if submits else None,
    }
    summary.append(item)
print("SUMMARY_JSON")
print(json.dumps(summary,indent=2))
if summary:
    largest=max(summary,key=lambda x:(x["completed_single_gpu"],x["file"]))
    print("LARGEST_COMPLETED_SINGLE_GPU",json.dumps(largest,sort_keys=True))
