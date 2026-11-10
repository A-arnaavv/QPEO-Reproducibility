#!/usr/bin/env python3
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

p=Path("audit/spot/job_info_df.csv")
counts=Counter()
types=defaultdict(Counter)
eligible=0
with p.open(newline="",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        try:
            cpu=float(r["cpu_request"]); gpu=float(r["gpu_request"]); workers=float(r["worker_num"]); dur=float(r["duration"])
        except Exception:
            continue
        if workers!=1 or not (0<gpu<=1) or cpu<0 or cpu>16 or dur<=0:
            continue
        model=r["gpu_model"]; typ=r["job_type"]
        counts[model]+=1; types[model][typ]+=1; eligible+=1
print("eligible",eligible)
print("MODEL_COUNTS",json.dumps({m:{"total":counts[m],**types[m]} for m in sorted(counts)},indent=2))
