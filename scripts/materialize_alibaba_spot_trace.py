#!/usr/bin/env python3
import csv, json, math
from collections import Counter
from pathlib import Path

ANCHOR=0.6260788463455607
SRC=Path("audit/spot/job_info_df.csv")
OUT=Path("data/alibaba_spot_a100_selected_128.csv")
CFG=Path("configs/trace_alibaba_spot_a100.json")
SOURCE_MODEL="A100-SXM4-80GB"
SOURCE_FP32_TFLOPS=19.5

rows=[]
with SRC.open(newline="",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        try:
            cpu=float(r["cpu_request"]); gpu=float(r["gpu_request"])
            workers=float(r["worker_num"]); duration=float(r["duration"])
            submit=float(r["submit_time"])
        except Exception:
            continue
        if r["gpu_model"]!=SOURCE_MODEL:
            continue
        if workers!=1 or not (0<gpu<=1) or cpu<0 or cpu>16 or duration<=0:
            continue
        if r["job_type"] not in ("HP","Spot"):
            continue
        rows.append(r)
rows.sort(key=lambda r:(float(r["submit_time"]),int(r["job_name"])))
n=len(rows)
if n!=83416:
    raise RuntimeError(f"Expected 83416 eligible A100 jobs from audit, got {n}")

windows={}
for m in (32,64,128):
    st=math.floor(ANCHOR*(n-m)); en=st+m-1
    block=rows[st:en+1]
    mix=Counter(r["job_type"] for r in block)
    windows[str(m)]={
        "start_index":st,"end_index":en,
        "first_task":block[0]["job_name"],"last_task":block[-1]["job_name"],
        "submission_span_hours":(float(block[-1]["submit_time"])-float(block[0]["submit_time"]))/3600.0,
        "job_type_counts":dict(mix),
    }

base=windows["128"]["start_index"]
for m in (32,64,128):
    windows[str(m)]["snapshot_start_index"]=windows[str(m)]["start_index"]-base
    windows[str(m)]["snapshot_end_index"]=windows[str(m)]["end_index"]-base

selected=rows[windows["128"]["start_index"]:windows["128"]["end_index"]+1]
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open("w",newline="",encoding="utf-8") as f:
    fields=list(rows[0])
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(selected)

cfg={
  "dataset_id":"alibaba_spot_gpu_2026_a100",
  "status":"locked-pre-results",
  "source":{
    "repository":"alibaba/clusterdata",
    "commit":"3858992ca41bf92082e6d6522ddf0bebd14e838b",
    "path":"cluster-trace-v2026-spot-gpu/job_info_df.csv",
    "blob_sha":"f2a0dcbc26b5a1a047e49a7efa886e86ec9cde32",
    "sha256":"113ccee4c28f5c3bbaaca974cd164b9280b7d4c39e53b745443b28eea05e03dd"
  },
  "selection":{
    "source_gpu_model":SOURCE_MODEL,
    "source_fp32_tflops":SOURCE_FP32_TFLOPS,
    "eligible_rule":"gpu_model==A100-SXM4-80GB, worker_num==1, 0<gpu_request<=1, 0<=cpu_request<=16, duration>0, job_type in {HP,Spot}",
    "eligible_count":n,
    "anchor_fraction":ANCHOR,
    "primary_size":64,
    "scalability_sizes":[32,64,128],
    "windows":windows,
    "snapshot":"data/alibaba_spot_a100_selected_128.csv"
  },
  "workload_mapping":{
    "gpu_demand":"gpu_request",
    "cpu_demand":"cpu_request",
    "memory_demand":0.0,
    "bandwidth_demand":0.0,
    "arrival_time":"submit_time relative to first job in selected window",
    "observed_runtime":"duration / 3600",
    "deadline_HP":"arrival + source duration",
    "deadline_Spot":"infinity (best-effort; excluded from SLA denominator)",
    "carbon_class":{"HP":"critical","Spot":"flexible"}
  },
  "service_time":{
    "primary_rule":"observed A100 duration * (19.5 FP32 TFLOPS / target_resource_fp32_tflops)",
    "source_fp32_tflops":SOURCE_FP32_TFLOPS,
    "source_spec":"NVIDIA A100 80GB SXM"
  },
  "qpeo_features":{
    "performance_affinity":"row-wise normalized inverse service time",
    "thermal_headroom":"1 - resource_max_power / max_resource_max_power",
    "dependency_matrix":"all zeros because the trace does not expose job dependency metadata"
  }
}
CFG.parent.mkdir(parents=True,exist_ok=True)
CFG.write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
print("eligible",n)
print(json.dumps(windows,indent=2))
