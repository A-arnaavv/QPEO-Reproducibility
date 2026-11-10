#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
from datetime import datetime

ANCHOR=0.6260788463455607
SRC=Path("audit/helios/data/Earth/cluster_log.csv")
OUT=Path("data/helios_earth_selected_128.csv")
CFG=Path("configs/trace_helios_earth.json")

with SRC.open(newline="",encoding="utf-8-sig") as f:
    rows=list(csv.DictReader(f))

eligible=[]
for r in rows:
    try:
        gpu=float(r["gpu_num"]); cpu=float(r["cpu_num"]); dur=float(r["duration"])
        submit=datetime.fromisoformat(r["submit_time"])
        start=datetime.fromisoformat(r["start_time"])
        end=datetime.fromisoformat(r["end_time"])
    except Exception:
        continue
    if (r.get("state") or "").strip().upper()!="COMPLETED":
        continue
    if gpu!=1 or dur<=0 or cpu>16 or cpu<0:
        continue
    if end<=start or start<submit:
        continue
    eligible.append((submit,int(r["job_id"]),r))
eligible.sort(key=lambda x:(x[0],x[1]))
n=len(eligible)
if n<128:
    raise RuntimeError(f"Need at least 128 eligible jobs, found {n}")

windows={}
for m in (32,64,128):
    st=math.floor(ANCHOR*(n-m)); en=st+m-1
    block=eligible[st:en+1]
    windows[str(m)]={
        "start_index":st,"end_index":en,
        "first_task":block[0][2]["job_id"],"last_task":block[-1][2]["job_id"],
        "submission_span_hours":(block[-1][0]-block[0][0]).total_seconds()/3600.0,
    }

base128=windows["128"]["start_index"]
for m in (32,64,128):
    windows[str(m)]["snapshot_start_index"]=windows[str(m)]["start_index"]-base128
    windows[str(m)]["snapshot_end_index"]=windows[str(m)]["end_index"]-base128

selected=[x[2] for x in eligible[windows["128"]["start_index"]:windows["128"]["end_index"]+1]]
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open("w",newline="",encoding="utf-8") as f:
    fields=list(rows[0].keys())
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(selected)

cfg={
  "dataset_id":"helios_earth",
  "status":"locked-pre-results",
  "source":{
    "repository":"S-Lab-System-Group/HeliosData",
    "ref":"master",
    "archive_path":"data.zip",
    "archive_blob_sha":"1a48d5e0fb816fe9bbf1ad7086e7dec90923e5c6",
    "archive_sha256":"3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac",
    "cluster":"Earth",
    "cluster_file":"data/Earth/cluster_log.csv"
  },
  "selection":{
    "cluster_rule":"cluster with largest completed single-GPU job pool",
    "eligible_rule":"state==COMPLETED, gpu_num==1, duration>0, 0<=cpu_num<=16, valid submit/start/end timestamps",
    "eligible_count":n,
    "anchor_fraction":ANCHOR,
    "primary_size":64,
    "scalability_sizes":[32,64,128],
    "windows":windows,
    "snapshot":"data/helios_earth_selected_128.csv"
  },
  "workload_mapping":{
    "gpu_demand":1.0,
    "cpu_demand":"cpu_num",
    "memory_demand":0.0,
    "bandwidth_demand":0.0,
    "arrival_time":"submit_time relative to first job in selected window",
    "observed_runtime":"duration / 3600 (source start-to-end)",
    "deadline":"source end_time relative to first selected submit_time",
    "carbon_class":"deferred for all jobs because Helios exposes no priority/SLA class"
  },
  "service_time":{
    "primary_rule":"observed duration / locked FP32 resource speed factor",
    "reference_hardware":"RTX A6000 normalized speed factor 1.0",
    "limitation":"source GPU model is not identified per job; this is a reconstruction proxy"
  },
  "qpeo_features":{
    "performance_affinity":"row-wise normalized inverse service time",
    "thermal_headroom":"1 - resource_max_power / max_resource_max_power",
    "dependency_matrix":"all zeros because the trace does not expose job dependency metadata"
  }
}
CFG.parent.mkdir(parents=True,exist_ok=True)
CFG.write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
print("eligible_count",n)
print(json.dumps(windows,indent=2))
