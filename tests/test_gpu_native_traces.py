import csv, json
from pathlib import Path
import numpy as np
import pytest
from qpeo.traces import load_helios_cluster_csv, load_acme_job_csv, load_philly_job_log, shard_multi_gpu_tasks

def test_helios_sharding(tmp_path: Path):
    p=tmp_path/"cluster_log.csv"
    fields=["job_id","user","vc","gpu_num","cpu_num","node_num","state","submit_time","start_time","end_time","duration","queue"]
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        w.writerow(dict(job_id="1",user="u",vc="v",gpu_num="4",cpu_num="16",node_num="1",state="COMPLETED",submit_time="2020-06-09 18:41:01",start_time="2020-06-09 18:41:01",end_time="2020-06-09 19:41:01",duration="3600",queue="0"))
    jobs=load_helios_cluster_csv(p); units,dep=shard_multi_gpu_tasks(jobs,max_scheduling_units=4)
    assert len(units)==4 and all(u.gpu_demand==1 for u in units)
    assert np.all(dep[np.triu_indices(4,1)]==1)

def test_acme_memory_split(tmp_path: Path):
    p=tmp_path/"trace_kalos.csv"
    fields=["job_id","user","node_num","gpu_num","cpu_num","mem_per_pod_GB","shared_mem_per_pod","type","state","submit_time","start_time","end_time","duration","queue","gpu_time"]
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        w.writerow(dict(job_id="a",user="u",node_num="1",gpu_num="2",cpu_num="8",mem_per_pod_GB="32",shared_mem_per_pod="0",type="LLM",state="COMPLETED",submit_time="2023-03-01 00:00:00+08:00",start_time="2023-03-01 00:00:10+08:00",end_time="2023-03-01 01:00:10+08:00",duration="3600",queue="10",gpu_time="7200"))
    jobs=load_acme_job_csv(p); units,_=shard_multi_gpu_tasks(jobs,max_scheduling_units=2)
    assert [u.memory_demand for u in units]==pytest.approx([16,16])

def test_philly_loader(tmp_path: Path):
    p=tmp_path/"jobs.json"
    p.write_text(json.dumps([{"status":"Pass","vc":"v","jobid":"j1","submitted_time":"2017-10-07 01:11:39","user":"u","attempts":[{"start_time":"2017-10-07 01:12:09","end_time":"2017-10-07 02:12:09","detail":[{"ip":"m1","gpus":["gpu0","gpu1"]}]}]}]))
    jobs=load_philly_job_log(p); units,dep=shard_multi_gpu_tasks(jobs,max_scheduling_units=2)
    assert len(jobs)==1 and jobs[0].gpu_demand==2
    assert len(units)==2 and dep[0,1]==1 and dep[1,0]==1
