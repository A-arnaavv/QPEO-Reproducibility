from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv, json
import numpy as np

from qpeo.carbon import CarbonReplay
from qpeo.resource_environment import load_resource_environment
from qpeo.scenarios import build_alibaba_gpu_v2023_scenario

def test_alibaba_scenario_fixture(tmp_path: Path):
    snap=tmp_path/'ali.csv'
    fields=['name','cpu_milli','memory_mib','num_gpu','gpu_milli','gpu_spec','qos','pod_phase','creation_time','deletion_time','scheduled_time']
    with snap.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,qos in enumerate(['LS','Burstable','LS']):
            creation=1000+i*100
            w.writerow(dict(name=f'p{i}',cpu_milli=4000,memory_mib=8192,num_gpu=1,gpu_milli=500,gpu_spec='',qos=qos,pod_phase='Succeeded',creation_time=creation,deletion_time=creation+400,scheduled_time=creation+40))
    cfg=tmp_path/'cfg.json'
    cfg.write_text(json.dumps({
        'dataset_id':'alibaba_gpu_v2023','eligible_count':3,
        'sampling':{'windows':{'3':{'start_index':0,'end_index':2,'first_task':'p0','last_task':'p2'}}},
        'workload_mapping':{'carbon_class_by_qos':{'LS':'critical','Burstable':'deferred','BE':'flexible'}},
    }))
    env=load_resource_environment(Path(__file__).parents[1]/'configs'/'resource_environment_v1.json')
    t0=datetime(2026,1,5,tzinfo=timezone.utc)
    names=tuple(s.name for s in env.slots)
    replay=CarbonReplay((t0,t0+timedelta(hours=1)),names,np.array([[100,120,140,160,180,200,220],[110,130,150,170,190,210,230]],float),'fixture',t0,t0+timedelta(hours=1))
    b=build_alibaba_gpu_v2023_scenario(snap,cfg,env,replay,size=3)
    assert b.task_ids==('p0','p1','p2')
    assert b.service_time_hours.shape==(3,7)
    assert b.qpeo_inputs.performance_affinity.shape==(3,7)
    assert b.qpeo_inputs.dependency_matrix.shape==(3,3)
    assert b.workloads[0].deadline>b.workloads[0].arrival_time
    assert b.problem.resources[0].memory_capacity==64.0
