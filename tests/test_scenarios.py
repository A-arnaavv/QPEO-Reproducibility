from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv, json
import numpy as np

from qpeo.carbon import CarbonReplay
from qpeo.resource_environment import load_resource_environment
from qpeo.scenarios import build_alibaba_gpu_v2023_scenario, build_elasticflow_itp_scenario, build_helios_earth_scenario

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


def test_locked_alibaba_snapshot_matches_protocol():
    root=Path(__file__).parents[1]
    env=load_resource_environment(root/'configs'/'resource_environment_v1.json')
    t0=datetime(2026,1,5,tzinfo=timezone.utc)
    names=tuple(slot.name for slot in env.slots)
    replay=CarbonReplay(
        (t0,t0+timedelta(hours=1)), names,
        np.array([[100,120,140,160,180,200,220],[110,130,150,170,190,210,230]],float),
        'fixture',t0,t0+timedelta(hours=1),
    )
    b=build_alibaba_gpu_v2023_scenario(
        root/'data'/'alibaba_gpu_v2023_eligible_succeeded.csv',
        root/'configs'/'trace_alibaba_gpu_v2023.json',
        env,replay,size=64,
    )
    assert len(b.task_ids)==64
    assert b.task_ids[0]=='openb-pod-3131'
    assert b.task_ids[-1]=='openb-pod-5985'
    assert b.service_time_hours.shape==(64,7)
    assert all(w.gpu_demand<=1.0 for w in b.workloads)
    assert all(w.memory_demand<=64.0 for w in b.workloads)


def test_locked_elasticflow_snapshot_matches_protocol():
    root=Path(__file__).parents[1]
    env=load_resource_environment(root/'configs'/'resource_environment_v1.json')
    t0=datetime(2026,1,5,tzinfo=timezone.utc)
    names=tuple(slot.name for slot in env.slots)
    replay=CarbonReplay(
        (t0,t0+timedelta(hours=1)),names,
        np.array([[100,120,140,160,180,200,220],[110,130,150,170,190,210,230]],float),
        'fixture',t0,t0+timedelta(hours=1),
    )
    b=build_elasticflow_itp_scenario(
        root/'data'/'elasticflow_cluster08_single_gpu.csv',
        root/'configs'/'trace_elasticflow_itp.json',
        env,replay,size=64,
    )
    assert len(b.task_ids)==64
    assert b.task_ids[0]=='fd68fa2d-ed9f-6fe1-0a85-877c7b6de8da'
    assert b.task_ids[-1]=='42dfd273-4486-8e16-9a12-9f4ce099e4b5'
    assert b.service_time_hours.shape==(64,7)
    assert all(w.gpu_demand==1.0 for w in b.workloads)
    assert all(w.cpu_demand==0.0 for w in b.workloads)
    assert all(w.memory_demand==0.0 for w in b.workloads)
    classes={w.carbon_class for w in b.workloads}
    assert classes


def test_locked_helios_snapshot_matches_protocol():
    root=Path(__file__).parents[1]
    env=load_resource_environment(root/'configs'/'resource_environment_v1.json')
    t0=datetime(2026,1,5,tzinfo=timezone.utc)
    names=tuple(slot.name for slot in env.slots)
    replay=CarbonReplay(
        (t0,t0+timedelta(hours=1)),names,
        np.array([[100,120,140,160,180,200,220],[110,130,150,170,190,210,230]],float),
        'fixture',t0,t0+timedelta(hours=1),
    )
    b=build_helios_earth_scenario(
        root/'data'/'helios_earth_selected_128.csv',
        root/'configs'/'trace_helios_earth.json',
        env,replay,size=64,
    )
    assert len(b.task_ids)==64
    assert b.task_ids[0]=='1738622'
    assert b.task_ids[-1]=='1739705'
    assert b.service_time_hours.shape==(64,7)
    assert all(w.gpu_demand==1.0 for w in b.workloads)
    assert all(w.memory_demand==0.0 for w in b.workloads)
    assert all(w.cpu_demand<=16.0 for w in b.workloads)
    assert all(w.carbon_class==CarbonClass.DEFERRED for w in b.workloads)
