import csv
from pathlib import Path

import numpy as np
import pytest

from qpeo.traces import (
    TraceMaterializationPolicy,
    load_alibaba_gpu_v2023_pods_csv,
    load_azure_functions_2021_csv,
    materialize_workloads,
    service_time_matrix_from_observed_runtime,
)
from qpeo.models import CarbonClass

def test_alibaba_loader(tmp_path: Path):
    p=tmp_path/'ali.csv'
    fields=['name','cpu_milli','memory_mib','num_gpu','gpu_milli','qos','creation_time','deletion_time','scheduled_time']
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        w.writerow(dict(name='p1',cpu_milli='4000',memory_mib='8192',num_gpu='1',gpu_milli='250',qos='LS',creation_time='100',scheduled_time='110',deletion_time='3710'))
    tasks=load_alibaba_gpu_v2023_pods_csv(p)
    assert len(tasks)==1
    assert tasks[0].gpu_demand == pytest.approx(.25)
    assert tasks[0].cpu_demand == pytest.approx(4.0)
    assert tasks[0].memory_demand == pytest.approx(8.0)
    assert tasks[0].runtime_hours == pytest.approx(1.0)

def test_azure_loader_requires_explicit_missing_resource_policy(tmp_path: Path):
    p=tmp_path/'az.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['app','func','end_timestamp','duration']); w.writeheader()
        w.writerow(dict(app='a',func='f',end_timestamp='100',duration='10'))
    tasks=load_azure_functions_2021_csv(p)
    policy=TraceMaterializationPolicy(
        fallback_cpu_demand=None,
        fallback_gpu_demand=None,
        fallback_memory_demand=None,
        fallback_bandwidth_demand=None,
        deadline_slack_multiplier=2,
        default_carbon_class=CarbonClass.DEFERRED,
    )
    with pytest.raises(ValueError, match='CPU demand'):
        materialize_workloads(tasks,policy)

def test_explicit_materialization_and_service_matrix(tmp_path: Path):
    p=tmp_path/'az.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['app','func','end_timestamp','duration']); w.writeheader()
        w.writerow(dict(app='a',func='f',end_timestamp='100',duration='10'))
    tasks=load_azure_functions_2021_csv(p)
    policy=TraceMaterializationPolicy(
        fallback_cpu_demand=1.0,
        fallback_gpu_demand=.2,
        fallback_memory_demand=1.0,
        fallback_bandwidth_demand=.1,
        deadline_slack_multiplier=2,
        default_carbon_class=CarbonClass.DEFERRED,
    )
    ws=materialize_workloads(tasks,policy)
    assert ws[0].gpu_demand == pytest.approx(.2)
    assert ws[0].deadline-ws[0].arrival_time == pytest.approx(20/3600)
    S=service_time_matrix_from_observed_runtime(tasks,[1.0,2.0])
    assert S.shape==(1,2)
    assert S[0,1] == pytest.approx((10/3600)/2)
