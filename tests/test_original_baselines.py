import numpy as np
import pytest
from qpeo.baselines import CommonBaselineConfig, run_pso,run_de,run_gwo,run_woa,run_abc,run_ga,run_hs,run_fa,run_aco
from qpeo.models import CarbonClass,ProblemInstance,Resource,Workload
from qpeo.simulator import PowerModel,SimulatorInputs
from qpeo.evaluator import check_constraints

def fixture_data():
    w=[Workload(f"w{i}",1,.2,1,1,i*.1,3+i*.1,CarbonClass.FLEXIBLE) for i in range(3)]
    r=[Resource("g0",1,8,200,1),Resource("g1",1,8,350,.8)]
    p=ProblemInstance(w,r,{CarbonClass.CRITICAL:250,CarbonClass.DEFERRED:400,CarbonClass.FLEXIBLE:500},2)
    s=SimulatorInputs(np.array([[1,1.2],[1.1,.9],[.8,1.0]],float),PowerModel(np.array([.05,.05]),np.array([.2,.18])),"allocated_gpu_hours")
    return p,s

@pytest.mark.parametrize("runner",[run_pso,run_de,run_gwo,run_woa,run_abc,run_ga,run_hs,run_fa,run_aco])
def test_original_baseline_runs(runner):
    p,s=fixture_data(); res=runner(p,s,[1/6]*6,config=CommonBaselineConfig(8,3),seed=4)
    assert len(res.final_assignments)==8
    assert len(res.best_scalar_history)==3
    assert all(check_constraints(p,X).feasible for X in res.final_assignments)
    assert len(res.archive.entries)>=1
