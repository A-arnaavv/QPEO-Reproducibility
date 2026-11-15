from datetime import datetime, timezone, timedelta
import numpy as np
from qpeo.carbon import CarbonReplay
from qpeo.experiment import run_full_suite
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.optimizer import QPEOInputs, ReconstructionQPEOSettings
from qpeo.simulator import PowerModel, SimulatorInputs

def make_one_resource_dynamic_case():
    problem=ProblemInstance(
        [Workload("w",1,.2,1,1,0,2,CarbonClass.FLEXIBLE)],
        [Resource("r0",1,8,100,1.0)],
        {CarbonClass.CRITICAL:500,CarbonClass.DEFERRED:500,CarbonClass.FLEXIBLE:500},
        1.0,
    )
    sim=SimulatorInputs(np.array([[1.0]]),PowerModel(np.array([0.0]),np.array([0.2])),"allocated_gpu_hours")
    t0=datetime(2026,1,5,tzinfo=timezone.utc)
    replay=CarbonReplay(
        (t0,t0+timedelta(minutes=30),t0+timedelta(minutes=60)),
        ("r0",),
        np.array([[100.0],[200.0],[300.0]]),
        "fixture",t0,t0+timedelta(minutes=60),
    )
    qinputs=QPEOInputs(
        simulator_inputs=sim,
        performance_affinity=np.array([[1.0]]),
        thermal_headroom=np.array([1.0]),
        dependency_matrix=np.zeros((1,1)),
        carbon_replay=replay,
    )
    settings=ReconstructionQPEOSettings([1/6]*6,1/3,1/3,1/3,.5,.25)
    return problem,qinputs,settings

def test_all_13_algorithms_use_execution_time_carbon():
    problem,qinputs,settings=make_one_resource_dynamic_case()
    suite=run_full_suite(problem,qinputs,settings,[1/6]*6,seeds=[7],population_size=4,max_iterations=2)
    assert len(suite.records)==13
    for rec in suite.records:
        hist=rec.result.guide_objective_history
        assert hist.shape==(3,6)
        assert np.allclose(hist[:,2],[150.0,150.0,150.0])
