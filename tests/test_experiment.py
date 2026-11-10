import numpy as np

from qpeo.baselines import CommonBaselineConfig
from qpeo.experiment import run_reviewer_critical_suite
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.optimizer import QPEOInputs, ReconstructionQPEOSettings
from qpeo.simulator import PowerModel, SimulatorInputs

def test_reviewer_critical_suite_runs_all_four():
    workloads = [
        Workload("w0", 1, .20, 1, 1, 0.0, 2.0, CarbonClass.FLEXIBLE),
        Workload("w1", 1, .20, 1, 1, 0.1, 2.2, CarbonClass.FLEXIBLE),
    ]
    resources = [
        Resource("g0", 1.0, 8, 200, 1.0),
        Resource("g1", 1.0, 8, 350, .8),
    ]
    problem = ProblemInstance(
        workloads, resources,
        {CarbonClass.CRITICAL:250, CarbonClass.DEFERRED:400, CarbonClass.FLEXIBLE:500},
        reference_cost=2.0,
    )
    sim = SimulatorInputs(
        service_time_hours=np.array([[1.0,1.2],[1.1,.9]]),
        power_model=PowerModel(np.array([.05,.05]),np.array([.2,.18])),
        billing_mode="allocated_gpu_hours",
    )
    qinputs = QPEOInputs(
        simulator_inputs=sim,
        performance_affinity=np.array([[1.0,.8],[.8,1.0]]),
        thermal_headroom=np.array([.8,.8]),
        dependency_matrix=np.zeros((2,2)),
    )
    settings = ReconstructionQPEOSettings(
        scalar_weights=[1/6]*6,
        alpha=.5, beta=.25, gamma=.25,
        dependency_threshold=.5,
        dependency_strength=.2,
    )
    suite = run_reviewer_critical_suite(
        problem, qinputs, settings, [1/6]*6,
        seeds=[1], population_size=6, max_iterations=2,
    )
    names = {r.algorithm for r in suite.records}
    assert names == {"QPEO","FPA","QIEA","NSGA-II"}
    assert len(suite.records) == 4
