import numpy as np

from qpeo.baselines import CommonBaselineConfig, run_fpa, run_nsga2, run_qiea
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.simulator import PowerModel, SimulatorInputs
from qpeo.evaluator import check_constraints

def toy_problem():
    workloads = [
        Workload("w0", 1, .25, 1, 1, 0.0, 3.0, CarbonClass.FLEXIBLE),
        Workload("w1", 1, .25, 1, 1, 0.2, 3.2, CarbonClass.FLEXIBLE),
        Workload("w2", 1, .25, 1, 1, 0.4, 3.4, CarbonClass.FLEXIBLE),
    ]
    resources = [
        Resource("g0", 1.0, 8, 200, 1.0),
        Resource("g1", 1.0, 8, 350, 0.8),
    ]
    problem = ProblemInstance(
        workloads, resources,
        {
            CarbonClass.CRITICAL: 250,
            CarbonClass.DEFERRED: 400,
            CarbonClass.FLEXIBLE: 500,
        },
        reference_cost=2.0,
    )
    sim = SimulatorInputs(
        service_time_hours=np.array([
            [1.0, 1.4],
            [1.2, 0.9],
            [0.8, 1.1],
        ]),
        power_model=PowerModel(
            idle_power_kw=np.array([0.05, 0.05]),
            dynamic_power_kw=np.array([0.20, 0.18]),
            power_exponent=1.0,
        ),
        billing_mode="allocated_gpu_hours",
    )
    return problem, sim

def _check_result(result, problem):
    assert len(result.final_assignments) == 8
    assert len(result.best_scalar_history) == 4
    assert result.guide_objective_history.shape == (5,6)
    for X in result.final_assignments:
        assert check_constraints(problem, X).feasible
    assert len(result.archive.entries) >= 1

def test_fpa_runs():
    problem, sim = toy_problem()
    result = run_fpa(problem, sim, [1/6]*6, config=CommonBaselineConfig(8,4), seed=3)
    _check_result(result, problem)

def test_qiea_runs():
    problem, sim = toy_problem()
    result = run_qiea(problem, sim, [1/6]*6, config=CommonBaselineConfig(8,4), seed=3)
    _check_result(result, problem)

def test_nsga2_runs():
    problem, sim = toy_problem()
    result = run_nsga2(problem, sim, [1/6]*6, config=CommonBaselineConfig(8,4), seed=3)
    _check_result(result, problem)
