import numpy as np
import pytest

from qpeo import (
    CarbonClass,
    PowerModel,
    ProblemInstance,
    Resource,
    SimulatorInputs,
    Workload,
    evaluate_objectives,
    simulate_fixed_assignment,
)

def make_problem():
    return ProblemInstance(
        workloads=[
            Workload("w1", 1, 0.4, 4, 1, 0.0, 5.0, CarbonClass.FLEXIBLE),
            Workload("w2", 1, 0.5, 6, 1, 1.0, 5.0, CarbonClass.DEFERRED),
        ],
        resources=[
            Resource("g1", 1.0, 16, 250, 1.0),
            Resource("g2", 1.0, 16, 400, 0.8),
        ],
        carbon_thresholds={
            CarbonClass.CRITICAL: 300,
            CarbonClass.DEFERRED: 450,
            CarbonClass.FLEXIBLE: 450,
        },
        reference_cost=3.0,
    )

def test_overlapping_energy_and_active_time():
    problem = make_problem()
    X = np.array([[1,0],[1,0]])
    inputs = SimulatorInputs(
        service_time_hours=np.array([[2.0,3.0],[2.0,2.5]]),
        power_model=PowerModel(
            idle_power_kw=np.array([0.10,0.08]),
            dynamic_power_kw=np.array([0.20,0.18]),
        ),
        billing_mode="active_resource_hours",
    )
    result = simulate_fixed_assignment(problem, X, inputs)
    assert result.energy_kwh_by_resource[0] == pytest.approx(0.66)
    assert result.resource_active_hours[0] == pytest.approx(3.0)
    assert result.allocated_gpu_hours[0] == pytest.approx(1.8)
    assert result.time_average_gpu_utilization[0] == pytest.approx(0.6)
    assert np.allclose(result.completion_times, [2.0,3.0])

def test_billing_mode_changes_cost_basis():
    problem = make_problem()
    X = np.array([[1,0],[1,0]])
    common = dict(
        service_time_hours=np.array([[2.0,3.0],[2.0,2.5]]),
        power_model=PowerModel(
            idle_power_kw=np.array([0.10,0.08]),
            dynamic_power_kw=np.array([0.20,0.18]),
        ),
    )
    active = simulate_fixed_assignment(problem, X, SimulatorInputs(**common, billing_mode="active_resource_hours"))
    allocated = simulate_fixed_assignment(problem, X, SimulatorInputs(**common, billing_mode="allocated_gpu_hours"))
    obj_active = evaluate_objectives(problem, X, active.state)
    obj_alloc = evaluate_objectives(problem, X, allocated.state)
    assert obj_active.f6_normalized_cost == pytest.approx(3.0/3.0)
    assert obj_alloc.f6_normalized_cost == pytest.approx(1.8/3.0)

def test_temporal_capacity_queues_aggregate_overflow():
    problem = make_problem()
    queued_problem = ProblemInstance(
        workloads=[
            Workload("w1",1,.7,4,1,0.0,5,CarbonClass.FLEXIBLE),
            Workload("w2",1,.6,6,1,0.25,5,CarbonClass.DEFERRED),
        ],
        resources=problem.resources,
        carbon_thresholds=problem.carbon_thresholds,
        reference_cost=problem.reference_cost,
    )
    X=np.array([[1,0],[1,0]])
    inputs=SimulatorInputs(
        service_time_hours=np.ones((2,2)),
        power_model=PowerModel(np.array([.1,.1]),np.array([.2,.2])),
        billing_mode="active_resource_hours",
    )
    result=simulate_fixed_assignment(queued_problem,X,inputs)
    assert result.start_times[0]==pytest.approx(0.0)
    assert result.start_times[1]==pytest.approx(1.0)
    assert result.completion_times[1]==pytest.approx(2.0)

def test_individual_resource_overflow_rejected():
    problem=make_problem()
    too_large=ProblemInstance(
        [Workload("big",1,1.2,4,1,0,5,CarbonClass.FLEXIBLE)],
        problem.resources,problem.carbon_thresholds,problem.reference_cost,
    )
    X=np.array([[1,0]])
    inputs=SimulatorInputs(
        service_time_hours=np.ones((1,2)),
        power_model=PowerModel(np.array([.1,.1]),np.array([.2,.2])),
        billing_mode="active_resource_hours",
    )
    with pytest.raises(ValueError,match="GPU-capacity"):
        simulate_fixed_assignment(too_large,X,inputs)
