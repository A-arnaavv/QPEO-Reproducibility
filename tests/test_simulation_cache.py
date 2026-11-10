import numpy as np
import pytest

from qpeo.carbon import with_carbon_intensities
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.simulator import (
    PowerModel, SimulatorInputs, clear_simulation_cache,
    simulate_fixed_assignment, simulation_cache_stats,
)


def _fixture():
    problem=ProblemInstance(
        workloads=[
            Workload("a",1,.5,2,0,0,4,CarbonClass.FLEXIBLE),
            Workload("b",1,.5,2,0,.5,4,CarbonClass.FLEXIBLE),
        ],
        resources=[
            Resource("r0",1,8,100,1,cpu_capacity=4),
            Resource("r1",1,8,200,1,cpu_capacity=4),
        ],
        carbon_thresholds={
            CarbonClass.CRITICAL:250,
            CarbonClass.DEFERRED:250,
            CarbonClass.FLEXIBLE:250,
        },
        reference_cost=1,
    )
    inputs=SimulatorInputs(
        service_time_hours=np.array([[1.,2.],[1.,2.]]),
        power_model=PowerModel(np.array([0.,0.]),np.array([.2,.2])),
        billing_mode="allocated_gpu_hours",
    )
    X=np.array([[1,0],[1,0]],dtype=int)
    return problem,inputs,X


def test_identical_assignment_reuses_only_schedule_simulation():
    problem,inputs,X=_fixture()
    clear_simulation_cache()
    a=simulate_fixed_assignment(problem,X,inputs)
    b=simulate_fixed_assignment(problem,X,inputs)
    st=simulation_cache_stats()
    assert st=={"hits":1,"misses":1,"entries":1}
    assert np.allclose(a.completion_times,b.completion_times)
    assert np.allclose(a.energy_kwh_by_resource,b.energy_kwh_by_resource)


def test_dynamic_carbon_can_hit_same_schedule_cache():
    problem,inputs,X=_fixture()
    clear_simulation_cache()
    a=simulate_fixed_assignment(problem,X,inputs)
    changed=with_carbon_intensities(problem,[150,225])
    b=simulate_fixed_assignment(changed,X,inputs)
    assert simulation_cache_stats()["hits"]==1
    assert np.allclose(a.completion_times,b.completion_times)


def test_carbon_feasibility_is_rechecked_before_cache_hit():
    problem,inputs,X=_fixture()
    clear_simulation_cache()
    simulate_fixed_assignment(problem,X,inputs)
    infeasible=with_carbon_intensities(problem,[300,200])
    with pytest.raises(ValueError,match="Carbon-threshold"):
        simulate_fixed_assignment(infeasible,X,inputs)
    # The infeasible call fails before it is counted as a cache hit.
    assert simulation_cache_stats()["hits"]==0
