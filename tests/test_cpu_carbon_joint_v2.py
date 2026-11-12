"""Integration regression: CPU queuing changes time-aligned carbon exposure.

This test exercises the actual simulator, not a 13-optimizer run. It uses a
synthetic two-job instance with a step change in forecast grid intensity.
"""
import numpy as np
import pytest

from qpeo.carbon_accounting import CarbonTimeSeries, account_simulation_result
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.simulator import PowerModel, SimulatorInputs, clear_simulation_cache, simulate_fixed_assignment


def _run_case(cpu_capacity):
    jobs = [
        Workload("a", 7., .3, 1., 0., 0., 5., CarbonClass.FLEXIBLE),
        Workload("b", 7., .3, 1., 0., .25, 5., CarbonClass.FLEXIBLE),
    ]
    problem=ProblemInstance(
        jobs, [Resource("r0", 1., 64., 100., 1., cpu_capacity=cpu_capacity)],
        {CarbonClass.CRITICAL: 500., CarbonClass.DEFERRED: 500.,
         CarbonClass.FLEXIBLE: 500.},
        1.,
    )
    inputs=SimulatorInputs(
        np.ones((2,1)), PowerModel(np.array([0.]), np.array([.3])),
        "allocated_gpu_hours",
        carbon_time_series=CarbonTimeSeries(
            np.array([[100.], [100.], [400.], [400.]]), .5,
            coverage_policy="strict",
        ),
    )
    clear_simulation_cache()
    X=np.ones((2,1),int)
    schedule=simulate_fixed_assignment(problem,X,inputs)
    carbon=account_simulation_result(problem,X,inputs,schedule,inputs.carbon_time_series)
    return schedule,carbon


def test_aggregate_cpu_queuing_changes_carbon_exposure_but_not_gpu_energy():
    constrained,a=_run_case(8.)
    unconstrained,b=_run_case(None)
    assert constrained.start_times == pytest.approx([0.,1.])
    assert unconstrained.start_times == pytest.approx([0.,.25])
    assert a.total_energy_kwh == pytest.approx(.18)
    assert b.total_energy_kwh == pytest.approx(.18)
    assert a.total_carbon_g == pytest.approx(45.)
    assert b.total_carbon_g == pytest.approx(24.75)
    assert a.energy_weighted_intensity_g_per_kwh == pytest.approx(250.)
    assert b.energy_weighted_intensity_g_per_kwh == pytest.approx(137.5)
