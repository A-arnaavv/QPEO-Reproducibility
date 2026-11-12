"""Synthetic full-suite check for the stationary carbon-v2 planning signal.

A planning grid signal (mean across a forecast week) is frozen for every
optimizer iteration. Evaluation of the carbon objective still integrates
forecast intensity along the candidate's scheduled execution intervals.
This is a synthetic test, not a primary workload trace experiment.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from qpeo.carbon import CarbonReplay, with_carbon_intensities
from qpeo.carbon_accounting import CarbonTimeSeries, account_simulation_result
from qpeo.experiment import run_full_suite
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.optimizer import QPEOInputs, ReconstructionQPEOSettings
from qpeo.simulator import PowerModel, SimulatorInputs, simulate_fixed_assignment


def test_full_suite_uses_fixed_planning_signal_and_execution_time_carbon():
    t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
    rows = np.array([[0., 200.], [200., 200.], [400., 100.]])
    replay = CarbonReplay(
        (t0, t0 + timedelta(minutes=30), t0 + timedelta(hours=1)),
        ("r0", "r1"),
        rows,
        "synthetic", t0, t0 + timedelta(hours=1),
    )
    workloads = [
        Workload("a", 2., .4, 1., 0., 0., 2., CarbonClass.FLEXIBLE),
        Workload("b", 2., .4, 1., 0., .25, 2., CarbonClass.FLEXIBLE),
    ]
    initial = ProblemInstance(
        workloads,
        [Resource("r0", 1., 8., 0., 1., cpu_capacity=3.),
         Resource("r1", 1., 8., 200., 1., cpu_capacity=3.)],
        {CarbonClass.CRITICAL: 500., CarbonClass.DEFERRED: 500.,
         CarbonClass.FLEXIBLE: 500.},
        1.,
    )
    fixed_signal = np.mean(replay.intensities, axis=0)
    planning_problem = with_carbon_intensities(initial, fixed_signal)
    assert [r.carbon_intensity for r in planning_problem.resources] == pytest.approx(
        [200., 500. / 3.]
    )
    forecast = CarbonTimeSeries(replay.intensities, .5, coverage_policy="repeat")
    simulator_inputs = SimulatorInputs(
        np.array([[1., 1.], [1., 1.]]),
        PowerModel(np.zeros(2), np.full(2, .3)),
        "allocated_gpu_hours",
        carbon_time_series=forecast,
    )
    qinputs = QPEOInputs(
        simulator_inputs=simulator_inputs,
        performance_affinity=np.ones((2, 2)),
        thermal_headroom=np.ones(2),
        dependency_matrix=np.zeros((2, 2)),
        carbon_replay=None,
    )
    settings = ReconstructionQPEOSettings([1 / 6] * 6, 1 / 3, 1 / 3, 1 / 3, .5, .25)
    suite = run_full_suite(
        planning_problem, qinputs, settings, [1 / 6] * 6,
        seeds=[7], population_size=4, max_iterations=2,
    )
    assert len(suite.records) == 13
    assert len(set(record.algorithm for record in suite.records)) == 13

    for record in suite.records:
        guide = record.result.guide_assignment
        simulation = simulate_fixed_assignment(planning_problem, guide, simulator_inputs)
        independent = account_simulation_result(
            planning_problem, guide, simulator_inputs, simulation, forecast
        )
        assert record.guide_objectives[1] == pytest.approx(independent.total_energy_kwh)
        assert record.guide_objectives[2] == pytest.approx(
            independent.energy_weighted_intensity_g_per_kwh
        )
        # A fixed planning signal (rather than a terminal forecast at 400/100)
        # remains attached to resources throughout the optimizer's search.
        assert [r.carbon_intensity for r in planning_problem.resources] == pytest.approx(
            fixed_signal
        )
