"""Opt-in carbon-v2 smoke test exercising all 13 algorithms.

This test is deliberately P=4, T=2 and uses a synthetic 1.5-hour schedule;
it is NOT an additional primary experimental result or an empirical benchmark.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from qpeo.carbon import CarbonReplay
from qpeo.carbon_accounting import CarbonTimeSeries
from qpeo.experiment import run_full_suite
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.optimizer import QPEOInputs, ReconstructionQPEOSettings
from qpeo.simulator import PowerModel, SimulatorInputs


def make_case(*, time_aligned: bool):
    t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
    replay = CarbonReplay(
        (t0, t0 + timedelta(minutes=30), t0 + timedelta(minutes=60)),
        ("r0",), np.array([[100.0], [200.0], [0.0]]),
        "synthetic", t0, t0 + timedelta(minutes=60),
    )
    problem = ProblemInstance(
        [Workload("w", 1, 1.0, 1, 0, 0, 2, CarbonClass.FLEXIBLE)],
        [Resource("r0", 1.0, 8, 100.0, 1.0)],
        {CarbonClass.CRITICAL: 500., CarbonClass.DEFERRED: 500.,
         CarbonClass.FLEXIBLE: 500.},
        1.0,
    )
    forecast = CarbonTimeSeries(
        np.array([[100.0], [200.0], [0.0]]), 0.5,
    ) if time_aligned else None
    sim_inputs = SimulatorInputs(
        service_time_hours=np.array([[1.5]]),
        power_model=PowerModel(np.array([0.0]), np.array([0.3])),
        billing_mode="allocated_gpu_hours",
        carbon_time_series=forecast,
    )
    qinputs = QPEOInputs(
        simulator_inputs=sim_inputs,
        performance_affinity=np.array([[1.0]]),
        thermal_headroom=np.array([1.0]),
        dependency_matrix=np.zeros((1, 1)),
        carbon_replay=replay,
    )
    settings = ReconstructionQPEOSettings([1 / 6] * 6, 1 / 3, 1 / 3, 1 / 3, .5, .25)
    return problem, qinputs, settings


def test_13_algorithm_opt_in_uses_execution_time_not_terminal_snapshot():
    problem, qinputs, settings = make_case(time_aligned=True)
    suite = run_full_suite(
        problem, qinputs, settings, [1 / 6] * 6,
        seeds=[7], population_size=4, max_iterations=2,
    )
    assert len(suite.records) == 13
    for record in suite.records:
        # One GPU at 0.3 kW for 1.5h: 0.45kWh,
        # forecast carbon: 0.15kWh*(100+200+0)=45g, hence 100g/kWh.
        assert record.guide_objectives[1] == pytest.approx(0.45)
        assert record.guide_objectives[2] == pytest.approx(100.0)


def test_legacy_mode_keeps_original_terminal_snapshot_semantics():
    problem, qinputs, settings = make_case(time_aligned=False)
    suite = run_full_suite(
        problem, qinputs, settings, [1 / 6] * 6,
        seeds=[7], population_size=4, max_iterations=2,
    )
    assert len(suite.records) == 13
    for record in suite.records:
        assert record.guide_objectives[2] == pytest.approx(0.0)
