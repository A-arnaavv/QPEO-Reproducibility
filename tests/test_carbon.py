"""Regression checks for the locked NESO carbon replay used in final runs."""

from pathlib import Path

import numpy as np

from qpeo.carbon import load_neso_regional_replay, with_carbon_intensities
from qpeo.models import CarbonClass, Workload
from qpeo.resource_environment import load_resource_environment


ROOT = Path(__file__).resolve().parents[1]


def _environment_and_replay():
    env = load_resource_environment(
        ROOT / "configs" / "resource_environment_v1.json"
    )
    replay = load_neso_regional_replay(
        ROOT / "data" / "carbon" / "neso_regional_2026-01-05_2026-01-11.csv",
        ROOT / "configs" / "carbon_replay_v1.json",
        expected_slot_names=[s.name for s in env.slots],
    )
    return env, replay


def test_locked_neso_carbon_replay_capacity_and_thresholds():
    env, replay = _environment_and_replay()
    assert replay.rows == 336
    assert replay.resources == len(env.slots) == 7
    assert replay.intensities.shape == (336, 7)
    assert np.all(np.isfinite(replay.intensities))
    assert np.all(replay.intensities >= 0)
    thresholds = replay.thresholds()
    assert np.all(
        (replay.intensities <= thresholds[CarbonClass.CRITICAL]).sum(axis=1) >= 1
    )
    assert np.all(
        (replay.intensities <= thresholds[CarbonClass.DEFERRED]).sum(axis=1) >= 3
    )
    assert np.all(
        (replay.intensities <= thresholds[CarbonClass.FLEXIBLE]).sum(axis=1) == 7
    )


def test_replay_iteration_endpoints_and_problem_carbon_refresh():
    env, replay = _environment_and_replay()
    first = replay.at_iteration(0, 500)
    last = replay.at_iteration(500, 500)
    assert np.array_equal(first, replay.intensities[0])
    assert np.array_equal(last, replay.intensities[-1])

    workload = Workload(
        "test", 1, 0.2, 1, 0, 0, 2, CarbonClass.FLEXIBLE
    )
    initial = env.build_problem([workload], first, replay.thresholds())
    updated = with_carbon_intensities(initial, last)
    assert np.allclose(
        [r.carbon_intensity for r in updated.resources], last
    )
    assert np.array_equal(
        [r.cost_per_gpu_hour for r in updated.resources],
        [r.cost_per_gpu_hour for r in initial.resources],
    )
