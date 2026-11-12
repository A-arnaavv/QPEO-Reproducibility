"""Regression checks using the real frozen NESO week in the QPEO repository.

These are accounting tests on synthetic constant-load schedules; they DO NOT
constitute complete 13-algorithm production-trace experiments.
"""
from pathlib import Path
import numpy as np
import pytest

from qpeo.carbon import load_neso_regional_replay
from qpeo.carbon_accounting import CarbonTimeSeries, integrate_time_aligned_carbon
from qpeo.resource_environment import load_resource_environment

ROOT = Path(__file__).resolve().parents[1]


def frozen_week():
    env = load_resource_environment(ROOT / "configs" / "resource_environment_v1.json")
    replay = load_neso_regional_replay(
        ROOT / "data" / "carbon" / "neso_regional_2026-01-05_2026-01-11.csv",
        ROOT / "configs" / "carbon_replay_v1.json",
        expected_slot_names=[s.name for s in env.slots],
    )
    assert replay.intensities.shape == (336, 7)
    assert replay.intensities[-1, 0] == pytest.approx(0.0)
    return replay


def synthetic_north_scotland(hours, *, mode):
    replay = frozen_week()
    forecast = CarbonTimeSeries(replay.intensities, .5, coverage_policy=mode)
    intervals = [[(0.0, hours, 1.0)]] + [[] for _ in range(6)]
    return integrate_time_aligned_carbon(
        intervals,
        gpu_capacities=np.ones(7),
        idle_power_kw=np.zeros(7),
        dynamic_power_kw=np.full(7, .3),
        power_exponent=1.0,
        forecast=forecast,
    )


def test_frozen_forecast_real_first_16_4_hours_not_terminal_zero():
    r = synthetic_north_scotland(16.4, mode="strict")
    assert r.total_energy_kwh == pytest.approx(4.92)
    assert r.total_carbon_g == pytest.approx(502.23)
    assert r.energy_weighted_intensity_g_per_kwh == pytest.approx(102.07926829268293)


def test_308h_strict_fails_and_explicit_repeat_is_calculable():
    with pytest.raises(ValueError, match="does not cover"):
        synthetic_north_scotland(308.0, mode="strict")
    r = synthetic_north_scotland(308.0, mode="repeat")
    assert r.total_energy_kwh == pytest.approx(92.4)
    assert r.total_carbon_g == pytest.approx(9298.35)
    assert r.energy_weighted_intensity_g_per_kwh == pytest.approx(100.63149350649351)
