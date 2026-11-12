"""Synthetic, exact-answer checks for optional time-integrated carbon accounting."""
import numpy as np
import pytest

from qpeo.carbon_accounting import CarbonTimeSeries, integrate_time_aligned_carbon


def compute(intervals, intensity, *, step=0.5, policy="strict", idle=0.0,
            dynamic=0.3, power=1.0):
    return integrate_time_aligned_carbon(
        [intervals], gpu_capacities=[1.0], idle_power_kw=[idle],
        dynamic_power_kw=[dynamic], power_exponent=power,
        forecast=CarbonTimeSeries(np.asarray(intensity, float).reshape(-1, 1), step,
                                  coverage_policy=policy),
    )


def test_overlapping_workloads_follow_actual_forecast_intervals():
    # GPU utilization is 0.5, 1.0, 0.5 in the three consecutive half hours.
    r = compute([(0.0, 1.0, 0.5), (0.5, 1.5, 0.5)], [100, 200, 300])
    assert r.total_energy_kwh == pytest.approx(0.3)
    assert r.total_carbon_g == pytest.approx(60.0)
    assert r.energy_weighted_intensity_g_per_kwh == pytest.approx(200.0)


def test_terminal_zero_does_not_erase_earlier_nonzero_exposure():
    r = compute([(0.0, 1.5, 1.0)], [100, 200, 0])
    assert r.total_energy_kwh == pytest.approx(0.45)
    assert r.total_carbon_g == pytest.approx(45.0)
    assert r.energy_weighted_intensity_g_per_kwh == pytest.approx(100.0)


def test_strict_policy_rejects_coverage_gap():
    with pytest.raises(ValueError, match="does not cover"):
        compute([(0.0, 2.0, 1.0)], [100, 200, 0])


def test_repeat_policy_is_explicit_and_wraps_forecast():
    r = compute([(0.0, 2.0, 1.0)], [100, 200, 0], policy="repeat")
    assert r.total_energy_kwh == pytest.approx(0.6)
    assert r.total_carbon_g == pytest.approx(60.0)
    assert r.energy_weighted_intensity_g_per_kwh == pytest.approx(100.0)


def test_nonzero_idle_component_only_during_workload_activity():
    r = compute([(0.5, 1.0, 0.5)], [100, 200], idle=0.1, dynamic=0.2)
    assert r.total_energy_kwh == pytest.approx(0.1)
    assert r.total_carbon_g == pytest.approx(20.0)


def test_parallel_resource_carbon_is_energy_weighted():
    forecast = CarbonTimeSeries(np.array([[0., 200.], [100., 200.]]), 0.5)
    r = integrate_time_aligned_carbon(
        [[(0., 1., 1.)], [(0., 1., 1.)]],
        gpu_capacities=[1., 1.], idle_power_kw=[0., 0.],
        dynamic_power_kw=[0.2, 0.4], power_exponent=1.0, forecast=forecast,
    )
    assert r.energy_kwh_by_resource == pytest.approx([0.2, 0.4])
    assert r.carbon_g_by_resource == pytest.approx([10., 80.])
    assert r.total_carbon_g == pytest.approx(90.)
    assert r.energy_weighted_intensity_g_per_kwh == pytest.approx(150.)


def test_illegal_gpu_overlap_rejected():
    with pytest.raises(ValueError, match="concurrent GPU"):
        compute([(0., 1., 0.7), (0.5, 1.5, 0.7)], [100, 100, 100])


def test_forecast_length_for_308_hour_schedule_requires_policy():
    # The frozen 336-row NESO schedule has 168 h of coverage.
    week = np.full((336, 1), 100.0)
    with pytest.raises(ValueError, match="does not cover"):
        integrate_time_aligned_carbon(
            [[(0., 308., 1.)]], gpu_capacities=[1.],
            idle_power_kw=[0.], dynamic_power_kw=[0.3], power_exponent=1.,
            forecast=CarbonTimeSeries(week, 0.5),
        )
    r = integrate_time_aligned_carbon(
        [[(0., 308., 1.)]], gpu_capacities=[1.],
        idle_power_kw=[0.], dynamic_power_kw=[0.3], power_exponent=1.,
        forecast=CarbonTimeSeries(week, 0.5, coverage_policy="repeat"),
    )
    assert r.total_energy_kwh == pytest.approx(92.4)
    assert r.total_carbon_g == pytest.approx(9240.)


def test_adapter_uses_actual_simulated_start_and_finish_times():
    from types import SimpleNamespace as NS
    from qpeo.carbon_accounting import account_simulation_result

    problem = NS(
        workloads=[NS(gpu_demand=0.5), NS(gpu_demand=0.5)],
        resources=[NS(gpu_capacity=1.0)],
    )
    X = np.array([[1], [1]])
    inputs = NS(power_model=NS(idle_power_kw=np.array([0.0]),
                               dynamic_power_kw=np.array([0.3]), power_exponent=1.0))
    result = NS(start_times=np.array([0.0, 0.5]),
                completion_times=np.array([1.0, 1.5]),
                energy_kwh_by_resource=np.array([0.3]))
    forecast = CarbonTimeSeries(np.array([[100.0], [200.0], [300.0]]), 0.5)
    audited = account_simulation_result(problem, X, inputs, result, forecast)
    assert audited.total_energy_kwh == pytest.approx(0.3)
    assert audited.total_carbon_g == pytest.approx(60.)
    assert audited.energy_weighted_intensity_g_per_kwh == pytest.approx(200.)


def test_adapter_rejects_energy_inconsistency():
    from types import SimpleNamespace as NS
    from qpeo.carbon_accounting import account_simulation_result
    problem = NS(workloads=[NS(gpu_demand=1.)], resources=[NS(gpu_capacity=1.)])
    sim = NS(start_times=np.array([0.]), completion_times=np.array([1.]),
             energy_kwh_by_resource=np.array([999.]))
    inputs = NS(power_model=NS(idle_power_kw=np.array([0.]),
                               dynamic_power_kw=np.array([0.3]), power_exponent=1.))
    with pytest.raises(RuntimeError, match="does not match"):
        account_simulation_result(problem, np.array([[1]]), inputs, sim,
                                  CarbonTimeSeries(np.array([[100.]]), 1.))


def test_forecast_origin_must_align_with_job_clock():
    forecast = CarbonTimeSeries(np.array([[100.]]), 1., forecast_start_hours=10.)
    with pytest.raises(ValueError, match="before the carbon"):
        integrate_time_aligned_carbon(
            [[(0., 1., 1.)]], gpu_capacities=[1.], idle_power_kw=[0.],
            dynamic_power_kw=[0.3], power_exponent=1., forecast=forecast,
        )


def test_forecast_defensive_copy_and_cached_integral():
    original = np.array([[50.0, 150.0], [100.0, 250.0]])
    forecast = CarbonTimeSeries(original, 0.5)
    original[0, 0] = 9999.0
    assert forecast.intensities_g_per_kwh[0, 0] == pytest.approx(50.0)
    assert not forecast.intensities_g_per_kwh.flags.writeable
    first = forecast.cumulative_intensity_hours
    assert forecast.cumulative_intensity_hours is first
    assert not first.flags.writeable
    assert np.allclose(first, [[0.0, 0.0], [25.0, 75.0], [75.0, 200.0]])
