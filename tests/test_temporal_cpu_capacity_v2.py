"""Regression tests for v2 scheduling's temporal aggregate CPU limits.

These are small deterministic simulator tests; they do not rerun any published
experimental cohort. v1 committed results must not be reused for v2 claims.
"""
import numpy as np
import pytest

from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.simulator import PowerModel, SimulatorInputs, clear_simulation_cache, simulate_fixed_assignment


def resource_case(cpu_capacity, cpu_demands, *, arrivals=None, gpu_demands=None):
    m = len(cpu_demands)
    arrival = arrivals if arrivals is not None else [0.0] * m
    gpu = gpu_demands if gpu_demands is not None else [0.3] * m
    w = [
        Workload(f"w{i}", float(cpu), float(gpu[i]), 1.0, 0.0,
                 float(arrival[i]), 100.0, CarbonClass.FLEXIBLE)
        for i, cpu in enumerate(cpu_demands)
    ]
    p = ProblemInstance(
        w, [Resource("r", 1.0, 64., 100., 1., cpu_capacity=cpu_capacity)],
        {CarbonClass.CRITICAL: 500., CarbonClass.DEFERRED: 500.,
         CarbonClass.FLEXIBLE: 500.},
        1.0,
    )
    x = np.ones((m, 1), dtype=int)
    sim = SimulatorInputs(np.ones((m, 1)),
                          PowerModel(np.zeros(1), np.array([0.3])),
                          "allocated_gpu_hours")
    clear_simulation_cache()
    return p, x, sim


def test_cpu_capacity_queues_jobs_even_when_gpu_and_memory_fit():
    # GPU overlap is possible (0.3+0.3<=1), but CPU overlap is not (7+7>8).
    p, x, sim = resource_case(8., [7., 7.], arrivals=[0., 0.25])
    out = simulate_fixed_assignment(p, x, sim)
    assert out.start_times == pytest.approx([0., 1.])
    assert out.completion_times == pytest.approx([1., 2.])


def test_cpu_capacity_allows_two_but_queues_third():
    p, x, sim = resource_case(8., [3., 3., 3.])
    out = simulate_fixed_assignment(p, x, sim)
    assert out.start_times == pytest.approx([0., 0., 1.])
    assert out.completion_times == pytest.approx([1., 1., 2.])


def test_none_cpu_capacity_preserves_gpu_bounded_overlaps():
    p, x, sim = resource_case(None, [7., 7.])
    out = simulate_fixed_assignment(p, x, sim)
    assert out.start_times == pytest.approx([0., 0.])


def test_individual_cpu_overflow_rejected():
    p, x, sim = resource_case(8., [9.])
    with pytest.raises(ValueError, match="CPU-capacity"):
        simulate_fixed_assignment(p, x, sim)


def test_cpu_only_queuing_still_gives_energy_from_real_intervals():
    p, x, sim = resource_case(8., [7., 7.], arrivals=[0., 0.25])
    out = simulate_fixed_assignment(p, x, sim)
    # Each 0.3 GPU fraction runs one hour; dynamic-only 0.3 kW at full util.
    assert out.energy_kwh_by_resource[0] == pytest.approx(0.18)
    assert out.allocated_gpu_hours[0] == pytest.approx(0.6)
