"""Unit checks for exact, mutually exclusive SLA miss attribution."""
import numpy as np
import pytest

from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from scripts.audit_carbon_v2_sla_miss_attribution import classify_schedule


def _test_problem():
    jobs = (
        Workload('intrinsic', 1., .5, 1., 0., 0., .5, CarbonClass.FLEXIBLE),
        Workload('slow_choice', 1., .5, 1., 0., 0., 1.5, CarbonClass.FLEXIBLE),
        Workload('waited', 1., .5, 1., 0., 0., 1.5, CarbonClass.FLEXIBLE),
        Workload('met', 1., .5, 1., 0., 0., 1.5, CarbonClass.FLEXIBLE),
        Workload('no_deadline', 1., .5, 1., 0., 0., float('inf'), CarbonClass.FLEXIBLE),
    )
    pb = ProblemInstance(
        jobs,
        (Resource('fast', 1., 8., 70., 1., cpu_capacity=4.),
         Resource('slow', 1., 8., 80., 1., cpu_capacity=4.)),
        {c: 100. for c in CarbonClass}, 1.,
    )
    service = np.array([[1., 2.]] * len(jobs), float)
    assignment = np.array([0, 1, 0, 0, 1], int)
    start = np.array([0., 0., .75, 0., 0.], float)
    finish = np.array([1., 2., 1.75, 1., 2.], float)
    return pb, service, assignment, start, finish


def test_exact_partition_of_unavoidable_assignment_delay_and_met():
    out = classify_schedule(*_test_problem())
    assert out['finite_deadlines'] == 4
    assert out['no_finite_deadline'] == 1
    assert out['missed'] == 3
    assert out['intrinsically_impossible'] == 1
    assert out['assigned_resource_too_slow'] == 1
    assert out['start_delay_on_fast_enough_assignment'] == 1
    assert out['met'] == 1
    assert out['numerical_boundary_misses'] == 0
    assert out['observed_f5_percent'] == pytest.approx(75.)


def test_roundoff_lateness_is_not_misreported_as_queue_miss():
    pb, service, assignment, start, finish = _test_problem()
    start[3] = 0.5 + 5e-10
    finish[3] = 1.5 + 5e-10
    out = classify_schedule(pb, service, assignment, start, finish)
    assert out['missed'] == 4
    assert out['numerical_boundary_misses'] == 1
    assert out['start_delay_on_fast_enough_assignment'] == 1
    assert out['observed_f5_percent'] == pytest.approx(100.)


def test_reject_ineligible_resource_and_wrong_service_duration():
    pb, service, assignment, start, finish = _test_problem()
    wrong_finish = finish.copy()
    wrong_finish[0] += .1
    with pytest.raises(ValueError, match='service intervals'):
        classify_schedule(pb, service, assignment, start, wrong_finish)
    from dataclasses import replace
    wrong_pb = replace(pb, resources=(pb.resources[0], replace(pb.resources[1], carbon_intensity=200.)))
    with pytest.raises(ValueError, match='eligibility'):
        classify_schedule(wrong_pb, service, assignment, start, finish)
