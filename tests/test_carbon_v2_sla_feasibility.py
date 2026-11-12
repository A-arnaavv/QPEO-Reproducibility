"""Validate the optimistic SLA bound and excluded infinite-deadline jobs."""
import numpy as np
import pytest

from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from scripts.audit_carbon_v2_sla_feasibility import audit_instance


def test_structural_sla_bound_rejects_ineligible_fastest_resource():
    pb = ProblemInstance(
        [Workload('c', 0., 1., 0., 0., 2., 3., CarbonClass.CRITICAL)],
        [Resource('fast-dirty', 1., 1., 400., 1.),
         Resource('slow-clean', 1., 1., 80., 1.)],
        {CarbonClass.CRITICAL: 100., CarbonClass.DEFERRED: 500., CarbonClass.FLEXIBLE: 500.}, 1.,
    )
    r=audit_instance(pb, np.array([[.5, 2.0]]))
    assert r['optimistically_impossible_jobs'] == 1
    assert r['optimistic_sla_violation_floor_pct'] == pytest.approx(100.)
    assert r['zero_slack_jobs'] == 0


def test_exact_deadline_and_infinite_deadline_handling():
    pb = ProblemInstance(
        [Workload('tight', 0., 1., 0., 0., 0., 1., CarbonClass.FLEXIBLE),
         Workload('spot', 0., 1., 0., 0., 0., float('inf'), CarbonClass.FLEXIBLE)],
        [Resource('r', 1., 1., 50., 1.)],
        {CarbonClass.CRITICAL: 100., CarbonClass.DEFERRED: 100., CarbonClass.FLEXIBLE: 100.}, 1.,
    )
    r=audit_instance(pb, np.array([[1.0], [3.0]]))
    assert r['finite_deadlines'] == 1
    assert r['no_deadline_jobs'] == 1
    assert r['zero_slack_jobs'] == 1
    assert r['optimistically_impossible_jobs'] == 0


def test_nonpositive_service_times_rejected():
    pb = ProblemInstance(
        [Workload('a', 0., 1., 0., 0., 0., 1., CarbonClass.FLEXIBLE)],
        [Resource('r', 1., 1., 50., 1.)],
        {CarbonClass.CRITICAL: 100., CarbonClass.DEFERRED: 100., CarbonClass.FLEXIBLE: 100.}, 1.,
    )
    with pytest.raises(ValueError, match='positive finite service'):
        audit_instance(pb, np.array([[0.]]))
