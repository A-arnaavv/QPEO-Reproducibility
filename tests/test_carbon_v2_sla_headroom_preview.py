"""Proposed SLA headroom policy does not alter locked trace instances or outputs."""
import numpy as np
import pytest

from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from scripts.audit_carbon_v2_sla_feasibility import audit_instance
from scripts.preview_carbon_v2_sla_headroom import (
    HEADROOM_FRACTIONS, make_headroom_variant, minimum_eligible_service_hours,
)


def _fixture():
    problem = ProblemInstance(
        workloads=(
            Workload("impossible", 1., 1., 1., 0., 0., 0.5, CarbonClass.CRITICAL),
            Workload("tight", 1., 1., 1., 0., 2., 3., CarbonClass.CRITICAL),
            Workload("unbounded", 1., 1., 1., 0., 0., float("inf"), CarbonClass.FLEXIBLE),
            Workload("loose", 1., 1., 1., 0., 0., 50., CarbonClass.FLEXIBLE),
        ),
        resources=(
            Resource("fast-ineligible", 1., 2., 400., 1., cpu_capacity=2.),
            Resource("eligible", 1., 2., 50., 1., cpu_capacity=2.),
        ),
        carbon_thresholds={c: 100. for c in CarbonClass},
        reference_cost=1.,
    )
    service = np.array([[0.1, 1.], [0.1, 1.], [0.1, 2.], [0.1, 1.]])
    return problem, service


def test_headroom_never_changes_locked_input_or_infinite_deadline():
    problem, service = _fixture()
    orig_deadlines = [w.deadline for w in problem.workloads]
    variant, report = make_headroom_variant(problem, service, 0.25)
    assert [w.deadline for w in problem.workloads] == orig_deadlines
    assert variant is not problem
    assert variant.resources is problem.resources
    assert [w.deadline for w in variant.workloads] == pytest.approx([1.25, 3.25, float("inf"), 50.])
    assert report["modified_finite_deadlines"] == 2
    assert report["unchanged_infinite_deadlines"] == 1
    assert report["optimistically_impossible_after"] == 0
    assert report["zero_slack_after"] == 0
    assert audit_instance(problem, service)["optimistically_impossible_jobs"] == 1


def test_headroom_nested_monotonic_and_separate_policy():
    problem, service = _fixture()
    deadlines = []
    for factor in HEADROOM_FRACTIONS:
        variant, diag = make_headroom_variant(problem, service, factor)
        deadlines.append([w.deadline for w in variant.workloads])
        assert diag["optimistically_impossible_after"] == 0
        assert diag["zero_slack_after"] == 0
    assert np.all(np.asarray(deadlines[1]) >= np.asarray(deadlines[0]))
    assert np.all(np.asarray(deadlines[2]) >= np.asarray(deadlines[1]))


def test_headroom_positive_margins_and_individually_eligible_resource_required():
    problem, service = _fixture()
    for factor in (0., -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            make_headroom_variant(problem, service, factor)
    with pytest.raises(ValueError, match="positive"):
        minimum_eligible_service_hours(problem, np.zeros_like(service))
    impossible = ProblemInstance(problem.workloads, (
        Resource("only-dirty", 1., 2., 400., 1., cpu_capacity=2.),
    ), problem.carbon_thresholds, 1.)
    with pytest.raises(ValueError, match="No individually eligible"):
        make_headroom_variant(impossible, service[:, :1], 0.25)
