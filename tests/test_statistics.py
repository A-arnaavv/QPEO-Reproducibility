import numpy as np
from qpeo.statistics import (
    approximate_hypervolume, bonferroni_alpha, friedman_test,
    iterations_to_fraction_of_final, mean_ci, nemenyi_posthoc,
    paired_cohens_dz, wilcoxon_against_reference,
)

def test_bonferroni_72():
    assert bonferroni_alpha(.01,72) == pytest_approx(.01/72)

def pytest_approx(x):
    import pytest
    return pytest.approx(x)

def test_mean_ci_and_effect():
    m=mean_ci([1,2,3,4]); assert m.n==4; assert m.ci_low<m.mean<m.ci_high
    d=paired_cohens_dz([1,1,1,1],[2,2.1,1.9,2.0]); assert d<0

def test_wilcoxon_friedman_nemenyi():
    samples={"A":[1,1.1,.9,1.0],"B":[2,2.1,1.9,2.0],"C":[3,3.1,2.9,3.0]}
    w=wilcoxon_against_reference(samples["A"],{"B":samples["B"]}); assert len(w)==1
    f=friedman_test(samples); assert f.average_ranks["A"] < f.average_ranks["C"]
    n=nemenyi_posthoc(samples); assert len(n)==3

def test_iterations_to_90_percent():
    assert iterations_to_fraction_of_final([10,5,2,1,0],fraction=.9,minimize=True)==4

def test_approx_hypervolume_ordering():
    ref=np.array([2.,2.]); ideal=np.array([0.,0.])
    hv1=approximate_hypervolume(np.array([[1.,1.]]),ref,ideal_point=ideal,samples_power=12)
    hv2=approximate_hypervolume(np.array([[.5,.5]]),ref,ideal_point=ideal,samples_power=12)
    assert hv2>hv1>0
