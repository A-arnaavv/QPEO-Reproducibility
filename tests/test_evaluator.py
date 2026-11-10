import numpy as np, pytest
from qpeo import *

@pytest.fixture
def problem():
    return ProblemInstance(
        [Workload("w1",1,.4,4,1,0,4,CarbonClass.FLEXIBLE),
         Workload("w2",1,.5,6,1,1,6,CarbonClass.DEFERRED)],
        [Resource("g1",1,16,250,1.0), Resource("g2",1,16,400,.8)],
        {CarbonClass.CRITICAL:300,CarbonClass.DEFERRED:450,CarbonClass.FLEXIBLE:450},
        2.0)

def test_feasible(problem):
    assert check_constraints(problem,np.array([[1,0],[0,1]])).feasible

def test_objectives(problem):
    X=np.array([[1,0],[0,1]])
    s=EvaluationState(np.array([3.5,5.0]),np.array([.9,1.1]),np.array([1.0,1.2]))
    o=evaluate_objectives(problem,X,s)
    assert o.f1_negative_gpu_util==pytest.approx(-.45)
    assert o.f2_energy_kwh==pytest.approx(2.0)
    assert o.f3_energy_weighted_carbon_intensity==pytest.approx((.9*250+1.1*400)/2)
    assert o.f4_makespan==pytest.approx(5.0)
    assert o.f5_sla_violation_percent==pytest.approx(0)
    assert o.f6_normalized_cost==pytest.approx((1+.8*1.2)/2)

def test_scalarization():
    F=np.array([[1,3,4,9,2,5],[2,1,8,5,4,3]],float)
    vals=scalar_fitness(normalize_population_objectives(F),[1/6]*6)
    assert vals.shape==(2,)

def test_dominance():
    assert dominates(np.ones(6),np.ones(6)*2)
