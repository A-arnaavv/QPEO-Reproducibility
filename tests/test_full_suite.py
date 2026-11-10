import numpy as np
from qpeo.experiment import run_full_suite
from qpeo.models import CarbonClass,ProblemInstance,Resource,Workload
from qpeo.optimizer import QPEOInputs,ReconstructionQPEOSettings
from qpeo.simulator import PowerModel,SimulatorInputs

def test_full_suite_has_13_algorithms():
    w=[Workload("a",1,.2,1,1,0,2,CarbonClass.FLEXIBLE),Workload("b",1,.2,1,1,.1,2.1,CarbonClass.FLEXIBLE)]
    r=[Resource("g0",1,8,200,1),Resource("g1",1,8,350,.8)]
    p=ProblemInstance(w,r,{CarbonClass.CRITICAL:250,CarbonClass.DEFERRED:400,CarbonClass.FLEXIBLE:500},2)
    sim=SimulatorInputs(np.array([[1,1.2],[1.1,.9]]),PowerModel(np.array([.05,.05]),np.array([.2,.18])),"allocated_gpu_hours")
    qi=QPEOInputs(sim,np.array([[1,.8],[.8,1]]),np.array([.8,.8]),np.zeros((2,2)))
    qs=ReconstructionQPEOSettings([1/6]*6,.5,.25,.25,.5,.2)
    suite=run_full_suite(p,qi,qs,[1/6]*6,seeds=[2],population_size=6,max_iterations=1)
    assert len(suite.algorithms)==13
    assert len(suite.records)==13
