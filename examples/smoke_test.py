import numpy as np
from qpeo import *
problem=ProblemInstance(
    [Workload("w1",2,.4,4,1,0,4,CarbonClass.FLEXIBLE),
     Workload("w2",4,.5,6,2,1,6,CarbonClass.DEFERRED)],
    [Resource("g1",1,16,250,1.0),Resource("g2",1,16,400,.8)],
    {CarbonClass.CRITICAL:300,CarbonClass.DEFERRED:450,CarbonClass.FLEXIBLE:450},
    2.0)
X=np.array([[1,0],[0,1]])
state=EvaluationState(np.array([3.5,5.0]),np.array([.9,1.1]),np.array([1.0,1.2]),np.array([1.0,1.2]))
print(check_constraints(problem,X))
print(evaluate_objectives(problem,X,state))
