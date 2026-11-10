import numpy as np
from qpeo import (
    CarbonClass, PowerModel, ProblemInstance, Resource, SimulatorInputs,
    Workload, evaluate_objectives, simulate_fixed_assignment,
)

problem = ProblemInstance(
    workloads=[
        Workload("train-A",4,0.40,8,2,0.0,4.0,CarbonClass.FLEXIBLE),
        Workload("infer-B",2,0.50,6,1,1.0,4.0,CarbonClass.DEFERRED),
    ],
    resources=[
        Resource("cloud-gpu",1.0,24,250,1.10),
        Resource("edge-gpu",1.0,16,400,0.70),
    ],
    carbon_thresholds={
        CarbonClass.CRITICAL:300,
        CarbonClass.DEFERRED:450,
        CarbonClass.FLEXIBLE:450,
    },
    reference_cost=3.0,
)
X=np.array([[1,0],[1,0]])
inputs=SimulatorInputs(
    service_time_hours=np.array([[2.0,3.5],[1.5,2.5]]),
    power_model=PowerModel(
        idle_power_kw=np.array([0.10,0.07]),
        dynamic_power_kw=np.array([0.25,0.18]),
    ),
    billing_mode="active_resource_hours",
)
result=simulate_fixed_assignment(problem,X,inputs)
print("completion:",result.completion_times)
print("energy:",result.energy_kwh_by_resource)
print("active GPU hours:",result.resource_active_hours)
print("objectives:",evaluate_objectives(problem,X,result.state).as_array())
