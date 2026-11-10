import numpy as np
from qpeo import (
    CarbonClass, ManuscriptQPEOConfig, PowerModel, ProblemInstance, QPEOInputs,
    ReconstructionQPEOSettings, Resource, SimulatorInputs, Workload, run_qpeo,
)

problem=ProblemInstance(
    workloads=[
        Workload("w1",1,.35,3,1,0.0,3.0,CarbonClass.FLEXIBLE),
        Workload("w2",1,.40,3,1,0.2,3.0,CarbonClass.DEFERRED),
    ],
    resources=[Resource("g1",1,10,220,1.0),Resource("g2",1,10,380,.7)],
    carbon_thresholds={CarbonClass.CRITICAL:250,CarbonClass.DEFERRED:450,CarbonClass.FLEXIBLE:450},
    reference_cost=3.0,
)
inputs=QPEOInputs(
    simulator_inputs=SimulatorInputs(
        service_time_hours=np.array([[1.0,1.5],[1.2,1.0]]),
        power_model=PowerModel(np.array([.08,.06]),np.array([.22,.18])),
        billing_mode="active_resource_hours",
    ),
    performance_affinity=np.array([[1.0,.6],[.7,1.0]]),
    thermal_headroom=np.array([.8,.9]),
    dependency_matrix=np.array([[0,.7],[.7,0]],float),
)
settings=ReconstructionQPEOSettings(
    scalar_weights=[1/6]*6,alpha=.5,beta=.2,gamma=.3,
    dependency_threshold=.5,dependency_strength=.2,
)
res=run_qpeo(problem,inputs,settings,config=ManuscriptQPEOConfig(population_size=8,max_iterations=10,stagnation_iterations=3),seed=42)
print("guide objectives:",res.guide_objectives.as_array())
print("guide assignment:\n",res.guide_assignment)
print("pareto archive size:",len(res.archive.entries))
print("entropy history:",np.round(res.entropy_history,4))
