import numpy as np

from qpeo import (
    CarbonClass, ManuscriptQPEOConfig, PowerModel, ProblemInstance, QPEOInputs,
    ReconstructionQPEOSettings, Resource, SimulatorInputs, Workload, run_qpeo,
)

def test_qpeo_end_to_end_small_instance():
    problem = ProblemInstance(
        workloads=[
            Workload("w1",1,.35,3,1,0.0,3.0,CarbonClass.FLEXIBLE),
            Workload("w2",1,.40,3,1,0.2,3.0,CarbonClass.DEFERRED),
        ],
        resources=[
            Resource("g1",1.0,10,220,1.0),
            Resource("g2",1.0,10,380,.7),
        ],
        carbon_thresholds={
            CarbonClass.CRITICAL:250,
            CarbonClass.DEFERRED:450,
            CarbonClass.FLEXIBLE:450,
        },
        reference_cost=3.0,
    )
    sim_inputs = SimulatorInputs(
        service_time_hours=np.array([[1.0,1.5],[1.2,1.0]]),
        power_model=PowerModel(np.array([.08,.06]),np.array([.22,.18])),
        billing_mode="active_resource_hours",
    )
    inputs = QPEOInputs(
        simulator_inputs=sim_inputs,
        performance_affinity=np.array([[1.0,.6],[.7,1.0]]),
        thermal_headroom=np.array([.8,.9]),
        dependency_matrix=np.array([[0,.7],[.7,0]],dtype=float),
    )
    settings = ReconstructionQPEOSettings(
        scalar_weights=[1/6]*6,
        alpha=.5,beta=.2,gamma=.3,
        dependency_threshold=.5,
        dependency_strength=.2,
    )
    cfg = ManuscriptQPEOConfig(population_size=6,max_iterations=4,stagnation_iterations=2)
    result = run_qpeo(problem,inputs,settings,config=cfg,seed=7)
    assert len(result.final_objectives) == 6
    assert len(result.archive.entries) >= 1
    assert result.entropy_history.shape == (4,)
    assert result.archive_size_history.shape == (5,)
    assert np.all(np.isfinite(result.guide_objectives.as_array()))
