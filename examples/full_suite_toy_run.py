"""Tiny end-to-end validation of all 13 algorithms.

This is NOT a paper experiment. It exists only to validate the rebuilt harness.
"""
import numpy as np

from qpeo.experiment import run_full_suite
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.optimizer import QPEOInputs, ReconstructionQPEOSettings
from qpeo.simulator import PowerModel, SimulatorInputs

workloads = [
    Workload(f"w{i}", 1.0, 0.18, 1.0, 1.0, i*0.1, 3.0+i*0.1, CarbonClass.FLEXIBLE)
    for i in range(4)
]
resources = [
    Resource("edge-low-carbon", 1.0, 8.0, 180.0, 1.1),
    Resource("cloud-fast", 1.0, 8.0, 330.0, 0.9),
    Resource("serverless", 1.0, 8.0, 420.0, 0.7),
]
problem = ProblemInstance(
    workloads, resources,
    {
        CarbonClass.CRITICAL: 250.0,
        CarbonClass.DEFERRED: 400.0,
        CarbonClass.FLEXIBLE: 500.0,
    },
    reference_cost=3.0,
)
service = np.array([
    [1.1, .8, .9],
    [1.0, .9, .8],
    [1.2, .7, .9],
    [.9, .8, 1.0],
])
sim = SimulatorInputs(
    service,
    PowerModel(np.array([.04,.06,.05]), np.array([.16,.24,.20]), 1.0),
    "allocated_gpu_hours",
)
perf = 1.0 / service
perf = perf / perf.max()
qinputs = QPEOInputs(
    sim,
    performance_affinity=perf,
    thermal_headroom=np.array([.9,.8,.85]),
    dependency_matrix=np.zeros((4,4)),
)
weights=[1/6]*6
settings=ReconstructionQPEOSettings(
    scalar_weights=weights,
    alpha=.5,beta=.25,gamma=.25,
    dependency_threshold=.5,dependency_strength=.2,
)

suite=run_full_suite(
    problem,qinputs,settings,weights,
    seeds=[11,22],population_size=10,max_iterations=5,
)
print("algorithm, seed, runtime_s, f1, f2, f3, f4, f5, f6, archive")
for r in suite.records:
    vals=", ".join(f"{v:.6g}" for v in r.guide_objectives)
    print(f"{r.algorithm}, {r.seed}, {r.runtime_seconds:.4f}, {vals}, {r.archive_size}")
