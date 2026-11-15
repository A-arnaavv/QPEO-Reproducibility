import numpy as np
import pytest

from qpeo import (
    CarbonClass,
    PowerModel,
    ProblemInstance,
    Resource,
    SimulatorInputs,
    Workload,
    simulate_fixed_assignment,
)


def test_concurrent_cpu_capacity_queues_second_workload():
    problem = ProblemInstance(
        workloads=[
            Workload("job1", 0.75, 0.2, 1.0, 0, 0.0, 10.0, CarbonClass.FLEXIBLE),
            Workload("job2", 0.75, 0.2, 1.0, 0, 0.0, 10.0, CarbonClass.FLEXIBLE),
        ],
        resources=[
            Resource("gpu1", 1.0, 64.0, 100.0, 1.0, cpu_capacity=1.0),
        ],
        carbon_thresholds={
            CarbonClass.CRITICAL: 200.0,
            CarbonClass.DEFERRED: 200.0,
            CarbonClass.FLEXIBLE: 200.0,
        },
        reference_cost=1.0,
    )

    inputs = SimulatorInputs(
        service_time_hours=np.array([[2.0], [2.0]]),
        power_model=PowerModel(
            idle_power_kw=np.array([0.0]),
            dynamic_power_kw=np.array([0.2]),
        ),
        billing_mode="allocated_gpu_hours",
    )

    assignment = np.ones((2, 1), dtype=int)
    result = simulate_fixed_assignment(problem, assignment, inputs)

    assert result.start_times == pytest.approx([0.0, 2.0])
    assert result.completion_times == pytest.approx([2.0, 4.0])
