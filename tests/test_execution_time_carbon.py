from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from qpeo import (
    CarbonClass,
    PowerModel,
    ProblemInstance,
    Resource,
    SimulatorInputs,
    Workload,
    evaluate_objectives,
    simulate_fixed_assignment,
)
from qpeo.carbon import CarbonReplay
from qpeo.simulator import clear_simulation_cache


@pytest.mark.parametrize(
    "arrival, expected_emissions, expected_intensity",
    [
        (0.0, 20.0, 200.0),
        (0.5, 30.0, 300.0),
    ],
)
def test_carbon_follows_execution_time(
    arrival, expected_emissions, expected_intensity
):
    clear_simulation_cache()

    start = datetime(2026, 1, 5, tzinfo=timezone.utc)

    replay = CarbonReplay(
        timestamps_utc=tuple(
            start + timedelta(minutes=30 * i)
            for i in range(4)
        ),
        slot_names=("gpu1",),
        intensities=np.array([
            [100.0],
            [300.0],
            [300.0],
            [300.0],
        ]),
        source_name="controlled-test",
        window_start_utc=start,
        window_end_utc=start + timedelta(hours=2),
    )
    replay.validate()

    problem = ProblemInstance(
        workloads=[
            Workload(
                "job1", 1.0, 0.2, 1.0, 0,
                arrival, arrival + 2.0,
                CarbonClass.FLEXIBLE,
            ),
        ],
        resources=[
            Resource(
                "gpu1", 1.0, 8.0, 100.0, 1.0,
                cpu_capacity=2.0,
            ),
        ],
        carbon_thresholds={
            CarbonClass.CRITICAL: 500.0,
            CarbonClass.DEFERRED: 500.0,
            CarbonClass.FLEXIBLE: 500.0,
        },
        reference_cost=1.0,
    )

    inputs = SimulatorInputs(
        service_time_hours=np.array([[1.0]]),
        power_model=PowerModel(
            idle_power_kw=np.array([0.0]),
            dynamic_power_kw=np.array([0.5]),
        ),
        billing_mode="allocated_gpu_hours",
    )

    assignment = np.array([[1]], dtype=int)

    result = simulate_fixed_assignment(
        problem,
        assignment,
        inputs,
        carbon_replay=replay,
    )

    assert result.energy_kwh_by_resource[0] == pytest.approx(
        0.1
    )

    assert result.carbon_emissions_g_by_resource[0] == (
        pytest.approx(expected_emissions)
    )

    objectives = evaluate_objectives(
        problem, assignment, result.state
    )

    assert objectives.f3_energy_weighted_carbon_intensity == (
        pytest.approx(expected_intensity)
    )
