from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from qpeo.baselines import (
    CommonBaselineConfig,
    _problem_for_iteration,
)
from qpeo.carbon import CarbonReplay
from qpeo.experiment import run_full_suite
from qpeo.models import (
    CarbonClass,
    ProblemInstance,
    Resource,
    Workload,
)
from qpeo.optimizer import (
    QPEOInputs,
    ReconstructionQPEOSettings,
    _qpeo_problem_for_iteration,
)
from qpeo.simulator import PowerModel, SimulatorInputs


def make_case():
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)

    replay = CarbonReplay(
        timestamps_utc=tuple(
            start + timedelta(minutes=30 * i)
            for i in range(3)
        ),
        slot_names=("r0",),
        intensities=np.array([
            [100.0],
            [200.0],
            [300.0],
        ]),
        source_name="controlled-test",
        window_start_utc=start,
        window_end_utc=start + timedelta(hours=1),
    )

    problem = ProblemInstance(
        workloads=[
            Workload(
                "job", 1.0, 0.2, 1.0, 0,
                0.0, 2.0, CarbonClass.FLEXIBLE,
            )
        ],
        resources=[
            Resource("r0", 1.0, 8.0, 100.0, 1.0),
        ],
        carbon_thresholds={
            cls: 500.0 for cls in CarbonClass
        },
        reference_cost=1.0,
    )

    simulator = SimulatorInputs(
        service_time_hours=np.array([[1.0]]),
        power_model=PowerModel(
            idle_power_kw=np.array([0.0]),
            dynamic_power_kw=np.array([0.2]),
        ),
        billing_mode="allocated_gpu_hours",
    )

    inputs = QPEOInputs(
        simulator_inputs=simulator,
        performance_affinity=np.array([[1.0]]),
        thermal_headroom=np.array([1.0]),
        dependency_matrix=np.zeros((1, 1)),
        carbon_replay=replay,
    )

    settings = ReconstructionQPEOSettings(
        [1 / 6] * 6,
        1 / 3, 1 / 3, 1 / 3,
        0.5, 0.25,
    )

    return problem, inputs, settings


def test_planning_intensity_remains_fixed():
    problem, inputs, _ = make_case()
    config = CommonBaselineConfig(
        population_size=4,
        max_iterations=2,
        carbon_replay=inputs.carbon_replay,
    )

    # Mean of the three observations: 200 gCO2e/kWh.
    for iteration in (0, 1, 2):
        qproblem = _qpeo_problem_for_iteration(
            problem, inputs, iteration, 2
        )
        bproblem = _problem_for_iteration(
            problem, config, iteration
        )

        assert qproblem.resources[0].carbon_intensity == (
            pytest.approx(200.0)
        )
        assert bproblem.resources[0].carbon_intensity == (
            pytest.approx(200.0)
        )


def test_all_13_algorithms_use_execution_time_carbon():
    problem, inputs, settings = make_case()

    suite = run_full_suite(
        problem,
        inputs,
        settings,
        [1 / 6] * 6,
        seeds=[7],
        population_size=4,
        max_iterations=2,
    )

    assert len(suite.records) == 13

    # The one-hour job consumes equal energy in each half-hour:
    # 0-0.5 h: 100 gCO2e/kWh
    # 0.5-1 h: 200 gCO2e/kWh
    # Energy-weighted result: 150 gCO2e/kWh.
    for record in suite.records:
        history = record.result.guide_objective_history

        assert history.shape == (3, 6)
        assert history[:, 2] == pytest.approx(
            [150.0, 150.0, 150.0]
        ), record.algorithm

def test_qpeo_refreshes_population_for_changing_kappa(monkeypatch):
    from qpeo.config import ManuscriptQPEOConfig
    from qpeo.operators import kappa_schedule
    import qpeo.optimizer as optimizer

    problem, inputs, settings = make_case()
    config = ManuscriptQPEOConfig(
        population_size=4,
        max_iterations=2,
    )

    observed_kappa = []
    original_evaluate = optimizer._evaluate_theta

    def recorded_evaluate(*args, **kwargs):
        observed_kappa.append(float(kwargs["kappa"]))
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        optimizer, "_evaluate_theta", recorded_evaluate
    )

    optimizer.run_qpeo(
        problem, inputs, settings,
        config=config, seed=7,
    )

    # Initial population + refresh and proposals per iteration.
    assert len(observed_kappa) == 20
    assert np.allclose(observed_kappa[:4], kappa_schedule(0))
    assert np.allclose(observed_kappa[4:12], kappa_schedule(1))
    assert np.allclose(observed_kappa[12:20], kappa_schedule(2))
