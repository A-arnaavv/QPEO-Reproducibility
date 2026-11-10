"""Ablation switches must not change the default full QPEO run."""
import numpy as np
import pytest

from qpeo import (
    CarbonClass, ManuscriptQPEOConfig, PowerModel, ProblemInstance,
    QPEOInputs, QPEOOperatorSwitches, ReconstructionQPEOSettings,
    Resource, SimulatorInputs, Workload, check_constraints, run_qpeo,
)


def _toy_case():
    p = ProblemInstance(
        [
            Workload("a", 1, .35, 2, 0, 0, 4, CarbonClass.FLEXIBLE),
            Workload("b", 1, .4, 2, 0, .1, 4, CarbonClass.DEFERRED),
        ],
        [Resource("cloud", 1, 8, 220, 1.0), Resource("edge", 1, 8, 380, .7)],
        {
            CarbonClass.CRITICAL: 250,
            CarbonClass.DEFERRED: 450,
            CarbonClass.FLEXIBLE: 450,
        },
        3.0,
    )
    inputs = QPEOInputs(
        simulator_inputs=SimulatorInputs(
            service_time_hours=np.array([[1.0, 1.3], [1.2, .9]]),
            power_model=PowerModel(np.array([.08, .06]), np.array([.22, .18])),
            billing_mode="allocated_gpu_hours",
        ),
        performance_affinity=np.array([[1.0, .7], [.75, 1.0]]),
        thermal_headroom=np.array([.7, .9]),
        dependency_matrix=np.array([[0, .8], [.8, 0]], float),
    )
    settings = ReconstructionQPEOSettings(
        scalar_weights=[1 / 6] * 6,
        alpha=.5,
        beta=.25,
        gamma=.25,
        dependency_threshold=.5,
        dependency_strength=.2,
    )
    cfg = ManuscriptQPEOConfig(
        population_size=6, max_iterations=3, stagnation_iterations=2
    )
    return p, inputs, settings, cfg


def test_default_switches_exactly_match_original_optimizer_behavior():
    p, inputs, settings, cfg = _toy_case()
    original = run_qpeo(p, inputs, settings, config=cfg, seed=42)
    explicit = run_qpeo(
        p, inputs, settings, config=cfg, seed=42,
        operator_switches=QPEOOperatorSwitches(),
    )
    np.testing.assert_array_equal(original.final_population, explicit.final_population)
    np.testing.assert_array_equal(
        original.guide_objective_history, explicit.guide_objective_history
    )
    np.testing.assert_array_equal(
        original.entropy_history, explicit.entropy_history
    )


@pytest.mark.parametrize("disabled", [
    "use_caqc_carbon_penalty",
    "use_ebws",
    "use_entropy_biodiversity",
    "use_global_qpso",
    "use_local_qpso",
])
def test_one_component_ablations_produce_feasible_objective_histories(disabled):
    p, inputs, settings, cfg = _toy_case()
    flags = {
        "use_caqc_carbon_penalty": True,
        "use_ebws": True,
        "use_entropy_biodiversity": True,
        "use_global_qpso": True,
        "use_local_qpso": True,
    }
    flags[disabled] = False
    result = run_qpeo(
        p, inputs, settings, config=cfg, seed=17,
        operator_switches=QPEOOperatorSwitches(**flags),
    )
    assert result.guide_objective_history.shape == (4, 6)
    assert np.all(np.isfinite(result.guide_objective_history))
    assert all(check_constraints(p, X).feasible for X in result.final_assignments)
    if disabled == "use_entropy_biodiversity":
        assert result.entropy_history.shape == (0,)
    else:
        assert result.entropy_history.shape == (3,)


def test_cannot_remove_both_pollination_modes():
    with pytest.raises(ValueError, match="At least one QPSO"):
        QPEOOperatorSwitches(
            use_global_qpso=False,
            use_local_qpso=False,
        ).validate()
