"""Random-search comparator must be reproducible and use the common decoder."""
import numpy as np

from qpeo.baselines import CommonBaselineConfig
from qpeo.evaluator import check_constraints
from qpeo.random_search import run_elitist_random_search
from qpeo.carbon import with_carbon_intensities
from scripts.run_final_experiment_shard import build_scenario


def test_elitist_random_search_runs_and_is_seed_deterministic():
    bundle, _, replay = build_scenario("alibaba_gpu_v2023", 32)
    config = CommonBaselineConfig(
        population_size=6, max_iterations=3,
        carbon_replay=bundle.qpeo_inputs.carbon_replay,
    )
    inputs = bundle.qpeo_inputs.simulator_inputs
    first = run_elitist_random_search(
        bundle.problem, inputs, [1/6]*6, config=config, seed=122
    )
    second = run_elitist_random_search(
        bundle.problem, inputs, [1/6]*6, config=config, seed=122
    )
    assert first.name == "Elitist Random Search"
    assert first.guide_objective_history.shape == (4, 6)
    assert first.archive_size_history.shape == (4,)
    assert first.best_scalar_history.shape == (3,)
    assert np.all(np.isfinite(first.guide_objective_history))
    np.testing.assert_array_equal(first.guide_objective_history, second.guide_objective_history)
    np.testing.assert_array_equal(first.final_population, second.final_population)
    at_end = with_carbon_intensities(
        bundle.problem, replay.at_iteration(3,3)
    )
    assert all(check_constraints(at_end, X).feasible for X in first.final_assignments)


def test_random_search_rejects_invalid_guide_weights():
    bundle, _, _ = build_scenario("alibaba_gpu_v2023", 32)
    config = CommonBaselineConfig(population_size=6, max_iterations=1)
    import pytest
    with pytest.raises(ValueError, match="scalar_weights"):
        run_elitist_random_search(
            bundle.problem, bundle.qpeo_inputs.simulator_inputs,
            [1, 0, 0], config=config, seed=3
        )
