"""Regression for the roundoff-only Helios GPU capacity overlap.

This reproduces initial QPEO candidate 31 from locked seed 104729.
The same candidate formerly produced two back-to-back intervals whose timestamps
differed by ~5e-17 hours and falsely raised an instantaneous utilization error.
"""

import numpy as np

from scripts.run_final_experiment_shard import build_scenario
from qpeo.config import ManuscriptQPEOConfig
from qpeo.optimizer import (
    ReconstructionQPEOSettings,
    _omega_for_problem,
    _qpeo_problem_for_iteration,
)
from qpeo.protocol import load_protocol
from qpeo.operators import assignment_probabilities, kappa_schedule
from qpeo.collapse import feasible_caqc_collapse
from qpeo.simulator import simulate_fixed_assignment


def test_locked_helios_seed0_candidate31_has_numerically_feasible_schedule():
    bundle, _, _ = build_scenario("helios_earth", 64)
    primary, _ = load_protocol(
        __import__("pathlib").Path(__file__).parents[1]
        / "configs"
        / "reconstruction_protocol.json"
    )
    settings = ReconstructionQPEOSettings(
        primary.scalar_guide_weights, primary.alpha, primary.beta,
        primary.gamma, primary.dependency_threshold,
        primary.dependency_strength,
    )
    cfg = ManuscriptQPEOConfig(population_size=50, max_iterations=500)
    problem = _qpeo_problem_for_iteration(
        bundle.problem, bundle.qpeo_inputs, 0, cfg.max_iterations
    )
    omega = _omega_for_problem(problem, bundle.qpeo_inputs, settings)
    rng = np.random.default_rng(104729)
    angles = rng.uniform(
        0, np.pi / 2, size=(50, 64 * len(problem.resources))
    )
    X = feasible_caqc_collapse(
        problem,
        assignment_probabilities(angles[31], omega),
        kappa=kappa_schedule(0),
        ci_ref=cfg.ci_ref,
    )
    result = simulate_fixed_assignment(
        problem, X, bundle.qpeo_inputs.simulator_inputs
    )
    assert np.all(np.isfinite(result.energy_kwh_by_resource))
    assert np.all(result.energy_kwh_by_resource >= 0)
    assert np.all(result.time_average_gpu_utilization <= 1.0 + 1e-9)
