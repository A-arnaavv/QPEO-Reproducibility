"""End-to-end smoke check for the isolated QPEO ablation harness."""
import math
from pathlib import Path

from qpeo.optimizer import ReconstructionQPEOSettings
from qpeo.protocol import load_protocol
from scripts.run_ablation_shard import VARIANTS, evaluate_ablation_variants
from scripts.run_final_experiment_shard import build_scenario


def test_five_ablations_run_on_locked_dynamic_carbon_scenario():
    bundle, _, _ = build_scenario("alibaba_gpu_v2023", 32)
    primary, _ = load_protocol(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "reconstruction_protocol.json"
    )
    settings = ReconstructionQPEOSettings(
        scalar_weights=primary.scalar_guide_weights,
        alpha=primary.alpha,
        beta=primary.beta,
        gamma=primary.gamma,
        dependency_threshold=primary.dependency_threshold,
        dependency_strength=primary.dependency_strength,
    )
    rows = evaluate_ablation_variants(
        bundle.problem, bundle.qpeo_inputs, settings,
        seed=primary.seeds[0], population_size=6, max_iterations=2,
    )
    assert len(rows) == len(VARIANTS) == 5
    assert {row["variant"] for row in rows} == set(VARIANTS)
    for row in rows:
        assert len(row["guide_objectives"]) == 6
        assert len(row["guide_objective_history"]) == 3
        assert all(
            math.isfinite(value)
            for values in row["guide_objective_history"] for value in values
        )
        assert row["simulation_cache_hits"] >= 0
        assert row["simulation_cache_misses"] >= 0
        assert row["runtime_seconds"] >= 0
    assert next(
        row for row in rows if row["variant"] == "no_entropy_bro"
    )["entropy_history"] == []
