"""Synthetic software-only fixtures for statistical analysis.

These numbers are NOT manuscript results and are never stored under results/raw.
"""

import numpy as np
import pytest

from scripts.analyze_final_results import (
    ALGORITHMS,
    GLOBAL_COMPARISONS,
    SCENARIO_COMPARISONS,
    SCENARIOS,
    _wilcoxon_paired,
    compute_statistics,
)


def synthetic_shards(seed_count=3):
    shards = []
    for scenario in SCENARIOS:
        for seed_index in range(seed_count):
            algorithms = []
            for a_index, name in enumerate(ALGORITHMS):
                algorithms.append({
                    "algorithm": name,
                    "runtime_seconds": float(10 + a_index + seed_index),
                    "archive_size": 3 + a_index,
                    "simulation_cache_hits": 100 + seed_index,
                    "simulation_cache_misses": 20 + a_index,
                    "guide_objectives": [
                        float(a_index + (k + 1) / 10 + seed_index * (k + 1) / 50)
                        for k in range(6)
                    ],
                })
            shards.append({
                "scenario": scenario, "seed_index": seed_index,
                "algorithms": algorithms,
            })
    return shards


def test_incomplete_collection_does_not_produce_inference():
    shards = synthetic_shards()
    with pytest.raises(ValueError, match="Incomplete shard count"):
        compute_statistics(shards[:-1], 3)


def test_complete_paired_statistical_report_shapes():
    results = compute_statistics(synthetic_shards(), 3)
    assert len(results["descriptives"]) == 4 * 9 * 13
    assert len(results["paired_wilcoxon"]) == 4 * 6 * 12
    assert len(results["friedman"]) == 4 * 6 * 13
    assert len(results["nemenyi"]) == 4 * 6 * (13 * 12 // 2)
    assert SCENARIO_COMPARISONS == 72
    assert GLOBAL_COMPARISONS == 288
    example = results["paired_wilcoxon"][0]
    assert example["baseline"] == "PSO"
    assert example["mean_qpeo_minus_baseline"] < 0
    assert example["p_bonferroni_288"] >= example["p_bonferroni_72"]
    assert 0 <= example["p_raw"] <= 1


def test_identical_paired_samples_have_p_one_and_effect_zero():
    v = np.array([1.0,2.0,3.0])
    statistic,p,dz = _wilcoxon_paired(v,v.copy())
    assert statistic == 0
    assert p == 1
    assert dz == 0
