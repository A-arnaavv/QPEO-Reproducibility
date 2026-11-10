import pytest

from scripts.analyze_ablation_results import (
    ALGORITHMS, COMPARE, COMPARISONS_PER_SCENARIO,
    GLOBAL_COMPARISONS, compute_ablation_stats,
)
from scripts.audit_final_shards import SCENARIOS
from scripts.run_ablation_shard import VARIANTS


def make_records(n=2):
    primary, ablation = [], []
    for scenario in SCENARIOS:
        for i in range(n):
            objective_ref = [1.0 + i / 10] * 6
            primary.append({
                "scenario": scenario,
                "seed_index": i,
                "seed": 100 + i,
                "algorithms": [
                    {
                        "algorithm": name,
                        "guide_objectives": (
                            objective_ref if name=="QPEO" else [2.0+i/10]*6
                        ),
                        "simulation_cache_hits": 10,
                        "simulation_cache_misses": 20,
                        "runtime_seconds": .1,
                        "archive_size": 2,
                    }
                    for name in ("QPEO","FPA")
                ],
            })
            ablation.append({
                "scenario": scenario,
                "seed_index": i,
                "seed": 100 + i,
                "variants": [
                    {
                        "variant": name,
                        "guide_objectives": [1.5+i/10]*6,
                        "simulation_cache_hits": 12,
                        "simulation_cache_misses": 22,
                        "runtime_seconds": .11,
                        "archive_size": 3,
                    }
                    for name in VARIANTS
                ],
                "secondary_baselines": [{
                    "variant": "elitist_random_search",
                    "guide_objectives": [1.7+i/10] * 6,
                    "simulation_cache_hits": 13,
                    "simulation_cache_misses": 23,
                    "runtime_seconds": .12,
                    "archive_size": 2,
                }],
            })
    return ablation, primary


def test_ablation_comparison_families_and_descriptive_counts():
    ablation, primary = make_records()
    result = compute_ablation_stats(ablation, primary, seed_count=2)
    assert len(ALGORITHMS)==8
    assert len(COMPARE)==7
    assert COMPARISONS_PER_SCENARIO==42
    assert GLOBAL_COMPARISONS==168
    assert len(result["descriptive_statistics"]) == len(SCENARIOS)*9*8
    assert len(result["paired_wilcoxon"]) == len(SCENARIOS)*6*7
    for row in result["paired_wilcoxon"]:
        assert row["n"]==2
        assert row["mean_qpeo_minus_comparator"] < 0
        assert row["qpeo_better_seed_count"] == 2
        assert row["comparator_better_seed_count"] == 0
        assert row["ties"] == 0
        assert 0 <= row["p_raw"] <= row["p_bonferroni_42"] <= 1
        assert 0 <= row["p_bonferroni_168"] <= 1
        assert row["p_bonferroni_168"] >= row["p_bonferroni_42"]
        assert row["comparator_type"] == (
            "classical_reference" if row["comparator"]=="classical_fpa"
            else "random_search_reference" if row["comparator"]=="elitist_random_search"
            else "operator_ablation"
        )


def test_refuses_partial_or_duplicated_seed_pairs():
    ablation, primary = make_records()
    with pytest.raises(ValueError, match="Complete paired"):
        compute_ablation_stats(ablation[:-1], primary, seed_count=2)
    with pytest.raises(ValueError, match="Complete paired"):
        compute_ablation_stats(ablation, primary + [primary[0]], seed_count=2)


def test_refuses_unpaired_seed_numbers_and_missing_variants():
    ablation, primary = make_records()
    bad=[dict(x) for x in ablation]
    bad[0]=dict(bad[0],seed=999)
    with pytest.raises(ValueError, match="Unpaired seed"):
        compute_ablation_stats(bad, primary, seed_count=2)
    bad=[dict(x) for x in ablation]
    bad[0]=dict(bad[0],variants=bad[0]["variants"][:-1])
    with pytest.raises(ValueError, match="Missing QPEO"):
        compute_ablation_stats(bad, primary, seed_count=2)
