"""Synthetic fixture tests: NEVER make a real campaign run."""
from __future__ import annotations

import json
import numpy as np
import pytest

from scripts.audit_final_shards import ALGORITHMS
from scripts.audit_carbon_v2_campaign import (
    expected_filename, validate_campaign_shard, audit_campaign,
    analyze_complete_campaign,
)
from scripts.run_carbon_v2_campaign import campaign_plan, run_one


def fake_shard():
    sample = {
        "schema_version": "qpeo-carbon-v2-test-shard-v1",
        "final_protocol": False,
        "scenario": "helios_earth",
        "seed_index": 0,
        "seed": 104729,
        "size": 64,
        "population_size": 50,
        "max_iterations": 500,
        "git_sha": "a"*40,
        "execution_provenance": {
            "source_git_sha": "a"*40,
            "source_has_tracked_changes": False,
        },
        "carbon_time_accounting": {
            "version": "v2", "coverage_policy": "repeat",
            "planning_signal": "fixed full-week regional mean intensity",
        },
        "algorithms": [],
    }
    for name in ALGORITHMS:
        sample["algorithms"].append({
            "algorithm": name, "seed": 104729,
            "guide_objectives": [-.3, 1.2, 100., 4., 0., 5.],
            "guide_carbon_g": 120.,
            "guide_carbon_intensity_g_per_kwh": 100.,
            "guide_objective_history": [[-.3,1.2,100.,4.,0.,5.] for _ in range(501)],
        })
    return sample


def put_shard(tmp_path, shard):
    directory = tmp_path / ("a"*12)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / expected_filename("helios_earth", 0, 104729)
    p.write_text(json.dumps(shard))
    return p


def test_plan_is_dry_and_contains_exact_120_paired_shards():
    plan = campaign_plan("f"*40)
    assert plan["status"] == "PLANNED_NOT_EXECUTED"
    assert len(plan["tasks"]) == 120
    assert plan["algorithm_records_expected"] == 1560
    assert len(set((x["scenario"], x["seed_index"]) for x in plan["tasks"])) == 120
    assert all("--execute" in x["invocation"] for x in plan["tasks"])
    assert all("test-P50-T500-N64-carbon-time-v2-repeat-fixedmean" in x["expected_file"] for x in plan["tasks"])


def test_audit_accepts_correct_metadata_without_deep_replay(tmp_path):
    p = put_shard(tmp_path, fake_shard())
    rec = validate_campaign_shard(p, "helios_earth", 0, 104729, deep=False)
    assert rec["git_sha"] == "a"*40
    report = audit_campaign(p.parent, deep=False)
    assert report["valid_shards"] == 1
    assert report["missing_shards"] == 119
    assert report["invalid_shards"] == 0
    assert not report["complete"]


def test_rejects_wrong_seed_invalid_carbon_identity_and_incomplete_history(tmp_path):
    d = fake_shard()
    p = put_shard(tmp_path, d)
    d["seed"] = 7
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="locked seed mismatch"):
        validate_campaign_shard(p, "helios_earth", 0, 104729, deep=False)
    d["seed"] = 104729
    d["algorithms"][0]["guide_carbon_g"] = 999.
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="identity failed"):
        validate_campaign_shard(p, "helios_earth", 0, 104729, deep=False)
    d["algorithms"][0]["guide_carbon_g"] = 120.
    d["algorithms"][0]["guide_objective_history"].pop()
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="501-point"):
        validate_campaign_shard(p, "helios_earth", 0, 104729, deep=False)


def test_mixed_protocol_file_in_campaign_dir_is_invalid(tmp_path):
    p = put_shard(tmp_path, fake_shard())
    (p.parent / "other__test-P4-T2-N64-carbon-time-v2-repeat-fixedmean.json").write_text("{}")
    report = audit_campaign(p.parent, deep=False)
    assert report["invalid_shards"] == 1
    assert any("Unexpected" in v["reason"] for v in report["invalid_files"])


def test_no_statistical_analysis_of_partial_or_fast_audited_campaign(tmp_path):
    p = put_shard(tmp_path, fake_shard())
    report = audit_campaign(p.parent, deep=False)
    with pytest.raises(ValueError, match="REFUSING"):
        analyze_complete_campaign(p.parent, tmp_path / "analysis", report)
    assert not (tmp_path / "analysis").exists()


def test_one_shard_without_explicit_execute_never_calls_optimizers():
    with pytest.raises(SystemExit, match="--execute"):
        run_one("helios_earth", 0, execute=False)
