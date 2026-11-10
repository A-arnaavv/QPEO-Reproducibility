import json

import pytest

from scripts.audit_final_shards import ALGORITHMS, audit, validate_shard


def sample_protocol():
    return {
        "protocol_version": "test-only",
        "primary": {
            "population_size": 4,
            "max_iterations": 2,
            "seeds": list(range(30)),
        },
    }


def sample_shard():
    return {
        "schema_version": "qpeo-final-shard-v1",
        "final_protocol": True,
        "scenario": "helios_earth",
        "seed_index": 0,
        "seed": 0,
        "size": 64,
        "population_size": 4,
        "max_iterations": 2,
        "git_sha": "fixture-commit",
        "algorithms": [
            {
                "algorithm": a,
                "seed": 0,
                "runtime_seconds": 0.1,
                "guide_objectives": [0.0] * 6,
                "guide_objective_history": [[0.0] * 6 for _ in range(3)],
                "simulation_cache_hits": 3,
                "simulation_cache_misses": 2,
            }
            for a in ALGORITHMS
        ],
    }


def test_audit_reports_one_valid_shard_and_119_missing(tmp_path):
    p = tmp_path / "helios_earth__seed00_0__final.json"
    p.write_text(json.dumps(sample_shard()))
    record = validate_shard(p, sample_protocol()["primary"])
    assert record["simulation_evaluation_calls_by_algorithm"]["QPEO"] == 5
    assert len(record["sha256"]) == 64
    report = audit(tmp_path, sample_protocol())
    assert report["expected_shards"] == 120
    assert report["valid_shards"] == 1
    assert report["missing_shards"] == 119
    assert report["invalid_shards"] == 0
    assert report["validated_algorithm_rows"] == 13


def test_rejects_unequal_seed_and_nonfinite_history(tmp_path):
    d = sample_shard()
    d["seed"] = 200
    p = tmp_path / "helios_earth__seed00_0__final.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Seed does not match"):
        validate_shard(p, sample_protocol()["primary"])

    d["seed"] = 0
    d["algorithms"][0]["guide_objective_history"][1][0] = float("nan")
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="non-finite objective history"):
        validate_shard(p, sample_protocol()["primary"])


def test_rejects_nonfinal_or_missing_algorithms(tmp_path):
    d = sample_shard()
    d["final_protocol"] = False
    p = tmp_path / "helios_earth__seed00_0__final.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="not marked final"):
        validate_shard(p, sample_protocol()["primary"])

    d["final_protocol"] = True
    d["algorithms"] = d["algorithms"][:-1]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="algorithm count"):
        validate_shard(p, sample_protocol()["primary"])
