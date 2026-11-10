import json
from pathlib import Path

from scripts.collect_final_artifacts import collect
from test_final_shard_audit import sample_protocol, sample_shard


def test_collect_accepts_valid_and_rejects_conflicting_existing(tmp_path):
    src = tmp_path / "artifacts"
    dst = tmp_path / "saved"
    src.mkdir()
    filename = "helios_earth__seed00_0__final.json"
    p = src / filename
    p.write_text(json.dumps(sample_shard()))
    first = collect(src, dst, sample_protocol())
    assert len(first["new_files"]) == 1
    assert first["coverage"]["valid_shards"] == 1

    second = collect(src, dst, sample_protocol())
    assert len(second["already_identical"]) == 1

    modified = sample_shard()
    modified["algorithms"][0]["guide_objectives"][0] = 0.1
    p.write_text(json.dumps(modified))
    third = collect(src, dst, sample_protocol())
    assert len(third["invalid_or_conflicting"]) == 1
    assert json.loads((dst / filename).read_text()) == sample_shard()


def test_collect_skips_nonfinal_without_corrupting_existing_results(tmp_path):
    src, dst = tmp_path / "artifacts", tmp_path / "saved"
    src.mkdir()
    raw = sample_shard()
    raw["final_protocol"] = False
    (src / "helios_earth__seed00_0__final.json").write_text(json.dumps(raw))
    report = collect(src, dst, sample_protocol())
    assert len(report["new_files"]) == 0
    assert len(report["invalid_or_conflicting"]) == 1
    assert report["coverage"]["valid_shards"] == 0
