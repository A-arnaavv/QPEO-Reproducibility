import json

import pytest

from scripts.audit_ablation_shards import (
    audit_ablation_directory, validate_ablation_shard,
)
from scripts.run_ablation_shard import VARIANTS
from test_final_shard_audit import sample_protocol, sample_shard


def ablation_fixture():
    return {
        "schema_version": "qpeo-ablation-shard-v1",
        "final_protocol": True,
        "scenario": "helios_earth",
        "dataset_id": "fixture-helios",
        "seed_index": 0,
        "seed": 0,
        "size": 64,
        "population_size": 4,
        "max_iterations": 2,
        "resource_environment_version": "fixture-v1",
        "carbon_source": "fixture-neso",
        "carbon_rows": 336,
        "variants": [
            {
                "variant":name,"seed":0,
                "runtime_seconds":.1,
                "guide_objectives":[0.0]*6,
                "guide_objective_history":[[0.0]*6]*3,
                "entropy_history":([] if name=="no_entropy_bro" else [0.5,0.4]),
                "simulation_cache_hits":4,
                "simulation_cache_misses":3
            }
            for name in VARIANTS
        ],
    }


def test_ablation_file_accepts_all_five_components(tmp_path):
    p=tmp_path/"helios_earth__seed00_0__ablation__final.json"
    p.write_text(json.dumps(ablation_fixture()))
    result=validate_ablation_shard(p,sample_protocol()["primary"])
    assert result["seed_index"]==0
    assert len(result["variants"])==5


def test_ablation_file_refuses_invalid_entropy_history(tmp_path):
    p=tmp_path/"helios_earth__seed00_0__ablation__final.json"
    d=ablation_fixture()
    d["variants"][2]["entropy_history"]=[.5]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="entropy-history length"):
        validate_ablation_shard(p,sample_protocol()["primary"])


def test_ablation_audit_pairs_exact_source_environment(tmp_path):
    abdir=tmp_path/"ablations"
    primdir=tmp_path/"primary"
    abdir.mkdir()
    primdir.mkdir()
    af=abdir/"helios_earth__seed00_0__ablation__final.json"
    pf=primdir/"helios_earth__seed00_0__final.json"
    ab=ablation_fixture()
    prim=sample_shard()
    for k in ("dataset_id","resource_environment_version","carbon_source","carbon_rows"):
        prim[k]=ab[k]
    af.write_text(json.dumps(ab))
    pf.write_text(json.dumps(prim))
    ok=audit_ablation_directory(abdir,primdir,sample_protocol()["primary"])
    assert ok["paired_valid_shards"]==1
    assert len(ok["missing_pairs"])==119
    assert ok["invalid_or_unpaired"]==[]

    ab["carbon_rows"]=335
    af.write_text(json.dumps(ab))
    bad=audit_ablation_directory(abdir,primdir,sample_protocol()["primary"])
    assert bad["paired_valid_shards"]==0
    assert len(bad["invalid_or_unpaired"])==1
    assert len(bad["missing_pairs"])==120
