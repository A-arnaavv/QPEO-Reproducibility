from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np

from qpeo.carbon import load_neso_regional_replay
from qpeo.experiment import run_full_suite
from qpeo.optimizer import ReconstructionQPEOSettings
from qpeo.protocol import load_protocol
from qpeo.resource_environment import load_resource_environment
from qpeo.scenarios import (
    build_alibaba_gpu_v2023_scenario,
    build_alibaba_spot_a100_scenario,
    build_elasticflow_itp_scenario,
    build_helios_earth_scenario,
)

ROOT=Path(__file__).resolve().parents[1]

SCENARIOS={
    "alibaba_gpu_v2023": (
        build_alibaba_gpu_v2023_scenario,
        ROOT/"data"/"alibaba_gpu_v2023_eligible_succeeded.csv",
        ROOT/"configs"/"trace_alibaba_gpu_v2023.json",
    ),
    "microsoft_elasticflow_itp_cluster08": (
        build_elasticflow_itp_scenario,
        ROOT/"data"/"elasticflow_cluster08_single_gpu.csv",
        ROOT/"configs"/"trace_elasticflow_itp.json",
    ),
    "helios_earth": (
        build_helios_earth_scenario,
        ROOT/"data"/"helios_earth_selected_128.csv",
        ROOT/"configs"/"trace_helios_earth.json",
    ),
    "alibaba_spot_gpu_2026_a100": (
        build_alibaba_spot_a100_scenario,
        ROOT/"data"/"alibaba_spot_a100_selected_128.csv",
        ROOT/"configs"/"trace_alibaba_spot_a100.json",
    ),
}


def _serialize_result(record):
    result=record.result
    payload={
        "algorithm":record.algorithm,
        "seed":record.seed,
        "runtime_seconds":record.runtime_seconds,
        "guide_objectives":record.guide_objectives.tolist(),
        "archive_size":record.archive_size,
        "guide_objective_history":np.asarray(result.guide_objective_history,dtype=float).tolist(),
        "archive_size_history":np.asarray(result.archive_size_history,dtype=int).tolist(),
    }
    if hasattr(result,"best_scalar_history"):
        payload["best_scalar_history"]=np.asarray(result.best_scalar_history,dtype=float).tolist()
    if hasattr(result,"entropy_history"):
        payload["entropy_history"]=np.asarray(result.entropy_history,dtype=float).tolist()
    return payload


def build_scenario(name:str,size:int):
    builder,snapshot,cfg=SCENARIOS[name]
    env=load_resource_environment(ROOT/"configs"/"resource_environment_v1.json")
    replay=load_neso_regional_replay(
        ROOT/"data"/"carbon"/"neso_regional_2026-01-05_2026-01-11.csv",
        ROOT/"configs"/"carbon_replay_v1.json",
        expected_slot_names=[s.name for s in env.slots],
    )
    return builder(snapshot,cfg,env,replay,size=size),env,replay


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenario",choices=sorted(SCENARIOS),required=True)
    ap.add_argument("--seed-index",type=int,required=True,help="0-based index into the locked 30-seed list")
    ap.add_argument("--size",type=int,default=64,choices=[32,64,128])
    ap.add_argument("--output-dir",default="results/raw")
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--population-size",type=int,default=None,help="Testing override; omit for locked final protocol")
    ap.add_argument("--max-iterations",type=int,default=None,help="Testing override; omit for locked final protocol")
    args=ap.parse_args()

    primary,raw_protocol=load_protocol(ROOT/"configs"/"reconstruction_protocol.json")
    if not (0<=args.seed_index<len(primary.seeds)):
        raise SystemExit(f"seed-index must be in [0,{len(primary.seeds)-1}]")
    seed=int(primary.seeds[args.seed_index])
    P=primary.population_size if args.population_size is None else int(args.population_size)
    T=primary.max_iterations if args.max_iterations is None else int(args.max_iterations)
    final_protocol=(args.population_size is None and args.max_iterations is None and args.size==64)

    outdir=ROOT/args.output_dir
    outdir.mkdir(parents=True,exist_ok=True)
    tag="final" if final_protocol else f"test-P{P}-T{T}-N{args.size}"
    outpath=outdir/f"{args.scenario}__seed{args.seed_index:02d}_{seed}__{tag}.json"
    if outpath.exists() and not args.force:
        print(f"exists, skipping: {outpath}")
        return

    bundle,env,replay=build_scenario(args.scenario,args.size)
    weights=tuple(primary.scalar_guide_weights)
    settings=ReconstructionQPEOSettings(
        scalar_weights=weights,
        alpha=primary.alpha,
        beta=primary.beta,
        gamma=primary.gamma,
        dependency_threshold=primary.dependency_threshold,
        dependency_strength=primary.dependency_strength,
    )
    suite=run_full_suite(
        bundle.problem,bundle.qpeo_inputs,settings,weights,
        seeds=[seed],population_size=P,max_iterations=T,
    )
    payload={
        "schema_version":"qpeo-final-shard-v1",
        "final_protocol":final_protocol,
        "scenario":args.scenario,
        "dataset_id":bundle.dataset_id,
        "size":args.size,
        "seed_index":args.seed_index,
        "seed":seed,
        "population_size":P,
        "max_iterations":T,
        "git_sha":os.environ.get("GITHUB_SHA"),
        "resource_environment_version":env.version,
        "carbon_source":replay.source_name,
        "carbon_rows":replay.rows,
        "algorithms":[_serialize_result(r) for r in suite.records],
    }
    if len(payload["algorithms"])!=13:
        raise RuntimeError("Expected exactly 13 algorithms in a full-suite shard")
    with tempfile.NamedTemporaryFile("w",dir=outdir,delete=False,encoding="utf-8",suffix=".tmp") as tmp:
        json.dump(payload,tmp,indent=2,allow_nan=False)
        tmp.write("\n")
        tmppath=Path(tmp.name)
    tmppath.replace(outpath)
    print(outpath)


if __name__=="__main__":
    main()
