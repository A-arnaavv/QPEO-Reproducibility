from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Callable

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np

from qpeo.carbon import load_neso_regional_replay, with_carbon_intensities
from qpeo.carbon_accounting import CarbonTimeSeries, account_simulation_result
from qpeo.experiment import run_full_suite
from qpeo.optimizer import ReconstructionQPEOSettings
from qpeo.protocol import load_protocol
from qpeo.resource_environment import load_resource_environment
from qpeo.simulator import simulate_fixed_assignment
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
        "simulation_cache_hits":int(record.cache_hits),
        "simulation_cache_misses":int(record.cache_misses),
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


def _local_provenance() -> dict:
    """Record exact revision and source/config status for a local validation run."""
    def git(*args):
        try:
            result=subprocess.run(
                ["git", "-C", str(ROOT), *args],
                check=True, capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    revision=os.environ.get("GITHUB_SHA") or git("rev-parse", "HEAD")
    # Include untracked code/config/tests: a ZIP-based local Git snapshot may
    # otherwise claim an exact source SHA while executing untracked Python files.
    dirty=git("status", "--porcelain", "--untracked-files=all", "--",
              "qpeo", "scripts", "configs", "tests", "data", "requirements.txt", "pyproject.toml")
    protocol_path=ROOT/"configs"/"carbon_v2_provisional_pilot_protocol.json"
    protocol_hash=(hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                   if protocol_path.exists() else None)
    return {
        "source_git_sha":revision,
        "source_has_tracked_changes":None if dirty is None else bool(dirty),
        "provisional_protocol_sha256":protocol_hash,
        "python_version":platform.python_version(),
        "numpy_version":np.__version__,
        "scipy_version":package_metadata.version("scipy"),
        "pytest_version":package_metadata.version("pytest"),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenario",choices=sorted(SCENARIOS),required=True)
    ap.add_argument("--seed-index",type=int,required=True,help="0-based index into the locked 30-seed list")
    ap.add_argument("--size",type=int,default=64,choices=[32,64,128])
    ap.add_argument("--output-dir",default="results/raw")
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--carbon-time-accounting",choices=("strict","repeat"),default=None,
                    help="Opt-in v2 execution-time carbon objective; NEVER overwrites locked v1 shards")
    ap.add_argument("--population-size",type=int,default=None,help="Testing override; omit for locked final protocol")
    ap.add_argument("--max-iterations",type=int,default=None,help="Testing override; omit for locked final protocol")
    args=ap.parse_args()

    primary,raw_protocol=load_protocol(ROOT/"configs"/"reconstruction_protocol.json")
    if not (0<=args.seed_index<len(primary.seeds)):
        raise SystemExit(f"seed-index must be in [0,{len(primary.seeds)-1}]")
    seed=int(primary.seeds[args.seed_index])
    P=primary.population_size if args.population_size is None else int(args.population_size)
    T=primary.max_iterations if args.max_iterations is None else int(args.max_iterations)
    final_protocol=(args.population_size is None and args.max_iterations is None and args.size==64
                    and args.carbon_time_accounting is None)

    outdir=ROOT/args.output_dir
    outdir.mkdir(parents=True,exist_ok=True)
    tag="final" if final_protocol else f"test-P{P}-T{T}-N{args.size}"
    if args.carbon_time_accounting is not None:
        tag+=f"-carbon-time-v2-{args.carbon_time_accounting}-fixedmean"
    outpath=outdir/f"{args.scenario}__seed{args.seed_index:02d}_{seed}__{tag}.json"
    if outpath.exists() and not args.force:
        print(f"exists, skipping: {outpath}")
        return

    bundle,env,replay=build_scenario(args.scenario,args.size)
    if args.carbon_time_accounting is not None:
        # First job arrival is mapped to the beginning of the frozen NESO week.
        # Strict coverage rejects horizons >168 h; repeating the week is explicitly
        # a counterfactual sensitivity assumption, not observed future forecasts.
        steps=np.diff([t.timestamp() for t in replay.timestamps_utc])/3600.0
        if len(steps)==0 or not np.allclose(steps,steps[0]) or steps[0]<=0:
            raise ValueError("Time-aligned accounting requires a uniformly spaced replay")
        carbon_time=CarbonTimeSeries(
            intensities_g_per_kwh=replay.intensities,
            step_hours=float(steps[0]),
            forecast_start_hours=0.0,
            coverage_policy=args.carbon_time_accounting,
        )
        # v2 is a ONE-SHOT, OFFLINE planning problem: the full forecast is
        # declared available before placement. Its regional time means are fixed
        # signals for CAQC/decoder eligibility and affinity during optimization.
        # There is no artificial advancement of grid time per optimizer iteration.
        # Time-resolved emissions remain evaluated from the actual job schedules.
        planning_ci=np.mean(replay.intensities,axis=0)
        bundle=replace(
            bundle,
            problem=with_carbon_intensities(bundle.problem,planning_ci),
            qpeo_inputs=replace(
                bundle.qpeo_inputs,
                carbon_replay=None,
                simulator_inputs=replace(
                    bundle.qpeo_inputs.simulator_inputs,
                    carbon_time_series=carbon_time,
                ),
            ),
        )
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
    algorithms=[_serialize_result(r) for r in suite.records]
    if args.carbon_time_accounting is not None:
        sim_inputs=bundle.qpeo_inputs.simulator_inputs
        final_problem=bundle.problem  # Same frozen planning model used by optimizers.
        for item,record in zip(algorithms,suite.records):
            X=record.result.guide_assignment
            sim=simulate_fixed_assignment(final_problem,X,sim_inputs)
            audit=account_simulation_result(
                final_problem,X,sim_inputs,sim,sim_inputs.carbon_time_series
            )
            # Retain the final guide's original scheduling intervals for verification.
            item["guide_assignment_indices"]=np.argmax(X,axis=1).tolist()
            item["guide_start_hours"]=sim.start_times.tolist()
            item["guide_completion_hours"]=sim.completion_times.tolist()
            item["guide_carbon_g"]=audit.total_carbon_g
            item["guide_carbon_intensity_g_per_kwh"]=audit.energy_weighted_intensity_g_per_kwh
    provenance=_local_provenance() if args.carbon_time_accounting is not None else None
    payload={
        "schema_version":"qpeo-carbon-v2-test-shard-v1" if args.carbon_time_accounting is not None else "qpeo-final-shard-v1",
        "final_protocol":final_protocol,
        "scenario":args.scenario,
        "dataset_id":bundle.dataset_id,
        "size":args.size,
        "seed_index":args.seed_index,
        "seed":seed,
        "population_size":P,
        "max_iterations":T,
        "git_sha":provenance["source_git_sha"] if provenance is not None else os.environ.get("GITHUB_SHA"),
        "execution_provenance":provenance,
        "resource_environment_version":env.version,
        "carbon_source":replay.source_name,
        "carbon_rows":replay.rows,
        "carbon_time_accounting":None if args.carbon_time_accounting is None else {
            "version":"v2", "coverage_policy":args.carbon_time_accounting,
            "forecast_clock_alignment":"first selected job arrival = first NESO forecast row",
            "planning_signal":"fixed full-week regional mean intensity",
            "planning_carbon_intensities_g_per_kwh":np.mean(replay.intensities,axis=0).tolist(),
            "decision_scope":"offline counterfactual one-shot scheduling with full forecast access",
            "carbon_constraint_scope":"resource forecast-week mean intensity, not per-execution-interval guarantee",
        },
        "algorithms":algorithms,
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
