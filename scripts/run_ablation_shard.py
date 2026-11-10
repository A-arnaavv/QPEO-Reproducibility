"""Run five single-component QPEO ablations against the locked real scenarios.

Separate feature branch; not part of the already-running primary experiment batch.
Each output contains five genuinely executed ablations, not reconstructed or
invented metric values. Full QPEO and classical FPA are reused from matching
primary-scenario seed results when the final ablation report is assembled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from qpeo.config import ManuscriptQPEOConfig
from qpeo.optimizer import (
    QPEOOperatorSwitches,
    ReconstructionQPEOSettings,
    run_qpeo,
)
from qpeo.protocol import load_protocol
from qpeo.simulator import clear_simulation_cache, simulation_cache_stats
from scripts.run_final_experiment_shard import SCENARIOS, build_scenario


VARIANTS: dict[str, QPEOOperatorSwitches] = {
    "no_caqc_penalty": QPEOOperatorSwitches(use_caqc_carbon_penalty=False),
    "no_ebws": QPEOOperatorSwitches(use_ebws=False),
    "no_entropy_bro": QPEOOperatorSwitches(use_entropy_biodiversity=False),
    "no_global_qpso": QPEOOperatorSwitches(use_global_qpso=False),
    "no_local_qpso": QPEOOperatorSwitches(use_local_qpso=False),
}


def evaluate_ablation_variants(
    problem,
    inputs,
    settings: ReconstructionQPEOSettings,
    *,
    seed: int,
    population_size: int,
    max_iterations: int,
) -> list[dict]:
    cfg = ManuscriptQPEOConfig(
        population_size=population_size,
        max_iterations=max_iterations,
    )
    records = []
    for variant, switches in VARIANTS.items():
        clear_simulation_cache()
        start = perf_counter()
        result = run_qpeo(
            problem,
            inputs,
            settings,
            config=cfg,
            seed=seed,
            operator_switches=switches,
        )
        elapsed = perf_counter() - start
        cache = simulation_cache_stats()
        records.append({
            "variant": variant,
            "seed": seed,
            "runtime_seconds": elapsed,
            "guide_objectives": result.guide_objectives.as_array().tolist(),
            "guide_objective_history": np.asarray(
                result.guide_objective_history, dtype=float
            ).tolist(),
            "archive_size": len(result.archive.entries),
            "archive_size_history": np.asarray(
                result.archive_size_history, dtype=int
            ).tolist(),
            "entropy_history": np.asarray(
                result.entropy_history, dtype=float
            ).tolist(),
            "simulation_cache_hits": int(cache["hits"]),
            "simulation_cache_misses": int(cache["misses"]),
        })
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    p.add_argument("--seed-index", required=True, type=int)
    p.add_argument("--size", type=int, default=64, choices=[32,64,128])
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--output-dir", default="results/ablations/raw")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    primary, _ = load_protocol(ROOT / "configs" / "reconstruction_protocol.json")
    if not 0 <= args.seed_index < len(primary.seeds):
        p.error("seed-index is outside locked 30-seed range")
    seed = int(primary.seeds[args.seed_index])
    P = primary.population_size if args.population_size is None else args.population_size
    T = primary.max_iterations if args.max_iterations is None else args.max_iterations
    is_final = (
        args.size == 64 and args.population_size is None
        and args.max_iterations is None
    )
    tag = "final" if is_final else f"test-P{P}-T{T}-N{args.size}"
    outdir = ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (
        f"{args.scenario}__seed{args.seed_index:02d}_{seed}"
        f"__ablation__{tag}.json"
    )
    if path.exists() and not args.force:
        print(f"exists, skipping: {path}")
        return

    bundle, env, replay = build_scenario(args.scenario, args.size)
    settings = ReconstructionQPEOSettings(
        scalar_weights=primary.scalar_guide_weights,
        alpha=primary.alpha,
        beta=primary.beta,
        gamma=primary.gamma,
        dependency_threshold=primary.dependency_threshold,
        dependency_strength=primary.dependency_strength,
    )
    records = evaluate_ablation_variants(
        bundle.problem,
        bundle.qpeo_inputs,
        settings,
        seed=seed,
        population_size=P,
        max_iterations=T,
    )
    payload = {
        "schema_version": "qpeo-ablation-shard-v1",
        "final_protocol": is_final,
        "scenario": args.scenario,
        "dataset_id": bundle.dataset_id,
        "seed_index": args.seed_index,
        "seed": seed,
        "size": args.size,
        "population_size": P,
        "max_iterations": T,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "resource_environment_version": env.version,
        "carbon_source": replay.source_name,
        "carbon_rows": replay.rows,
        "variants": records,
    }
    if len(records) != len(VARIANTS):
        raise RuntimeError("Expected five ablation variants")
    with tempfile.NamedTemporaryFile(
        "w", dir=outdir, delete=False, encoding="utf-8", suffix=".tmp"
    ) as tmp:
        json.dump(payload, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    print(path)


if __name__ == "__main__":
    main()
