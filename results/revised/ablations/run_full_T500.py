import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from qpeo.config import ManuscriptQPEOConfig
from qpeo.optimizer import run_qpeo, ReconstructionQPEOSettings
from qpeo.protocol import load_protocol
from qpeo.simulator import clear_simulation_cache
from scripts.run_revised_experiment_shard import SCENARIOS, build_scenario

P = 50
T = 500
N = 64

primary, _ = load_protocol("configs/revised_protocol_20261115.json")

settings = ReconstructionQPEOSettings(
    scalar_weights=primary.scalar_guide_weights,
    alpha=primary.alpha,
    beta=primary.beta,
    gamma=primary.gamma,
    dependency_threshold=primary.dependency_threshold,
    dependency_strength=primary.dependency_strength,
)

sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True
).strip()

output_dir = Path("results/revised/ablations/full_T500_3seeds")
output_dir.mkdir(parents=True, exist_ok=True)

variants = {
    "full_QPEO": {},
    "without_EBPP": {
        "stagnation_iterations": T + 1,
    },
    "without_CAQC_penalty": {
        "kappa_min": 0.0,
        "kappa_max": 0.0,
    },
}

completed = 0

for scenario in SCENARIOS:
    print(f"\nSCENARIO: {scenario}", flush=True)

    bundle, _, replay = build_scenario(scenario, N)

    sim = replace(
        bundle.qpeo_inputs.simulator_inputs,
        carbon_replay=replay,
    )
    inputs = replace(
        bundle.qpeo_inputs,
        simulator_inputs=sim,
    )

    for seed_index in range(3):
        seed = int(primary.seeds[seed_index])

        for variant, overrides in variants.items():
            filename = (
                f"{scenario}__seed{seed_index:02d}"
                f"__{variant}.json"
            )
            path = output_dir / filename

            if path.exists():
                saved = json.loads(path.read_text())
                assert saved["source_git_sha"] == sha
                assert saved["scenario"] == scenario
                assert saved["seed"] == seed
                assert saved["variant"] == variant
                assert saved["population_size"] == P
                assert saved["max_iterations"] == T
                assert saved["workload_count"] == N
                assert len(saved["objectives"]) == 6
                assert np.all(np.isfinite(saved["objectives"]))

                completed += 1
                print(
                    f"  SKIPPED completed: seed={seed} "
                    f"variant={variant}",
                    flush=True,
                )
                continue

            print(
                f"  RUNNING: seed={seed} "
                f"variant={variant}",
                flush=True,
            )

            clear_simulation_cache()

            config = ManuscriptQPEOConfig(
                population_size=P,
                max_iterations=T,
                **overrides,
            )

            start = time.perf_counter()

            result = run_qpeo(
                bundle.problem,
                inputs,
                settings,
                config=config,
                seed=seed,
            )

            elapsed = time.perf_counter() - start

            f = np.asarray(
                result.guide_objectives.as_array(),
                dtype=float,
            )

            assert f.shape == (6,)
            assert np.all(np.isfinite(f))

            record = {
                "schema": "qpeo-ablation-T500-v1",
                "source_git_sha": sha,
                "scenario": scenario,
                "seed_index": seed_index,
                "seed": seed,
                "variant": variant,
                "variant_overrides": overrides,
                "workload_count": N,
                "population_size": P,
                "max_iterations": T,
                "objectives": f.tolist(),
                "emissions_g": float(f[1] * f[2]),
                "elapsed_seconds": float(elapsed),
                "archive_size": len(result.archive.entries),
            }

            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(record, indent=2) + "\n"
            )
            temporary.replace(path)

            completed += 1

            print(
                f"  SAVED: seed={seed} variant={variant} "
                f"emissions={record['emissions_g']:.2f} g "
                f"runtime={elapsed:.1f}s "
                f"progress={completed}/36",
                flush=True,
            )

print("\nFULL-BUDGET ABLATIONS COMPLETE")
print("Validated runs:", completed, "/ 36")
print("Source commit:", sha)
print("Results:", output_dir)
