# QPEO reproducibility rebuild — v5


## Current manuscript benchmark — corrected evaluation (16 November 2026)

The results reported in the current QPEO manuscript are based on the completed corrected scheduling and carbon-accounting protocol.

The validated comparative evaluation comprises four trace-derived workload scenarios, thirteen algorithms (QPEO and twelve baselines), thirty matched random seeds per scenario, and six scheduling objectives. All algorithms use a population size of P=50 and T=500 optimisation iterations.

The corrected benchmark contains 120 validated scenario–seed experiments and 1,560 algorithm-level results. Across 288 paired Wilcoxon comparisons with study-wide Bonferroni correction at familywise alpha=0.01, 81 significant comparisons favour QPEO, 80 favour baselines, and 127 are not statistically significant.

The authoritative corrected results are located at:

- `results/revised/raw/` — 120 validated experimental records.
- `results/revised/analysis/complete_cohort_20261116/` — final statistical outputs, tables, and figures.

The earlier results in `results/raw/` and `results/analysis/` belong to the historical pre-correction evaluation. Their statistical totals (76/108/104) are superseded and must not be used to interpret the current manuscript.

The reported findings concern reconstructed trace-driven scheduling simulations, not physical cloud–edge deployments. Energy-weighted carbon intensity and total modelled carbon emissions are distinct metrics.


This repository reconstructs the experimental implementation of **Quantum Pollination
Ecology Optimization (QPEO)** from the current manuscript after loss of the original
codebase.

## Core rule
The project distinguishes between:
1. **manuscript-specified behavior**, implemented directly; and
2. **lost-code choices**, which are never silently guessed and are recorded as explicit
   reconstruction conventions in `RECONSTRUCTION_LOG.md`.

## Implemented in v5
- workload/resource data model
- six manuscript objectives and five formal constraints
- explicit service-time + power simulator
- QESR/QPSO/CAQC/EBWS/EBPP/BRO operators
- feasibility-aware QPEO realization and Pareto archive
- reconstructed QPEO Algorithm 1 loop
- trace materialization/audit layer
- matched-budget baseline harness
- original nine reconstructed baselines: PSO, DE, GWO, WOA, ABC, GA, HS, FA, ACO
- reviewer-critical baselines: FPA, QIEA-style baseline, NSGA-II
- paired Wilcoxon, Cohen d_z, Friedman, Nemenyi, confidence intervals, Bonferroni
- explicit-reference Sobol hypervolume estimator
- 13-algorithm experiment suite

## Historical pre-correction benchmark (10 November 2026; superseded)
**Primary computation and validation are complete.** All 120 locked
scenario-seed shards are committed in `results/raw/`: four trace-based simulated
scenarios, 30 predeclared seeds each, 13 algorithms, 1,560 algorithm runs.
`results/final_shard_audit.json` reports 120/120 valid files, zero missing and
zero invalid shards. Validated aggregation and statistical outputs are in
`results/final_summary_long.csv` and `results/analysis/`, including 288 paired
QPEO-versus-baseline Wilcoxon comparisons.

**Interpretation is mixed, not universal superiority.** Under the 288-test
Bonferroni correction at familywise alpha 0.01, 76 significant comparisons
favor QPEO, 108 favor baselines and 104 are not significant. QPEO is strongest
on *energy-weighted carbon intensity*, which is not total emissions. Energy,
makespan, utilization, SLA and modeled cost must be discussed independently.
The original implementation was lost: these are **independently reconstructed,
trace-based simulation results**, not replicas of the original measurements.
The equal population/iteration budget is not an equal evaluation-call budget.
Performance, power and cost rely on the disclosed proxies in
`docs/LOCKED_PRIMARY_METHODS.md`.

**Source provenance:** `results/reproducibility/seed00_parity.json` documents
canonical rechecks of three earlier seed-0 cohorts. Alibaba and Microsoft
were equivalent; the original Alibaba Spot seed-0 cohort differed materially
for QPEO, PSO and NSGA-II, so the original was archived and the canonical
result retained. Helios seed 0 was computed under an earlier source commit
with the same experiment logic; later changes included documentation and
dependency pinning. That dependency/source provenance must remain disclosed.

**Historical checklist (10 November 2026; superseded):**
- [ ] Complete and report carefully scoped ablations, sensitivity and robustness
      checks, or justify transparently any studies not performed; do not
      silently change the locked primary protocol
- [ ] Integrate objective-level results, paired effects, rankings and limitations
      into the manuscript; avoid unsupported hypervolume or stationary
      convergence claims
- [ ] Check dataset/baseline citations, bibliography, figures, equations and
      computational-budget fairness disclosures
- [ ] Perform final provenance review and supervisor/co-author manuscript review

The historical development notes below describe earlier stages. The locked
configuration files, validated raw shards and analysis reports are authoritative
for the final primary study.

## Tests
```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Toy integration check
```bash
python examples/full_suite_toy_run.py
```
This toy script is only a software validation and must not be cited as a manuscript
experiment.

See `DATASET_AUDIT.md` and `RECONSTRUCTION_LOG.md` before running paper experiments.


## v6 pre-registered reconstruction protocol
`configs/reconstruction_protocol.json` freezes all outcome-independent choices that can
be fixed before the real trace experiments: 30 deterministic seeds, equal primary
six-objective guide weights, equal primary ecological-compatibility weights,
dependency settings, and sensitivity grids. Hardware- and trace-dependent settings
remain explicitly unresolved and must be fixed before the final runs.


### Raw convergence histories
Every optimizer now records the six raw guide-objective values at initialization and
after every iteration (T+1 rows). Convergence-to-90% analyses must use these raw,
consistently scaled histories rather than the changing within-population scalar score.


## v7 resource environment
`configs/resource_environment_v1.json` locks the reconstructed heterogeneous resource
pool before final experiments. NVIDIA-published memory, maximum power, and FP32
throughput values are used for the 4090/A6000/A5000/Jetson Orin hardware classes.
FP32 is only a reproducible service-speed proxy; normalized cost indices remain
explicit reconstruction assumptions and are accompanied by sensitivity profiles.


## v8 carbon replay
`configs/carbon_replay_v1.json` pre-registers a fixed one-week NESO regional carbon
replay and deterministic resource-to-region mapping. Carbon thresholds use a
feasibility-guaranteed 1/3/all-resource rule rather than being tuned against algorithm
results.


## v9 dynamic carbon integration
The carbon replay is now consumed inside the optimization loops. QPEO and all 12
baselines receive the same carbon vector at each iteration. Current populations are
re-evaluated after each refresh, and current-state Pareto archives prevent stale carbon
objectives from contaminating comparisons across time.


## v11 Alibaba GPU trace scenario
The first real trace scenario is locked from Alibaba GPU Trace v2023. The repository
stores the 178-job eligible Succeeded single/fractional-GPU pool together with the
upstream Git blob SHA. A deterministic contiguous 64-job block is the primary
experiment, with nested 32/128-job scalability windows. Source execution time and
end-to-end latency are retained; no synthetic Alibaba deadline multiplier is used.


## TRACE2 Microsoft ElasticFlow ITP
The second real scenario uses Microsoft's public ElasticFlow/ITP traces. Cluster08 is
selected by a pre-results rule (largest positive-duration single-GPU pool). All 12,558
eligible raw jobs join exactly to the authors' deadline-bearing experiment trace.
Primary evaluation uses a deterministic contiguous 64-job window with nested 32/128
scalability windows. Raw arrivals, durations and GPU demand are retained; SLA
deadlines come from the authors' published experiment trace. Unreported CPU, memory
and bandwidth demands are explicitly zero/non-binding rather than fabricated.


## TRACE3 Helios Earth
The third production scenario uses SenseTime's public Helios traces. Earth is selected
by a pre-results rule (largest completed single-GPU pool). After requiring CPU<=16
for representability in the reconstructed resource pool, 293,903 jobs remain. The
primary experiment uses a deterministic 64-job window with nested 32/128 windows.
GPU and CPU demand, arrivals, execution duration, queueing and completion timestamps
come from the trace. Memory/bandwidth are explicitly zero/non-binding because Helios
does not publish them; carbon sensitivity is neutral/deferred for all jobs rather than
being inferred from behavior.


## TRACE4 Alibaba Spot A100
The fourth locked scenario uses Alibaba's 2026 Spot-GPU trace restricted to
A100-SXM4-80GB single-worker jobs. The pre-results 64-job primary window contains
both HP and Spot jobs. HP jobs are SLA-constrained and critical; Spot jobs are
best-effort, flexible, and excluded from the SLA denominator. Service times are
scaled from the source A100 FP32 throughput to each reconstructed target resource.


## Carbon data used in the final experiment
The locked primary carbon replay is the **NESO regional forecast** defined in
`configs/carbon_replay_v1.json`, with its immutable snapshot under `data/carbon/`.
The earlier Electricity Maps v8 prototype is superseded and is not part of the
final experimental protocol.
