# QPEO reproducibility rebuild — v5

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

## Important status
**No paper result values have been fabricated or back-filled.** The current code is a
reconstructed experimental framework. Real paper results must be regenerated after:
- public trace windows are fixed,
- missing trace fields are handled by explicit documented policies,
- service-time / power / price models are fixed,
- undocumented QPEO parameters are selected and reported,
- all algorithms are run for the final 30-seed protocol.

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
