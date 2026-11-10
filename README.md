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
replay and deterministic resource-to-region mapping. Carbon thresholds are derived
from replay tertiles (33rd/67th percentile plus a flexible maximum) rather than tuned
against algorithm results.
