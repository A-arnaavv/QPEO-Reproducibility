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
