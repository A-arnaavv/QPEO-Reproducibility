# QPEO Component-Ablation and EBWS Diagnostic Results

## Full-budget component ablations

The directory `full_T500_3seeds/` contains 36 validated JSON records:

- Four trace-derived scheduling scenarios
- Three matched random seeds per scenario
- Three variants: full_QPEO, without_EBPP, and without_CAQC_penalty
- Population size P=50
- Optimisation iterations T=500
- Workload count N=64

The experiments were generated using source commit:

220ec284d24d92c07ccd129f593c3e4352216401

This original commit belongs to a separate development history and is not currently available as a commit on the public reproducibility branch.

A comparison against the corrected-validation-20261115 branch confirmed that the tracked QPEO implementation, experimental scripts, configurations, workload data, and dependency specifications were unchanged. The only difference in those inspected paths was the removal of a cloud batch-launch manifest.

The original ablation runner is preserved as `run_full_T500.py`. Its saved-result validation checks the original source commit, so executing it against existing records from a later Git revision may fail that provenance assertion.

These three-seed ablation results provide descriptive component-level evidence, not statistically generalisable proof of improvement.

## EBWS synthetic-dependency diagnostic

The file `ebws_synthetic_chain_seed00_T100.json` records a separate single-seed diagnostic of nonzero workload-dependency coupling.

This diagnostic demonstrates sensitivity to EBWS but does not establish general scheduling improvements or precedence-aware scheduling effectiveness.

The four primary benchmark scenarios use zero-valued dependency matrices, and the simulator does not enforce workload-precedence constraints.
