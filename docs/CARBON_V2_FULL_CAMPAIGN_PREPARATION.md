# Carbon-v2 corrected full-cohort campaign — preparation and audit

**Prepared:** 12 November 2026  
**Status:** Candidate protocol and campaign tools created; **0 of 120** full corrected shards executed.  
**Source:** `carbon-accounting-v2-validation` GitHub branch.  
**Do not merge new statistical results into the manuscript until all 120 shards have been independently validated.**

## Scope

- Four trace scenarios: Alibaba GPU v2023, Microsoft ElasticFlow, SenseTime Helios Earth, and Alibaba Spot-GPU A100 (2026).
- 64 selected jobs per scenario; 13 algorithms; 30 predefined seeds; 50 population members; 500 iterations.
- Expected **120** scenario-seed shards with **1,560** algorithm records.
- Candidate study explicitly uses the corrected aggregate CPU/GPU/RAM scheduling logic, a fixed forecast-week regional mean as the offline planning/eligibility signal, and **time-aligned energy-weighted carbon intensity** along the simulated workload schedule. The carbon week is repeated beyond 168 hours as a **synthetic counterfactual**, not an observed forecast.
- Reuse the original fixed seeds and budgets for comparability, while marking this as a **changed evaluation protocol** (not a claimed prospective preregistration).

## Files added

- `configs/carbon_v2_candidate_full_campaign.json`: complete candidate methodology, constraints and publication gates.
- `scripts/run_carbon_v2_campaign.py`: read-only 120-task `plan`; optional **one-shard-only** `run-one --execute` with pinned dependency, branch and clean-checkout checks.
- `scripts/audit_carbon_v2_campaign.py`: strict 120-shard, 1,560-algorithm-record audit with independent schedule, CPU/GPU/RAM capacity, energy and carbon replay. It **refuses** to run descriptive and inferential analyses on incomplete/mixed/fast-audited cohorts.
- `tests/test_carbon_v2_campaign.py`: synthetic regression tests against missing, corrupted and inconsistent shard metadata. No optimization jobs are run by the test suite.

## Safe commands

Review the plan without launching computation:

```bash
python scripts/run_carbon_v2_campaign.py plan
```

To audit how many **full** shards have actually been produced (audit only; no optimization):

```bash
python scripts/audit_carbon_v2_campaign.py \
  --input-dir results/carbon_v2_campaign/<source_git_sha12> \
  --fast-audit
```

To opt in to exactly **one full-size** shard after coauthor approval of the candidate protocol and a pinned local environment:

```bash
python scripts/run_carbon_v2_campaign.py run-one \
  --scenario microsoft_elasticflow_itp_cluster08 \
  --seed-index 0 \
  --execute
```

The `run-one` command **never** dispatches GitHub Actions and does not overwrite the legacy v1 output directory. It refuses to run if the pinned dependencies do not match, the Git checkout is dirty, or the selected branch is not `carbon-accounting-v2-validation`.

After all 120 shards exist, **only then**:

```bash
python scripts/audit_carbon_v2_campaign.py \
  --input-dir results/carbon_v2_campaign/<source_git_sha12> \
  --analyze
```

The analyzer produces separate Carbon-v2 descriptive statistics, paired Wilcoxon, Friedman, Nemenyi, a per-run modeled carbon-mass CSV, a carbon-mass descriptive table, and a methodology manifest. It **never** modifies the original `results/analysis/` output or the current manuscript. It does not claim that modeled carbon mass is measured physical emissions.

## Testing done for campaign preparation

Executed **110/110** full-repository pytest tests locally, including the six new campaign-control tests, in the existing validation workspace. Also checked a dry-run 120-task plan, a zero-shard audit showing 120 missing records and zero invalid records, and the explicit `--execute` guard. **No full-size Carbon-v2 shard was executed by this preparation step.**

### Remaining gates

1. Coauthor review of the changed offline carbon planning model, the weekly repetition policy and the claims that can be supported.
2. Pinned environment (repo requirements: NumPy 2.4.6, SciPy 1.17.1, pytest 9.2.0) on the machine that executes the full campaign.
3. Freeze one committed Git source revision across all 120 shard runs.
4. Execute all shards under the same code/configuration and validate independently; a missing or corrupted shard prevents inferential analysis.
5. Regenerate the revised Tables 4–6, affected figures and statistical discussion from **only** the new complete cohort. Older v1 numbers must not be mixed with corrected results.
6. Evaluate ablations, robustness or hypervolume separately if stated in the manuscript; this primary six-objective cohort alone is not evidence for unrun experiments.

**Scientific caution:** The energy/carbon outputs are counterfactual simulated GPU-side estimates; do not call them direct datacentre power measurements, observed UK emissions, verified annual carbon savings, or renewable-energy utilization percentages.