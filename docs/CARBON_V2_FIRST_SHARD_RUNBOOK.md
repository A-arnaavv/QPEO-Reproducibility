# QPEO Carbon-v2 — first corrected full-size shard

**Status: prepared, not executed.** The corrected full-study campaign remains at **0 of 120 shards**. The most recent local repository test suite passed **112 of 112 tests**, but the runtime has NumPy 2.3.5, SciPy 1.17.0 and pytest 9.0.2 rather than the exact required 2.4.6, 1.17.1 and 9.2.0. No matching installation wheels are available here, and this runtime cannot fetch them. **Do not call any unpinned run publication-grade.**

## What is now available

The `carbon-accounting-v2-validation` branch contains an opt-in local setup and single-shard launcher:

`scripts/carbon_v2_first_shard_local.sh`

It checks the correct branch, rejects tracked or untracked source edits, records the Git revision, requires Python **3.13** and the exact packages from `requirements.txt`, reruns the complete test suite, and launches **one** dataset–seed pair only after an explicit `--execute-one` command. It also invokes the read-only campaign audit afterward. It never dispatches GitHub Actions or overwrites the original v1 results.

## Commands on an internet-connected Mac or Linux computer

Open Terminal. Use a fresh clone, not the earlier downloaded ZIP, to preserve the real Git revision:

```bash
git clone --branch carbon-accounting-v2-validation \
  https://github.com/A-arnaavv/QPEO-Reproducibility.git
cd QPEO-Reproducibility
```

Install Python 3.13 beforehand if your computer does not have the `python3.13` command. Prepare the isolated environment and run the tests **without starting experiments**:

```bash
bash scripts/carbon_v2_first_shard_local.sh --install
```

Only after the corrected candidate protocol has been approved, deliberately run the first full-size shard on **your own local machine**:

```bash
bash scripts/carbon_v2_first_shard_local.sh --execute-one
```

This uses Microsoft ElasticFlow, seed index 0 (seed 104729), all 13 algorithms, population 50, **500** iterations, and the disclosed offline fixed-mean planning / time-integrated repeated-week carbon policy. It is CPU-intensive and may take considerable time; it is *not* the 120-shard campaign.

Result file, inside the checkout:

`results/carbon_v2_campaign/<git-sha-first-12>/microsoft_elasticflow_itp_cluster08__seed00_104729__test-P50-T500-N64-carbon-time-v2-repeat-fixedmean.json`

The command also creates an audit report under `results/carbon_v2_campaign/analysis/<git-sha-first-12>/campaign_audit.json`. The audit should show **1 valid, 119 missing, 0 invalid**. Missing shards are expected at this stage. It is **not** the final statistical analysis.

## What to return here

Upload the single JSON shard and `campaign_audit.json`, or a ZIP containing both, together with any command error output if something fails. The result can then be independently inspected. Do **not** upload the old 120 v1 shards or merge v1 and v2 numbers.

## Remaining scientific gates

The repeated 168-hour UK carbon forecast is a counterfactual modeling assumption, not later-week measured grid intensity; resource carbon-class eligibility uses a fixed forecast-week mean and does not guarantee an instantaneous emissions limit. Even after one full shard passes, the protocol still requires coauthor review, complete corrected 120-shard cohort, paired statistics, and regenerated figures/tables before manuscript claims can be updated.
