# QPEO Carbon-v2 — first corrected full-size shard

**Status: prepared, not executed.** The corrected full-study campaign remains at **0 of 120 shards**. The most recent local repository test suite passed **113 of 113 tests**, but the runtime has NumPy 2.3.5, SciPy 1.17.0 and pytest 9.0.2 rather than the exact required 2.4.6, 1.17.1 and 9.2.0. No matching installation wheels are available here, and this runtime cannot fetch them. **Do not call any unpinned run publication-grade.**

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


## Latest local readiness check (12 November 2026)

- The safe first-shard launcher now generates a **deep** collection audit without `--fast-audit`, so it independently replays the newly completed shard when run. A regression test was added.
- Full local repository regression suite (against the uploaded source snapshot plus these launcher changes): **113/113 passed**.
- The local test container has Python 3.13.5, but NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2. Exact locked versions are NumPy 2.4.6, SciPy 1.17.1, pytest 9.2.0. Required wheels were unavailable from the local cache and network-backed install is not available here; no full-size shard was run.
- A read-only *empty-cohort* audit confirmed **0 valid, 120 missing, 0 invalid**, as expected before running a full shard. The empty-cohort audit is not evidence that any experiment has completed.
- The next action remains setting up the pinned Python 3.13 environment on an internet-connected computer (`bash scripts/carbon_v2_first_shard_local.sh --install`), followed by one explicitly approved `--execute-one` full-size experiment. The full 120-shard study and manuscript results remain pending.


## Provenance gate strengthened (12 November 2026, subsequent audit)

The corrected primary-campaign runner was audited further. Two gaps were fixed on the same validation branch:

1. **Reuse of existing output:** The runner now calls the strict campaign-level `validate_campaign_shard(..., deep=True)` for any existing or newly created shard. A merely pilot-valid JSON with incorrect P, T, history length, filename, or seed cannot be silently reused as a full-size result.
2. **Runtime provenance:** The runner explicitly requires Python 3.13.x. Each carbon-v2 output records Python, NumPy, SciPy and pytest versions. The campaign auditor verifies these recorded values against the pinned requirements (`numpy==2.4.6`, `scipy==1.17.1`, `pytest==9.2.0`). It rejects an unpinned run even when the numbers and simulated schedule seem internally consistent.

Three new regression tests cover the Python version guard, dependency provenance, and mislabeled pilot reuse. **The complete local test suite passes 116/116 tests.** The four modified source/test files were independently confirmed to have **identical Git blob SHA-1s** between the locally tested tree and the GitHub validation branch.

Official PyPI release pages confirm these package versions exist, but the current isolated container cannot download them, so the first publication-grade shard remains **NOT EXECUTED (0/120)**. Running tests with the container's NumPy 2.3.5, SciPy 1.17.0 and pytest 9.0.2 does not satisfy the locked campaign requirement.

To prepare a **fresh, clean clone** on a Mac/Linux machine with package access:

```bash
git clone --branch carbon-accounting-v2-validation https://github.com/A-arnaavv/QPEO-Reproducibility.git
cd QPEO-Reproducibility
bash scripts/carbon_v2_first_shard_local.sh --install
```

This is **setup/testing only**, not a full-size experimental run. After methodology approval, deliberately execute **exactly one** corrected shard:

```bash
bash scripts/carbon_v2_first_shard_local.sh --execute-one
```

The runner now requires **full campaign-level and schedule-level checks** for that one shard, and the launcher performs a deep collection audit. The collection is expected to show **1 valid, 119 missing, 0 invalid** until further runs occur. Share the single saved JSON shard and campaign audit for independent review; do not substitute the earlier P=50/T=20 exploratory pilot.


## Dataset integrity gate (12 November 2026)

The first full-size shard, its underlying runner, and its recorded execution provenance now treat the entire **data/** tree as part of the source-cleanliness gate, alongside qpeo/, scripts/, configs/, tests/, requirements.txt and pyproject.toml. This blocks execution if any committed input trace, the frozen NESO forecast CSV, or the archived NESO raw JSON is modified, or if any untracked data files are present. The source Git SHA therefore identifies the data as well as the algorithm code and configuration. Two new regression tests explicitly verify this requirement.

**Actual local tests:** 118 / 118 passed in Python 3.13.5 with the *unpinned* container packages (NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2). These tests verify the new gate but do NOT satisfy the publication-run environment. The source and regression files match their GitHub committed blob hashes exactly.

**Campaign state:** 0/120 full-size shards executed. The first full-size shard still requires a fresh clean Git checkout and the three pinned package versions. Do not relabel the earlier one-seed P=50/T=20 exploratory test as P=50/T=500 evidence; do not modify the manuscript's numerical tables until the corrected complete study is independently audited.
