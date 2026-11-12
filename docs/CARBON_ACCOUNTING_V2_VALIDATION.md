# Carbon-accounting v2 — isolated validation branch

**Status, 12 November 2026:** The corrected integration code is committed. **11 local pure-function/adapter tests passed**; two additional all-algorithm integration smoke tests were committed but have **not yet been executed**. The 13-algorithm 30-seed primary experiment has not been rerun.

The original main branch and its locked v1 experimental results remain unchanged.

## Diagnosis

In v1, the reported objective f3 is the modeled energy by resource weighted by the grid carbon-intensity forecast **at the final optimizer iteration**. The North Scotland terminal forecast has intensity zero, which explains several zero-intensity results with positive energy. It does not represent energy-associated carbon exposure over the jobs' execution intervals.

## Implemented correction (opt-in only)

- qpeo/carbon_accounting.py implements CarbonTimeSeries, integrate_time_aligned_carbon and account_simulation_result.
- Actual simulated job start/finish times, assigned resources and GPU fractions determine an instantaneous resource-utilization profile.
- Active GPU power is integrated against the appropriate forecast intervals, producing modeled total grams CO2e and energy-weighted carbon intensity.
- Forecast rows are interpreted as piecewise constant 30-minute intervals. The first selected workload arrival is aligned with the first NESO forecast row. This is a **counterfactual modeling alignment**, not a claim that the workloads ran in the UK in January 2026.
- The default STRICT coverage policy rejects any workload outside the frozen 168-hour forecast. The optional REPEAT policy cycles the week and is explicitly a **synthetic sensitivity assumption**, not observed later weeks.
- Per-resource energy is checked against the original simulator's calculated power/energy before accepting a result.
- qpeo/simulator.py adds optional carbon_time_series to SimulatorInputs; both QPEO and all baseline evaluators use the new f3 only when it is supplied. Otherwise v1 behavior is retained unchanged.
- scripts/run_final_experiment_shard.py adds the option --carbon-time-accounting strict|repeat and forces test-tagged output filenames. These outputs also retain guide assignments, start/end times, modeled carbon grams, and intensity for independent inspection. No locked __final.json artifacts are overwritten.

## Formula and units

For resource j, power P_j(u_j(t)) is in kW, duration t in hours, and forecast carbon intensity CI_j(t) in gCO2e/kWh:

    E_j = integral_active P_j(u_j(t)) dt              (kWh)
    C_j = integral_active P_j(u_j(t)) CI_j(t) dt      (gCO2e)
    C_total = sum_j C_j
    f3 = C_total / sum_j E_j                          (gCO2e/kWh)

The primary GPU-only power model excludes host and inactive idle power. Values are **forecast-based modeled exposure**, not measured physical emissions.

## Tests actually performed

The local file tests/test_time_aligned_carbon_accounting.py passed **11 out of 11** tests, including overlapping fractional workloads, terminal zero forecast, strict vs repeat coverage for a 308 h simulated workload, multi-resource energy weighting, capacity failures, nonzero active-only idle power, energy parity, and forecast time origin. These tests use analytical synthetic data and run without requiring an expensive experimental campaign.

tests/test_time_aligned_carbon_full_suite.py contains two **additional unexecuted** integration tests intended to exercise v1 and v2 semantics for all 13 algorithms at population 4, iterations 2. They are not counted among the 11 passed tests.

## Local pilot commands (NOT run in this audit)

Unit and small full-suite test:

    python -m pytest -q tests/test_time_aligned_carbon_accounting.py tests/test_time_aligned_carbon_full_suite.py

Small Microsoft pilot with 13 algorithms, one seed, population 4, two iterations, strict coverage:

    python scripts/run_final_experiment_shard.py --scenario microsoft_elasticflow_itp_cluster08 --seed-index 0 --population-size 4 --max-iterations 2 --carbon-time-accounting strict --output-dir results/carbon_v2_pilot

This generates a test-tagged carbon-v2 file, not a new primary result.

## Publication gates still unresolved

1. Execute the two new end-to-end integration tests and a small real-trace pilot in a local or otherwise explicitly approved environment. Do not automatically dispatch GitHub Actions.
2. The frozen NESO week has only **168 hours** of coverage; the Alibaba v2023 main scenario is about **308 hours** long. Strict mode must reject that mismatch. Either obtain a sufficiently long actual forecast or explicitly preregister the repeat-policy as a counterfactual sensitivity test.
3. **Updated:** In the opt-in v2 shard runner, the decoder and CAQC use the **fixed full-week regional mean forecast** for offline planning at every search iteration. The objective integrates intensity over actual simulated execution intervals. Thresholds are therefore **planning-average** resource eligibility checks, not per-interval execution guarantees. This is a modeled offline counterfactual, not a real-time dispatch demonstration.
4. Aggregate simultaneous CPU demand is not constrained by the current simulator, and EBWS has no dependency edges in the four primary traces. The carbon patch does not resolve these separate scientific issues.
5. After validation and a locked v2 protocol, the full 13-algorithm study must be rerun, then all objectives, statistics, figures and manuscript claims regenerated. Updating only Table 6 would be misleading because changing f3 changes the optimizer's solution choices.

**Do not insert the v1 carbon-intensity numbers into a manuscript Table 6 labelled as time-aligned carbon exposure or emissions.**


## Local follow-up validation (12 November 2026; completed in the analysis runtime)

The local tests/test_time_aligned_carbon_accounting.py suite was **rerun and 11/11 tests passed**.

Two additional local **synthetic constant-load** pilot scripts evaluated the exact frozen NESO North Scotland forecast values (NOT the full QPEO optimizer or source workload trace) using the committed qpeo/carbon_accounting.py:

|Period|Policy|Modeled energy|Modeled carbon|Energy-weighted intensity|Status|
|---|---|---:|---:|---:|---|
|First 16.4 hours of NESO week|strict|4.920 kWh|502.230 gCO2e|102.079 gCO2e/kWh|PASS|
|First 16.4 hours|repeat|4.920 kWh|502.230 gCO2e|102.079 gCO2e/kWh|PASS; same as strict inside forecast|
|308 hours, including forecast overflow|strict|—|—|—|PASS: rejects unsupported forecast horizon|
|308 hours, including periodic replay|repeat|92.400 kWh|9,298.350 gCO2e|100.631 gCO2e/kWh|PASS|

These values were cross-checked with a separate direct sum of actual 336 half-hour North Scotland forecast rows. **They are validation fixtures, not benchmark/QPEO results.** The terminal NESO value was zero, whereas the time-aligned 16.4-hour exposure was positive.

A provisional consistent **repeat** policy across all four scenarios is recorded in configs/carbon_v2_provisional_pilot_protocol.json. It is intended for *controlled counterfactual simulations*, not real subsequent-week grid forecasts. The configuration is **not locked for publication** until the full implementation is tested end-to-end, coauthors approve its assumptions, and the corrected experimental campaign is rerun. Strict mode remains useful to validate coverage for scenarios shorter than 168 hours.

**Full 13-optimizer integration status:** The two synthetic optimizer smoke tests have been committed but NOT executed in this analysis environment, which has only the carbon-accounting module, not a full runnable checkout of the private GitHub repository. Neither the small real-trace optimizer pilot nor full 1,560-run v2 study has been executed. Do not confuse accounting fixtures with algorithm performance results. No GitHub Actions were launched.


## Additional CPU-scheduling validation (12 November 2026)

The validation branch's qpeo/simulator.py was modified to check **concurrent aggregate CPU demand**, in addition to concurrent GPU fraction and memory demand, when scheduling overlapping jobs. Its event intervals now retain (start, finish, gpu_demand, memory_demand, cpu_demand). Resources without CPU limits continue to use infinite CPU capacity. This change will affect scheduling, makespan, energy allocation, and time-aligned carbon exposure, and therefore the legacy numerical results cannot be retained as corrected v2 outputs.

The local test environment imported the existing **exact carbon_accounting.py** and a **verbatim extraction of the new CPU interval scheduling helpers**. The local test command passed **17/17 tests**: 11 carbon accounting tests, 5 temporal CPU helper tests, and one joint CPU scheduling + carbon integration test. These are unit/analytical checks, **not a run against a full checkout of all 13 algorithms**.

Joint synthetic validation:
- Two jobs each use 7 CPU units and 0.3 GPU for 1 hour, arriving at 0.0 and 0.25 h. The resource has 8 CPU units, 1 GPU, and a 0.3 kW peak power proxy.
- Carbon intensity is 100 gCO2e/kWh for the first hour, then 400 gCO2e/kWh for the second hour.
- CPU-constrained execution queues starts to (0.0, 1.0) h, producing 0.18 kWh, **45.00 gCO2e**, **250 gCO2e/kWh**.
- With unlimited CPU, starts are (0.0, 0.25) h, producing the same 0.18 kWh, **24.75 gCO2e**, **137.5 gCO2e/kWh**.
- This demonstrates why time-aligned emissions must follow the **actual corrected** schedule. It does not demonstrate real-world or full-method performance.

GitHub regression tests committed:
- tests/test_temporal_cpu_capacity_v2.py (five full-simulator tests).
- tests/test_cpu_carbon_joint_v2.py (one full-simulator plus carbon-adapter test).

**These two new GitHub regression suites have not been run against the complete repository code in this environment.** The full optimizer integration suite and real-trace pilot remain unexecuted. No GitHub Actions were launched.

**Resolved in the opt-in v2 pilot runner:** Decoder feasibility is evaluated against a fixed regional **mean forecast-week** intensity, not an iteration-indexed value. The carbon outcome integrates time-aligned intensity over scheduled execution intervals. Regional mean eligibility does **not** imply that all job-time intensity values satisfy a strict class threshold, and the manuscript must avoid claiming that guarantee.

Once a runnable full repository checkout is available: first run python -m pytest -q tests/test_temporal_cpu_capacity_v2.py tests/test_cpu_carbon_joint_v2.py tests/test_time_aligned_carbon_full_suite.py; then run scripts/run_final_experiment_shard.py with --population-size 4 --max-iterations 2 --carbon-time-accounting repeat for a single real-trace seed, inspect the persisted guide schedules and carbon totals, and only then consider the full cohort.


## Preflight and optimization improvements

- `qpeo/carbon_accounting.py` now takes a defensive read-only copy of the forecast matrix and caches its piecewise-constant cumulative intensity integral once per `CarbonTimeSeries`. This prevents rebuilding the 336-row table for every candidate and preserves numerical results. New immutability regression added.
- Locally executed pure-function and extracted CPU-helper checks: **18/18 passed** (12 accounting, five temporal CPU helper, one joint CPU–carbon example); full-repository 13-algorithm smoke tests still **not executed locally**.
- Synthetic 64-job/7-resource local integration stress: 2,000 calls took approximately **0.58 s** with cached integral vs **0.64 s** with forced recalculation; both returned identical modeled carbon to six decimal places. These figures are a microbenchmark of the accounting routine, not end-to-end QPEO timing.
- A read-only `scripts/validate_carbon_v2_pilot.py` now checks saved outputs for all 13 algorithms: assignment indices, 64 workload intervals, recorded service times, arrival constraints, simultaneous CPU/GPU/RAM capacities, and recomputed modeled energy and carbon against saved values.
- `scripts/run_carbon_v2_preflight.sh` supplies a self-contained local workflow for five small test modules, a **one-seed P=4/T=2** Microsoft trace pilot, and the validator. It does not launch Actions, run the full experimental cohort, or overwrite v1 `__final.json` results.

**Important:** The output validator and preflight shell script are committed but have **not been executed against a complete local checkout** in this environment. They are safeguards awaiting the real integration pilot, not evidence that the 13 optimizers have passed corrected end-to-end tests. Publication remains blocked until the pilot and complete corrected cohort are verified.


## Fixed offline planning signal — pilot methodological decision (12 November 2026)

The v2 branch now implements an explicit **offline fixed-plan / time-varying-execution** distinction in scripts/run_final_experiment_shard.py. For all 13 algorithms in the v2 pilot, the fixed planning vector is the 336-row arithmetic mean of the seven regional forecast series: [95.06, 95.33, 302.10, 154.33, 254.05, 191.78, 223.76] gCO2e/kWh (rounded). A workload is planning-eligible on 4/7, 6/7 and 7/7 resource regions for the Critical, Deferred and Flexible thresholds respectively. These values are not direct execution guarantees. The optimizer does **not** map its iteration index to new forecast timestamps in v2, removing artificial nonstationarity.

The carbon objective remains time-integrated GPU-side energy-weighted intensity along each simulated schedule. For runs longer than the frozen 168-hour forecast, explicitly repeating the week is a **counterfactual sensitivity policy**. No claimed real-time UK-grid deployment or measured CO2 reduction follows.

The updated test-tagged output records **fixedmean** in its filename and stores the planning vector and decision scope in metadata; the separate validator recomputes the schedule under the same fixed planning vector and checks timing, CPU/GPU/memory capacity and carbon/energy parity. The full-suite small synthetic test tests/test_carbon_v2_fixed_planning_full_suite.py and scripts/run_carbon_v2_preflight.sh have been updated. **These new end-to-end tests and the full trace pilot have NOT been executed** against a complete checked-out repository; no GitHub Actions were run. Original v1 outputs remain unchanged and must not be relabeled v2.
