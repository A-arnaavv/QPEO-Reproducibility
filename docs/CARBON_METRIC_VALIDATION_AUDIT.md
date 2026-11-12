# Carbon-metric validation audit — 12 November 2026

**Decision: HOLD replacing manuscript Table 6 until its metric and evaluation scope are explicitly decided.** Do not present old or reconstructed carbon-intensity numbers as real emissions, renewable utilization, or annual CO2 savings. This is a source-backed audit of the existing completed runs; it does not modify source code or results and does not start compute jobs.

## 1. Code tracing

- `configs/carbon_replay_v1.json` and `qpeo/carbon.py::CarbonReplay.at_iteration(t,T)` map optimization iteration `t` to a row of the frozen NESO Jan 5–11 2026 regional **forecast** by `round(t*(K-1)/T)`.
- `qpeo/optimizer.py` and `qpeo/baselines.py` refresh the problem's carbon-intensity vector each iteration. On iteration **500**, all methods see the **final 2026-01-11 23:30 UTC forecast row**.
- `qpeo/evaluator.py::evaluate_objectives` defines `f3 = dot(E_resource, CI_resource)/sum(E_resource)` (for positive energy); `CI_resource` is from **one snapshot**, **not** a time integral of per-job resource power over carbon forecasts across the actual job schedule.
- Thus saved final guide `f3` is **terminal-snapshot energy-weighted regional forecast intensity**. The optimizer reacts to changing forecast signals across its search, but final `f3` does **not** measure time-aligned workload emissions and does not integrate the preceding weekly forecast history.

### Final frozen forecast row, January 11 2026, 23:30 UTC

|Region/resource region|Forecast gCO2/kWh|
|---|---:|
|North Scotland|0|
|North West England|46|
|South Wales|131|
|East England|73|
|South England|144|
|London|100|
|South East England|162|

Across the 336 in-window half-hour rows, North Scotland has **30** zero forecasts, and its arithmetic time-mean is **95.06 gCO2/kWh**. The final-time value **0** is unusual relative to the week-average but originates in the frozen forecast data; it is not an explicit missing-data flag. This is a forecast, not observed physical emissions at the non-UK workload sites.

## 2. Direct checks of the 1,560 algorithm×scenario×seed summary rows

Source: `results/final_summary_long.csv`, columns `f2_energy_kwh` and `f3_carbon_intensity`.

**Both zero carbon-intensity and strictly positive modeled energy occur in the SAME runs:**

- Alibaba GPU v2023: WOA **30/30** seeds have `f3=0` and `f2=21.19691667 kWh`; ACO **30/30** also have `f3=0` and `f2=21.19691667 kWh`.
- Helios Earth: WOA **30/30** seeds `f3=0`, energy `15.66441667 kWh`; ACO **6/30** `f3=0`, same positive energy.
- Alibaba Spot-GPU A100: WOA **30/30** seeds `f3=0`, energy `42.91011951 kWh`.
- Microsoft ElasticFlow: WOA **11/30** seeds `f3=0`, energy `9.04641667 kWh`.

The `f3=0` cases are mathematically consistent with allocation of all energy to the only zero-intensity resource **at the final forecast row**. These summary values alone cannot establish every original resource assignment, but no zero-energy explanation applies.

## 3. Why this does not validate carbon-footprint claims

- The final carbon metric uses **one forecast time**, regardless of workload arrival/completion. It is an optimization-decision snapshot proxy, not a workload-period carbon measurement.
- The main Alibaba GPU v2023 makespan is approximately **308 hours**, longer than the fixed **168-hour** replay window. There is no documented mapping from the full execution timeline to adequate contemporaneous carbon forecasts.
- In these counterfactual tests, UK intensity signals are mapped to seven logical resources even though the source traces are heterogeneous and not from these exact UK grid regions.
- A zero value in the terminal forecast can lead to a zero `f3` despite significant energy demand; that is **not** proof of zero environmental impact.
- Carbon **intensity** in gCO2/kWh and **total emissions** in gCO2 are different quantities. The latter requires integrating energy consumption by resource and time against applicable grid-intensity forecasts.
- The source inputs for the completed cohort also omit temporal aggregate CPU-capacity enforcement; this separate limitation still needs correction or accurate scoping.

## 4. Can current exported raw shards be recalculated without rerunning?

**No, not for a time-aligned carbon objective.** `scripts/run_final_experiment_shard.py::_serialize_result` writes guide objective vectors, histories, archive size, runtime, and cache counters. It does **not** serialize the winning `guide_assignment`, final per-resource energy vectors, job-level start/end intervals, or complete Pareto fronts. `results/final_summary_long.csv` similarly exports only scalar objectives. Thus a schedule-aware carbon recalculation cannot be recovered simply from the existing 120 saved result files.

## 5. Two scientifically distinct next paths

**Path A — preferred for primary claims of sustainable scheduling and carbon savings:**
1. Specify how the job-replay clock aligns with carbon-intensity timestamps for all job intervals. Use a carbon source covering the longest simulated workload horizon or predeclare and justify a repeating/rebased forecast policy (and evaluate this as a sensitivity condition). Do not silently reuse a 168h forecast for >300h jobs.
2. Correct/check time-indexed energy accounting and (if claiming CPU-GPU co-scheduling) enforce aggregate CPU capacity at each overlapping interval.
3. Define objective `f3` as a clear reproducible carbon criterion, e.g. total modeled operational carbon mass `sum_{j,t} E_{j,t}(kWh)*CI_{j,t}(gCO2/kWh)`, or time-aligned energy-weighted intensity `sum(E*CI)/sum(E)` across the full simulated schedule, clearly specifying forecasts vs actuals.
4. Implement tests with known synthetic schedules and zero/nonzero forecast intervals; validate a one-scenario/one-seed pilot at no paid cloud cost **before** any full rerun.
5. Re-execute all 13 methods under a common corrected protocol, and regenerate **all** six-objective results/statistics, figures, and ablation where needed. Changing an optimization objective or feasibility constraints affects search decisions and cannot be repaired merely by re-labeling existing carbon values.

**Path B — fast but much narrower evidence, no rerun:**
- Preserve current `f3` numbers as **terminal-snapshot energy-weighted carbon-intensity forecast proxies**, including the zero entries. Label and discuss their dependence on the Jan 11 endpoint and forecast zero. Do **not** call them time-integrated emissions, carbon footprint, real energy environmental impacts, observed power, or renewable share; do not infer annual savings. Explain that terminal-snapshot comparisons need not generalize to different forecast times. Do not assert broad superior carbon-aware scheduling effectiveness from this proxy alone.
- The results in other objectives may be discussed under their separately disclosed model limitations. Reviewers may reasonably find this carbon evaluation insufficient for the paper's sustainability claims.

**Recommendation:** For the intended revised Scientific Reports primary evaluation, **Path A is scientifically stronger**. Mark current GitHub metrics/tables as **provisional** until model audit and corrected protocol are settled. Do not silently insert zeros or invent replacement numbers.

## Evidence pointers

- `qpeo/carbon.py`; `qpeo/evaluator.py`; `qpeo/optimizer.py`; `qpeo/baselines.py`.
- `data/carbon/neso_regional_2026-01-05_2026-01-11.csv`; `configs/carbon_replay_v1.json`.
- `scripts/run_final_experiment_shard.py`; `results/final_summary_long.csv`.
- Official NESO regional-intensity forecast dataset: https://www.neso.energy/data-portal/regional-carbon-intensity-forecast
