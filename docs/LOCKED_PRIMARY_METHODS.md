# Revised QPEO Experimental Methods — Source-Locked Draft

**Purpose:** Methods text for the reconstructed Scientific Reports revision.
This document describes the **implemented, locked** primary experiment and its
limitations, not the irretrievably lost implementation or original paper numbers.
It is not a results section. Do not infer missing performance statistics from it.

## Evaluation setting

We evaluate the reconstructed QPEO approach against twelve optimizers: PSO,
DE, GWO, WOA, ABC, GA, HS, FA, ACO, FPA, a QIEA-style implementation, and
NSGA-II. Every optimizer runs over the same reconstructed scheduling instance
for a given trace scenario and random-seed index. The primary design uses a
population of 50, 500 iterations, and 30 predeclared seeds per algorithm for
each of four trace scenarios: Alibaba GPU Trace v2023, Microsoft ElasticFlow
(cluster08, single-GPU workloads), Helios Earth, and Alibaba Spot-GPU 2026
(A100 workloads). Each primary instance contains 64 eligible workload units.
The full design comprises `4 scenarios × 30 seeds × 13 algorithms = 1,560`
algorithm runs, stored as 120 scenario-seed result files.

The trace windows and filtering rules are fixed in
`configs/reconstruction_protocol.json` and the four
`configs/trace_*.json` files, with immutable normalized trace snapshots
under `data/`. For the primary scheduling model, eligible jobs have one
GPU or fractional single-GPU demand. Source-specific CPU, memory, deadline,
and class mappings differ where raw fields are absent or inapplicable and
must be reported separately. Do not describe the four datasets as having
identical native resource or QoS fields. These are **trace-based simulated
scheduling scenarios**; they are not measurements of QPEO deployed on all
seven hardware systems.

## Reconstructed heterogeneous resource pool

Each scheduling scenario uses the same seven logical resource slots:
one cloud RTX A6000, one cloud RTX A5000, one cloud RTX 4090, two edge
Jetson AGX Orin 64GB slots, and two serverless RTX A5000 slots.
Per-resource GPU capacity is one; the locked configuration specifies
CPU capacities, 64 GB host/system memory, GPU power proxies, and cost
indices in `configs/resource_environment_v1.json`.

Service duration on each resource is estimated from trace-observed
runtime using fixed relative FP32 throughput factors against RTX A6000
(38.7 TFLOPS). FP32 throughput is a transparent **modeling proxy**, not
a measured application-level throughput factor for each workload or
hardware platform. The primary resource power model uses incremental
GPU-side power `P(u)=P_max u`, zero idle-power fraction, and power
exponent one. It excludes host and idle power. Resource cost uses
dimensionless, fixed relative hourly cost indices: **not observed cloud
billing prices and not USD**. Normalized cost must be interpreted as
a reconstructed cost proxy.

For mixed-resource scheduling, the simulator enforces simultaneous
GPU, CPU and memory capacity over execution intervals as specified,
queues workloads when capacity is occupied, and permits fractional-GPU
overlap when feasible. Each workload receives exactly one logical
resource assignment. The evaluator reports six minimized objectives:
negative GPU utilization; modeled GPU energy; energy-weighted carbon
intensity; makespan; SLA-violation percentage; and normalized resource
cost. Dynamic carbon and feasibility are refreshed as scheduled.

## Fixed dynamic-carbon replay

The primary carbon policy uses the frozen National Energy System Operator
(NESO) **regional carbon-intensity forecast**, not the superseded
Electricity Maps development prototype. The snapshot covers
**5–11 January 2026**, with **336 half-hour observations**. Seven
predeclared UK regions map one-to-one to the seven logical resource slots.
At iteration `t` of `T` the shared replay index is
`round(t × (K−1)/T)`, for `K=336`. All algorithms receive the
same iteration-aligned carbon state for a given experiment.

The carbon-class thresholds are determined from worst-case row order
statistics over the frozen replay, guaranteeing at least one eligible
region for critical, three for deferred, and all seven for flexible
classes at each replay step. In the locked snapshot, the rounded
thresholds documented in the protocol are **192, 281 and 391
gCO2/kWh**, respectively (exact values include a numerical epsilon).
Missing values within a region are forward-filled; leading gaps are
back-filled from the first available observation. This is a controlled
**forecast replay**—not contemporaneous measured emissions at the
original workload sites.

For QPEO, ecological compatibility is recomputed after each carbon
refresh. For QPEO and all twelve baselines, current objective values
and the archive are updated to prevent comparisons with stale
carbon-dependent values.

## Experimental control and analysis

Primary QPEO scalar-guide weights assign `1/6` to each objective.
Ecological performance, thermal and carbon affinity weights are each
`1/3`. Dependency threshold and strength are fixed at `0.5`
and `0.25`. The complete seed list is recorded in
`configs/reconstruction_protocol.json`. Runs use a per-algorithm
cold simulation cache; identical assignments may reuse deterministic
schedule simulation but are reevaluated under the current carbon
state. The raw guide-objective vector is stored at initialization
and after each iteration (501 vectors per algorithm). All result
files contain runtime and simulator-cache hit/miss counts.

**Equal population size and iteration count are not equal objective
function evaluation budgets**. The comparison is P/T matched, and
method-specific simulation calls must be reported. Some algorithms
evaluate more candidates per iteration than others. Do not call
these runs NFE-matched.

Once all 120 final-protocol files are validated, compute the mean,
sample SD, 95% confidence interval and median for each metric.
Paired QPEO-versus-baseline Wilcoxon tests use matching seed indices,
alongside Cohen's paired `d_z`, per-scenario Friedman ranks and
Nemenyi pairwise rank comparisons. All six objectives are minimized.
Report raw `p` values, Bonferroni-72 values within each scenario,
and the more conservative Bonferroni-288 correction across all
four scenarios; the preregistered familywise alpha is `0.01`.
No incomplete seed collection will be used to make significance claims.

### Limitations requiring explicit disclosure

1. **Reconstruction:** Original source code and earlier reported numbers
   were lost. New results must be called independently reconstructed
   experiments, not replication of the original measurements.
2. **Performance and power proxies:** FP32-based service speeds and
   modeled GPU-only power/cost are transparent assumptions, not
   application benchmarks, direct wattmeter measurements, or invoices.
3. **Carbon replay:** Time-varying NESO forecasts are external to the
   calendar dates and geographies of the four source traces; this is a
   controlled scheduling benchmark, not a factual geo-placement trace.
4. **Convergence:** The objective landscape changes over iterations.
   Ordinary stationary time-to-90%-of-final-performance curves cannot
   be claimed from the saved histories without a separate, clearly
   defined dynamic-environment convergence metric.
5. **Hypervolume:** Current result shards save archive sizes and
   guide objectives but not complete per-run Pareto objective sets.
   Do not publish hypervolume estimates reconstructed from sizes.
6. **Provenance:** Three initial seed-0 files predate a simulator
   floating-point timestamp-tolerance correction (see the audit
   below). Their values require an equivalence check/recheck before
   treating all 120 shards as one implementation cohort.
7. **Trace heterogeneity:** Not every trace supplies native memory,
   CPU, deadlines or class labels. Report per-source reconstruction
   rules, rather than implying all variables were directly observed.

## Source revision audit and original seed-0 files

Canonical parallel-batch source commit:
`7d1efcf0980620e66db01b4d7c32d80081038471`.

| Scenario seed 0 | Recorded source SHA | Difference from canonical batch |
|---|---|---|
| Alibaba GPU v2023 | `c4bd0e16b3ec2d150de00d4885315cd361820c90` | Predates the simulator's sub-nanosecond timestamp tolerance, plus subsequent documentation/dependency changes |
| Microsoft ElasticFlow | `981145939269876c9de0ca12d97cf28b9ca4a511` | Predates the same timestamp tolerance |
| Alibaba Spot-GPU 2026 | `981145939269876c9de0ca12d97cf28b9ca4a511` | Predates the same timestamp tolerance |
| Helios Earth | `254b170a95fec1c631d49ff0ed1c91c71e48e9cc` | Already includes the corrected simulator; later changes are documentation/dependency/reporting related |

The GitHub compare API shows the relevant simulator difference is the
replacement of a strictly positive interval test with an epsilon test
at event boundaries, and an analogous epsilon in the temporal capacity
window. This correction rejects true resource overflow but ignores
floating-point overlaps shorter than `1e−12` hours. No other optimizer
source modules changed in these comparisons, although the dependency
file was subsequently pinned.

**Resolution policy:** Keep original seed-0 artifacts immutable for
provenance. Before final significance analysis, recheck the three
pre-tolerance seed-0 scenarios using the canonical solver and pinned
dependencies. If objective outputs are equivalent within a declared
numerical tolerance, document the comparison. If any substantive
difference exists, use canonical reruns and preserve the original
artifacts separately. Do not mix materially different methods in the
same statistical sample.

## Authoritative configuration notice

The current primary protocol is defined by
`configs/reconstruction_protocol.json`,
`configs/resource_environment_v1.json`,
`configs/carbon_replay_v1.json`, and the four
`configs/trace_*.json` manifests. Older experimental development
documents such as `configs/gpu_native_datasets.json`,
`configs/trace_materialization.json`, and
`configs/resource_pool.json` may describe superseded Acme/Philly,
16-resource or MSRP-based proposals. They must **not** be cited as
describing the locked four-scenario, seven-resource experiment.

## Source material for eventual manuscript citations

Cite the original publications/releases for Alibaba GPU v2023,
Microsoft ElasticFlow, SenseTime Helios, Alibaba Spot-GPU and NESO,
as well as the original 12 baseline algorithm literature. Add verified
bibliographic metadata rather than inventing author/title/DOI entries.

**Status:** Source-locked methodological draft; no claims of superiority
or statistical significance are made here.
