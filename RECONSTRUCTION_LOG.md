# Reconstruction log

## Explicitly specified in the manuscript
- Population size P = 50
- Maximum iterations T = 500
- Levy exponent lambda = 1.5
- Global/local switching probability p_s = 0.8
- kappa_min = 0.2
- kappa_max = 2.0
- tau_carbon = 150
- tau_H = log2(P)/2 (reported as 2.82 for P=50)
- T_stag = 20
- delta_bio = 0.3
- CI_ref = 475 gCO2eq/kWh
- six objective formulas
- five formal feasibility constraints
- QPSO global/local equations
- CAQC score equation
- EBWS update equation
- EBPP entropy and BRO equations

## Not sufficiently specified / lost with the original code
These must NOT be presented as recovered original settings:
- six scalar-guidance weights w_k
- ecological compatibility weights alpha, beta, gamma
- exact definitions/scaling of Perf(i,j) and Therm(j)
- EBWS dependency threshold delta_dep
- EBWS coupling strength phi
- exact construction of rho_n from theta_n
- exact workload-dependency measure d(i,i')
- carbon thresholds for critical/deferred/flexible workload classes
- feasibility-repair rule after CAQC argmax
- trace preprocessing and sampling windows
- execution-time model
- resource power/energy model
- infrastructure price table and reference cost construction
- random-seed policy
- hypervolume reference point

## Reconstruction convention R1: candidate density proxy
The manuscript states rho_n is D x D and also parameterizes each candidate by a
D-dimensional angle vector. It does not provide the mapping between theta_n and the
mixture {p_k, |psi_nk>}.

For the rebuild, the operator module exposes an explicitly named
`candidate_density_proxy()` using a normalized amplitude vector proportional to
sin(theta), with rho_n = |psi_n><psi_n|. This is a *reconstruction convention*, not
a claim about the lost implementation.

This convention is useful because the population mean can have rank up to P, making
the manuscript's log2(P)-based entropy scale internally coherent when D >= P.

## Reconstruction convention R2: no silent defaults for undocumented weights
The operator functions require alpha/beta/gamma, delta_dep, phi, and scalar weights
to be supplied explicitly. A later experiment configuration may choose neutral values,
but they will be labelled as reconstruction settings and reported in the revised paper.

## Reconstruction convention S1-v10: temporal-capacity execution semantics
The manuscript's original Eqs. (2)-(3) summed all assigned workload demand without
a time index, which makes realistic multi-arrival traces infeasible even when jobs can
execute sequentially. The rebuilt simulator therefore treats X as a resource-assignment
decision and enforces GPU and memory capacity over simulated time. Workloads are
processed in arrival order and start at the earliest time their assigned resource has
sufficient GPU and memory capacity for the full service interval. The revised paper
must update Eqs. (2)-(3) to their time-indexed form and explain this correction.

## Reconstruction convention S2: explicit service-time matrix
The lost code's execution-time model is unavailable. The simulator therefore requires
`service_time_hours[i,j]` as an input. Later trace adapters / hardware characterization
must produce this matrix, and the revised paper must report how it is constructed.

## Reconstruction convention S3: explicit utilization-based power model
Energy is reconstructed using an explicit per-resource power curve
`P(u) = P_idle + P_dynamic * u^q` integrated over overlapping workload intervals.
`P_idle`, `P_dynamic`, and exponent `q` are experiment inputs and are not claimed to
be original recovered settings.

## Reconstruction convention S4: explicit GPU-hour accounting mode
The manuscript defines cost using occupied GPU time but does not specify whether
fractional GPU allocation or active device wall-clock time is billed. The simulator
therefore requires an explicit choice between `active_resource_hours` and
`allocated_gpu_hours`. The chosen mode must be reported in the revised experiment
configuration.

## Reconstruction convention C1: feasibility-aware CAQC repair
Equation (17) independently selects the maximum-scoring resource for each workload,
but the lost implementation's capacity-conflict repair is not documented. The rebuild
uses deterministic greedy placement: more constrained workloads are placed first and
resources are considered in descending CAQC score order subject to Eqs. (2)-(6).
This rule is explicitly identified as reconstructed and its limitation is surfaced by a
`NoFeasibleAssignment` exception rather than silently violating constraints.

## Reconstructed Algorithm 1 execution semantics
The v3 optimizer follows the manuscript pseudocode: initialize the angle population,
construct CAQC assignments, evaluate objectives, maintain a non-dominated archive,
select a scalar guide from normalized objectives, apply global/local QPSO, then EBWS,
accept only dominating proposals, compute population entropy, and trigger BRO for
stagnant candidates below the entropy threshold. BRO-modified angle vectors are
re-evaluated on the following iteration, matching the order shown in Algorithm 1.

## Dataset audit finding D1: manuscript dataset labels require correction
The reconstruction checked the official public repositories rather than assuming the
old Table 2 descriptions were accurate. Azure Public Dataset V2 is a VM trace, not a
serverless trace; the relevant Azure serverless sources are the Azure Functions traces.
The clearly documented Alibaba GPU scheduling release is GPU v2023; Alibaba's 2022
release is not the same GPU trace. No official public Kubernetes HPA production trace
matching the manuscript's stated scale was identified during the audit. See
`DATASET_AUDIT.md`.

## Trace-layer rule T1: no silent filling of absent resource fields
Public traces differ in what they expose. The trace materializer requires explicit
fallbacks for missing CPU/GPU/memory/bandwidth values and an explicit deadline slack
policy. If a fallback is absent, materialization fails instead of inventing a value.

## Reconstruction convention B1: common baseline decoder
The rebuilt comparison suite represents non-QPEO candidate solutions as MxN score
matrices. A shared deterministic feasibility decoder maps those scores to one-hot
workload-resource assignments while enforcing the same formal GPU, memory,
exactly-one, binary, and carbon-threshold constraints used by QPEO. Unlike CAQC,
the common decoder does not apply a QPEO-specific carbon penalty. This is a fair
reconstruction protocol, not a recovered detail of the lost baseline code.

## Reviewer-critical baseline protocols
- **FPA-B1:** continuous MxN random-key scores, standard global Levy pollination
  and local pairwise pollination, common feasibility decoder.
- **NSGAII-B1:** continuous MxN random-key scores, non-dominated sorting,
  crowding distance, SBX crossover, polynomial mutation, common decoder.
- **QIEA-B1:** MxN q-bit angle representation with sin^2 observation scores and
  a rotation update toward the current scalar guide's one-hot assignment. This is
  explicitly described as a *QIEA-style scheduling baseline* because the manuscript
  cites the QIEA family rather than a fully specified scheduling implementation.

All three baselines use the same ProblemInstance, SimulatorInputs, population size,
iteration limit, random-seed list, objective evaluator, and formal feasibility
constraints as QPEO.

## Original-nine baseline reconstruction protocols
The lost baseline wrapper code could not be recovered. The rebuild therefore uses
transparent textbook-style variants with a shared continuous MxN score representation
and the common feasibility decoder B1: PSO, DE/rand/1/bin, GWO, WOA, ABC,
real-coded GA, Harmony Search, Firefly Algorithm, and assignment-pheromone ACO.
Their algorithm-specific constants are explicit function arguments. These are
reconstructed comparison implementations and must not be described as byte-for-byte
recovery of the original experiments.

## Statistical-analysis reconstruction protocol
The analysis module uses paired Wilcoxon signed-rank tests, paired standardized mean
difference d_z, Friedman ranking, asymptotic Nemenyi post-hoc comparisons, t-based
95% confidence intervals, and explicit Bonferroni family-wise correction. With 12
baselines and six primary objectives, alpha=0.01 gives 72 pairwise objective tests and
a corrected threshold of 1.3889e-4.

The lost hypervolume reference point is not documented. Hypervolume therefore
requires an explicit reportable reference point. The current implementation provides
a deterministic Sobol quasi-Monte-Carlo estimator for minimization hypervolume and
will not silently choose the old unknown reference point.

Convergence-to-90% must be computed from a raw consistently scaled metric history;
per-iteration min-max scalar fitness is not suitable because its scale changes with the
population. Raw objective histories will be added to experiment outputs before final
paper convergence values are regenerated.


## Resource-environment protocol RENV1
The rebuilt cloud--edge--serverless resource pool is locked before the final trace
runs in `configs/resource_environment_v1.json`. Hardware memory, maximum GPU/module
power, and FP32 throughput are taken from NVIDIA specifications for RTX 4090,
RTX A6000, RTX A5000, and Jetson AGX Orin 64GB. FP32 throughput is used only as a
transparent relative service-speed proxy normalized to RTX A6000; it is not presented
as an application-specific measured speedup.

The primary power model is GPU-side incremental power, P(u)=Pmax*u, with zero idle
term so that the primary model uses only sourced maximum GPU/module power rather
than inventing unsourced idle values. Idle fractions of 0.10 and 0.20 are predeclared
sensitivity settings.

The primary infrastructure-cost coefficients are dimensionless reconstruction cost
indices rather than USD prices. Two additional cost profiles (performance-proportional
and power-proportional) are predeclared for sensitivity analysis. This avoids implying
that the lost implementation's provider-specific hourly prices were recovered.


## Carbon-replay protocol CARB1
The rebuilt experiment uses NESO's openly available GB regional carbon-intensity
forecast as a reproducible carbon replay rather than requiring private WattTime or
Electricity Maps credentials. The locked primary window is the first complete ISO
week of 2026 (2026-01-05 through 2026-01-11 UTC). Seven resource slots are mapped
in fixed slot order to predeclared GB regions; this mapping is locked before the
optimizer results are generated.

Carbon-sensitivity thresholds are derived from the replay, not tuned to QPEO:
critical = the worst-case row minimum, deferred = the worst-case third-lowest regional
intensity, and flexible = the replay-wide maximum, each with numerical epsilon. This
guarantees at least 1, 3, and all resources respectively remain carbon-feasible at
every replay step. Optimizer iteration t is mapped
linearly to the replay row index round(t*(K-1)/T). Missing regional values use a
predeclared within-region forward-fill/back-fill rule and a region with no observations
causes the experiment to fail.


## Dynamic-carbon protocol DYN1
The locked carbon replay is applied to every optimizer, not only QPEO. At iteration
t, the common problem instance is refreshed with the same replay vector for QPEO and
all twelve baselines. Existing populations are re-decoded/re-evaluated under the new
carbon state before the iteration's search update. QPEO additionally recomputes the
ecological-compatibility matrix because Eq. (10) contains carbon intensity.

Because objective f3 and carbon-feasibility constraints change with the grid state, a
historical Pareto archive would contain stale objective values. Under DYN1 the archive
is therefore rebuilt from the current population after each carbon refresh and then
updated within the iteration. PSO personal-best positions are also re-evaluated under
the current carbon state before selecting its global guide. This is a reconstruction
choice for nonstationary optimization and is applied symmetrically across methods.

## Time-aware utilisation objective U1-v10
The original f1 formula used aggregate assigned GPU demand divided by capacity and
can exceed 100% when sequential jobs are assigned to the same resource. For trace
replay, f1 is reconstructed as negative time-average GPU utilisation over the global
scheduling horizon: for each resource, integrate allocated GPU fraction-hours and
divide by H = max_i C_i - min_i a_i, then average across resources. This value is in
[-1,0] and remains meaningful for queued/sequential workloads. The manuscript must
replace the old f1 expression before final submission.


## Memory-semantics correction MEM1-v11
Alibaba's `memory_mib` field is host/main-memory request. The scheduler's formal
`MEMcap` constraint therefore uses per-slot host/unified memory capacity rather than
NVIDIA VRAM capacity. The reconstructed environment locks 64 GiB host/unified memory
for each slot, matching the manuscript's 64 GB experimental memory baseline and the
64 GB Orin class. GPU VRAM (24/48/64 GB) remains recorded as hardware metadata but
is not used as the trace's main-memory constraint.
