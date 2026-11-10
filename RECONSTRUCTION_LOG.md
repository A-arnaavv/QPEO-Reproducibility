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


## Alibaba trace protocol TRACE1-v11
The primary Alibaba scenario uses the official GPU v2023 default pod trace. The
upstream file is pinned by Git blob SHA
`bd82505791c723c5b11486ef457fb338529912c0`. Eligibility is fixed before results:
Succeeded pods only, `num_gpu=1`, `0<gpu_milli<=1000`, positive scheduled runtime,
CPU request <=16 cores, and host-memory request <=64 GiB. This leaves 178 jobs.

Jobs are sorted by creation time. A single pre-registered anchor from
`numpy.random.default_rng(20261110).random() = 0.6260788463455607` determines
contiguous nested windows: 32 jobs (indices 91--122), 64 jobs (71--134, primary),
and 128 jobs (31--158). The primary block begins at `openb-pod-3131` and ends at
`openb-pod-5985`.

GPU, CPU and host-memory demands are taken directly from the trace. Bandwidth is set
to zero because the trace does not expose a bandwidth request and bandwidth is not a
hard constraint in the current reconstruction. QoS maps to carbon sensitivity as
LS=critical, Burstable=deferred, BE=flexible. Deadline is the source-observed
end-to-end interval from creation to deletion; service-time baseline is the observed
scheduled-to-deletion runtime. Cross-resource service times use the pre-registered
FP32 relative-speed proxy. Inter-job dependency is set to zero because this trace does
not expose dependency metadata.


## Microsoft ElasticFlow trace protocol TRACE2-v12
TRACE2 uses the official `microsoft/elasticflow-traces` repository. The raw archive is
pinned by Git blob SHA `8147e243486d25a21abb12c1e50cbcf90aa72ebc` and SHA-256
`67398e1a89245c50ae0881bf997e6cda0c95f9f47980af5bc56d67998965669f`.
The authors' experiment archive is pinned by Git blob SHA
`bf9e3ccae14b35410a213b2528c02e71906f8890` and SHA-256
`e9c20b73de61d2702aef086aebc995cbc432affbd0f09969cf30879c5d1a83c4`.

Among the ten clusters, Cluster08 is selected because it has the largest number of
positive-duration single-GPU raw jobs (12,558). Every one of those jobs has an exact
`job_id` match in the authors' experiment trace. Jobs are sorted by submission time
and sampled with the same pre-registered anchor fraction used for TRACE1. The primary
64-job window is indices 7822--7885, from
`fd68fa2d-ed9f-6fe1-0a85-877c7b6de8da` through
`42dfd273-4486-8e16-9a12-9f4ce099e4b5`; 32- and 128-job nested windows are
also locked.

Arrival time, duration, and one-GPU demand come from the raw production trace.
Deadline comes from the matching ElasticFlow experiment trace and is therefore an
author-generated experimental SLA field, not a raw production measurement. Its
slack-to-duration ratio is split at precomputed 33rd/67th percentiles
(0.8275201174 and 1.1668952957) to map jobs to critical/deferred/flexible carbon
classes before optimizer results are observed. CPU, memory and bandwidth demand are
set to zero/non-binding because the raw trace does not publish those fields. Service
time uses the same documented FP32 relative-speed reconstruction proxy as TRACE1.


## Helios Earth trace protocol TRACE3-v13
TRACE3 uses the official SenseTime Helios release, pinned by Git blob SHA
`1a48d5e0fb816fe9bbf1ad7086e7dec90923e5c6` and archive SHA-256
`3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac`.
Among Earth, Saturn, Uranus and Venus, Earth is selected by the pre-results rule
"largest completed single-GPU pool." Eligibility is fixed as COMPLETED, gpu_num=1,
duration>0, valid submit/start/end times, and 0<=cpu_num<=16, leaving 293,903 jobs.

The same anchor fraction used by TRACE1/TRACE2 selects nested 32/64/128 contiguous
submission-time windows. The primary 64-job window is full-pool indices
183966--184029, from job 1738622 through 1739705. GPU demand and CPU demand are
source fields. Memory and bandwidth are zero/non-binding because Helios does not
publish those requests. Deadline is the observed source end time relative to the
selected block's first submission; base runtime is source start-to-end duration.
Because Helios does not expose a workload priority/SLA class, all jobs are assigned
the neutral deferred carbon-sensitivity class rather than inferring urgency from queue
or runtime behavior.


## SLA-domain correction SLA1-v14
The SLA objective now evaluates only workloads with finite deadlines. Workloads with
`deadline=inf` are explicitly best-effort and are excluded from both the numerator and
denominator of the SLA-violation percentage. This is required for Spot workloads in
TRACE4, which are opportunistic and do not carry the same strict SLO semantics as HP
jobs. Existing TRACE1--TRACE3 workloads all retain finite deadlines, so their SLA
metric is unchanged. The revised manuscript must define an SLA-constrained subset
(W_{SLA}) in the formal f5 expression.


## Trace-derived deadline and missing-resource policy
The revised GPU-native traces expose submission/queue/run timing, so the SLA deadline
proxy is defined as the original observed production completion time:
deadline_i = arrival_i + observed_queue_i + observed_runtime_i. No tuned slack
multiplier is used.

For traces lacking per-job memory, the final imputation value will be the pooled median
native GiB-per-GPU scheduling unit measured from Alibaba GPU v2023 and Acme Kalos
after the same validity filtering and sharding. Missing CPU demand is similarly imputed
from the pooled native CPU-per-GPU median across Alibaba, Helios, and Acme. These
values are computed and frozen before optimizer execution. Bandwidth is set to zero
when unavailable because the primary traces lack a consistent bandwidth-request field
and bandwidth is not a current formal hard constraint.


## Final-run reproducibility and computational-budget caveat
The locked primary experiment matches all 13 algorithms on the same scenario,
seed index, population size P=50, and iteration count T=500. **This is not an
identical objective-evaluation budget**: evolutionary, swarm and constructive
methods perform different numbers of candidate evaluations per iteration. Every
final result therefore preserves per-method simulator cache hits and misses;
`hits + misses` reports simulator invocations as a diagnostic (including repeated
assignments), alongside algorithm runtime. Do not describe the primary comparison
as NFE-matched in the manuscript; use an explicit evaluation-matched sensitivity
study if that claim is required.

The final study uses the locked NESO carbon replay and four trace configurations
named in `configs/reconstruction_protocol.json`. Earlier exploration drafts and
prototype configs (including the Acme/Philly primary-trace proposal and
Electricity Maps prototype) are superseded, not additional primary datasets.
NumPy 2.4.6, SciPy 1.17.1 and pytest 9.2.0 are pinned to the versions used by the
successful final-protocol GitHub runner jobs.
