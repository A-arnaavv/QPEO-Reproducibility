# QPEO manuscript-replacement validation audit (12 November 2026)

**Decision:** Prospectively use the reproducible *independently reconstructed* GitHub study as primary evidence for the revised Scientific Reports QPEO paper, **only after appropriate validation and manuscript scope corrections**. The earlier values in `Article_Title__6_(2).pdf` belong to a different, unverified original experiment and cannot be pooled with the reconstruction. The original manuscript file is not altered by this audit.

## 1. Supported experiment facts

- Four frozen trace-derived scenarios: `alibaba_gpu_v2023`, `microsoft_elasticflow_itp_cluster08`, `helios_earth`, and `alibaba_spot_gpu_2026_a100`.
- Each primary scenario has 64 selected workloads from a larger filtered pool.
- Thirteen optimizers (QPEO and 12 baselines including FPA, QIEA-style, NSGA-II); population `P=50`, 500 iterations, 30 predefined seeds per scenario.
- 120 scenario×seed result files, 1,560 algorithm×scenario×seed runs, source-locked and statistically analyzed in `results/analysis/`.
- Six minimized guide-solution objectives (`qpeo/evaluator.py`): negative **time-averaged GPU utilization across all seven resource slots**; modeled GPU energy (kWh); **energy-weighted carbon intensity** (gCO2e/kWh, *not total emissions*); simulated makespan (hours); simulated percentage of deadline misses among finite-deadline workloads; relative, dimensionless normalized cost proxy.
- Bonferroni over all 288 QPEO-vs-baseline paired tests at familywise alpha 0.01: 76 significant comparisons favor QPEO, 108 favor baselines, 104 not significant.
- Runtime records are wall-clock times to run algorithms; the iteration budgets match, **objective/simulator evaluation calls do not necessarily match**.
- The carbon schedule replays a fixed **January 5–11, 2026 UK NESO forecast** across seven logical slots; the underlying workloads originate elsewhere and at other dates. These are simulated counterfactual carbon-policy exposures, not observed location-specific emissions.

## 2. Code-level validation findings

### A. High SLA/deadline-miss rates are unsurprising under the chosen deadlines

`qpeo/scenarios.py` builds:
- Alibaba v2023: deadline at observed deletion/end-to-end completion time;
- ElasticFlow: original authors' **generated experimental deadline**, not a directly measured raw-production SLA;
- Helios Earth: deadline at historical observed end time;
- Alibaba Spot: HP deadline = arrival + observed source duration, while Spot jobs have infinite deadline and are **excluded from the SLA denominator**.

`qpeo/evaluator.py` computes deadline misses as `100 × mean(completion > deadline)` **among finite deadlines only**. Workload execution duration is adjusted using FP32 hardware proxies and resources may queue. The high observed misses are therefore *model- and deadline-definition-dependent*, not verifiable service-level violations in real deployments. Do not title this metric 'production SLA compliance' without explicit qualification.

### B. EBWS is inactive on all four primary scenarios

Every scenario builder in `qpeo/scenarios.py` sets `dependency=np.zeros((size,size))` because the traces lack recorded inter-job dependencies. `qpeo/operators.py::ebws_update` only couples elements with `dependency > dependency_threshold`. With all-zero matrices it is an identity transform on all primary runs.

Therefore, the primary experiment **does not test a benefit of EBWS**. Reusing the original manuscript's EBWS ablation numbers or claiming empirical validation of EBWS from these 120 output files is impermissible. A separate, transparently specified dependency-enriched experiment or ablation would be necessary for that claim.

### C. Temporal aggregate CPU capacity is not enforced

`qpeo/evaluator.py::check_constraints` verifies **individual-job** CPU demand fits the assigned resource CPU capacity, along with per-job GPU/memory and carbon constraints. `qpeo/simulator.py::_window_capacity_ok` and `_earliest_feasible_start` check **concurrent GPU and memory**, but have no CPU demand/capacity inputs; scheduled intervals store only time, GPU, memory. Hence **simultaneous CPU capacity is not enforced** for multiple overlapping jobs.

Do **not** report these results as proof of full CPU–GPU co-scheduling under concurrent CPU constraints. This finding does not prove any particular saved schedule exceeded CPU capacity; it establishes that the model never checked it. To claim coupled CPU capacity validation, correct the simulator and recompute affected outputs. A narrower, accurately described model can instead report per-job CPU compatibility with temporal GPU/memory scheduling, subject to author/editor assessment.

### D. GPU utilization and units differ from the existing manuscript

`qpeo/evaluator.py` defines `f1 = -mean(allocated_gpu_fraction_hours / global_scheduling_horizon)` over seven slots. Values such as `-0.0416` correspond to 4.16% **horizon-averaged all-slot utilization** under this model; they are not directly comparable to the original paper's 91.3% reported utilization. Makespan `f4` is measured in **hours**, not seconds. Source traces and work amounts differ, so absolute energy values are not comparable to the original.

### E. Carbon, energy, and cost are proxies

`configs/resource_environment_v1.json`: service duration scaled by nominal FP32 TFLOPS; dynamic GPU-only power `P(u)=P_max·u`, zero idle and host power; relative unitless hourly cost indices. Accordingly:
- **No whole-datacenter energy, annual tCO2 savings, renewable utilization, or measured GPU electrical consumption** is established.
- Carbon **intensity** is not carbon **mass**; if emissions are desired, explicitly define and compute energy × intensity with aligned units and time assumptions.
- Runtime from Python is not measured production scheduling latency.

### F. Complexity analysis must match actual implementation

`qpeo/operators.py::population_entropy_from_angles` forms a `P×P` Gram matrix from the `P×D` angle population and eigendecomposes that matrix. It avoids explicitly storing/eigendecomposing a `D×D` population-mean density matrix in the primary optimizer. The dominant entropy cost of the implemented low-rank method is approximately `O(P²D + P³)`, memory `O(PD + P²)` (ignoring other scheduling operations), **not the manuscript's claimed dense entropy eigendecomposition `O(D³)` as an implemented runtime cost**. Reconcile the asymptotic analysis, including scheduling simulation and repeated carbon-refresh evaluations. The original theorem-level entropy and global-convergence statements require separate mathematical review.

### G. Additional evidence gaps

- Primary result files do not preserve complete per-run Pareto-front objective vectors. The predeclared hypervolume indicator **cannot currently be reconstructed or reported**.
- Objective histories exist but carbon-dependent landscape changes by iteration: stationary convergence-to-final comparisons require a separate justified protocol.
- No validated new **component-wise QPEO ablation** results are committed for the locked four-scenario primary protocol; do not transfer Table 9 numbers from the original study.
- Report selected *guide-solution* objectives rather than claiming the present statistics characterize the full Pareto set.
- Seed-zero and package-version provenance differences are logged in `results/reproducibility/seed00_parity.json`; canonical Spot source differences were resolved by replacing the affected seed-zero result, not claiming original parity.

## 3. Required manuscript replacement map

1. **Abstract/Introduction/Contributions:** change overbroad claims of universal improvement and EBWS-supported performance. Describe an independently reconstructed classical quantum-inspired scheduler and carbon/operational trade-offs, including limitations.
2. **Section 3 formal model:** retain theoretical target formulation only after clearly distinguishing the more limited *implemented* CPU and memory constraints, SLA proxy, and cost/carbon definitions.
3. **Section 4 QPEO components:** describe EBWS as a proposed operator **not empirically activated by the primary traces**.
4. **Section 5 algorithm and complexity:** reconcile actual entropy Gram implementation; do not preserve unsupported global-optimum and all-iteration entropy guarantees without a valid proof.
5. **Section 6.1 and Table 2:** update to the actual four frozen trace-derived scenarios; keep organization/formatting while changing rows and claims.
6. **Section 6.2 and Table 3:** update to the seven logical resources and actual NumPy/SciPy classical Python simulation stack. Do not imply measured CUDA/CloudSim/iFogSim/physical GPU execution; P/T matched, not NFE matched.
7. **Section 6.3:** define the six metrics and statistical family precisely; do not claim measured hypervolume or comparable stationary convergence.
8. **Section 7 / Tables 4–7 / Figures 4–9:** regenerate only from `results/analysis/` and source-backed outputs. Remove unsupported latency, renewable percentage, actual tCO2 savings, PUE, DCGM, throughput, CPU–GPU overhead, and unverified other metrics.
9. **Section 8 / Tables 8–9 / Figures 10–11:** replace statistical results and describe ablation as outstanding until independently supported; distinguish significance from practical magnitude. Existing ablation results are not validated by this reconstructed cohort.
10. **Sections 9–10 / response letter:** accurately discuss mixed wins/losses, objective trade-offs, high deadline misses, proxy modeling, and future EBWS/real-hardware validation. Reply to Reviewer 2, Comment 9 with the actual 13-algorithm design and limitations.

## 4. Publication-readiness decision

**Current computational results: complete and reproducible.**
**Current manuscript evidence: NOT yet validated for resubmission.**

Highest-priority unresolved choices:
- Either validate concurrent CPU handling via a corrected simulator and recompute affected runs, or explicitly narrow the claims to individual CPU fit plus temporally constrained GPU/memory.
- Acknowledge that EBWS is untested here; only a separate data-defined dependency test could validate it.
- Decide whether to run focused ablation/sensitivity experiments for the lock-step reconstructed model. They are important because the original manuscript claims empirical component contributions; no substitution of old results is valid.
- Replace the manuscript coherently, not by mixing original and reconstructed rows or silently retaining unsupported values.

**Budget safeguard:** No new GitHub Actions or other compute campaigns should be dispatched as part of this audit; the earlier $10 Actions budget may have been removed. Code/doc changes should not silently trigger expensive workflows.
