# Reviewer 2, Comment 9 — Original-results-preserving expansion audit

**Status: comparability evidence pending; no original-manuscript baseline extension has been completed.**

**Authoritative manuscript:** `Article_Title__6_(2).pdf` (user-provided manuscript, not included in this repository). This file is the manuscript whose pre-existing numerical measurements and previous reviewer edits are intended to be retained **subject to their original evidentiary support**.

**Purpose:** Address the reviewer request for more comprehensive state-of-the-art comparisons, additional performance analyses, and deeper interpretation, while preserving the existing evaluation where scientifically justified. Do not silently replace the original results with the independent reconstruction in this repository.

## A. What the existing manuscript reports

- Section 6.1 / Table 2 names **Google Cluster Traces v3, Alibaba Cluster Trace 2022, Azure Public Dataset v2, and Kubernetes HPA traces**.
- Section 6.2 / Table 3 states `P=50`, `T=500`, `30` runs and describes a hardware/software stack including Python, CUDA, CloudSim Plus, and iFogSim. These descriptions must be checked against actual execution records, not used to infer that such hardware was necessarily used.
- The main comparisons in Tables 4–8 give values for QPEO and **nine** baseline algorithms: PSO, DE, GWO, WOA, ABC, GA, HS, FA, ACO.
- A paragraph has **already been added** in Section 6.2 naming **FPA, QIEA, and NSGA-II**, but the main performance tables have **not** yet been extended with measured results for these methods.
- NSGA-II is mentioned in Table 3's software frameworks but is absent from the original main performance result tables.
- Table 9 includes a **Classical FPA** *ablation* row for Google only. This is **not** a complete FPA baseline dataset for all four original scenarios and objectives; only reuse it as a matched baseline if the raw experiment definition and records substantiate equivalence.

## B. What this repository establishes — and does not establish

This repository **independently reconstructed** QPEO and 12 optimizers from the paper after loss of the original code. It contains 120 valid scenario-seed files and a complete 13-algorithm, 30-seed, four-scenario analysis.

**The reconstructed four scenarios differ from the manuscript's four datasets:**
Alibaba GPU v2023, Microsoft ElasticFlow ITP, Helios Earth, Alibaba Spot-GPU 2026 A100. The reconstruction also has its own selected 64-workload windows, seven logical resources, dynamic UK NESO carbon replay, modeled GPU-only power, and dimensionless cost proxy.

Consequently, the reconstructed FPA, QIEA, and NSGA-II measurements **cannot be pasted into the original manuscript's Google/Alibaba 2022/Azure/Kubernetes tables** as if obtained under the same conditions. These are different experiments and metric realizations.

## C. Evidence needed to extend the *original* result tables validly

1. **Original inputs:** exact dataset releases, filtered workload rows/windows, original trace files or immutable subset manifests, preprocessing mappings, seeds, and data licenses/source citations.
2. **Original evaluation implementation:** resource pool and computational environment actually used; schedule simulator or original hardware measurements; objective, SLA/deadline, energy, carbon, cost, utilization, and runtime definitions; feasibility/constraint handling. Preserve measurement units.
3. **Original experiment provenance:** raw per-run outcomes for QPEO and the original nine baselines, ideally objective vectors and timing/history by each of 30 seeds and four scenarios; script versions, parameter configurations, and original statistics/figure-generation code. The original tables alone are insufficient for valid paired reanalysis.
4. **New matched runs:** standalone FPA, specified QIEA variant, and NSGA-II evaluated on the exact **same original workload instances and evaluation model**. Match `P=50`, `T=500`, and `30` independently seeded runs, while disclosing any difference in simulator/objective-evaluation calls.
5. **Outputs:** all six existing objectives with original units/definitions and each additional metric the paper actually reports (e.g. throughput, latency, power, PUE, renewable usage) *only if the same measurement methodology supports those additional claims*. No estimated or invented cells.
6. **Statistical reanalysis:** compute descriptive statistics and suitable paired comparisons from raw runs. There are `12 × 6 = 72` QPEO-versus-baseline objective hypotheses *per prespecified aggregate family*; at familywise alpha `0.01`, Bonferroni threshold `0.01/72 = 1.3889e-4`. If all four dataset-specific families are treated jointly, `12 × 6 × 4 = 288`, threshold `0.01/288 = 3.4722e-5`. Define the family in advance and state it unambiguously. Recompute Friedman, post-hoc comparisons, confidence intervals, and effect sizes from actual data. Do not simply reuse prior 9-baseline ranks or p-values.
7. **Reviewer interpretation:** characterize where QPEO wins, ties, or loses against FPA/QIEA/NSGA-II; discuss novelty vs FPA and QIEA, the extra algorithmic overhead, and the boundaries of generalizability. The existence of three extra baselines by itself does not fully answer the request for stronger performance analysis and discussion.

## D. Existing manuscript consistency issues to investigate

- Table 5 displays PSO energy `847.3 kWh`, QPEO `578.2 kWh`. Their arithmetic relative reduction is `(847.3-578.2)/847.3 = 31.8%`, whereas the text claims `41.2%`. Verify source records and calculation before editing.
- Table 6 displays GWO carbon intensity `334.2` and QPEO `252.1 gCO2eq/kWh`, corresponding to a `24.6%` arithmetic reduction, versus the `21.4%` textual claim. Verify source records.
- Section 6.2 currently speaks as if all three additional algorithms were **already evaluated on the manuscript's original datasets**. This must be supported by matched outputs before submission; otherwise its tense/claim needs correction.
- Existing claims of universal significant superiority and all-large effect sizes are **claims in the manuscript**, not verified by the independently reconstructed experiment. Do not transfer significance or empirical support between studies.
- The original Section 6.3 mentions hypervolume, time-to-convergence, and runtime; do not represent absent objective-front records or unspecified histories as observed.

## E. Execution gate and outcome

**Do not initiate more GitHub Actions runs or spend additional Actions budget by default.** First obtain/verify the original records and establish that all baseline outputs can be computed on the *same original experimental conditions*. Start with an inexpensive local pilot only after comparability is established.

If the original results cannot be substantiated, retaining them unchanged as proven primary measurements is **not** defensible. Options then include recovering original evidence, executing a coherent fresh comparable study, or clearly presenting the independently reconstructed study as a separate supplementary simulation; the last option does **not** supply missing direct comparisons in the existing primary tables.

**Current status:** audit of the present repo tree and available manuscript materials found no original Google/Alibaba 2022/Azure/Kubernetes 30-seed raw-run archive or original implementation. This is an evidence gap, not proof those files do not exist elsewhere.
