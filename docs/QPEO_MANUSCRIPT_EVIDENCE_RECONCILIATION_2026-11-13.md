# Manuscript evidence reconciliation — 13 November 2026

**Source reviewed:** `Article_Title__6_-2.pdf` (44-page QPEO draft created 12 November 2026). **Status:** read-only review, not a corrected full-study result. The corrected Carbon-v2 P=50/T=500, 30 seeds × four scenarios remains **0/120 shards executed**.

## Submission-blocking contradictions

1. **Abstract (p.2), Conclusion (p.38), Data Availability (p.40)** name Google/Alibaba 2022/Azure/Kubernetes with nine baselines/six scenarios, while Section 6.1 (p.24) names **Alibaba GPU v2023, Microsoft ElasticFlow ITP, SenseTime Helios Earth, Alibaba Spot-GPU 2026** and **13 algorithms (QPEO plus twelve baselines)** over four simulated scenarios.
2. **Abstract (p.2)** claims 91.3% mean GPU utilization, whereas the reported scenario-specific QPEO means in Section 7.1 (p.27) are 4.16%, 39.09%, 38.99% and 37.39% for the seven-slot resource pool. Without a different explicitly defined measurement, these are inconsistent.
3. **Section 7.4, Table 7 and Figure 8 (pp.31–32)** report 2.1% QPEO SLA violations and 118 ms end-to-end latency, alongside throughput, cold-start and co-scheduling-overhead values. The 13-algorithm reconstructed study stores six guide objectives, not independently verified end-to-end millisecond latency; its corrected one-seed pilot yields very different *deadline-proxy* results. Pilots are **not** full-study data, so neither source authorizes substituting pilot numbers for this table.
4. **Table 6/Figure 6 (pp.29–30)** report 82.7% renewable use and 847 tCO2e annual avoided emissions. The reconstructed trace-driven GPU power/forecast model does not establish physical renewables or annual emissions avoided. Omit unsupported claims until independent measurements and derivation are supplied.
5. **Ablation Table 9 and Figure 11 (pp.35–36)** refer to Google Cluster Traces and 92.1% QPEO GPU utilization, a different and not independently verified experimental cohort from the four reconstructed primary scenarios. Treat as a separate unverified study unless complete source run records are supplied.
6. **Methodology p.13 and Table 3 p.25** describe static resource-capacity sums/real-time carbon, and per-job CPU compatibility. The corrected simulator enforces concurrent **CPU/GPU/RAM** capacity, while fixed forecast-week mean carbon intensity is used for **offline** eligibility and actual scheduled-interval forecast integration defines modeled f3. The 168-hour week is periodically repeated only as disclosed counterfactual extrapolation.
7. **Runtime p.25:** manuscript lists Python 3.11; corrected candidate full campaign requires Python 3.13, NumPy 2.4.6, SciPy 1.17.1, pytest 9.2.0. Don't claim execution of the target run until it occurs.
8. **Convergence pp.32–33:** stationary time-to-convergence claims do not follow from legacy nonstationary carbon-replay history. Regenerate with a defined corrected objective or omit.

## Verified deadline-proxy feasibility findings (diagnostic only)

| Scenario | Finite deadlines | Impossible even with zero queue | Zero slack |
|---|---:|---:|---:|
| Alibaba GPU v2023 | 64 | 0 | 53 |
| Microsoft ElasticFlow ITP | 64 | **26** (40.625% floor) | 0 |
| SenseTime Helios Earth | 64 | 0 | **64** |
| Alibaba Spot-GPU A100 | **44** (20 unconstrained Spot jobs excluded) | 0 | 0 |

The Microsoft one-seed **P=50/T=20 exploratory** guide missed 61/64 deadlines (26 individually impossible, 19 too-slow assigned resources, 16 delayed starts, 3 met). All 91 algorithm rows from seven saved **pilot** shards exactly matched the reconstructed f5; these are neither independent 30-seed contrasts nor the corrected primary study.

**Methodology wording to adopt:** Define f5 as 100 × (number of finite-deadline jobs whose simulated completion exceeds their original deadline) / (number of finite-deadline jobs). Call it a **reconstructed deadline-proxy violation percentage**, not contractual SLA compliance. Explicitly state dataset-specific deadline provenance and denominators and report optimistic no-contention lower bounds. Keep original deadlines in the candidate primary protocol; headroom modifications require a separately approved sensitivity cohort.

## Required order

Reconcile dataset/baseline count and section text; withhold unsupported annual-emissions, renewable-use, network-latency, cold-start and Google ablation numerical claims; reconcile Eq. (2)–(6) with the concurrent-capacity simulator and offline carbon policy; obtain coauthor review of SLA and carbon semantics; execute all 120 corrected pinned-environment shards; independently audit; regenerate Tables/Figures/abstract/conclusion from actual corrected results. **Do not mix v1 and v2 results.**

Additional detailed paste-ready replacement paragraphs are available in the local manuscript-audit report. The PDF and old experiment files were not edited.

**Source diagnostics:** `docs/CARBON_V2_SLA_FEASIBILITY_AUDIT_2026-11-12.md` and `docs/CARBON_V2_SLA_MISS_ATTRIBUTION_2026-11-13.md`.
