# Carbon-v2 — read-only SLA miss attribution (13 November 2026)

**Status:** Diagnostic analysis of saved single-seed pilot guide schedules; **not** publication-grade corrected primary results. No optimizer or paid GitHub Actions runs were launched.

## Audit scope

The new script, scripts/audit_carbon_v2_sla_miss_attribution.py, independently reconstructs original finite-deadline f5 values from saved assignments, start times and completion times. Inputs are the previously validated six pilot-shard ZIP and Microsoft P=50/T=20 exploratory-shard ZIP. It audits **seven test shards × thirteen algorithms = 91 algorithm outputs** and checks model eligibility, fixed service durations, arrival times, and saved f5 equality. Every row passed, and there were zero numerical-boundary-only misses.

The categories are exhaustive and mutually exclusive:
- **Individual infeasibility:** fastest eligible service at arrival would miss the deadline regardless of queue.
- **Assigned resource too slow:** another individually eligible resource has short enough service, but the chosen resource does not, even without waiting.
- **Start delay:** the chosen resource has short enough service in isolation but the saved start/finish time still misses.
- **Numerical-only boundary:** less than or equal to 1e-8 hours past the deadline (zero observed).

Classification is an **individual counterfactual**, not proof that moving all jobs to faster resources could meet the deadlines jointly under shared resource capacities.

## QPEO counts in the saved pilots

| Trace and pilot | Finite | Misses | Individual impossibility | Too-slow assignment | Start-delay-associated | Met |
|---|---:|---:|---:|---:|---:|---:|
| Alibaba GPU v2023 P=4/T=2 | 64 | 43 | 0 | 42 | 1 | 21 |
| Microsoft ElasticFlow P=4/T=2 | 64 | 63 | 26 | 22 | 15 | 1 |
| Helios Earth P=4/T=2 | 64 | 61 | 0 | 51 | 10 | 3 |
| Alibaba Spot-GPU P=4/T=2 | 44 | 36 | 0 | 11 | 25 | 8 |
| Microsoft ElasticFlow P=8/T=10 | 64 | 62 | 26 | 24 | 12 | 2 |
| Microsoft ElasticFlow P=50/T=20 | 64 | 61 | 26 | 19 | 16 | 3 |

The Helios strict-policy P=4/T=2 shard has the same guide schedule as the repeat-policy shard and is not a separate experimental replicate.

The Microsoft P=50/T=20 QPEO pilot's f5 is 61/64=**95.3125%**, including 26/64=**40.625%** that were already individually impossible under fixed carbon-week mean eligibility. The remaining 35 misses are attributable to 19 too-slow selected resources and 16 delayed starts, but cannot simply be assumed globally avoidable.

## Scientific interpretation and status

- The four datasets have different deadline origins and denominators. Spot-GPU has **44 finite deadlines** rather than 64. Keep f5 labeled a **reconstructed deadline-proxy violation metric**, not contractual SLA compliance.
- Retain original deadlines in the candidate primary protocol. A headroom sensitivity must be separately approved and tagged; this analysis does not authorize its execution.
- The 91 records come from **one seed per pilot**, with smaller budgets. Do not use them for significance tests or claims that QPEO outperforms other algorithms.
- Full P=50/T=500 corrected study: **0/120** shards executed. Original traces/results/manuscript numerical claims untouched. Pinned-environment primary campaign remains pending.
- Three focused attribution tests passed, and an isolated earlier source snapshot plus already committed operator/cache fixes and these tests passed 105/105; the latest entire remote test suite was not rerun in the pinned environment here.

## Reproduce

From a clean checkout of the Carbon-v2 validation branch, with the two existing pilot evidence ZIPs supplied as arguments:

    python scripts/audit_carbon_v2_sla_miss_attribution.py \
      QPEO_Carbon_v2_Validated_Pilot_Evidence.zip \
      QPEO_Carbon_v2_P50_T20_Exploratory_Evidence.zip \
      --output-dir results/diagnostics/sla_miss_attribution

The generated CSV provides all 91 per-algorithm counts; JSON includes source-shard SHA-256 fingerprints and method definitions. Unit tests are in tests/test_carbon_v2_sla_miss_attribution.py.
