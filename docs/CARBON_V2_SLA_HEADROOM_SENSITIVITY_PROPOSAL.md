# Carbon-v2 SLA deadline-headroom sensitivity proposal

**Status: coauthor-review proposal; read-only diagnostic, no optimizer experiments authorized.** Date: 12 November 2026.

## Why

Four locked workload scenarios have structurally different deadline semantics. The original Carbon-v2 audit found 26/64 individually impossible finite deadlines for Microsoft ElasticFlow (no queue, under fixed mean-carbon eligibility), 53/64 zero-slack jobs in Alibaba v2023, and 64/64 zero-slack jobs in Helios. Alibaba Spot-GPU has 44 finite deadlines and 20 jobs with infinite deadlines.

**Primary study recommendation:** Preserve original deadlines and data. Describe f5 as the *reconstructed deadline-proxy violation percentage*, report the per-scenario finite-deadline denominators and optimistic feasibility lower bound, and avoid claims about contractual SLA compliance.

## Separate synthetic policy (for coauthor approval)

For every job i with finite original deadline d_i and arrival a_i, find m_i = the fastest service time among resources individually eligible under existing GPU, CPU, memory and frozen full-week-mean carbon limits. The proposed separate sensitivity deadline is:

    d_i(alpha) = max(d_i, a_i + (1 + alpha) * m_i)

Proposed margin grid: **alpha = 0.25, 0.50, 1.00**, expressed as uncontended headroom fractions of m_i. Never tighten deadlines. Preserve infinite deadlines and leave all arrivals, job demands, service times, resources, costs, forecast, carbon thresholds, and original source files unchanged. These settings were proposed **after inspecting feasibility results** and must not be represented as preregistered independently of those findings. The variants create synthetic deadlines, not real or contractual SLAs.

## Verified read-only preview

| Workload trace | Finite deadlines | Baseline individually impossible | Baseline zero slack | Modified at 25% / 50% / 100% |
|---|---:|---:|---:|---|
| Alibaba v2023 | 64 | 0 | 53 | 64 / 64 / 64 |
| Microsoft ElasticFlow | 64 | 26 | 0 | 40 / 40 / 40 |
| Helios Earth | 64 | 0 | 64 | 64 / 64 / 64 |
| Alibaba Spot-GPU | 44 | 0 | 0 | 0 / 0 / 44 |

All variants eliminate the **optimistic individual no-queue** deadline impossibility by construction. This does not prove any actual joint schedule can meet its deadlines, nor indicate optimizer performance. At 100% headroom, the maximum extension is 20.73 hours for Alibaba v2023, 1.61 hours for Microsoft, 3.90 hours for Helios, and 0.18 hours for Spot-GPU. Spot jobs with infinite deadlines stay excluded from f5, and the 25% and 50% variants leave the 44 finite Spot-GPU deadlines unchanged.

## Future work gate

1. Obtain coauthor approval on the original-deadline primary reporting and any separate sensitivity grid. Do not choose margins retroactively based on winning optimizer results.
2. If approved, create a **separate opt-in optimizer run mode**, output naming, configuration hash, reproducibility manifest and audit. Do not overwrite any historical results or mix primary and sensitivity cohorts.
3. Validate a tiny separately tagged pilot before larger execution. Only after that, decide whether full-seed sensitivity runs are warranted. Three full-budget four-dataset sensitivity cohorts would add up to 360 shards; none are being started.
4. Use matched seeds and within-scenario/within-alpha comparisons; report effect sizes and deadline-increase magnitudes. Set correction for multiple tests before confirmatory analysis.

## Reproduction

    python scripts/preview_carbon_v2_sla_headroom.py

The script only writes a JSON **diagnostic preview** (not optimizer results). Tests are in tests/test_carbon_v2_sla_headroom_preview.py. The complete local test suite passed **124/124 tests** using the available unpinned environment.

**Full corrected primary study: 0/120 shards executed.** Pinned dependencies and methodological approval remain outstanding. Original source traces, v1 shards, and manuscript numerical results are unchanged.
