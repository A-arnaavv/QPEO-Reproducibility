# Carbon-v2 optimistic SLA feasibility audit — 12 November 2026

**Status: read-only diagnostic; no full-size campaign shard executed.** The manuscript's SLA claims require review before authorizing the corrected 120-shard comparison.

## Why this was checked

The first pilot had extremely high deadline-violation rates. To distinguish optimizer behavior from dataset/protocol limitations, we computed the earliest theoretically possible completion time **per job**, assuming no competing jobs and no queue. The only individually eligible resources were those satisfying job-level GPU/CPU/RAM capacities and the frozen forecast-week **mean** carbon class threshold.

For every workload with finite deadline, the optimistic available slack is:

    deadline_i - arrival_i - min(service_time_ij over eligible resources j)

A negative value below -1e-8 h proves the deadline cannot be met under this model, even with no contention. Absolute slack at or below 1e-8 h means the job has essentially **no queueing slack**. This is a necessary-condition diagnostic, not an achievable aggregate schedule or optimizer benchmark.

## Locked 64-workload scenario findings

| Dataset | Finite deadlines | Structurally impossible | Minimum f5 | Zero slack | Positive slack |
|---|---:|---:|---:|---:|---:|
| Alibaba GPU v2023 | 64 | 0 | 0% | **53** | 11 |
| Microsoft ElasticFlow | 64 | **26** | **40.625%** | 0 | 38 |
| Helios Earth | 64 | 0 | 0% | **64** | 0 |
| Alibaba Spot-GPU A100 | **44** | 0 | 0% | 0 | 44 |

All 64 jobs in each scenario have at least one individually eligible resource. Microsoft: **all 13 critical jobs** and **13/27 deferred jobs** are structurally late under the candidate carbon rule; none of its 24 flexible jobs is structurally late. Without the carbon eligibility restriction, no Microsoft job is individually impossible in the no-queue limit.

## Source-derived deadline semantics

- **Alibaba v2023:** qpeo/scenarios.py computes the deadline from the recorded deletion_time and creation_time, i.e., observed trace end behavior. It is not an independently specified contractual SLA.
- **Helios:** qpeo/scenarios.py uses recorded end_time as the deadline. In the selected scenario, modeled fastest eligible duration equals recorded duration for all 64 jobs, leaving zero queue slack.
- **Microsoft:** qpeo/scenarios.py uses the trace deadline field, while class mapping and region-mean carbon eligibility make 26 jobs impossible by the optimistic no-queue test.
- **Alibaba Spot:** deadlines are synthesized for 44 HP jobs as arrival plus recorded base runtime; the remaining 20 Spot jobs have infinite deadlines and are excluded from f5.

These deadline definitions vary materially between datasets. qpeo/evaluator.py calculates f5 as the percentage of *finite-deadline* jobs whose modeled completion time exceeds their deadline.

## Scientific recommendation

**Do not start all 120 expensive corrected shards without explicitly agreeing how to interpret the deadline metric.** For the existing locked candidate, call f5 *reconstructed deadline-proxy violation percentage*, disclose all finite-deadline denominators, and report the optimistic lower bound and zero-slack counts. It is not support for real contractual SLA compliance.

A genuinely positive-slack deadline sensitivity (predeclared rule and multiple margins) could be a **separate** experiment after coauthor agreement. Changing deadlines affects optimization and must not be made silently or retrospectively. Keep the unchanged source data and old locked v1 results intact. No publication-grade conclusions follow from the existing small pilots.

## Reproduction

Run the **read-only** script from a Carbon-v2 branch checkout:

    python scripts/audit_carbon_v2_sla_feasibility.py

It writes results/diagnostics/carbon_v2_sla_feasibility.json, reconstructing the same trace snapshots, resource environment and region-mean eligibility as the candidate protocol. Regression tests are in tests/test_carbon_v2_sla_feasibility.py. The complete **local** suite passed **121/121 tests**, using the available *unpinned* environment. This does not satisfy the pinned full-campaign runtime or substitute for the primary experiment.

**Campaign status:** 0 of 120 corrected full-size shards executed. No GitHub Actions/cloud jobs triggered; manuscript numerical results unchanged.
