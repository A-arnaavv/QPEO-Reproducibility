# Carbon-v2 end-to-end validation — 12 November 2026

**Status: LOCAL PILOT VALIDATED, PRIMARY EXPERIMENT NOT RERUN.**

The source checkout was obtained from the uploaded `QPEO-Reproducibility-carbon-accounting-v2-validation.zip`, extracted into an isolated local workspace and tested without triggering GitHub Actions. The two discovered defects were fixed locally, all tests rerun, and exactly those four source/test files updated on this validation branch.

## Two issues discovered by actual testing

1. `qpeo/operators.py::ecological_compatibility` rejected an all-zero carbon forecast due to `ci_max <= 0`. The corrected function interprets a valid all-zero regional vector as an equal carbon preference across resources, retaining a finite compatibility factor. An explicit invalid `ci_max` remains an error.
2. `qpeo/simulator.py` memoization keyed cached schedules by Python object IDs but did not keep the key owners alive. Temporary `SimulatorInputs` instances could reuse an ID and produce a cached state with the **wrong cost billing convention**. The cache now retains owner references for its lifetime and releases them when cleared.

Regression tests were added in `tests/test_operators.py` and `tests/test_simulation_cache.py`.

## Actual test results

- Initial targeted full-suite tests: 21 passed / 2 failed, exposing the all-zero intensity edge case.
- After first fix, full test suite: 102 passed / 1 failed, exposing the cache collision.
- **Final: 104 / 104 full-repository pytest tests passed locally.**
- **Six independent trace-based pilot shards passed the read-only pilot validator, with 13 algorithms in each output (78 audited algorithm results):**

| Scenario | Forecast policy | Population | Iterations | Seed | Validation |
|---|---|---:|---:|---:|---|
| Microsoft ElasticFlow | repeat | 4 | 2 | 104729 | PASS |
| Alibaba GPU v2023 | repeat | 4 | 2 | 104729 | PASS |
| SenseTime Helios Earth | repeat | 4 | 2 | 104729 | PASS |
| Alibaba Spot-GPU 2026 | repeat | 4 | 2 | 104729 | PASS |
| Microsoft ElasticFlow | repeat | 8 | 10 | 104729 | PASS |
| SenseTime Helios Earth | strict | 4 | 2 | 104729 | PASS |

A strict policy run on Alibaba v2023 correctly rejected a schedule extending beyond the 168-hour recorded week.

The validator replays final schedules under the same fixed-mean regional planning signal, verifies arrival times, service durations and concurrent CPU/GPU/RAM limits, and independently recalculates modeled energy and time-integrated forecast carbon.

## Reproducibility / restrictions

- The isolated local Git snapshot was `84d76c52fa6385772e89396b582459bc22c4f131`; it is NOT the remote GitHub commit identifier. The four corrected files were verified byte-for-byte equivalent between the tested local tree and the GitHub branch.
- The local runtime used Python 3.11+ with NumPy **2.3.5**, SciPy **1.17.0**, pytest **9.0.2**; repository pinned requirements are NumPy 2.4.6, SciPy 1.17.1 and pytest 9.2.0. Re-run in the pinned environment before a publication-grade campaign.
- The model is **offline** and assumes a fixed week-mean UK regional forecast signal for planning. The repeat-policy extrapolation is a synthetic counterfactual, not measured later-week carbon intensity.
- These smoke tests do **not** support claims that QPEO outperforms other algorithms, emissions avoided, renewable utilization, or generalizability.
- The full corrected **13 algorithms × 30 seeds × 4 datasets**, population 50, 500 iterations study has **not** been rerun; do not replace manuscript tables or figures with pilot output.

No paid cloud jobs, GitHub Actions, or primary study reruns were launched. Original locked results remain untouched.
