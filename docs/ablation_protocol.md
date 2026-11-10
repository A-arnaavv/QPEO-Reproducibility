# QPEO component-ablation protocol (pre-results)

Status: **implementation preparation only**. The primary 4 × 30 × 13
experiments are already executing from their separately locked commit and will
not be modified by work on this branch.

## Inputs held fixed

Each ablation uses exactly the same four locked trace scenarios, seven logical
resource slots, immutable NESO carbon replay, population size `P=50`,
iteration limit `T=500`, 30 seed values, simulation cache semantics,
power/price model, hard feasibility constraints, and six minimization objectives.
Only the targeted operator family changes. No experimental output has been
produced or interpreted while defining these variants.

| Identifier | Change relative to full QPEO | Interpretation |
|---|---|---|
| `full` | All QPEO operators active | Reference |
| `no_caqc_penalty` | Set CAQC exponential carbon-penalty strength `kappa=0`, retaining ecological affinity and *hard carbon feasibility* | Isolates **CAQC carbon penalty**, not all carbon awareness |
| `no_ebws` | Skip ecology-based workload-sharing update | Isolates EBWS |
| `no_entropy_bro` | Skip population entropy computation and biodiversity restoration | Isolates entropy-triggered biodiversity mechanism; entropy history is absent, never replaced by a fabricated value |
| `no_global_qpso` | Use local QPSO on every proposal | Isolates global QPSO |
| `no_local_qpso` | Use global QPSO on every proposal | Isolates local QPSO |

The parameterized implementation uses `QPEOOperatorSwitches`. The default
switch configuration preserves full QPEO exactly and is checked with an
identical-seed regression test.

**Do not mislabel `no_caqc_penalty` as removal of all carbon awareness:**
resource compatibility still contains a carbon component, the constraints
still enforce carbon-class thresholds, and the environment's carbon trace
remains dynamic for every variant.

## Additional comparisons

The existing full-suite **FPA** rows serve as a classical FPA
reference, using the exact same locked seed, scenario and carbon replay.
A separate **elitist random-search** comparator is now implemented in
`qpeo/random_search.py`. It independently samples uniform random-key
scheduling scores, uses the shared non-QPEO feasible decoder, and retains
the best candidates under the changing NESO carbon replay. Its proposal
distribution does not invoke QPEO angles, CAQC, EBWS or pollination.

This is deliberately called *elitist random search*, not pure independent
Monte Carlo search, because it preserves incumbents across iterations. It
refreshes the previous population under each new carbon state and evaluates
a fresh random population of the same size. Thus it evaluates roughly 2P
candidates per iteration and its simulator-call counts will be reported.
Neither FPA nor random search is a QPEO operator ablation.

With this comparator the report has eight total rows (full, five single-factor
ablations, FPA, random search), seven comparisons against full QPEO across
six objectives, and 42 paired tests per scenario (168 across four scenarios).
Use per-scenario Bonferroni-42 and global Bonferroni-168 alongside raw values.
Random search is **implemented and smoke-tested but has not yet been run
at full scale**.

## Statistical comparisons

Use the same paired seed indices, report mean, sample SD, 95% CI, and
paired effects across all six objectives. Clearly declare correction across
the ablation comparison family. Expose runtime and simulator-call totals,
since matching `P` and `T` does not imply equal evaluation counts.

The optimization environment changes over the 500 iterations due to the NESO
carbon replay. Raw guide-objective histories are available but should not
be interpreted as ordinary stationary-objective convergence curves.

Hypervolume cannot be computed from archive **sizes**; it requires genuine
per-run archive objective vectors under a declared common normalization and
reference point. Do not infer hypervolume values from saved primary
archive-size fields.

## Execution gate

Do not start costly ablation runs until:
1. the running 120-shard primary collection has completed or its missing
   shards are explicitly identified;
2. the default-versus-explicit-default exact-equivalence test passes;
3. every ablation produces a formally feasible assignment and finite objective
   history in a locked scenario smoke test;
4. no unapproved changes to the original primary run protocol are made.
