from __future__ import annotations

import numpy as np

from .models import ProblemInstance
from .operators import caqc_scores


class NoFeasibleAssignment(RuntimeError):
    pass


def feasible_caqc_collapse(
    problem: ProblemInstance,
    probabilities: np.ndarray,
    *,
    kappa: float,
    ci_ref: float = 475.0,
) -> np.ndarray:
    P = np.asarray(probabilities, dtype=float)
    M, N = len(problem.workloads), len(problem.resources)
    if P.shape != (M, N):
        raise ValueError(f"probabilities must have shape {(M, N)}.")

    ci = np.asarray([r.carbon_intensity for r in problem.resources], dtype=float)
    scores = caqc_scores(P, ci, kappa=kappa, ci_ref=ci_ref)

    individually_feasible: list[np.ndarray] = []
    difficulty: list[tuple[int, float, int]] = []
    for i, w in enumerate(problem.workloads):
        threshold = float(problem.carbon_thresholds[w.carbon_class])
        mask = np.asarray([
            (w.gpu_demand <= r.gpu_capacity + 1e-12)
            and (w.memory_demand <= r.memory_capacity + 1e-12)
            and (r.carbon_intensity <= threshold + 1e-12)
            for r in problem.resources
        ], dtype=bool)
        if not np.any(mask):
            raise NoFeasibleAssignment(
                f"Workload {w.name!r} has no individually feasible resource."
            )
        individually_feasible.append(mask)
        max_gpu = max(float(r.gpu_capacity) for r in problem.resources)
        max_mem = max(float(r.memory_capacity) for r in problem.resources)
        demand = float(w.gpu_demand / max_gpu + w.memory_demand / max_mem)
        difficulty.append((int(mask.sum()), -demand, i))

    order = [i for _, _, i in sorted(difficulty)]
    X = np.zeros((M, N), dtype=int)

    for i in order:
        allowed = individually_feasible[i]
        resource_order = np.argsort(-scores[i], kind="stable")
        chosen = next((int(j) for j in resource_order if allowed[j]), None)
        if chosen is None:
            raise NoFeasibleAssignment(f"Workload {problem.workloads[i].name!r} has no feasible resource.")
        X[i, chosen] = 1

    return X
