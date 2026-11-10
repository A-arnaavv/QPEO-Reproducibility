from __future__ import annotations

import numpy as np

from .collapse import NoFeasibleAssignment
from .models import ProblemInstance


def feasible_score_decode(problem: ProblemInstance, scores: np.ndarray) -> np.ndarray:
    S = np.asarray(scores, dtype=float)
    M, N = len(problem.workloads), len(problem.resources)
    if S.shape != (M, N):
        raise ValueError(f"scores must have shape {(M, N)}.")
    if np.any(~np.isfinite(S)):
        raise ValueError("scores must be finite.")

    gpu_remaining = np.asarray([r.gpu_capacity for r in problem.resources], dtype=float)
    mem_remaining = np.asarray([r.memory_capacity for r in problem.resources], dtype=float)

    individually_feasible: list[np.ndarray] = []
    difficulty: list[tuple[int, float, int]] = []
    max_gpu = max(float(r.gpu_capacity) for r in problem.resources)
    max_mem = max(float(r.memory_capacity) for r in problem.resources)

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
        demand = float(w.gpu_demand / max_gpu + w.memory_demand / max_mem)
        difficulty.append((int(mask.sum()), -demand, i))

    order = [i for _, _, i in sorted(difficulty)]
    X = np.zeros((M, N), dtype=int)

    for i in order:
        w = problem.workloads[i]
        resource_order = np.argsort(-S[i], kind="stable")
        placed = False
        for j in resource_order:
            if not individually_feasible[i][j]:
                continue
            if (
                w.gpu_demand <= gpu_remaining[j] + 1e-12
                and w.memory_demand <= mem_remaining[j] + 1e-12
            ):
                X[i, j] = 1
                gpu_remaining[j] -= float(w.gpu_demand)
                mem_remaining[j] -= float(w.memory_demand)
                placed = True
                break
        if not placed:
            raise NoFeasibleAssignment(
                "Common greedy decoder could not construct a feasible assignment. "
                "A feasible assignment may still exist; this is a reconstruction-protocol limitation."
            )

    return X
