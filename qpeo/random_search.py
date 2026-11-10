"""Random-key scheduling search with no QPEO operator or CAQC decoder.

Random scores are independent U[0,1] for each candidate and resource.
The shared baseline feasible decoder is applied under the same carbon
replay and static resource requirements as FPA/PSO/DE. At each iteration
a fresh population is sampled independently, while the best P feasible
candidates from the union with prior incumbents survive. This is
explicitly *elitist random search*, not unqualified Monte Carlo sampling.

Each iteration evaluates the old P candidates again under the updated
carbon state plus P new candidates, matching the *evaluation structure*
of many baseline methods without asserting all methods are NFE-matched.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .archive import ParetoArchive
from .baseline_common import BaselineResult, evaluate_scores, guide_index, scalar_values
from .baselines import CommonBaselineConfig
from .carbon import with_carbon_intensities
from .models import ObjectiveVector, ProblemInstance
from .simulator import SimulatorInputs


def _at_iteration(problem: ProblemInstance, cfg: CommonBaselineConfig, t: int):
    replay = cfg.carbon_replay
    if replay is None:
        return problem
    slot_names = tuple(r.name for r in problem.resources)
    if replay.slot_names != slot_names:
        raise ValueError("Carbon replay slot order differs from problem resources")
    return with_carbon_intensities(problem, replay.at_iteration(t, cfg.max_iterations))


def _evaluate_population(
    problem: ProblemInstance,
    inputs: SimulatorInputs,
    population: np.ndarray,
) -> tuple[list[np.ndarray], list[ObjectiveVector]]:
    assignments, objectives = [], []
    for scores in population:
        X, obj = evaluate_scores(problem, inputs, scores)
        assignments.append(X)
        objectives.append(obj)
    return assignments, objectives


def _archive(
    population: np.ndarray,
    assignments: list[np.ndarray],
    objectives: list[ObjectiveVector],
) -> ParetoArchive:
    archive = ParetoArchive()
    for scores, X, obj in zip(population, assignments, objectives):
        archive.update(scores.reshape(-1), X, obj)
    return archive


def run_elitist_random_search(
    problem: ProblemInstance,
    simulator_inputs: SimulatorInputs,
    scalar_weights: Sequence[float],
    *,
    config: CommonBaselineConfig | None = None,
    seed: int = 0,
) -> BaselineResult:
    cfg = config or CommonBaselineConfig()
    cfg.validate()
    M, N = len(problem.workloads), len(problem.resources)
    simulator_inputs.validate(M, N)
    weights = np.asarray(scalar_weights, float)
    if weights.shape != (6,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("scalar_weights must contain six nonnegative values summing to 1")
    rng = np.random.default_rng(seed)

    pop = rng.uniform(0, 1, size=(cfg.population_size, M, N))
    current = _at_iteration(problem, cfg, 0)
    assignments, objectives = _evaluate_population(current, simulator_inputs, pop)
    archive = _archive(pop, assignments, objectives)
    best_scalar_history: list[float] = []
    guide_history = [objectives[guide_index(objectives, weights)].as_array().copy()]
    archive_history = [len(archive.entries)]

    for t in range(1, cfg.max_iterations + 1):
        current = _at_iteration(problem, cfg, t)
        assignments, objectives = _evaluate_population(
            current, simulator_inputs, pop
        )
        best_scalar_history.append(float(np.min(scalar_values(objectives, weights))))

        new_pop = rng.uniform(0, 1, size=(cfg.population_size, M, N))
        new_X, new_F = _evaluate_population(current, simulator_inputs, new_pop)
        combined_pop = np.concatenate([pop, new_pop], axis=0)
        combined_X = assignments + new_X
        combined_F = objectives + new_F
        values = scalar_values(combined_F, weights)
        selected = np.argsort(values, kind="stable")[:cfg.population_size]
        pop = combined_pop[selected].copy()
        assignments = [combined_X[k] for k in selected]
        objectives = [combined_F[k] for k in selected]
        archive = _archive(pop, assignments, objectives)
        guide_history.append(
            objectives[guide_index(objectives, weights)].as_array().copy()
        )
        archive_history.append(len(archive.entries))

    return BaselineResult(
        name="Elitist Random Search",
        archive=archive,
        final_population=pop.copy(),
        final_assignments=tuple(np.asarray(x, dtype=int).copy() for x in assignments),
        final_objectives=tuple(objectives),
        guide_index=guide_index(objectives, weights),
        best_scalar_history=np.asarray(best_scalar_history, dtype=float),
        guide_objective_history=np.asarray(guide_history, dtype=float),
        archive_size_history=np.asarray(archive_history, dtype=int),
        seed=int(seed),
    )
