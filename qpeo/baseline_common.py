from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .archive import ParetoArchive
from .carbon_accounting import account_simulation_result
from .decoder import feasible_score_decode
from .evaluator import evaluate_objectives, normalize_population_objectives, scalar_fitness
from .models import ObjectiveVector, ProblemInstance
from .simulator import SimulatorInputs, simulate_fixed_assignment


@dataclass(frozen=True)
class BaselineResult:
    name: str
    archive: ParetoArchive
    final_population: np.ndarray
    final_assignments: tuple[np.ndarray, ...]
    final_objectives: tuple[ObjectiveVector, ...]
    guide_index: int
    best_scalar_history: np.ndarray
    guide_objective_history: np.ndarray
    archive_size_history: np.ndarray
    seed: int

    @property
    def guide_objectives(self) -> ObjectiveVector:
        return self.final_objectives[self.guide_index]

    @property
    def guide_assignment(self) -> np.ndarray:
        return self.final_assignments[self.guide_index]


def evaluate_scores(
    problem: ProblemInstance,
    simulator_inputs: SimulatorInputs,
    scores: np.ndarray,
) -> tuple[np.ndarray, ObjectiveVector]:
    X = feasible_score_decode(problem, scores)
    sim = simulate_fixed_assignment(problem, X, simulator_inputs)
    obj = evaluate_objectives(problem, X, sim.state)
    if simulator_inputs.carbon_time_series is not None:
        accounting = account_simulation_result(
            problem, X, simulator_inputs, sim, simulator_inputs.carbon_time_series
        )
        obj = replace(
            obj,
            f3_energy_weighted_carbon_intensity=accounting.energy_weighted_intensity_g_per_kwh,
        )
    return X, obj


def objective_matrix(objectives: Sequence[ObjectiveVector]) -> np.ndarray:
    return np.vstack([o.as_array() for o in objectives])


def scalar_values(objectives: Sequence[ObjectiveVector], weights: Sequence[float]) -> np.ndarray:
    Fhat = normalize_population_objectives(objective_matrix(objectives))
    return np.asarray(scalar_fitness(Fhat, weights), dtype=float)


def guide_index(objectives: Sequence[ObjectiveVector], weights: Sequence[float]) -> int:
    return int(np.argmin(scalar_values(objectives, weights)))
