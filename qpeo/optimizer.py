from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .archive import ParetoArchive
from .collapse import feasible_caqc_collapse
from .carbon import CarbonReplay, with_carbon_intensities
from .config import ManuscriptQPEOConfig
from .evaluator import (
    dominates,
    evaluate_objectives,
    normalize_population_objectives,
    scalar_fitness,
)
from .models import ObjectiveVector, ProblemInstance
from .operators import (
    assignment_probabilities,
    biodiversity_restore,
    candidate_density_proxy,
    ebws_update,
    ecological_compatibility,
    global_qpso_update,
    kappa_schedule,
    local_qpso_update,
    von_neumann_entropy,
    population_entropy_from_angles,
)
from .simulator import SimulatorInputs, simulate_fixed_assignment


@dataclass(frozen=True)
class ReconstructionQPEOSettings:
    scalar_weights: Sequence[float]
    alpha: float
    beta: float
    gamma: float
    dependency_threshold: float
    dependency_strength: float

    def validate(self) -> None:
        w = np.asarray(self.scalar_weights, dtype=float)
        if w.shape != (6,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
            raise ValueError("scalar_weights must be 6 non-negative values summing to 1.")
        if min(self.alpha, self.beta, self.gamma) < 0 or not np.isclose(
            self.alpha + self.beta + self.gamma, 1.0
        ):
            raise ValueError("alpha, beta, gamma must be non-negative and sum to 1.")
        if not (0 <= self.dependency_threshold <= 1):
            raise ValueError("dependency_threshold must be in [0,1].")
        if not (0 <= self.dependency_strength <= 1):
            raise ValueError("dependency_strength must be in [0,1].")


@dataclass(frozen=True)
class QPEOInputs:
    simulator_inputs: SimulatorInputs
    performance_affinity: np.ndarray
    thermal_headroom: np.ndarray
    dependency_matrix: np.ndarray
    carbon_replay: CarbonReplay | None = None

    def validate(self, M: int, N: int) -> None:
        self.simulator_inputs.validate(M, N)
        if np.asarray(self.performance_affinity).shape != (M, N):
            raise ValueError("performance_affinity must have shape (M,N).")
        if np.asarray(self.thermal_headroom).shape != (N,):
            raise ValueError("thermal_headroom must have shape (N,).")
        if np.asarray(self.dependency_matrix).shape != (M, M):
            raise ValueError("dependency_matrix must have shape (M,M).")
        if self.carbon_replay is not None:
            self.carbon_replay.validate()
            if self.carbon_replay.resources != N:
                raise ValueError("carbon replay resource count must match problem resources")


@dataclass(frozen=True)
class QPEOResult:
    archive: ParetoArchive
    final_population: np.ndarray
    final_assignments: tuple[np.ndarray, ...]
    final_objectives: tuple[ObjectiveVector, ...]
    guide_index: int
    entropy_history: np.ndarray
    guide_objective_history: np.ndarray
    archive_size_history: np.ndarray
    seed: int

    @property
    def guide_objectives(self) -> ObjectiveVector:
        return self.final_objectives[self.guide_index]

    @property
    def guide_assignment(self) -> np.ndarray:
        return self.final_assignments[self.guide_index]


def _evaluate_theta(
    problem: ProblemInstance,
    theta: np.ndarray,
    omega: np.ndarray,
    simulator_inputs: SimulatorInputs,
    *,
    kappa: float,
    ci_ref: float,
) -> tuple[np.ndarray, ObjectiveVector]:
    probs = assignment_probabilities(theta, omega)
    X = feasible_caqc_collapse(
        problem, probs, kappa=kappa, ci_ref=ci_ref
    )
    sim = simulate_fixed_assignment(problem, X, simulator_inputs)
    obj = evaluate_objectives(problem, X, sim.state)
    return X, obj


def _guide_index(objectives: Sequence[ObjectiveVector], weights: Sequence[float]) -> int:
    F = np.vstack([o.as_array() for o in objectives])
    Fhat = normalize_population_objectives(F)
    values = scalar_fitness(Fhat, weights)
    return int(np.argmin(values))


def _omega_for_problem(problem: ProblemInstance, inputs: QPEOInputs, settings: ReconstructionQPEOSettings) -> np.ndarray:
    ci = np.asarray([r.carbon_intensity for r in problem.resources], dtype=float)
    return ecological_compatibility(
        np.asarray(inputs.performance_affinity, dtype=float),
        np.asarray(inputs.thermal_headroom, dtype=float),
        ci,
        alpha=settings.alpha, beta=settings.beta, gamma=settings.gamma,
    )


def _qpeo_problem_for_iteration(base_problem: ProblemInstance, inputs: QPEOInputs, t: int, max_iterations: int) -> ProblemInstance:
    replay = inputs.carbon_replay
    if replay is None:
        return base_problem
    expected = tuple(r.name for r in base_problem.resources)
    if replay.slot_names != expected:
        raise ValueError("carbon replay slot order must match problem resource names")
    return with_carbon_intensities(base_problem, replay.planning_mean())


def run_qpeo(
    problem: ProblemInstance,
    inputs: QPEOInputs,
    settings: ReconstructionQPEOSettings,
    *,
    config: ManuscriptQPEOConfig | None = None,
    seed: int = 0,
) -> QPEOResult:
    cfg = config or ManuscriptQPEOConfig()
    settings.validate()
    M, N = len(problem.workloads), len(problem.resources)
    D = M * N
    inputs.validate(M, N)
    if cfg.population_size < 3:
        raise ValueError("population_size must be at least 3 for local QPSO sampling.")

    base_problem = problem
    problem = _qpeo_problem_for_iteration(base_problem, inputs, 0, cfg.max_iterations)
    rng = np.random.default_rng(seed)
    population = rng.uniform(0.0, np.pi / 2, size=(cfg.population_size, D))

    omega = _omega_for_problem(problem, inputs, settings)

    assignments: list[np.ndarray] = []
    objectives: list[ObjectiveVector] = []
    archive = ParetoArchive()
    kappa0 = kappa_schedule(
        0,
        kappa_min=cfg.kappa_min,
        kappa_max=cfg.kappa_max,
        tau_carbon=cfg.tau_carbon,
    )
    for n in range(cfg.population_size):
        X, obj = _evaluate_theta(
            problem, population[n], omega, inputs.simulator_inputs,
            kappa=kappa0, ci_ref=cfg.ci_ref,
        )
        assignments.append(X)
        objectives.append(obj)
        archive.update(population[n], X, obj)

    guide = _guide_index(objectives, settings.scalar_weights)
    stagnation = np.zeros(cfg.population_size, dtype=int)
    entropy_history: list[float] = []
    guide_objective_history: list[np.ndarray] = [objectives[guide].as_array().copy()]
    archive_sizes: list[int] = [len(archive.entries)]

    all_indices = np.arange(cfg.population_size)

    for t in range(1, cfg.max_iterations + 1):
        kappa = kappa_schedule(
            t,
            kappa_min=cfg.kappa_min,
            kappa_max=cfg.kappa_max,
            tau_carbon=cfg.tau_carbon,
        )
        # Fixed planning carbon needs no population refresh.
        theta_best = population[guide].copy()

        for n in range(cfg.population_size):
            if rng.uniform() < cfg.global_switch_probability:
                proposal = global_qpso_update(
                    population[n], theta_best, rng, lam=cfg.levy_lambda
                )
            else:
                choices = all_indices[all_indices != n]
                m, l = rng.choice(choices, size=2, replace=False)
                proposal = local_qpso_update(
                    population[n], population[m], population[l], rng
                )

            proposal_2d = ebws_update(
                proposal.reshape(M, N),
                np.asarray(inputs.dependency_matrix, dtype=float),
                delta_dep=settings.dependency_threshold,
                phi=settings.dependency_strength,
            )
            proposal = proposal_2d.reshape(D)

            X_new, obj_new = _evaluate_theta(
                problem, proposal, omega, inputs.simulator_inputs,
                kappa=kappa, ci_ref=cfg.ci_ref,
            )

            if dominates(obj_new, objectives[n]):
                population[n] = proposal
                assignments[n] = X_new
                objectives[n] = obj_new
                stagnation[n] = 0
                archive.update(proposal, X_new, obj_new)
            else:
                stagnation[n] += 1

        entropy = population_entropy_from_angles(population)
        entropy_history.append(entropy)

        if entropy < cfg.entropy_threshold:
            stagnant_mask = stagnation >= cfg.stagnation_iterations
            if np.any(stagnant_mask):
                population = biodiversity_restore(
                    population,
                    population[_guide_index(objectives, settings.scalar_weights)],
                    stagnant_mask,
                    rng,
                    delta_bio=cfg.biodiversity_amplitude,
                )
                stagnation[stagnant_mask] = 0

        guide = _guide_index(objectives, settings.scalar_weights)
        guide_objective_history.append(objectives[guide].as_array().copy())
        archive_sizes.append(len(archive.entries))

    return QPEOResult(
        archive=archive,
        final_population=population.copy(),
        final_assignments=tuple(np.asarray(x, dtype=int).copy() for x in assignments),
        final_objectives=tuple(objectives),
        guide_index=guide,
        entropy_history=np.asarray(entropy_history, dtype=float),
        guide_objective_history=np.asarray(guide_objective_history, dtype=float),
        archive_size_history=np.asarray(archive_sizes, dtype=int),
        seed=seed,
    )
