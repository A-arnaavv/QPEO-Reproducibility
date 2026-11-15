from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np

from .archive import ParetoArchive
from .carbon import CarbonReplay, with_carbon_intensities
from .baseline_common import BaselineResult, evaluate_scores, guide_index, objective_matrix, scalar_values
from .models import ObjectiveVector, ProblemInstance
from .operators import mantegna_levy_step
from .simulator import SimulatorInputs


@dataclass(frozen=True)
class CommonBaselineConfig:
    population_size: int = 50
    max_iterations: int = 500
    carbon_replay: CarbonReplay | None = None

    def validate(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")


def _problem_for_iteration(base_problem: ProblemInstance, cfg: CommonBaselineConfig, t: int) -> ProblemInstance:
    replay = cfg.carbon_replay
    if replay is None:
        return base_problem
    expected = tuple(r.name for r in base_problem.resources)
    if replay.slot_names != expected:
        raise ValueError("carbon replay slot order must match problem resource names")
    return with_carbon_intensities(base_problem, replay.planning_mean())


def _evaluate_population(problem: ProblemInstance, simulator_inputs: SimulatorInputs, pop: np.ndarray, transform=None):
    assignments: list[np.ndarray] = []
    objectives: list[ObjectiveVector] = []
    for i in range(len(pop)):
        scores = pop[i] if transform is None else transform(pop[i])
        X, obj = evaluate_scores(problem, simulator_inputs, scores)
        assignments.append(X); objectives.append(obj)
    return assignments, objectives


def _dynamic_refresh(base_problem, cfg, t, simulator_inputs, pop, transform=None):
    if cfg.carbon_replay is None:
        return None
    current = _problem_for_iteration(base_problem, cfg, t)
    assignments, objectives = _evaluate_population(current, simulator_inputs, pop, transform)
    archive = _initialize_archive(pop, assignments, objectives)
    return current, assignments, objectives, archive


def _validate_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (6,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("scalar_weights must be 6 non-negative values summing to 1.")
    return w


def _initialize_archive(
    pop: np.ndarray,
    assignments: list[np.ndarray],
    objectives: list[ObjectiveVector],
) -> ParetoArchive:
    archive = ParetoArchive()
    for k in range(len(objectives)):
        archive.update(pop[k].reshape(-1), assignments[k], objectives[k])
    return archive


def _final_result(
    name: str,
    pop: np.ndarray,
    assignments: list[np.ndarray],
    objectives: list[ObjectiveVector],
    weights: Sequence[float],
    archive: ParetoArchive,
    best_history: list[float],
    guide_history: list[np.ndarray],
    archive_history: list[int],
    seed: int,
) -> BaselineResult:
    g = guide_index(objectives, weights)
    return BaselineResult(
        name=name,
        archive=archive,
        final_population=np.asarray(pop, dtype=float).copy(),
        final_assignments=tuple(np.asarray(x, dtype=int).copy() for x in assignments),
        final_objectives=tuple(objectives),
        guide_index=g,
        best_scalar_history=np.asarray(best_history, dtype=float),
        guide_objective_history=np.asarray(guide_history, dtype=float),
        archive_size_history=np.asarray(archive_history, dtype=int),
        seed=seed,
    )


def run_fpa(
    problem: ProblemInstance,
    simulator_inputs: SimulatorInputs,
    scalar_weights: Sequence[float],
    *,
    config: CommonBaselineConfig | None = None,
    switch_probability: float = 0.8,
    levy_lambda: float = 1.5,
    seed: int = 0,
) -> BaselineResult:
    """Continuous Flower Pollination Algorithm baseline with a common decoder.

    Reconstruction protocol FPA-B1: each flower is an MxN score matrix in [0,1].
    Global and local pollination follow the standard FPA structure. Candidate score
    matrices are converted to feasible schedules by the shared baseline decoder.
    """
    cfg = config or CommonBaselineConfig()
    cfg.validate(); _validate_weights(scalar_weights)
    base_problem = problem
    problem = _problem_for_iteration(base_problem, cfg, 0)
    if not (0 <= switch_probability <= 1):
        raise ValueError("switch_probability must be in [0,1].")
    M, N = len(problem.workloads), len(problem.resources)
    rng = np.random.default_rng(seed)
    pop = rng.uniform(0.0, 1.0, size=(cfg.population_size, M, N))

    assignments: list[np.ndarray] = []
    objectives: list[ObjectiveVector] = []
    for k in range(cfg.population_size):
        X, obj = evaluate_scores(problem, simulator_inputs, pop[k])
        assignments.append(X); objectives.append(obj)
    archive = _initialize_archive(pop, assignments, objectives)
    best_history: list[float] = []
    guide_history: list[np.ndarray] = [objectives[guide_index(objectives, scalar_weights)].as_array().copy()]
    archive_history: list[int] = [len(archive.entries)]

    for t in range(1, cfg.max_iterations + 1):
        refreshed = _dynamic_refresh(base_problem, cfg, t, simulator_inputs, pop)
        if refreshed is not None:
            problem, assignments, objectives, archive = refreshed
        values = scalar_values(objectives, scalar_weights)
        best_idx = int(np.argmin(values))
        best = pop[best_idx].copy()
        best_history.append(float(values[best_idx]))

        for i in range(cfg.population_size):
            if rng.uniform() < switch_probability:
                L = mantegna_levy_step(rng, levy_lambda)
                proposal = pop[i] + L * (best - pop[i])
            else:
                choices = np.arange(cfg.population_size)
                choices = choices[choices != i]
                j, k = rng.choice(choices, size=2, replace=False)
                eps = rng.uniform()
                proposal = pop[i] + eps * (pop[j] - pop[k])
            proposal = np.clip(proposal, 0.0, 1.0)
            X_new, obj_new = evaluate_scores(problem, simulator_inputs, proposal)

            pair = [objectives[i], obj_new]
            pair_vals = scalar_values(pair, scalar_weights)
            if pair_vals[1] < pair_vals[0]:
                pop[i] = proposal
                assignments[i] = X_new
                objectives[i] = obj_new
                archive.update(proposal.reshape(-1), X_new, obj_new)

        guide_history.append(objectives[guide_index(objectives, scalar_weights)].as_array().copy())
        archive_history.append(len(archive.entries))

    return _final_result(
        "FPA", pop, assignments, objectives, scalar_weights, archive,
        best_history, guide_history, archive_history, seed,
    )


def _nondominated_sort(F: np.ndarray) -> list[np.ndarray]:
    n = F.shape[0]
    dominates_set: list[list[int]] = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    first: list[int] = []
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            p_dom_q = np.all(F[p] <= F[q]) and np.any(F[p] < F[q])
            q_dom_p = np.all(F[q] <= F[p]) and np.any(F[q] < F[p])
            if p_dom_q:
                dominates_set[p].append(q)
            elif q_dom_p:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            first.append(p)

    fronts: list[np.ndarray] = []
    current = first
    while current:
        fronts.append(np.asarray(current, dtype=int))
        nxt: list[int] = []
        for p in current:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    nxt.append(q)
        current = nxt
    return fronts


def _crowding_distance(F: np.ndarray, front: np.ndarray) -> np.ndarray:
    m = F.shape[1]
    size = len(front)
    if size == 0:
        return np.empty(0)
    if size <= 2:
        return np.full(size, np.inf)
    dist = np.zeros(size, dtype=float)
    sub = F[front]
    for k in range(m):
        order = np.argsort(sub[:, k], kind="stable")
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        lo, hi = sub[order[0], k], sub[order[-1], k]
        span = hi - lo
        if span <= 1e-15:
            continue
        for r in range(1, size - 1):
            dist[order[r]] += (sub[order[r + 1], k] - sub[order[r - 1], k]) / span
    return dist


def _rank_and_crowding(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fronts = _nondominated_sort(F)
    rank = np.full(F.shape[0], np.iinfo(np.int32).max, dtype=int)
    crowd = np.zeros(F.shape[0], dtype=float)
    for r, front in enumerate(fronts):
        rank[front] = r
        crowd[front] = _crowding_distance(F, front)
    return rank, crowd


def _binary_tournament(rng: np.random.Generator, rank: np.ndarray, crowd: np.ndarray) -> int:
    a, b = rng.integers(0, len(rank), size=2)
    if rank[a] < rank[b]: return int(a)
    if rank[b] < rank[a]: return int(b)
    if crowd[a] > crowd[b]: return int(a)
    if crowd[b] > crowd[a]: return int(b)
    return int(a if rng.uniform() < 0.5 else b)


def _sbx(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    eta_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(size=p1.shape)
    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta_c + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)),
    )
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def _polynomial_mutation(x: np.ndarray, rng: np.random.Generator, eta_m: float, p: float) -> np.ndarray:
    y = x.copy()
    mask = rng.uniform(size=x.shape) < p
    if not np.any(mask):
        return y
    u = rng.uniform(size=x.shape)
    delta = np.where(
        u < 0.5,
        (2 * u) ** (1.0 / (eta_m + 1.0)) - 1.0,
        1.0 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1.0)),
    )
    y[mask] += delta[mask]
    return np.clip(y, 0.0, 1.0)


def run_nsga2(
    problem: ProblemInstance,
    simulator_inputs: SimulatorInputs,
    scalar_weights: Sequence[float],
    *,
    config: CommonBaselineConfig | None = None,
    crossover_probability: float = 0.9,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    seed: int = 0,
) -> BaselineResult:
    """NSGA-II baseline over MxN continuous random-key score matrices."""
    cfg = config or CommonBaselineConfig()
    cfg.validate(); _validate_weights(scalar_weights)
    base_problem = problem
    problem = _problem_for_iteration(base_problem, cfg, 0)
    M, N = len(problem.workloads), len(problem.resources)
    D = M * N
    rng = np.random.default_rng(seed)
    pop = rng.uniform(0.0, 1.0, size=(cfg.population_size, M, N))

    assignments: list[np.ndarray] = []
    objectives: list[ObjectiveVector] = []
    for i in range(cfg.population_size):
        X, obj = evaluate_scores(problem, simulator_inputs, pop[i])
        assignments.append(X); objectives.append(obj)
    archive = _initialize_archive(pop, assignments, objectives)
    best_history: list[float] = []
    guide_history: list[np.ndarray] = [objectives[guide_index(objectives, scalar_weights)].as_array().copy()]
    archive_history: list[int] = [len(archive.entries)]

    for t in range(1, cfg.max_iterations + 1):
        refreshed = _dynamic_refresh(base_problem, cfg, t, simulator_inputs, pop)
        if refreshed is not None:
            problem, assignments, objectives, archive = refreshed
        F = objective_matrix(objectives)
        rank, crowd = _rank_and_crowding(F)
        best_history.append(float(np.min(scalar_values(objectives, scalar_weights))))

        children: list[np.ndarray] = []
        while len(children) < cfg.population_size:
            i1 = _binary_tournament(rng, rank, crowd)
            i2 = _binary_tournament(rng, rank, crowd)
            p1, p2 = pop[i1], pop[i2]
            if rng.uniform() < crossover_probability:
                c1, c2 = _sbx(p1, p2, rng, eta_c)
            else:
                c1, c2 = p1.copy(), p2.copy()
            mut_p = 1.0 / D
            children.append(_polynomial_mutation(c1, rng, eta_m, mut_p))
            if len(children) < cfg.population_size:
                children.append(_polynomial_mutation(c2, rng, eta_m, mut_p))
        child_pop = np.stack(children)

        child_assignments: list[np.ndarray] = []
        child_objectives: list[ObjectiveVector] = []
        for i in range(cfg.population_size):
            X, obj = evaluate_scores(problem, simulator_inputs, child_pop[i])
            child_assignments.append(X); child_objectives.append(obj)
            archive.update(child_pop[i].reshape(-1), X, obj)

        combined_pop = np.concatenate([pop, child_pop], axis=0)
        combined_assignments = assignments + child_assignments
        combined_objectives = objectives + child_objectives
        combined_F = objective_matrix(combined_objectives)
        fronts = _nondominated_sort(combined_F)
        selected: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= cfg.population_size:
                selected.extend(front.tolist())
            else:
                dist = _crowding_distance(combined_F, front)
                order = np.argsort(-dist, kind="stable")
                need = cfg.population_size - len(selected)
                selected.extend(front[order[:need]].tolist())
                break
        idx = np.asarray(selected, dtype=int)
        pop = combined_pop[idx]
        assignments = [combined_assignments[i] for i in idx]
        objectives = [combined_objectives[i] for i in idx]
        guide_history.append(objectives[guide_index(objectives, scalar_weights)].as_array().copy())
        archive_history.append(len(archive.entries))

    return _final_result(
        "NSGA-II", pop, assignments, objectives, scalar_weights, archive,
        best_history, guide_history, archive_history, seed,
    )


def run_qiea(
    problem: ProblemInstance,
    simulator_inputs: SimulatorInputs,
    scalar_weights: Sequence[float],
    *,
    config: CommonBaselineConfig | None = None,
    rotation_angle: float = 0.05 * math.pi,
    mutation_probability: float = 0.01,
    seed: int = 0,
) -> BaselineResult:
    """QIEA-style baseline adapted to one-hot scheduling.

    Each candidate contains MxN q-bit angles. Observation probabilities are sin^2(theta).
    The current scalar guide supplies the target one-hot bits; each angle is rotated
    toward the target bit. This is explicitly a QIEA-style scheduling reconstruction,
    not a claim that one canonical QIEA implementation was recovered.
    """
    cfg = config or CommonBaselineConfig()
    cfg.validate(); _validate_weights(scalar_weights)
    base_problem = problem
    problem = _problem_for_iteration(base_problem, cfg, 0)
    if rotation_angle <= 0:
        raise ValueError("rotation_angle must be > 0.")
    if not (0 <= mutation_probability <= 1):
        raise ValueError("mutation_probability must be in [0,1].")
    M, N = len(problem.workloads), len(problem.resources)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, np.pi / 2, size=(cfg.population_size, M, N))

    assignments: list[np.ndarray] = []
    objectives: list[ObjectiveVector] = []
    for i in range(cfg.population_size):
        scores = np.sin(theta[i]) ** 2
        X, obj = evaluate_scores(problem, simulator_inputs, scores)
        assignments.append(X); objectives.append(obj)
    archive = _initialize_archive(theta, assignments, objectives)
    best_history: list[float] = []
    guide_history: list[np.ndarray] = [objectives[guide_index(objectives, scalar_weights)].as_array().copy()]
    archive_history: list[int] = [len(archive.entries)]

    for t in range(1, cfg.max_iterations + 1):
        refreshed = _dynamic_refresh(base_problem, cfg, t, simulator_inputs, theta, lambda z: np.sin(z) ** 2)
        if refreshed is not None:
            problem, assignments, objectives, archive = refreshed
        vals = scalar_values(objectives, scalar_weights)
        g = int(np.argmin(vals))
        guide_bits = assignments[g].astype(float)
        best_history.append(float(vals[g]))

        for i in range(cfg.population_size):
            current_bits = assignments[i].astype(float)
            direction = guide_bits - current_bits
            proposal = theta[i] + rotation_angle * direction
            mask = rng.uniform(size=(M, N)) < mutation_probability
            proposal[mask] = (np.pi / 2) - proposal[mask]
            proposal = np.clip(proposal, 0.0, np.pi / 2)
            X_new, obj_new = evaluate_scores(problem, simulator_inputs, np.sin(proposal) ** 2)
            pair_vals = scalar_values([objectives[i], obj_new], scalar_weights)
            if pair_vals[1] < pair_vals[0]:
                theta[i] = proposal
                assignments[i] = X_new
                objectives[i] = obj_new
                archive.update(proposal.reshape(-1), X_new, obj_new)

        guide_history.append(objectives[guide_index(objectives, scalar_weights)].as_array().copy())
        archive_history.append(len(archive.entries))

    return _final_result(
        "QIEA", theta, assignments, objectives, scalar_weights, archive,
        best_history, guide_history, archive_history, seed,
    )


def _population_candidate_better(
    objectives: list[ObjectiveVector],
    idx: int,
    candidate: ObjectiveVector,
    weights: Sequence[float],
) -> bool:
    augmented = list(objectives) + [candidate]
    vals = scalar_values(augmented, weights)
    return bool(vals[-1] < vals[idx])


def run_pso(
    problem: ProblemInstance,
    simulator_inputs: SimulatorInputs,
    scalar_weights: Sequence[float],
    *,
    config: CommonBaselineConfig | None = None,
    inertia: float = 0.729,
    cognitive: float = 1.49445,
    social: float = 1.49445,
    seed: int = 0,
) -> BaselineResult:
    """PSO reconstruction over continuous MxN scheduling scores."""
    cfg = config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem = problem; problem = _problem_for_iteration(base_problem, cfg, 0)
    M,N=len(problem.workloads),len(problem.resources); rng=np.random.default_rng(seed)
    pop=rng.uniform(0,1,(cfg.population_size,M,N)); vel=np.zeros_like(pop)
    assignments=[]; objectives=[]
    for i in range(cfg.population_size):
        X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    pbest=pop.copy(); pbest_obj=list(objectives)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for t in range(1, cfg.max_iterations + 1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
            _,pbest_obj=_evaluate_population(problem,simulator_inputs,pbest)
        pvals=scalar_values(pbest_obj,scalar_weights); g=int(np.argmin(pvals)); gbest=pbest[g].copy(); bh.append(float(pvals[g]))
        for i in range(cfg.population_size):
            r1=rng.uniform(size=(M,N)); r2=rng.uniform(size=(M,N))
            vel[i]=inertia*vel[i]+cognitive*r1*(pbest[i]-pop[i])+social*r2*(gbest-pop[i])
            pop[i]=np.clip(pop[i]+vel[i],0,1)
            X,o=evaluate_scores(problem,simulator_inputs,pop[i])
            assignments[i]=X; objectives[i]=o
            if _population_candidate_better(pbest_obj,i,o,scalar_weights):
                pbest[i]=pop[i].copy(); pbest_obj[i]=o
            archive.update(pop[i].reshape(-1),X,o)
        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())
        ah.append(len(archive.entries))
    return _final_result("PSO",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)


def run_de(
    problem: ProblemInstance,
    simulator_inputs: SimulatorInputs,
    scalar_weights: Sequence[float],
    *,
    config: CommonBaselineConfig | None = None,
    differential_weight: float = 0.5,
    crossover_rate: float = 0.9,
    seed: int = 0,
) -> BaselineResult:
    """DE/rand/1/bin reconstruction."""
    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
    M,N=len(problem.workloads),len(problem.resources); D=M*N; rng=np.random.default_rng(seed)
    pop=rng.uniform(0,1,(cfg.population_size,M,N)); assignments=[]; objectives=[]
    for i in range(cfg.population_size):
        X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for t in range(1, cfg.max_iterations + 1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        bh.append(float(np.min(scalar_values(objectives,scalar_weights))))
        for i in range(cfg.population_size):
            choices=np.delete(np.arange(cfg.population_size),i); a,b,c=rng.choice(choices,3,replace=False)
            mutant=np.clip(pop[a]+differential_weight*(pop[b]-pop[c]),0,1)
            mask=rng.uniform(size=(M,N))<crossover_rate
            mask.reshape(-1)[rng.integers(D)]=True
            trial=np.where(mask,mutant,pop[i])
            X,o=evaluate_scores(problem,simulator_inputs,trial)
            if _population_candidate_better(objectives,i,o,scalar_weights):
                pop[i]=trial; assignments[i]=X; objectives[i]=o; archive.update(trial.reshape(-1),X,o)
        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())
        ah.append(len(archive.entries))
    return _final_result("DE",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)


def run_gwo(problem, simulator_inputs, scalar_weights, *, config=None, seed=0) -> BaselineResult:
    """Grey Wolf Optimizer reconstruction."""
    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
    M,N=len(problem.workloads),len(problem.resources); rng=np.random.default_rng(seed)
    pop=rng.uniform(0,1,(cfg.population_size,M,N)); assignments=[]; objectives=[]
    for i in range(cfg.population_size): X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for step in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,step,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); order=np.argsort(vals); alpha,beta,delta=[pop[k].copy() for k in order[:3]]; bh.append(float(vals[order[0]]))
        t=step-1; a=2.0*(1.0-t/max(cfg.max_iterations-1,1)); newpop=[]
        for i in range(cfg.population_size):
            parts=[]
            for leader in (alpha,beta,delta):
                r1=rng.uniform(size=(M,N)); r2=rng.uniform(size=(M,N)); A=2*a*r1-a; C=2*r2
                parts.append(leader-A*np.abs(C*leader-pop[i]))
            newpop.append(np.clip(np.mean(parts,axis=0),0,1))
        for i,p in enumerate(newpop):
            X,o=evaluate_scores(problem,simulator_inputs,p); pop[i]=p; assignments[i]=X; objectives[i]=o; archive.update(p.reshape(-1),X,o)
        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())
        ah.append(len(archive.entries))
    return _final_result("GWO",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)


def run_woa(problem, simulator_inputs, scalar_weights, *, config=None, b:float=1.0, seed=0) -> BaselineResult:
    """Whale Optimization Algorithm reconstruction."""
    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
    M,N=len(problem.workloads),len(problem.resources); rng=np.random.default_rng(seed)
    pop=rng.uniform(0,1,(cfg.population_size,M,N)); assignments=[]; objectives=[]
    for i in range(cfg.population_size): X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for step in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,step,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); leader=pop[int(np.argmin(vals))].copy(); bh.append(float(np.min(vals))); t=step-1; a=2-2*t/max(cfg.max_iterations-1,1)
        proposals=[]
        for i in range(cfg.population_size):
            r1,r2,p=float(rng.uniform()),float(rng.uniform()),float(rng.uniform()); A=2*a*r1-a; C=2*r2
            if p<0.5:
                if abs(A)<1:
                    x=leader-A*np.abs(C*leader-pop[i])
                else:
                    rand=pop[rng.integers(cfg.population_size)]; x=rand-A*np.abs(C*rand-pop[i])
            else:
                l=float(rng.uniform(-1,1)); D=np.abs(leader-pop[i]); x=D*np.exp(b*l)*np.cos(2*np.pi*l)+leader
            proposals.append(np.clip(x,0,1))
        for i,pv in enumerate(proposals):
            X,o=evaluate_scores(problem,simulator_inputs,pv); pop[i]=pv; assignments[i]=X; objectives[i]=o; archive.update(pv.reshape(-1),X,o)
        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())
        ah.append(len(archive.entries))
    return _final_result("WOA",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)


def run_abc(problem, simulator_inputs, scalar_weights, *, config=None, limit:int=20, seed=0) -> BaselineResult:
    """Artificial Bee Colony reconstruction."""
    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
    M,N=len(problem.workloads),len(problem.resources); rng=np.random.default_rng(seed)
    pop=rng.uniform(0,1,(cfg.population_size,M,N)); trials=np.zeros(cfg.population_size,int); assignments=[]; objectives=[]
    for i in range(cfg.population_size): X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for t in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        bh.append(float(np.min(scalar_values(objectives,scalar_weights))))
        for i in range(cfg.population_size):
            k=int(rng.choice(np.delete(np.arange(cfg.population_size),i))); phi=rng.uniform(-1,1,size=(M,N)); cand=np.clip(pop[i]+phi*(pop[i]-pop[k]),0,1)
            X,o=evaluate_scores(problem,simulator_inputs,cand)
            if _population_candidate_better(objectives,i,o,scalar_weights): pop[i]=cand; assignments[i]=X; objectives[i]=o; trials[i]=0; archive.update(cand.reshape(-1),X,o)
            else: trials[i]+=1
        vals=scalar_values(objectives,scalar_weights); fitness=1/(1+vals-np.min(vals)); probs=fitness/fitness.sum()
        for _bee in range(cfg.population_size):
            i=int(rng.choice(cfg.population_size,p=probs)); k=int(rng.choice(np.delete(np.arange(cfg.population_size),i))); phi=rng.uniform(-1,1,size=(M,N)); cand=np.clip(pop[i]+phi*(pop[i]-pop[k]),0,1)
            X,o=evaluate_scores(problem,simulator_inputs,cand)
            if _population_candidate_better(objectives,i,o,scalar_weights): pop[i]=cand; assignments[i]=X; objectives[i]=o; trials[i]=0; archive.update(cand.reshape(-1),X,o)
            else: trials[i]+=1
        for i in np.where(trials>=limit)[0]:
            pop[i]=rng.uniform(0,1,(M,N)); X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments[i]=X; objectives[i]=o; trials[i]=0; archive.update(pop[i].reshape(-1),X,o)
        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())
        ah.append(len(archive.entries))
    return _final_result("ABC",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)


def run_ga(problem, simulator_inputs, scalar_weights, *, config=None, crossover_probability=.9, mutation_probability=None, seed=0) -> BaselineResult:
    """Real-coded genetic algorithm reconstruction."""
    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
    M,N=len(problem.workloads),len(problem.resources); D=M*N; rng=np.random.default_rng(seed); mp=(1/D if mutation_probability is None else mutation_probability)
    pop=rng.uniform(0,1,(cfg.population_size,M,N)); assignments=[]; objectives=[]
    for i in range(cfg.population_size): X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for t in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); children=[]
        def tournament():
            a,b=rng.integers(0,cfg.population_size,2); return int(a if vals[a]<=vals[b] else b)
        while len(children)<cfg.population_size:
            p1,p2=pop[tournament()],pop[tournament()]
            if rng.uniform()<crossover_probability:
                alpha=rng.uniform(size=(M,N)); c1=alpha*p1+(1-alpha)*p2; c2=alpha*p2+(1-alpha)*p1
            else: c1,c2=p1.copy(),p2.copy()
            for c in (c1,c2):
                mask=rng.uniform(size=(M,N))<mp; c[mask]=np.clip(c[mask]+rng.normal(0,.1,size=mask.sum()),0,1); children.append(c)
                if len(children)>=cfg.population_size: break
        child_objs=[]; child_X=[]
        for c in children:
            X,o=evaluate_scores(problem,simulator_inputs,c); child_X.append(X); child_objs.append(o); archive.update(c.reshape(-1),X,o)
        combo=np.concatenate([pop,np.stack(children)],axis=0); comboX=assignments+child_X; comboO=objectives+child_objs; cvals=scalar_values(comboO,scalar_weights); keep=np.argsort(cvals)[:cfg.population_size]
        pop=combo[keep]; assignments=[comboX[i] for i in keep]; objectives=[comboO[i] for i in keep]; gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy()); ah.append(len(archive.entries))
    return _final_result("GA",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)


def run_hs(problem, simulator_inputs, scalar_weights, *, config=None, hmcr=.9, par=.3, bandwidth=.05, seed=0) -> BaselineResult:
    """Harmony Search reconstruction; population acts as harmony memory."""
    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
    M,N=len(problem.workloads),len(problem.resources); rng=np.random.default_rng(seed)
    pop=rng.uniform(0,1,(cfg.population_size,M,N)); assignments=[]; objectives=[]
    for i in range(cfg.population_size): X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for t in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); new=np.empty((M,N))
        for idx in np.ndindex(M,N):
            if rng.uniform()<hmcr:
                new[idx]=pop[rng.integers(cfg.population_size)][idx]
                if rng.uniform()<par: new[idx]+=rng.uniform(-bandwidth,bandwidth)
            else: new[idx]=rng.uniform()
        new=np.clip(new,0,1); X,o=evaluate_scores(problem,simulator_inputs,new)
        augmented=objectives+[o]; av=scalar_values(augmented,scalar_weights); worst=int(np.argmax(av[:-1]))
        if av[-1]<av[worst]: pop[worst]=new; assignments[worst]=X; objectives[worst]=o; archive.update(new.reshape(-1),X,o)
        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())
        ah.append(len(archive.entries))
    return _final_result("HS",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)


def run_fa(problem, simulator_inputs, scalar_weights, *, config=None, beta0=1.0, gamma=1.0, alpha=0.2, seed=0) -> BaselineResult:
    """Firefly Algorithm reconstruction."""
    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
    M,N=len(problem.workloads),len(problem.resources); rng=np.random.default_rng(seed)
    pop=rng.uniform(0,1,(cfg.population_size,M,N)); assignments=[]; objectives=[]
    for i in range(cfg.population_size): X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for step in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,step,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        t=step-1; vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); order=np.argsort(vals); newpop=pop.copy(); alpha_t=alpha*(.97**t)
        for ii in range(cfg.population_size):
            i=order[ii]
            for jj in range(ii):
                j=order[jj]; r2=float(np.sum((newpop[i]-pop[j])**2)); beta=beta0*np.exp(-gamma*r2)
                newpop[i]=np.clip(newpop[i]+beta*(pop[j]-newpop[i])+alpha_t*(rng.uniform(size=(M,N))-.5),0,1)
        for i in range(cfg.population_size):
            X,o=evaluate_scores(problem,simulator_inputs,newpop[i]); pop[i]=newpop[i]; assignments[i]=X; objectives[i]=o; archive.update(pop[i].reshape(-1),X,o)
        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())
        ah.append(len(archive.entries))
    return _final_result("FA",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)


def run_aco(problem, simulator_inputs, scalar_weights, *, config=None, evaporation=.2, reinforcement=1.0, seed=0) -> BaselineResult:
    """Assignment-pheromone ACO reconstruction using the common decoder."""
    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
    M,N=len(problem.workloads),len(problem.resources); rng=np.random.default_rng(seed); pheromone=np.ones((M,N),float)
    pop=np.empty((cfg.population_size,M,N)); assignments=[]; objectives=[]
    def construct():
        noise=-np.log(-np.log(np.clip(rng.uniform(size=(M,N)),1e-12,1-1e-12)))
        return np.maximum(1e-12,np.log(pheromone+1e-12)+noise)
    for i in range(cfg.population_size):
        pop[i]=construct(); X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o)
    archive=_initialize_archive(pop,assignments,objectives); bh=[]; gh=[objectives[guide_index(objectives,scalar_weights)].as_array().copy()]; ah=[len(archive.entries)]
    for t in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); g=int(np.argmin(vals)); bh.append(float(vals[g])); pheromone*=(1-evaporation); pheromone+=reinforcement*assignments[g]; pheromone=np.clip(pheromone,1e-6,None)
        assignments=[]; objectives=[]
        for i in range(cfg.population_size):
            pop[i]=construct(); X,o=evaluate_scores(problem,simulator_inputs,pop[i]); assignments.append(X); objectives.append(o); archive.update(pop[i].reshape(-1),X,o)
        gh.append(objectives[guide_index(objectives,scalar_weights)].as_array().copy())
        ah.append(len(archive.entries))
    return _final_result("ACO",pop,assignments,objectives,scalar_weights,archive,bh,gh,ah,seed)
