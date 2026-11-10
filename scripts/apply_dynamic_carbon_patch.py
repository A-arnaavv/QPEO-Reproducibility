from pathlib import Path

def must_replace(path, old, new, count=None):
    p=Path(path)
    s=p.read_text()
    if old not in s:
        raise RuntimeError(f"Expected pattern not found in {path}: {old[:100]!r}")
    p.write_text(s.replace(old,new) if count is None else s.replace(old,new,count))

# ---- baselines.py: dynamic carbon applied symmetrically to all 12 baselines ----
p=Path("qpeo/baselines.py")
s=p.read_text()
s=s.replace("from .archive import ParetoArchive\n", "from .archive import ParetoArchive\nfrom .carbon import CarbonReplay, with_carbon_intensities\n")
s=s.replace("""class CommonBaselineConfig:
    population_size: int = 50
    max_iterations: int = 500
""","""class CommonBaselineConfig:
    population_size: int = 50
    max_iterations: int = 500
    carbon_replay: CarbonReplay | None = None
""")
anchor="""def _validate_weights(weights: Sequence[float]) -> np.ndarray:
"""
helper="""def _problem_for_iteration(base_problem: ProblemInstance, cfg: CommonBaselineConfig, t: int) -> ProblemInstance:
    replay = cfg.carbon_replay
    if replay is None:
        return base_problem
    expected = tuple(r.name for r in base_problem.resources)
    if replay.slot_names != expected:
        raise ValueError("carbon replay slot order must match problem resource names")
    return with_carbon_intensities(base_problem, replay.at_iteration(t, cfg.max_iterations))


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


"""
if helper not in s:
    s=s.replace(anchor,helper+anchor)

patterns=[
("""    cfg = config or CommonBaselineConfig()
    cfg.validate(); _validate_weights(scalar_weights)
""","""    cfg = config or CommonBaselineConfig()
    cfg.validate(); _validate_weights(scalar_weights)
    base_problem = problem
    problem = _problem_for_iteration(base_problem, cfg, 0)
"""),
("""    cfg = config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
""","""    cfg = config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem = problem; problem = _problem_for_iteration(base_problem, cfg, 0)
"""),
("""    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
""","""    cfg=config or CommonBaselineConfig(); cfg.validate(); _validate_weights(scalar_weights)
    base_problem=problem; problem=_problem_for_iteration(base_problem,cfg,0)
""")]
for old,new in patterns:
    s=s.replace(old,new)

s=s.replace("""    for _ in range(cfg.max_iterations):
        values = scalar_values(objectives, scalar_weights)
""","""    for t in range(1, cfg.max_iterations + 1):
        refreshed = _dynamic_refresh(base_problem, cfg, t, simulator_inputs, pop)
        if refreshed is not None:
            problem, assignments, objectives, archive = refreshed
        values = scalar_values(objectives, scalar_weights)
""",1)
s=s.replace("""    for _ in range(cfg.max_iterations):
        F = objective_matrix(objectives)
""","""    for t in range(1, cfg.max_iterations + 1):
        refreshed = _dynamic_refresh(base_problem, cfg, t, simulator_inputs, pop)
        if refreshed is not None:
            problem, assignments, objectives, archive = refreshed
        F = objective_matrix(objectives)
""",1)
s=s.replace("""    for _ in range(cfg.max_iterations):
        vals = scalar_values(objectives, scalar_weights)
""","""    for t in range(1, cfg.max_iterations + 1):
        refreshed = _dynamic_refresh(base_problem, cfg, t, simulator_inputs, theta, lambda z: np.sin(z) ** 2)
        if refreshed is not None:
            problem, assignments, objectives, archive = refreshed
        vals = scalar_values(objectives, scalar_weights)
""",1)
s=s.replace("""    for _ in range(cfg.max_iterations):
        pvals=scalar_values(pbest_obj,scalar_weights); g=int(np.argmin(pvals)); gbest=pbest[g].copy(); bh.append(float(pvals[g]))
""","""    for t in range(1, cfg.max_iterations + 1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
            _,pbest_obj=_evaluate_population(problem,simulator_inputs,pbest)
        pvals=scalar_values(pbest_obj,scalar_weights); g=int(np.argmin(pvals)); gbest=pbest[g].copy(); bh.append(float(pvals[g]))
""",1)
s=s.replace("""    for _ in range(cfg.max_iterations):
        bh.append(float(np.min(scalar_values(objectives,scalar_weights))))
""","""    for t in range(1, cfg.max_iterations + 1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        bh.append(float(np.min(scalar_values(objectives,scalar_weights))))
""",1)
s=s.replace("""    for t in range(cfg.max_iterations):
        vals=scalar_values(objectives,scalar_weights); order=np.argsort(vals); alpha,beta,delta=[pop[k].copy() for k in order[:3]]; bh.append(float(vals[order[0]]))
        a=2.0*(1.0-t/max(cfg.max_iterations-1,1)); newpop=[]
""","""    for step in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,step,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); order=np.argsort(vals); alpha,beta,delta=[pop[k].copy() for k in order[:3]]; bh.append(float(vals[order[0]]))
        t=step-1; a=2.0*(1.0-t/max(cfg.max_iterations-1,1)); newpop=[]
""",1)
s=s.replace("""    for t in range(cfg.max_iterations):
        vals=scalar_values(objectives,scalar_weights); leader=pop[int(np.argmin(vals))].copy(); bh.append(float(np.min(vals))); a=2-2*t/max(cfg.max_iterations-1,1)
""","""    for step in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,step,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); leader=pop[int(np.argmin(vals))].copy(); bh.append(float(np.min(vals))); t=step-1; a=2-2*t/max(cfg.max_iterations-1,1)
""",1)
s=s.replace("""    for _ in range(cfg.max_iterations):
        bh.append(float(np.min(scalar_values(objectives,scalar_weights))))
        # employed bees
""","""    for t in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        bh.append(float(np.min(scalar_values(objectives,scalar_weights))))
        # employed bees
""",1)
s=s.replace("""    for _ in range(cfg.max_iterations):
        vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); children=[]
""","""    for t in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); children=[]
""",1)
s=s.replace("""    for _ in range(cfg.max_iterations):
        vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); new=np.empty((M,N))
""","""    for t in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); new=np.empty((M,N))
""",1)
s=s.replace("""    for t in range(cfg.max_iterations):
        vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); order=np.argsort(vals); newpop=pop.copy(); alpha_t=alpha*(.97**t)
""","""    for step in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,step,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        t=step-1; vals=scalar_values(objectives,scalar_weights); bh.append(float(np.min(vals))); order=np.argsort(vals); newpop=pop.copy(); alpha_t=alpha*(.97**t)
""",1)
s=s.replace("""    for _ in range(cfg.max_iterations):
        vals=scalar_values(objectives,scalar_weights); g=int(np.argmin(vals)); bh.append(float(vals[g])); pheromone*=(1-evaporation); pheromone+=reinforcement*assignments[g]; pheromone=np.clip(pheromone,1e-6,None)
""","""    for t in range(1,cfg.max_iterations+1):
        refreshed=_dynamic_refresh(base_problem,cfg,t,simulator_inputs,pop)
        if refreshed is not None:
            problem,assignments,objectives,archive=refreshed
        vals=scalar_values(objectives,scalar_weights); g=int(np.argmin(vals)); bh.append(float(vals[g])); pheromone*=(1-evaporation); pheromone+=reinforcement*assignments[g]; pheromone=np.clip(pheromone,1e-6,None)
""",1)
p.write_text(s)

# ---- optimizer.py: QPEO carbon refresh, Omega refresh, current-state archive ----
p=Path("qpeo/optimizer.py")
s=p.read_text()
s=s.replace("from .collapse import feasible_caqc_collapse\n","from .collapse import feasible_caqc_collapse\nfrom .carbon import CarbonReplay, with_carbon_intensities\n")
s=s.replace("""class QPEOInputs:
    simulator_inputs: SimulatorInputs
    performance_affinity: np.ndarray
    thermal_headroom: np.ndarray
    dependency_matrix: np.ndarray
""","""class QPEOInputs:
    simulator_inputs: SimulatorInputs
    performance_affinity: np.ndarray
    thermal_headroom: np.ndarray
    dependency_matrix: np.ndarray
    carbon_replay: CarbonReplay | None = None
""")
s=s.replace("""        if np.asarray(self.dependency_matrix).shape != (M, M):
            raise ValueError("dependency_matrix must have shape (M,M).")
""","""        if np.asarray(self.dependency_matrix).shape != (M, M):
            raise ValueError("dependency_matrix must have shape (M,M).")
        if self.carbon_replay is not None:
            self.carbon_replay.validate()
            if self.carbon_replay.resources != N:
                raise ValueError("carbon replay resource count must match problem resources")
""")
anchor="def run_qpeo(\n"
helper="""def _omega_for_problem(problem: ProblemInstance, inputs: QPEOInputs, settings: ReconstructionQPEOSettings) -> np.ndarray:
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
    return with_carbon_intensities(base_problem, replay.at_iteration(t, max_iterations))


"""
if helper not in s:
    s=s.replace(anchor,helper+anchor)
s=s.replace("""    if cfg.population_size < 3:
        raise ValueError("population_size must be at least 3 for local QPSO sampling.")

    rng = np.random.default_rng(seed)
""","""    if cfg.population_size < 3:
        raise ValueError("population_size must be at least 3 for local QPSO sampling.")

    base_problem = problem
    problem = _qpeo_problem_for_iteration(base_problem, inputs, 0, cfg.max_iterations)
    rng = np.random.default_rng(seed)
""")
old="""    ci = np.asarray([r.carbon_intensity for r in problem.resources], dtype=float)
    omega = ecological_compatibility(
        np.asarray(inputs.performance_affinity, dtype=float),
        np.asarray(inputs.thermal_headroom, dtype=float),
        ci,
        alpha=settings.alpha,
        beta=settings.beta,
        gamma=settings.gamma,
    )
"""
s=s.replace(old,"    omega = _omega_for_problem(problem, inputs, settings)\n")
old="""        kappa = kappa_schedule(
            t,
            kappa_min=cfg.kappa_min,
            kappa_max=cfg.kappa_max,
            tau_carbon=cfg.tau_carbon,
        )
        theta_best = population[guide].copy()
"""
new="""        kappa = kappa_schedule(
            t,
            kappa_min=cfg.kappa_min,
            kappa_max=cfg.kappa_max,
            tau_carbon=cfg.tau_carbon,
        )
        if inputs.carbon_replay is not None:
            problem = _qpeo_problem_for_iteration(base_problem, inputs, t, cfg.max_iterations)
            omega = _omega_for_problem(problem, inputs, settings)
            assignments = []
            objectives = []
            archive = ParetoArchive()
            for n in range(cfg.population_size):
                X_now, obj_now = _evaluate_theta(
                    problem, population[n], omega, inputs.simulator_inputs,
                    kappa=kappa, ci_ref=cfg.ci_ref,
                )
                assignments.append(X_now)
                objectives.append(obj_now)
                archive.update(population[n], X_now, obj_now)
            guide = _guide_index(objectives, settings.scalar_weights)
        theta_best = population[guide].copy()
"""
if old not in s:
    raise RuntimeError("QPEO loop pattern not found")
s=s.replace(old,new)
p.write_text(s)

# ---- experiment.py: give every baseline the same replay ----
p=Path("qpeo/experiment.py")
s=p.read_text()
old="records=[]; bcfg=CommonBaselineConfig(population_size,max_iterations); qcfg=ManuscriptQPEOConfig(population_size=population_size,max_iterations=max_iterations)"
new="records=[]; bcfg=CommonBaselineConfig(population_size,max_iterations,qpeo_inputs.carbon_replay); qcfg=ManuscriptQPEOConfig(population_size=population_size,max_iterations=max_iterations)"
if old not in s:
    raise RuntimeError("Experiment config pattern not found")
p.write_text(s.replace(old,new))
