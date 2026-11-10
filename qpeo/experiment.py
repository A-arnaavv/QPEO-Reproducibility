from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence, Any

import numpy as np

from .baselines import (
    CommonBaselineConfig, run_abc, run_aco, run_de, run_fa, run_fpa, run_ga,
    run_gwo, run_hs, run_nsga2, run_pso, run_qiea, run_woa,
)
from .config import ManuscriptQPEOConfig
from .models import ProblemInstance
from .optimizer import QPEOInputs, ReconstructionQPEOSettings, run_qpeo
from .simulator import SimulatorInputs, clear_simulation_cache, simulation_cache_stats


@dataclass(frozen=True)
class RunRecord:
    algorithm: str
    seed: int
    runtime_seconds: float
    guide_objectives: np.ndarray
    archive_size: int
    result: Any
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass(frozen=True)
class ExperimentSuite:
    records: tuple[RunRecord, ...]
    def by_algorithm(self, name: str) -> tuple[RunRecord, ...]:
        return tuple(r for r in self.records if r.algorithm == name)
    @property
    def algorithms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.algorithm for r in self.records))


def _run_suite(problem, qpeo_inputs, qpeo_settings, scalar_weights, *, seeds, population_size, max_iterations, names):
    simulator_inputs: SimulatorInputs = qpeo_inputs.simulator_inputs
    records=[]; bcfg=CommonBaselineConfig(population_size,max_iterations,qpeo_inputs.carbon_replay); qcfg=ManuscriptQPEOConfig(population_size=population_size,max_iterations=max_iterations)
    runners={
        "QPEO": lambda s: run_qpeo(problem,qpeo_inputs,qpeo_settings,config=qcfg,seed=s),
        "PSO": lambda s: run_pso(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "DE": lambda s: run_de(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "GWO": lambda s: run_gwo(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "WOA": lambda s: run_woa(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "ABC": lambda s: run_abc(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "GA": lambda s: run_ga(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "HS": lambda s: run_hs(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "FA": lambda s: run_fa(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "ACO": lambda s: run_aco(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "FPA": lambda s: run_fpa(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "QIEA": lambda s: run_qiea(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
        "NSGA-II": lambda s: run_nsga2(problem,simulator_inputs,scalar_weights,config=bcfg,seed=s),
    }
    for seed in seeds:
        for name in names:
            # A cold cache per algorithm prevents cross-method runtime advantages.
            clear_simulation_cache()
            t0=perf_counter(); result=runners[name](int(seed)); elapsed=perf_counter()-t0
            cache_stats=simulation_cache_stats()
            records.append(RunRecord(
                name,int(seed),float(elapsed),result.guide_objectives.as_array().copy(),
                len(result.archive.entries),result,
                cache_hits=cache_stats["hits"],cache_misses=cache_stats["misses"],
            ))
    return ExperimentSuite(tuple(records))


def run_reviewer_critical_suite(problem:ProblemInstance,qpeo_inputs:QPEOInputs,qpeo_settings:ReconstructionQPEOSettings,scalar_weights:Sequence[float],*,seeds:Sequence[int],population_size:int=50,max_iterations:int=500)->ExperimentSuite:
    return _run_suite(problem,qpeo_inputs,qpeo_settings,scalar_weights,seeds=seeds,population_size=population_size,max_iterations=max_iterations,names=("QPEO","FPA","QIEA","NSGA-II"))


def run_full_suite(problem:ProblemInstance,qpeo_inputs:QPEOInputs,qpeo_settings:ReconstructionQPEOSettings,scalar_weights:Sequence[float],*,seeds:Sequence[int],population_size:int=50,max_iterations:int=500)->ExperimentSuite:
    """Run all 13 methods with matched P, T, problem instances and seed list.\n\n    The algorithms may make different numbers of simulation/evaluation calls per\n    iteration. The experimental report must distinguish iteration matching from\n    evaluation-budget matching and report per-method simulator calls and runtimes.\n    """
    names=("QPEO","PSO","DE","GWO","WOA","ABC","GA","HS","FA","ACO","FPA","QIEA","NSGA-II")
    return _run_suite(problem,qpeo_inputs,qpeo_settings,scalar_weights,seeds=seeds,population_size=population_size,max_iterations=max_iterations,names=names)
