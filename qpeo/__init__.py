"""QPEO reproducibility rebuild."""
from .models import CarbonClass, ConstraintReport, EvaluationState, ObjectiveVector, ProblemInstance, Resource, Workload
from .evaluator import check_constraints, dominates, evaluate_objectives, normalize_population_objectives, scalar_fitness
from .config import ManuscriptQPEOConfig
from .operators import (
    assignment_probabilities, biodiversity_restore, candidate_density_proxy,
    caqc_scores, caqc_unconstrained_collapse, ebws_update, ecological_compatibility,
    global_qpso_update, kappa_schedule, local_qpso_update, mantegna_levy_step,
    qbit_components, validate_angles, von_neumann_entropy,
)
from .simulator import PowerModel, SimulationResult, SimulatorInputs, simulate_fixed_assignment
from .collapse import NoFeasibleAssignment, feasible_caqc_collapse
from .archive import ParetoArchive, ParetoEntry
from .optimizer import QPEOInputs, QPEOResult, ReconstructionQPEOSettings, run_qpeo

__all__ = [
    "CarbonClass", "ConstraintReport", "EvaluationState", "ObjectiveVector", "ProblemInstance", "Resource", "Workload",
    "check_constraints", "dominates", "evaluate_objectives", "normalize_population_objectives", "scalar_fitness",
    "ManuscriptQPEOConfig", "assignment_probabilities", "biodiversity_restore", "candidate_density_proxy",
    "caqc_scores", "caqc_unconstrained_collapse", "ebws_update", "ecological_compatibility", "global_qpso_update",
    "kappa_schedule", "local_qpso_update", "mantegna_levy_step", "qbit_components", "validate_angles",
    "von_neumann_entropy", "PowerModel", "SimulationResult", "SimulatorInputs", "simulate_fixed_assignment",
    "NoFeasibleAssignment", "feasible_caqc_collapse", "ParetoArchive", "ParetoEntry", "QPEOInputs", "QPEOResult",
    "ReconstructionQPEOSettings", "run_qpeo",
]

from .traces import (
    RawTraceTask, TraceMaterializationPolicy, load_alibaba_gpu_v2023_pods_csv,
    load_azure_functions_2021_csv, load_canonical_trace_csv,
    materialize_workloads, service_time_matrix_from_observed_runtime,
)
__all__ += ["RawTraceTask", "TraceMaterializationPolicy", "load_alibaba_gpu_v2023_pods_csv", "load_azure_functions_2021_csv", "load_canonical_trace_csv", "materialize_workloads", "service_time_matrix_from_observed_runtime"]

from .decoder import feasible_score_decode
from .baseline_common import BaselineResult
from .baselines import CommonBaselineConfig, run_fpa, run_nsga2, run_qiea
from .experiment import RunRecord, ExperimentSuite, run_reviewer_critical_suite
from .baselines import run_pso, run_de, run_gwo, run_woa, run_abc, run_ga, run_hs, run_fa, run_aco
from .experiment import run_full_suite
from .statistics import (
    mean_ci, paired_cohens_dz, wilcoxon_against_reference, bonferroni_alpha,
    friedman_test, nemenyi_posthoc, iterations_to_fraction_of_final,
    approximate_hypervolume,
)

from .protocol import PrimaryProtocol, load_protocol, pooled_normalization_transform, normalize_with_transform
__all__ += ["PrimaryProtocol", "load_protocol", "pooled_normalization_transform", "normalize_with_transform"]

from .resource_environment import (
    HardwareClassSpec, ResourceSlotSpec, ResourceEnvironment, load_resource_environment,
)
__all__ += ["HardwareClassSpec", "ResourceSlotSpec", "ResourceEnvironment", "load_resource_environment"]

from .carbon import CarbonReplay, load_carbon_replay_config, load_neso_regional_replay, with_carbon_intensities
__all__ += ["CarbonReplay", "load_carbon_replay_config", "load_neso_regional_replay", "with_carbon_intensities"]
