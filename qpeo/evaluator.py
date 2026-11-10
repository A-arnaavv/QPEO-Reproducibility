from __future__ import annotations
from typing import Sequence
import numpy as np
from .models import ConstraintReport, EvaluationState, ObjectiveVector, ProblemInstance

def _assignment_matrix(X,M,N):
    X=np.asarray(X)
    if X.shape!=(M,N): raise ValueError(f"X must have shape {(M,N)}, got {X.shape}")
    return X

def check_constraints(problem:ProblemInstance,X:np.ndarray,*,atol:float=1e-12)->ConstraintReport:
    M,N=len(problem.workloads),len(problem.resources)
    X=_assignment_matrix(X,M,N)
    binary_ok=bool(np.all(np.isclose(X,0,atol=atol)|np.isclose(X,1,atol=atol)))
    exactly_one_ok=bool(np.all(np.isclose(X.sum(axis=1),1,atol=atol)))
    gpu_d=np.asarray([w.gpu_demand for w in problem.workloads],float)
    mem_d=np.asarray([w.memory_demand for w in problem.workloads],float)
    gpu_cap=np.asarray([r.gpu_capacity for r in problem.resources],float)
    mem_cap=np.asarray([r.memory_capacity for r in problem.resources],float)
    # Temporal capacity is enforced by the simulator. At assignment level, each
    # workload must individually fit its selected resource.
    gpu_ok=bool(np.all(gpu_d[:,None]*X<=gpu_cap[None,:]+atol))
    mem_ok=bool(np.all(mem_d[:,None]*X<=mem_cap[None,:]+atol))
    carbon_ok=True; cviol=[]
    for i,w in enumerate(problem.workloads):
        thr=float(problem.carbon_thresholds[w.carbon_class])
        for j,r in enumerate(problem.resources):
            if r.carbon_intensity*float(X[i,j])>thr+atol:
                carbon_ok=False
                cviol.append(f"{w.name}->{r.name}")
    violations=[]
    if not binary_ok: violations.append("Assignment matrix is not binary.")
    if not exactly_one_ok: violations.append("Each workload must be assigned exactly once.")
    if not gpu_ok: violations.append("GPU-capacity constraint violated.")
    if not mem_ok: violations.append("Memory-capacity constraint violated.")
    if not carbon_ok: violations.append("Carbon-threshold constraint violated: "+", ".join(cviol))
    return ConstraintReport(not violations,binary_ok,exactly_one_ok,gpu_ok,mem_ok,carbon_ok,tuple(violations))

def evaluate_objectives(problem:ProblemInstance,X:np.ndarray,state:EvaluationState,*,require_feasible=True,zero_energy_carbon_value=0.0)->ObjectiveVector:
    M,N=len(problem.workloads),len(problem.resources)
    X=_assignment_matrix(X,M,N).astype(float)
    state.validate(M,N)
    if require_feasible:
        rep=check_constraints(problem,X)
        if not rep.feasible: raise ValueError("Infeasible schedule: "+" | ".join(rep.violations))
    arrivals=np.asarray([w.arrival_time for w in problem.workloads],float)
    completion=np.asarray(state.completion_times,float)
    horizon=float(completion.max()-arrivals.min())
    if horizon<=0: raise ValueError("Scheduling horizon must be > 0.")
    # Time-aware average GPU utilisation over the global scheduling horizon.
    # allocated_gpu_fraction_hours already integrates gpu_demand / gpu_capacity.
    allocated=np.asarray(state.allocated_gpu_fraction_hours,float)
    f1=-float(np.mean(allocated/horizon))
    energy=np.asarray(state.energy_kwh_by_resource,float)
    f2=float(energy.sum())
    ci=np.asarray([r.carbon_intensity for r in problem.resources],float)
    f3=float(np.dot(energy,ci)/f2) if f2>0 else float(zero_energy_carbon_value)
    f4=horizon
    deadlines=np.asarray([w.deadline for w in problem.workloads],float)
    sla_mask=np.isfinite(deadlines)
    f5=float(100*np.mean(completion[sla_mask]>deadlines[sla_mask])) if np.any(sla_mask) else 0.0
    costs=np.asarray([r.cost_per_gpu_hour for r in problem.resources],float)
    occupied=np.asarray(state.occupied_gpu_hours,float)
    f6=float(np.dot(costs,occupied)/problem.reference_cost)
    return ObjectiveVector(f1,f2,f3,f4,f5,f6)

def normalize_population_objectives(objective_matrix:np.ndarray,*,epsilon=1e-12)->np.ndarray:
    F=np.asarray(objective_matrix,float)
    if F.ndim!=2 or F.shape[1]!=6: raise ValueError("objective_matrix must have shape (P,6)")
    mn,mx=F.min(0),F.max(0)
    return (F-mn)/(mx-mn+epsilon)

def scalar_fitness(normalized_objectives:np.ndarray,weights:Sequence[float])->np.ndarray:
    F=np.asarray(normalized_objectives,float); w=np.asarray(weights,float)
    if w.shape!=(6,) or np.any(w<0) or not np.isclose(w.sum(),1): raise ValueError("weights must be 6 non-negative values summing to 1")
    if F.ndim==1 and F.shape!=(6,): raise ValueError("single vector must have shape (6,)")
    if F.ndim==2 and F.shape[1]!=6: raise ValueError("population must have 6 objectives")
    return F@w

def dominates(a,b)->bool:
    aa=a.as_array() if isinstance(a,ObjectiveVector) else np.asarray(a,float)
    bb=b.as_array() if isinstance(b,ObjectiveVector) else np.asarray(b,float)
    return bool(np.all(aa<=bb) and np.any(aa<bb))
