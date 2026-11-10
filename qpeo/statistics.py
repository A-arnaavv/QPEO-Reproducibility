from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.stats import qmc


OBJECTIVE_NAMES = (
    "gpu_util_min_form",
    "energy_kwh",
    "carbon_intensity",
    "makespan",
    "sla_violation_percent",
    "normalized_cost",
)


@dataclass(frozen=True)
class MeanCI:
    mean: float
    sd: float
    sem: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class PairwiseTest:
    baseline: str
    statistic: float
    p_value: float
    effect_dz: float


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    average_ranks: Mapping[str, float]


@dataclass(frozen=True)
class NemenyiPair:
    algorithm_a: str
    algorithm_b: str
    rank_difference: float
    q_statistic: float
    p_value: float


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> MeanCI:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("At least two scalar observations are required.")
    mean = float(np.mean(x)); sd = float(np.std(x, ddof=1)); sem = float(stats.sem(x))
    q = float(stats.t.ppf((1 + confidence) / 2, df=x.size - 1))
    half = q * sem
    return MeanCI(mean, sd, sem, mean - half, mean + half, int(x.size))


def paired_cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    aa=np.asarray(a,float); bb=np.asarray(b,float)
    if aa.shape != bb.shape or aa.ndim != 1 or aa.size < 2:
        raise ValueError("Paired samples must be same-shape 1-D arrays with n>=2.")
    d=aa-bb; sd=float(np.std(d,ddof=1))
    if sd <= 1e-15:
        if abs(float(np.mean(d))) <= 1e-15: return 0.0
        return float(np.sign(np.mean(d))*np.inf)
    return float(np.mean(d)/sd)


def wilcoxon_against_reference(
    reference: Sequence[float],
    baselines: Mapping[str, Sequence[float]],
    *,
    zero_method: str = "wilcox",
) -> tuple[PairwiseTest, ...]:
    ref=np.asarray(reference,float)
    out=[]
    for name, vals in baselines.items():
        x=np.asarray(vals,float)
        if x.shape != ref.shape:
            raise ValueError(f"{name}: paired sample shape mismatch.")
        try:
            res=stats.wilcoxon(ref,x,zero_method=zero_method,alternative="two-sided",method="auto")
            stat=float(res.statistic); p=float(res.pvalue)
        except ValueError:
            stat=0.0; p=1.0
        out.append(PairwiseTest(name,stat,p,paired_cohens_dz(ref,x)))
    return tuple(out)


def bonferroni_alpha(familywise_alpha: float, n_comparisons: int) -> float:
    if not (0 < familywise_alpha < 1) or n_comparisons < 1:
        raise ValueError("Require 0<alpha<1 and n_comparisons>=1.")
    return float(familywise_alpha/n_comparisons)


def _rank_rows(values: np.ndarray) -> np.ndarray:
    return np.vstack([stats.rankdata(row, method="average") for row in values])


def friedman_test(samples: Mapping[str, Sequence[float]]) -> FriedmanResult:
    names=list(samples)
    if len(names)<3: raise ValueError("Friedman test requires >=3 algorithms.")
    arrays=[np.asarray(samples[n],float) for n in names]
    shape=arrays[0].shape
    if len(shape)!=1 or shape[0]<2 or any(a.shape!=shape for a in arrays):
        raise ValueError("All algorithm samples must be paired 1-D arrays with equal length >=2.")
    res=stats.friedmanchisquare(*arrays)
    matrix=np.column_stack(arrays)
    avg=np.mean(_rank_rows(matrix),axis=0)
    return FriedmanResult(float(res.statistic),float(res.pvalue),{n:float(r) for n,r in zip(names,avg)})


def nemenyi_posthoc(samples: Mapping[str, Sequence[float]]) -> tuple[NemenyiPair, ...]:
    fr=friedman_test(samples); names=list(samples); k=len(names); N=len(next(iter(samples.values())))
    se=math_sqrt(k*(k+1)/(6.0*N))
    pairs=[]
    for a,b in combinations(names,2):
        diff=abs(fr.average_ranks[a]-fr.average_ranks[b]); q=diff/se if se>0 else np.inf
        p=float(stats.studentized_range.sf(q*np.sqrt(2.0),k,np.inf))
        pairs.append(NemenyiPair(a,b,float(diff),float(q),p))
    return tuple(pairs)


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x))


def iterations_to_fraction_of_final(
    history: Sequence[float],
    *,
    fraction: float = 0.90,
    minimize: bool = True,
) -> int:
    h=np.asarray(history,float)
    if h.ndim!=1 or h.size<1 or not (0<fraction<=1):
        raise ValueError("Invalid history or fraction.")
    start=float(h[0]); final=float(h[-1])
    target=start+fraction*(final-start)
    if minimize:
        hits=np.where(h<=target+1e-15)[0]
    else:
        hits=np.where(h>=target-1e-15)[0]
    return int(hits[0]+1) if hits.size else int(h.size)


def _nondominated(points: np.ndarray) -> np.ndarray:
    P=np.asarray(points,float); keep=np.ones(P.shape[0],bool)
    for i in range(P.shape[0]):
        if not keep[i]: continue
        for j in range(P.shape[0]):
            if i==j: continue
            if np.all(P[j]<=P[i]) and np.any(P[j]<P[i]):
                keep[i]=False; break
    return P[keep]


def approximate_hypervolume(
    points: np.ndarray,
    reference_point: Sequence[float],
    *,
    ideal_point: Sequence[float] | None = None,
    samples_power: int = 15,
    scramble_seed: int = 0,
) -> float:
    P=np.asarray(points,float); ref=np.asarray(reference_point,float)
    if P.ndim!=2 or P.shape[0]<1 or ref.shape!=(P.shape[1],):
        raise ValueError("points must be (n,d) and reference_point shape (d,).")
    ideal=np.min(P,axis=0) if ideal_point is None else np.asarray(ideal_point,float)
    if ideal.shape!=ref.shape or np.any(ref<=ideal):
        raise ValueError("reference_point must be strictly worse (larger) than ideal_point in every minimization dimension.")
    P=_nondominated(P)
    P=np.minimum(np.maximum(P,ideal),ref)
    sampler=qmc.Sobol(d=P.shape[1],scramble=True,seed=scramble_seed)
    U=sampler.random_base2(samples_power)
    X=ideal+U*(ref-ideal)
    dominated=np.zeros(X.shape[0],dtype=bool)
    for p in P:
        dominated |= np.all(X>=p,axis=1)
    box=float(np.prod(ref-ideal))
    return float(np.mean(dominated)*box)
