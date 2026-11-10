from __future__ import annotations
import math
from typing import Iterable
import numpy as np

def validate_angles(theta: np.ndarray) -> np.ndarray:
    theta=np.asarray(theta,dtype=float)
    if np.any(theta<0) or np.any(theta>np.pi/2):
        raise ValueError("QES angle values must lie in [0, pi/2].")
    return theta

def qbit_components(theta: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Eq. (8): |q> = cos(theta)|0> + sin(theta)|1>."""
    t=validate_angles(theta)
    return np.cos(t),np.sin(t)

def ecological_compatibility(
    performance_affinity: np.ndarray,
    thermal_headroom: np.ndarray,
    carbon_intensity: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    ci_max: float | None=None,
) -> np.ndarray:
    """Eq. (10). Undocumented alpha/beta/gamma are explicit required inputs."""
    if min(alpha,beta,gamma)<0 or not np.isclose(alpha+beta+gamma,1.0):
        raise ValueError("alpha, beta, gamma must be non-negative and sum to 1.")
    perf=np.asarray(performance_affinity,float)
    therm=np.asarray(thermal_headroom,float)
    ci=np.asarray(carbon_intensity,float)
    if perf.ndim!=2:
        raise ValueError("performance_affinity must have shape (M,N).")
    M,N=perf.shape
    if therm.shape!=(N,) or ci.shape!=(N,):
        raise ValueError("thermal_headroom and carbon_intensity must have shape (N,).")
    cmax=float(ci_max if ci_max is not None else np.max(ci))
    if cmax<=0: raise ValueError("ci_max must be > 0.")
    return alpha*perf + beta*therm[None,:] + gamma*(1.0-ci[None,:]/cmax)

def assignment_probabilities(theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Eq. (9): P_n(i,j)=sin^2(theta_n(i*N+j))*Omega(i,j)."""
    t=validate_angles(theta).reshape(-1)
    om=np.asarray(omega,float)
    M,N=om.shape
    if t.size!=M*N:
        raise ValueError(f"theta has {t.size} elements but omega implies D={M*N}.")
    return np.sin(t.reshape(M,N))**2 * om

def candidate_density_proxy(theta: np.ndarray) -> np.ndarray:
    """
    Reconstruction convention R1, not fully specified by the manuscript.

    psi is the normalized vector proportional to sin(theta);
    rho = |psi><psi|. The population mean may then be mixed.
    """
    t=validate_angles(theta).reshape(-1)
    amp=np.sin(t)
    norm=np.linalg.norm(amp)
    if norm<=1e-15:
        amp=np.ones_like(amp)/math.sqrt(amp.size)
    else:
        amp=amp/norm
    return np.outer(amp,amp)

def von_neumann_entropy(rhos: Iterable[np.ndarray]) -> float:
    """Eq. (20) applied to the mean population density matrix."""
    mats=[np.asarray(r,float) for r in rhos]
    if not mats: raise ValueError("At least one density matrix is required.")
    shape=mats[0].shape
    if len(shape)!=2 or shape[0]!=shape[1] or any(m.shape!=shape for m in mats):
        raise ValueError("All density matrices must be square and have the same shape.")
    mean=np.mean(mats,axis=0)
    mean=(mean+mean.T)/2
    tr=float(np.trace(mean))
    if tr<=0: raise ValueError("Mean density matrix must have positive trace.")
    mean=mean/tr
    vals=np.linalg.eigvalsh(mean)
    vals=np.clip(vals,0.0,1.0)
    vals=vals[vals>1e-15]
    return float(-np.sum(vals*np.log2(vals)))

def mantegna_levy_step(rng: np.random.Generator, lam: float=1.5) -> float:
    """Eqs. (14)-(15), scalar L(lambda) consistent with Eq. (13)."""
    if not (0<lam<2): raise ValueError("lam must lie in (0,2).")
    sigma_u=((math.gamma(1+lam)*math.sin(math.pi*lam/2))/
             (math.gamma((1+lam)/2)*lam*2**((lam-1)/2)))**(1/lam)
    u=rng.normal(0,sigma_u)
    v=rng.normal(0,1)
    return float(u/(abs(v)**(1/lam)+1e-15))

def global_qpso_update(theta: np.ndarray, theta_best: np.ndarray, rng: np.random.Generator, *, lam: float=1.5) -> np.ndarray:
    """Eq. (13)."""
    t=np.asarray(theta,float); b=np.asarray(theta_best,float)
    if t.shape!=b.shape: raise ValueError("theta and theta_best must have same shape.")
    L=mantegna_levy_step(rng,lam)
    return np.clip(t+L*(b-t),0,np.pi/2)

def local_qpso_update(theta: np.ndarray, theta_m: np.ndarray, theta_l: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Eq. (16)."""
    t=np.asarray(theta,float); m=np.asarray(theta_m,float); l=np.asarray(theta_l,float)
    if not (t.shape==m.shape==l.shape): raise ValueError("All angle vectors must have same shape.")
    eps=float(rng.uniform(0,1))
    return np.clip(t+eps*(m-l),0,np.pi/2)

def kappa_schedule(t: int, *, kappa_min: float=.2, kappa_max: float=2.0, tau_carbon: float=150.0) -> float:
    """Eq. (18)."""
    if t<0 or tau_carbon<=0: raise ValueError("t>=0 and tau_carbon>0 required.")
    return float(kappa_min+(kappa_max-kappa_min)*(1-math.exp(-t/tau_carbon)))

def caqc_scores(probabilities: np.ndarray, carbon_intensity: np.ndarray, *, kappa: float, ci_ref: float=475.0) -> np.ndarray:
    """Eq. (17) score before argmax/feasibility repair."""
    P=np.asarray(probabilities,float); ci=np.asarray(carbon_intensity,float)
    if P.ndim!=2 or ci.shape!=(P.shape[1],): raise ValueError("Shape mismatch.")
    if ci_ref<=0: raise ValueError("ci_ref must be > 0.")
    return P*np.exp(-kappa*ci[None,:]/ci_ref)

def caqc_unconstrained_collapse(probabilities: np.ndarray, carbon_intensity: np.ndarray, *, kappa: float, ci_ref: float=475.0) -> np.ndarray:
    scores=caqc_scores(probabilities,carbon_intensity,kappa=kappa,ci_ref=ci_ref)
    M,N=scores.shape
    X=np.zeros((M,N),dtype=int)
    X[np.arange(M),np.argmax(scores,axis=1)]=1
    return X

def ebws_update(
    theta: np.ndarray,
    dependency: np.ndarray,
    *,
    delta_dep: float,
    phi: float,
) -> np.ndarray:
    T=np.asarray(theta,float).copy()
    dep=np.asarray(dependency,float)
    M,N=T.shape
    if dep.shape!=(M,M): raise ValueError("dependency must have shape (M,M).")
    if not (0<=delta_dep<=1 and 0<=phi<=1): raise ValueError("delta_dep and phi must be in [0,1].")
    base=T.copy()
    active = np.where(dep > delta_dep, dep, 0.0)
    np.fill_diagonal(active, 0.0)
    incoming = active.T @ base
    strength = active.sum(axis=0)[:, None]
    T = base + phi * (incoming - strength * base)
    return np.clip(T,0,np.pi/2)

def biodiversity_restore(
    population: np.ndarray,
    theta_best: np.ndarray,
    stagnant_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    delta_bio: float=.3,
) -> np.ndarray:
    """Eq. (21), clipped back to the valid angle domain."""
    pop=np.asarray(population,float).copy()
    best=np.asarray(theta_best,float)
    mask=np.asarray(stagnant_mask,bool)
    if pop.ndim!=2 or best.shape!=(pop.shape[1],) or mask.shape!=(pop.shape[0],):
        raise ValueError("Population, best vector, or stagnant mask shape mismatch.")
    count=int(mask.sum())
    if count:
        perturb=rng.uniform(0,2*np.pi,size=(count,pop.shape[1]))-np.pi
        pop[mask]=best[None,:]+delta_bio*perturb
        pop[mask]=np.clip(pop[mask],0,np.pi/2)
    return pop


def population_entropy_from_angles(population: np.ndarray) -> float:
    """
    Exact low-rank von Neumann entropy for reconstruction convention R1.

    For rho_bar = (1/P) sum_n |psi_n><psi_n|, the non-zero eigenvalues
    equal those of the P x P Gram matrix G = Psi Psi^T / P.
    """
    pop = np.asarray(population, dtype=float)
    if pop.ndim != 2 or pop.shape[0] < 1 or pop.shape[1] < 1:
        raise ValueError("population must have shape (P,D).")
    validate_angles(pop)
    amp = np.sin(pop)
    norms = np.linalg.norm(amp, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-15
    if np.any(zero):
        amp[zero] = 1.0 / math.sqrt(pop.shape[1])
        norms[zero] = 1.0
    psi = amp / norms
    gram = (psi @ psi.T) / pop.shape[0]
    gram = (gram + gram.T) / 2.0
    vals = np.linalg.eigvalsh(gram)
    vals = np.clip(vals, 0.0, 1.0)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log2(vals)))
