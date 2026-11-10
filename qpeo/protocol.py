from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PrimaryProtocol:
    population_size: int
    max_iterations: int
    runs_per_algorithm: int
    scalar_guide_weights: tuple[float, ...]
    alpha: float
    beta: float
    gamma: float
    dependency_threshold: float
    dependency_strength: float
    billing_mode: str
    seeds: tuple[int, ...]

    def validate(self) -> None:
        w = np.asarray(self.scalar_guide_weights, dtype=float)
        if w.shape != (6,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
            raise ValueError("scalar_guide_weights must be six non-negative values summing to one")
        eco = np.asarray([self.alpha, self.beta, self.gamma], dtype=float)
        if np.any(eco < 0) or not np.isclose(eco.sum(), 1.0):
            raise ValueError("alpha/beta/gamma must be non-negative and sum to one")
        if len(self.seeds) != self.runs_per_algorithm or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed list must contain exactly runs_per_algorithm unique seeds")
        if self.population_size < 4 or self.max_iterations < 1:
            raise ValueError("invalid computational budget")
        if not (0 <= self.dependency_threshold <= 1 and 0 <= self.dependency_strength <= 1):
            raise ValueError("dependency settings must be in [0,1]")


def load_protocol(path: str | Path) -> tuple[PrimaryProtocol, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    p = data["primary"]
    e = p["ecological_compatibility"]
    primary = PrimaryProtocol(
        population_size=int(p["population_size"]),
        max_iterations=int(p["max_iterations"]),
        runs_per_algorithm=int(p["runs_per_algorithm"]),
        scalar_guide_weights=tuple(float(x) for x in p["scalar_guide_weights"]),
        alpha=float(e["alpha_performance"]),
        beta=float(e["beta_thermal"]),
        gamma=float(e["gamma_carbon"]),
        dependency_threshold=float(p["dependency_threshold"]),
        dependency_strength=float(p["dependency_strength"]),
        billing_mode=str(p["billing_mode"]),
        seeds=tuple(int(x) for x in p["seeds"]),
    )
    primary.validate()
    return primary, data


def pooled_normalization_transform(objective_matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mats = [np.asarray(m, dtype=float) for m in objective_matrices]
    if not mats or any(m.ndim != 2 or m.shape[1] != 6 for m in mats):
        raise ValueError("Each objective matrix must have shape (n,6)")
    pooled = np.vstack(mats)
    if np.any(~np.isfinite(pooled)):
        raise ValueError("Objective values must be finite")
    return pooled.min(axis=0), pooled.max(axis=0)


def normalize_with_transform(F: np.ndarray, lo: np.ndarray, hi: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if F.ndim != 2 or F.shape[1] != 6 or lo.shape != (6,) or hi.shape != (6,):
        raise ValueError("Expected F=(n,6), lo=(6,), hi=(6,)")
    return (F - lo) / (hi - lo + epsilon)
