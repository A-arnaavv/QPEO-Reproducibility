from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .evaluator import dominates
from .models import ObjectiveVector


@dataclass(frozen=True)
class ParetoEntry:
    theta: np.ndarray
    assignment: np.ndarray
    objectives: ObjectiveVector


@dataclass
class ParetoArchive:
    entries: list[ParetoEntry] = field(default_factory=list)

    def update(self, theta: np.ndarray, assignment: np.ndarray, objectives: ObjectiveVector) -> bool:
        candidate = objectives.as_array()
        for entry in self.entries:
            current = entry.objectives.as_array()
            if np.allclose(candidate, current, rtol=1e-10, atol=1e-12):
                return False
            if dominates(current, candidate):
                return False

        self.entries = [
            e for e in self.entries
            if not dominates(candidate, e.objectives.as_array())
        ]
        self.entries.append(ParetoEntry(
            np.asarray(theta, dtype=float).copy(),
            np.asarray(assignment, dtype=int).copy(),
            objectives,
        ))
        return True

    def objective_matrix(self) -> np.ndarray:
        if not self.entries:
            return np.empty((0, 6), dtype=float)
        return np.vstack([e.objectives.as_array() for e in self.entries])
