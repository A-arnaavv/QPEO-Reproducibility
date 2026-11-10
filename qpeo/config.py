from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ManuscriptQPEOConfig:
    population_size: int = 50
    max_iterations: int = 500
    levy_lambda: float = 1.5
    global_switch_probability: float = 0.8
    kappa_min: float = 0.2
    kappa_max: float = 2.0
    tau_carbon: float = 150.0
    stagnation_iterations: int = 20
    biodiversity_amplitude: float = 0.3
    ci_ref: float = 475.0

    @property
    def entropy_threshold(self) -> float:
        return math.log2(self.population_size) / 2.0
