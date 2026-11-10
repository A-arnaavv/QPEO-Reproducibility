from pathlib import Path
import numpy as np
import pytest

from qpeo.protocol import load_protocol, pooled_normalization_transform, normalize_with_transform


def test_pre_registered_protocol_loads():
    path = Path(__file__).parents[1] / "configs" / "reconstruction_protocol.json"
    p, raw = load_protocol(path)
    assert p.population_size == 50
    assert p.max_iterations == 500
    assert p.runs_per_algorithm == 30
    assert len(p.seeds) == 30
    assert len(set(p.seeds)) == 30
    assert sum(p.scalar_guide_weights) == pytest.approx(1.0)
    assert p.alpha + p.beta + p.gamma == pytest.approx(1.0)
    assert raw["status"] == "pre-results"


def test_pooled_hv_normalization_rule():
    A = np.array([[1, 10, 2, 8, 3, 5], [2, 8, 4, 7, 4, 4]], float)
    B = np.array([[0, 12, 3, 6, 2, 7]], float)
    lo, hi = pooled_normalization_transform([A, B])
    assert np.allclose(lo, [0, 8, 2, 6, 2, 4])
    assert np.allclose(hi, [2, 12, 4, 8, 4, 7])
    norm = normalize_with_transform(A, lo, hi)
    assert np.all(norm >= 0)
    assert np.all(norm <= 1)
