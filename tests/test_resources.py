from pathlib import Path
import pytest
from qpeo.resources import ResourcePoolSpec, build_resource_pool, load_resource_config, fp32_normalized_speed_factors

def test_vendor_templates_and_primary_counts():
    path = Path(__file__).parents[1] / "configs" / "resource_pool.json"
    templates, raw = load_resource_config(path)
    assert templates["a5000"].memory_gib == 24
    assert templates["a5000"].max_power_w == 230
    assert templates["a6000"].memory_gib == 48
    assert templates["a6000"].max_power_w == 300
    assert templates["rtx4090"].memory_gib == 24
    assert templates["rtx4090"].max_power_w == 450
    assert templates["agx_orin_64"].memory_gib == 64
    assert templates["agx_orin_64"].max_power_w == 60
    assert sum(raw["primary_topology_reconstruction"]["counts"].values()) == 16

def test_pool_builds_with_explicit_reconstruction_values():
    path = Path(__file__).parents[1] / "configs" / "resource_pool.json"
    templates, raw = load_resource_config(path)
    spec = ResourcePoolSpec(
        counts=raw["primary_topology_reconstruction"]["counts"],
        idle_fraction_of_max=.2,
        cost_per_gpu_hour={"a5000":1.0,"a6000":1.2,"rtx4090":1.1,"agx_orin_64":.3},
        speed_factor={"a5000":1.0,"a6000":1.2,"rtx4090":1.5,"agx_orin_64":.35},
    )
    resources, power, speed, keys = build_resource_pool(
        templates, spec, carbon_intensity_by_layer={"cloud":300,"edge":180,"serverless":360}
    )
    assert len(resources) == 16
    assert power.idle_power_kw.shape == (16,)
    assert power.dynamic_power_kw.shape == (16,)
    assert speed.shape == (16,)
    a5000_i = keys.index("a5000")
    assert power.idle_power_kw[a5000_i] + power.dynamic_power_kw[a5000_i] == pytest.approx(.230)
    orin_i = keys.index("agx_orin_64")
    assert power.idle_power_kw[orin_i] + power.dynamic_power_kw[orin_i] == pytest.approx(.060)


def test_fp32_speed_factors_match_published_values():
    path = Path(__file__).parents[1] / "configs" / "resource_pool.json"
    templates, raw = load_resource_config(path)
    factors = fp32_normalized_speed_factors(templates)
    assert factors["a5000"] == pytest.approx(1.0)
    assert factors["a6000"] == pytest.approx(38.7/27.8)
    assert factors["rtx4090"] == pytest.approx(82.6/27.8)
    assert factors["agx_orin_64"] == pytest.approx(5.3/27.8)
    assert raw["primary_topology_reconstruction"]["idle_fraction_of_max"] == pytest.approx(.20)
