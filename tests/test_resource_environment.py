from pathlib import Path
import numpy as np
import pytest
from qpeo.models import CarbonClass, Workload
from qpeo.resource_environment import load_resource_environment

def env():
    return load_resource_environment(Path(__file__).parents[1] / "configs" / "resource_environment_v1.json")

def test_locked_hardware_spec_values():
    e=env()
    assert e.hardware_classes["rtx_4090"].gpu_memory_gb == pytest.approx(24)
    assert e.hardware_classes["rtx_4090"].max_gpu_power_kw == pytest.approx(.450)
    assert e.hardware_classes["rtx_4090"].fp32_tflops == pytest.approx(82.6)
    assert e.hardware_classes["rtx_a6000"].gpu_memory_gb == pytest.approx(48)
    assert e.hardware_classes["rtx_a6000"].max_gpu_power_kw == pytest.approx(.300)
    assert e.hardware_classes["rtx_a6000"].fp32_tflops == pytest.approx(38.7)
    assert e.hardware_classes["rtx_a5000"].gpu_memory_gb == pytest.approx(24)
    assert e.hardware_classes["rtx_a5000"].max_gpu_power_kw == pytest.approx(.230)
    assert e.hardware_classes["rtx_a5000"].fp32_tflops == pytest.approx(27.8)
    assert e.hardware_classes["jetson_agx_orin_64gb"].gpu_memory_gb == pytest.approx(64)
    assert e.hardware_classes["jetson_agx_orin_64gb"].max_gpu_power_kw == pytest.approx(.060)
    assert e.hardware_classes["jetson_agx_orin_64gb"].fp32_tflops == pytest.approx(5.3)

def test_speed_factors_are_a6000_normalized():
    e=env(); speed=e.speed_factors(); names=[s.name for s in e.slots]
    assert speed[names.index("cloud-a6000-0")] == pytest.approx(1.0)
    assert speed[names.index("cloud-a5000-0")] == pytest.approx(27.8/38.7)
    assert speed[names.index("cloud-4090-0")] == pytest.approx(82.6/38.7)
    assert speed[names.index("edge-orin-0")] == pytest.approx(5.3/38.7)

def test_primary_power_model_hits_documented_max_at_full_utilization():
    e=env(); p=e.power_model()
    assert np.allclose(p.idle_power_kw,0)
    assert np.allclose(p.dynamic_power_kw,e.max_power_kw())
    p20=e.power_model(idle_fraction=.2)
    assert np.allclose(p20.idle_power_kw+p20.dynamic_power_kw,e.max_power_kw())

def test_cost_profiles_are_explicit_and_positive():
    e=env()
    for profile in ["primary_tier_index","performance_proportional","power_proportional"]:
        c=e.cost_indices(profile); assert c.shape==(7,); assert np.all(c>0)

def test_build_problem_requires_external_carbon_values():
    e=env(); workloads=[Workload("w",1,.2,2,1,0,2,CarbonClass.FLEXIBLE)]
    ci={s.name:300.0 for s in e.slots}
    problem=e.build_problem(workloads,ci,{CarbonClass.CRITICAL:250,CarbonClass.DEFERRED:400,CarbonClass.FLEXIBLE:500})
    assert len(problem.resources)==7
    assert problem.reference_cost==pytest.approx(1.0)
    assert problem.resources[0].memory_capacity==pytest.approx(64.0)


def test_formal_memory_capacity_uses_host_memory_not_vram():
    e=env()
    ci={slot.name:300.0 for slot in e.slots}
    workloads=[Workload("w",1,.2,48,1,0,2,CarbonClass.FLEXIBLE)]
    p=e.build_problem(workloads,ci,{CarbonClass.CRITICAL:350,CarbonClass.DEFERRED:400,CarbonClass.FLEXIBLE:500})
    assert all(r.memory_capacity==pytest.approx(64.0) for r in p.resources)
    assert e.hardware_classes["rtx_a5000"].gpu_memory_gb==pytest.approx(24.0)
