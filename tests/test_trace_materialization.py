import pytest
from qpeo.models import CarbonClass
from qpeo.traces import RawTraceTask, NativeTraceMaterializationPolicy, observed_queue_hours, trace_derived_deadline, pooled_native_per_gpu_median, materialize_gpu_native_workloads

def task(name,gpu,cpu,mem,q,runtime,arrival=0):
    return RawTraceTask(name,arrival,runtime,cpu,gpu,mem,None,None,{"queue_hours":str(q)})

def test_trace_deadline_uses_observed_completion():
    t=task("a",1,4,16,.5,2.0,arrival=1.0)
    assert observed_queue_hours(t)==pytest.approx(.5)
    assert trace_derived_deadline(t)==pytest.approx(3.5)

def test_pooled_native_medians():
    ts=[task("a",1,4,16,.1,1),task("b",.5,2,4,.1,1),task("c",2,8,64,.1,1)]
    assert pooled_native_per_gpu_median(ts,field="cpu_demand")==pytest.approx(4.0)
    assert pooled_native_per_gpu_median(ts,field="memory_demand")==pytest.approx(16.0)

def test_materialization_imputes_only_missing_fields():
    ts=[task("known",1,4,12,.2,1),task("missing",1,None,None,.3,2)]
    policy=NativeTraceMaterializationPolicy(16,4,0,CarbonClass.DEFERRED)
    ws=materialize_gpu_native_workloads(ts,policy)
    assert ws[0].memory_demand==pytest.approx(12)
    assert ws[1].memory_demand==pytest.approx(16)
    assert ws[1].cpu_demand==pytest.approx(4)
    assert ws[1].deadline==pytest.approx(2.3)
