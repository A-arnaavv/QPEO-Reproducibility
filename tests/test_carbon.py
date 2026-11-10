from pathlib import Path
import csv
from datetime import datetime, timezone
import numpy as np
import pytest
from qpeo.carbon import CarbonReplay, load_neso_regional_replay, with_carbon_intensities
from qpeo.models import CarbonClass, ProblemInstance, Resource, Workload
from qpeo.resource_environment import load_resource_environment

def test_iteration_mapping_and_threshold_order():
    replay=CarbonReplay(
        timestamps_utc=tuple(datetime(2026,1,5,h,tzinfo=timezone.utc) for h in range(4)),
        slot_names=("a","b"),
        intensities=np.array([[100,200],[120,220],[140,240],[160,260]],float),
        source_name="fixture",
        window_start_utc=datetime(2026,1,5,tzinfo=timezone.utc),
        window_end_utc=datetime(2026,1,5,3,tzinfo=timezone.utc),
    )
    replay.validate()
    assert np.allclose(replay.at_iteration(0,6),[100,200])
    assert np.allclose(replay.at_iteration(6,6),[160,260])
    th=replay.thresholds()
    assert th[CarbonClass.CRITICAL]<=th[CarbonClass.DEFERRED]<=th[CarbonClass.FLEXIBLE]
    for row in replay.intensities:
        assert np.sum(row<=th[CarbonClass.CRITICAL])>=1
        assert np.sum(row<=th[CarbonClass.DEFERRED])>=min(3,replay.resources)
        assert np.sum(row<=th[CarbonClass.FLEXIBLE])==replay.resources

def test_neso_loader_filters_locked_window_and_fills_missing(tmp_path:Path):
    cfg=Path(__file__).parents[1]/"configs"/"carbon_replay_v1.json"
    env=load_resource_environment(Path(__file__).parents[1]/"configs"/"resource_environment_v1.json")
    columns=["Date & Time","North Scotland Carbon Intensity Forecast","North West England Carbon Intensity Forecast","South Wales Carbon Intensity Forecast","East England Carbon Intensity Forecast","South England Carbon Intensity Forecast","London Carbon Intensity Forecast","South East England Carbon Intensity Forecast"]
    p=tmp_path/"carbon.csv"
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader()
        w.writerow({c:("100" if c!="Date & Time" else "2026-01-04T23:30:00+00:00") for c in columns})
        row={c:"200" for c in columns}; row["Date & Time"]="2026-01-05T00:00:00+00:00"; row[columns[2]]=""; w.writerow(row)
        row={c:"220" for c in columns}; row["Date & Time"]="2026-01-05T00:30:00+00:00"; w.writerow(row)
    replay=load_neso_regional_replay(p,cfg,expected_slot_names=[s.name for s in env.slots])
    assert replay.rows==2
    assert replay.intensities.shape==(2,7)
    assert replay.intensities[0,1]==pytest.approx(220.0)

def test_with_carbon_intensities_preserves_other_resource_fields():
    p=ProblemInstance(
        [Workload("w",1,.2,2,1,0,2,CarbonClass.FLEXIBLE)],
        [Resource("r",1,24,300,1.1,cpu_capacity=8)],
        {CarbonClass.CRITICAL:100,CarbonClass.DEFERRED:300,CarbonClass.FLEXIBLE:500},
        1,
    )
    q=with_carbon_intensities(p,[150])
    assert q.resources[0].carbon_intensity==pytest.approx(150)
    assert q.resources[0].memory_capacity==pytest.approx(24)
    assert q.resources[0].cost_per_gpu_hour==pytest.approx(1.1)
