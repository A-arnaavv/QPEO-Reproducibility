from pathlib import Path
import numpy as np
import pytest
from qpeo.carbon import *

def test_policy():
    p, raw = load_carbon_policy(Path(__file__).parents[1]/"configs"/"carbon_benchmark.json")
    assert p.layer_zones["cloud"]=="US-MIDA-PJM"
    assert p.layer_zones["serverless"]=="US-CAL-CISO"
    assert p.layer_zones["edge"]=="IN-NO"
    q=electricity_maps_past_range_query(p,"cloud")
    assert q["temporalGranularity"]=="hourly"
    assert q["emissionFactorType"]=="lifecycle"
    assert q["flowTraced"] is True
    assert q["disableCallerLookup"] is True

def test_snapshot_roundtrip(tmp_path):
    times=tuple(f"h{i}" for i in range(168))
    s=CarbonSeries("US-MIDA-PJM",times,np.linspace(250,450,168),np.array([False]*160+[True]*8))
    path=tmp_path/"c.csv"
    digest=save_carbon_snapshot_csv(path,s)
    assert len(digest)==64
    loaded=load_carbon_snapshot_csv(path,"US-MIDA-PJM")
    assert np.allclose(loaded.intensity_gco2eq_per_kwh,s.intensity_gco2eq_per_kwh)
    assert estimated_fraction(loaded)==pytest.approx(8/168)
