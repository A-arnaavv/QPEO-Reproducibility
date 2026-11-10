from pathlib import Path
import json
import numpy as np

from qpeo.carbon import load_neso_regional_replay
from qpeo.experiment import run_full_suite
from qpeo.optimizer import ReconstructionQPEOSettings
from qpeo.protocol import load_protocol
from qpeo.resource_environment import load_resource_environment
from qpeo.scenarios import build_alibaba_spot_a100_scenario

def test_real_trace_full_suite_smoke():
    root=Path(__file__).parents[1]
    primary,_=load_protocol(root/'configs'/'reconstruction_protocol.json')
    env=load_resource_environment(root/'configs'/'resource_environment_v1.json')
    replay=load_neso_regional_replay(
        root/'data'/'carbon'/'neso_regional_2026-01-05_2026-01-11.csv',
        root/'configs'/'carbon_replay_v1.json',
        expected_slot_names=[s.name for s in env.slots],
    )
    bundle=build_alibaba_spot_a100_scenario(
        root/'data'/'alibaba_spot_a100_selected_128.csv',
        root/'configs'/'trace_alibaba_spot_a100.json',
        env,replay,size=32,
    )
    weights=tuple(primary.scalar_guide_weights)
    settings=ReconstructionQPEOSettings(
        weights,primary.alpha,primary.beta,primary.gamma,
        primary.dependency_threshold,primary.dependency_strength,
    )
    suite=run_full_suite(
        bundle.problem,bundle.qpeo_inputs,settings,weights,
        seeds=[primary.seeds[0]],population_size=4,max_iterations=1,
    )
    assert len(suite.records)==13
    assert len(suite.algorithms)==13
    assert all(np.all(np.isfinite(r.guide_objectives)) for r in suite.records)
