"""Read-only decomposition of saved Carbon-v2 pilot deadline misses.

Uses original trace deadlines and fixed-mean Carbon-v2 eligibility. Distinguishes
individual infeasibility, selected resource's uncontended service time, and
lateness introduced by waiting on an otherwise fast-enough assigned resource.
Does NOT infer that the remaining deadlines are jointly achievable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from qpeo.carbon import with_carbon_intensities
from scripts.run_final_experiment_shard import SCENARIOS, build_scenario

TOL_H = 1e-8


def calculate_eligibility(problem) -> np.ndarray:
    """Individual job/resource fit under the frozen Carbon-v2 planning rule."""
    return np.asarray([
        [w.gpu_demand <= r.gpu_capacity + 1e-12
         and w.cpu_demand <= (np.inf if r.cpu_capacity is None else r.cpu_capacity) + 1e-12
         and w.memory_demand <= r.memory_capacity + 1e-12
         and r.carbon_intensity <= problem.carbon_thresholds[w.carbon_class] + 1e-12
         for r in problem.resources]
        for w in problem.workloads
    ], dtype=bool)


def classify_schedule(problem, service_hours: np.ndarray, assignment: np.ndarray,
                      start: np.ndarray, finish: np.ndarray, *, tol_h: float = TOL_H) -> dict:
    """Attribute each recorded strict deadline miss to mutually exclusive categories.

    'assigned_resource_too_slow' means the selected resource cannot finish on
    time at the job's arrival; another individually eligible resource *could*.
    'start_delay' means the assigned resource would suffice at arrival, but the
    recorded execution misses. Neither is a global scheduling counterfactual.
    """
    m, n = len(problem.workloads), len(problem.resources)
    duration = np.asarray(service_hours, float)
    index = np.asarray(assignment)
    begin = np.asarray(start, float)
    end = np.asarray(finish, float)
    if (duration.shape != (m, n) or not np.all(np.isfinite(duration))
            or np.any(duration <= 0)):
        raise ValueError('invalid service-time matrix')
    if index.shape != (m,) or not np.issubdtype(index.dtype, np.integer):
        raise ValueError('assignment must contain one integer resource index per job')
    if np.any(index < 0) or np.any(index >= n):
        raise ValueError('invalid resource indices')
    if begin.shape != (m,) or end.shape != (m,) or not np.all(np.isfinite(begin)) or not np.all(np.isfinite(end)):
        raise ValueError('invalid saved job intervals')
    if not np.isfinite(tol_h) or tol_h < 0:
        raise ValueError('invalid time tolerance')
    arrivals = np.asarray([w.arrival_time for w in problem.workloads], float)
    deadlines = np.asarray([w.deadline for w in problem.workloads], float)
    valid_deadlines = np.isfinite(deadlines)
    if not np.all(np.isfinite(arrivals)) or np.any(np.isnan(deadlines)) or np.any(np.isneginf(deadlines)):
        raise ValueError('invalid arrivals or deadlines')
    if np.any(begin < arrivals - tol_h):
        raise ValueError('recorded start time precedes job arrival')
    if not np.allclose(end-begin, duration[np.arange(m), index], rtol=1e-9, atol=1e-9):
        raise ValueError('recorded service intervals disagree with fixed service times')
    eligible = calculate_eligibility(problem)
    if np.any(~eligible.any(axis=1)):
        raise ValueError('at least one job has no eligible resource')
    if np.any(~eligible[np.arange(m), index]):
        raise ValueError('saved assignment violates individual resource/carbon eligibility')
    best_duration = np.min(np.where(eligible, duration, np.inf), axis=1)
    chosen_duration = duration[np.arange(m), index]
    miss = valid_deadlines & (end > deadlines)  # EXACT existing f5 convention
    structural = valid_deadlines & (arrivals + best_duration > deadlines + tol_h)
    if np.any(structural & ~miss):
        raise ValueError('structurally impossible deadline recorded as met')
    boundary = miss & (end <= deadlines + tol_h)
    intrinsic = miss & structural & ~boundary
    chosen_slow = miss & ~structural & (arrivals + chosen_duration > deadlines + tol_h) & ~boundary
    delayed = miss & ~structural & ~chosen_slow & ~boundary
    met = valid_deadlines & ~miss
    counts = {
        'finite_deadlines': int(valid_deadlines.sum()),
        'no_finite_deadline': int((~valid_deadlines).sum()),
        'missed': int(miss.sum()),
        'met': int(met.sum()),
        'intrinsically_impossible': int(intrinsic.sum()),
        'assigned_resource_too_slow': int(chosen_slow.sum()),
        'start_delay_on_fast_enough_assignment': int(delayed.sum()),
        'numerical_boundary_misses': int(boundary.sum()),
        'optimistic_impossible_jobs_in_instance': int(structural.sum()),
        'observed_f5_percent': float(miss.sum()*100/valid_deadlines.sum()) if valid_deadlines.any() else 0.0,
        'mean_wait_hours_all_jobs': float(np.mean(begin-arrivals)),
        'median_wait_hours_all_jobs': float(np.median(begin-arrivals)),
    }
    if counts['missed'] != sum(counts[k] for k in (
        'intrinsically_impossible', 'assigned_resource_too_slow',
        'start_delay_on_fast_enough_assignment', 'numerical_boundary_misses')):
        raise AssertionError('deadline misses do not partition exactly')
    return counts


def audit_shard(payload: dict, *, source_file: str, content_sha256: str) -> list[dict]:
    if (payload.get('schema_version') != 'qpeo-carbon-v2-test-shard-v1'
            or not isinstance(payload.get('carbon_time_accounting'), dict)
            or payload['carbon_time_accounting'].get('planning_signal') != 'fixed full-week regional mean intensity'):
        raise ValueError('only a fixed-mean Carbon-v2 test shard can be audited')
    scenario = payload['scenario']
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    size = int(payload['size'])
    bundle, _env, forecast = build_scenario(scenario, size)
    problem = with_carbon_intensities(bundle.problem, forecast.intensities.mean(axis=0))
    algorithms = payload['algorithms']
    if len(algorithms) != 13 or len({a['algorithm'] for a in algorithms}) != 13:
        raise ValueError('not a complete 13-algorithm pilot')
    rows = []
    for record in algorithms:
        counts = classify_schedule(
            problem, bundle.service_time_hours,
            np.asarray(record['guide_assignment_indices'], dtype=int),
            np.asarray(record['guide_start_hours'], dtype=float),
            np.asarray(record['guide_completion_hours'], dtype=float),
        )
        f5 = float(record['guide_objectives'][4])
        if not np.isclose(f5, counts['observed_f5_percent'], rtol=0, atol=1e-8):
            raise AssertionError(f"{scenario}/{record['algorithm']}: saved f5 differs from reconstructed misses")
        rows.append({
            'source_file': source_file, 'source_sha256': content_sha256,
            'scenario': scenario, 'seed': payload['seed'],
            'population': payload['population_size'], 'iterations': payload['max_iterations'],
            'coverage_policy': payload['carbon_time_accounting']['coverage_policy'],
            'algorithm': record['algorithm'], 'recorded_f5_percent': f5,
            **counts,
        })
    return rows


def read_all(inputs: list[Path]) -> list[dict]:
    rows = []
    for path in inputs:
        if path.suffix.lower() == '.zip':
            with ZipFile(path) as z:
                names = [n for n in z.namelist() if n.endswith('.json') and '__seed' in Path(n).name]
                if not names:
                    raise ValueError(f'no Carbon-v2 shards found in {path}')
                for name in names:
                    raw = z.read(name)
                    rows.extend(audit_shard(json.loads(raw), source_file=f'{path.name}::{name}',
                                            content_sha256=hashlib.sha256(raw).hexdigest()))
        elif path.suffix.lower() == '.json':
            raw = path.read_bytes()
            rows.extend(audit_shard(json.loads(raw), source_file=str(path),
                                    content_sha256=hashlib.sha256(raw).hexdigest()))
        else:
            raise ValueError(f'expected ZIP or JSON, got {path}')
    if len({(r['source_sha256'], r['algorithm']) for r in rows}) != len(rows):
        raise ValueError('same pilot was supplied more than once')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=Path, nargs='+', help='saved pilot evidence ZIP(s) or shard JSON(s)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = read_all(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'sla_miss_attribution.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'status': 'READ_ONLY_PILOT_DIAGNOSTIC_NOT_FULL_CAMPAIGN',
        'rows': len(rows), 'shards': len(rows)//13,
        'source_files': [str(p) for p in args.inputs],
        'method': 'Saved guide schedules, original finite deadlines, best individually eligible service time under fixed carbon week mean. Exactly reproduces recorded f5.',
        'classification': {
            'intrinsically_impossible': 'Even fastest eligible service at arrival finishes after original deadline',
            'assigned_resource_too_slow': 'Some other resource is individually fast enough, but selected resource is not, even without waiting',
            'start_delay_on_fast_enough_assignment': 'Selected resource could finish on time if started at arrival; recorded start/finish misses deadline',
            'numerical_boundary_misses': 'Saved f5 counts a strictly positive lateness no greater than 1e-8 h',
        },
        'limitations': ['No optimizer was run', 'Assignment-only and start-delay classes are explanatory, not promises that an alternate globally feasible schedule exists', 'Pilots have one seed and do not support superiority claims', 'Carbon week repetition is a counterfactual model', 'Do not treat synthetic proxy deadlines as contractual SLAs'],
        'algorithm_rows': rows,
    }
    (args.output_dir / 'sla_miss_attribution.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"Verified {report['shards']} test shards and {len(rows)} algorithm results; reconstructed all reported f5 values")
    print(args.output_dir)


if __name__ == '__main__':
    main()
