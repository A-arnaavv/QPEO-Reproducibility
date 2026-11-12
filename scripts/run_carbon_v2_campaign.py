"""Candidate Carbon-v2 full campaign plan and opt-in ONE-SHARD runner.

Plan mode is harmless (read only). run-one REQUIRES --execute, pinned
versions, exact validation branch, and a clean tracked checkout.
No GitHub Actions is dispatched. No v1 files are modified.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_final_shards import SCENARIOS
from scripts.audit_carbon_v2_campaign import TAG, _read_protocol, expected_filename
from scripts.validate_carbon_v2_pilot import validate_pilot

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs/carbon_v2_candidate_full_campaign.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=True, timeout=20).stdout.strip()


def _locked_environment() -> None:
    versions = {line.split("==")[0].strip(): line.split("==")[1].strip()
                for line in (ROOT / "requirements.txt").read_text().splitlines()
                if "==" in line}
    errors = []
    for name in ("numpy", "scipy", "pytest"):
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "NOT INSTALLED"
        if actual != versions[name]:
            errors.append(f"{name}: installed {actual}, required {versions[name]}")
    if errors:
        raise SystemExit("REFUSING unpinned full campaign: " + "; ".join(errors))


def campaign_plan(source_sha: str = "<GIT_SHA>") -> dict:
    primary = _read_protocol()
    seeds = primary["seeds"]
    tasks = []
    for scenario in SCENARIOS:
        for index, seed in enumerate(seeds):
            tasks.append({
                "scenario": scenario, "seed_index": index, "seed": seed,
                "expected_file": expected_filename(scenario, index, seed),
                "invocation": f"python scripts/run_carbon_v2_campaign.py run-one --scenario {scenario} --seed-index {index} --execute",
            })
    return {"status": "PLANNED_NOT_EXECUTED", "source_revision": source_sha,
            "protocol_file": "configs/carbon_v2_candidate_full_campaign.json",
            "candidate_protocol": json.loads(CANDIDATE.read_text()),
            "task_count": len(tasks), "algorithm_records_expected": len(tasks) * 13,
            "tasks": tasks}


def run_one(scenario: str, seed_index: int, *, execute: bool) -> dict:
    if not execute:
        raise SystemExit("REFUSING to run: add --execute to opt into exactly ONE full shard")
    if scenario not in SCENARIOS:
        raise SystemExit("Unknown scenario")
    primary = _read_protocol()
    if not 0 <= seed_index < len(primary["seeds"]):
        raise SystemExit("Seed index outside locked 0..29")
    _locked_environment()
    current_branch = _git("branch", "--show-current")
    if current_branch != "carbon-accounting-v2-validation":
        raise SystemExit("REFUSING: must run from the carbon-accounting-v2-validation branch")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("REFUSING: tracked Git checkout has uncommitted changes")
    git_sha = _git("rev-parse", "HEAD")
    if len(git_sha) != 40:
        raise SystemExit("REFUSING: invalid Git revision")
    output_dir = ROOT / "results/carbon_v2_campaign" / git_sha[:12]
    seed = primary["seeds"][seed_index]
    outpath = output_dir / expected_filename(scenario, seed_index, seed)
    if outpath.exists():
        # Never overwrite or resume opaque shards. An existing validated shard
        # is reused only if it passes independent schedule reconstruction.
        validated = validate_pilot(outpath)
        if validated["algorithms_verified"] != 13:
            raise SystemExit("Existing shard did not validate")
        return {"status": "existing-valid-shard", "file": str(outpath)}
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/run_final_experiment_shard.py"),
           "--scenario", scenario, "--seed-index", str(seed_index),
           "--population-size", "50", "--max-iterations", "500",
           "--carbon-time-accounting", "repeat", "--output-dir", str(output_dir)]
    print("EXECUTING one full corrected shard:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not outpath.exists():
        raise RuntimeError("Experiment finished but the expected shard was not created")
    audited = validate_pilot(outpath)
    if audited["algorithms_verified"] != 13:
        raise RuntimeError("Incomplete post-execution schedule validation")
    return {"status": "one-shard-executed-and-validated", "file": str(outpath),
            "git_sha": git_sha, "algorithms": 13}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Output 120-shard execution plan. NEVER computes")
    plan.add_argument("--output", type=Path, default=ROOT / "results/carbon_v2_campaign_plan.json")
    one = sub.add_parser("run-one", help="Execute exactly one full shard (explicit opt in)")
    one.add_argument("--scenario", required=True, choices=SCENARIOS)
    one.add_argument("--seed-index", required=True, type=int)
    one.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.command == "plan":
        try:
            sha = _git("rev-parse", "HEAD")
        except (OSError, subprocess.SubprocessError):
            sha = "<GIT_SHA>"
        data = campaign_plan(sha)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print(f"PLAN ONLY: {data['task_count']} scenario × seed shards, "
              f"{data['algorithm_records_expected']} algorithm records")
        print(args.output)
    else:
        result = run_one(args.scenario, args.seed_index, execute=args.execute)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
