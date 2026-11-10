"""Recover validated full-protocol JSON shards from GitHub Action artifacts.

The input directory contains unzipped per-job artifacts. Validates every file,
never overwrites a nonidentical previously committed result, and leaves an
explicit report of missing or invalid files. No experiment values are imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.audit_final_shards import audit, validate_shard


def collect(source: Path, destination: Path, protocol: dict) -> dict:
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    primary = protocol["primary"]
    records = {
        "new_files": [],
        "already_identical": [],
        "invalid_or_conflicting": [],
    }
    for path in sorted(source.glob("*__final.json")):
        try:
            metadata = validate_shard(path, primary)
            target = destination / path.name
            if target.exists():
                old = hashlib.sha256(target.read_bytes()).hexdigest()
                if old != metadata["sha256"]:
                    raise ValueError(
                        f"Existing result differs; refusing overwrite: {path.name}"
                    )
                records["already_identical"].append(metadata)
            else:
                shutil.copy2(path, target)
                records["new_files"].append(metadata)
        except Exception as exc:
            records["invalid_or_conflicting"].append(
                {"filename": path.name, "error": str(exc)}
            )
    records["coverage"] = audit(destination, protocol)
    records["source_artifact_count"] = len(list(source.glob("*__final.json")))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--protocol", default="configs/reconstruction_protocol.json")
    parser.add_argument("--report", default="results/artifact_collection_report.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / args.protocol).read_text(encoding="utf-8"))
    report = collect(Path(args.source), root / args.output_dir, protocol)
    p = root / args.report
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    coverage = report["coverage"]
    print(
        "FINAL ARTIFACT COLLECTION:",
        len(report["new_files"]), "added,",
        len(report["already_identical"]), "unchanged,",
        len(report["invalid_or_conflicting"]), "invalid/conflicting;",
        coverage["valid_shards"], "of", coverage["expected_shards"], "validated"
    )
    if report["invalid_or_conflicting"]:
        print("INVALID FILES:", json.dumps(report["invalid_or_conflicting"], indent=2))


if __name__ == "__main__":
    main()
