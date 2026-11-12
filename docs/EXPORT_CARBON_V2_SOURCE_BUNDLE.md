# Export a small Carbon-v2 source bundle for local validation

This repository's full GitHub archive contains large legacy experimental results. To make **just the corrected code and reproducible test inputs** available for independent local execution without GitHub Actions or paid computation, export the source-only subset from a checkout of the `carbon-accounting-v2-validation` branch:

```bash
git switch carbon-accounting-v2-validation
git status --short
git archive --format=zip --output=../QPEO_Carbon_v2_Source.zip HEAD \
  qpeo scripts tests configs data requirements.txt pyproject.toml
```

The ZIP includes all 13 algorithm implementations, the simulator, Carbon-v2 accounting, the frozen four trace snapshots, locked carbon forecast, the small pilot runner, and its tests. It excludes `results/raw` and the older expensive experimental-output archives. The frozen raw NESO JSON is included as a provenance input but is not required to execute the pilot. **No network, GitHub Actions, or cloud GPU is needed to run tests on the exported source.**

For a local test workspace after extracting the ZIP, a repository checkout or a local Git snapshot is required because the pilot validator now records a verifiable revision:

```bash
python -m pip install -r requirements.txt
# Run the preflight inside a real git checkout, not from an unversioned ZIP:
bash scripts/run_carbon_v2_preflight.sh
```

When validating an extracted ZIP in a separate isolated directory, create a **local-only** Git snapshot with no remote (it will have its own hash, not the original GitHub commit): `git init`, `git add .`, and `git -c user.name='Local Validation' -c user.email='local@example.invalid' commit -m 'Local source snapshot'`. This local commit is for provenance only; do not push it to the repository. Then run the preflight. Alternatively run the preflight directly in an existing clean clone.

**Do not treat the tiny P=4, T=2, 1-seed smoke output as primary scientific evidence.** It exists only to expose implementation, runtime, constraint, and accounting errors. The new corrected full study must later be rerun with its own locked protocol and independently analyzed before manuscript tables are changed.

To share the code for validation in ChatGPT, attach `QPEO_Carbon_v2_Source.zip`; do not upload the older 120-shard results or the entire repository history.
