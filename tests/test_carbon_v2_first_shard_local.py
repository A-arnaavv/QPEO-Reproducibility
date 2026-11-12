"""Safe shell-runner guards: never launch a computation in these tests."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "carbon_v2_first_shard_local.sh"


def test_shell_launcher_refuses_missing_pinned_environment(tmp_path):
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("git and bash required for offline shell-runner test")
    root = tmp_path / "test_repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
    (root / "requirements.txt").write_text("numpy==2.4.6\nscipy==1.17.1\npytest==9.2.0\n")
    (root / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True,
                              capture_output=True, text=True)
    git("init", "-b", "carbon-accounting-v2-validation")
    git("add", "scripts", "requirements.txt", "pyproject.toml")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-m", "Local test")
    for mode in ("--check", "--execute-one"):
        result = subprocess.run(["bash", str(root / "scripts" / SCRIPT.name), mode],
                                cwd=root, capture_output=True, text=True, check=False)
        assert result.returncode == 3
        assert "Pinned environment missing" in result.stderr
        assert "EXECUTING" not in (result.stdout + result.stderr)


def test_shell_launcher_requests_deep_first_shard_audit():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "scripts/audit_carbon_v2_campaign.py --input-dir" in source
    assert "--fast-audit" not in source


def test_first_shard_launcher_tracks_input_data_tree():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "qpeo scripts configs tests data requirements.txt pyproject.toml" in source
