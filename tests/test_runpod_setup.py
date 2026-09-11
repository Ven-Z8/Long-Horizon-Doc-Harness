from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_runpod_scripts_are_executable_and_have_valid_shell_syntax() -> None:
    scripts = [
        ROOT / "scripts" / "runpod" / "bootstrap.sh",
        ROOT / "scripts" / "runpod" / "verify.sh",
    ]
    for script in scripts:
        assert script.is_file(), script
        assert script.stat().st_mode & 0o111, script
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_bootstrap_dry_run_is_non_destructive(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "runpod" / "bootstrap.sh"),
            "--repo-dir",
            str(repo),
            "--skip-repo",
            "--skip-dependencies",
            "--skip-models",
            "--skip-data",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (repo / "models").exists()
