from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run_pai(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "aegis.cli", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _episode_outcomes(report_dir: Path) -> list[dict]:
    lines = (report_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    return [
        json.loads(line)
        for line in lines
        if json.loads(line)["event"] == "episode_end"
    ]


def _report_dict(report_dir: Path) -> dict:
    return json.loads((report_dir / "report.json").read_text(encoding="utf-8"))


class TestConfigValidation:
    def test_valid_config_ok(self) -> None:
        result = _run_pai(["validate"])
        assert result.returncode == 0, result.stderr
        assert "config OK" in result.stdout

    def test_invalid_config_fails_with_exit_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("eval:\n  episodes: -5\n", encoding="utf-8")
        result = _run_pai(["validate", "--config", str(bad)])
        assert result.returncode == 2, result.stdout
        assert "config error" in result.stderr

    def test_missing_model_config_fails_with_exit_2(self) -> None:
        result = _run_pai(["validate", "--model", "does-not-exist"])
        assert result.returncode == 2
        assert "config error" in result.stderr


class TestDeterminism:
    """Same seed -> identical episode outcomes across two runs (core POC claim)."""

    @pytest.mark.parametrize("model", ["random", "scripted"])
    def test_identical_outcomes_same_seed(self, model: str, tmp_path: Path) -> None:
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        r1 = _run_pai(
            ["eval", "--model", model, "--episodes", "3", "--seed", "7", "--output-dir", str(out1)]
        )
        r2 = _run_pai(
            ["eval", "--model", model, "--episodes", "3", "--seed", "7", "--output-dir", str(out2)]
        )
        assert r1.returncode == 0, r1.stderr
        assert r2.returncode == 0, r2.stderr

        # Find the run subdirs (one per run).
        d1 = sorted(p for p in out1.iterdir() if p.is_dir())[0]
        d2 = sorted(p for p in out2.iterdir() if p.is_dir())[0]

        o1 = _episode_outcomes(d1)
        o2 = _episode_outcomes(d2)
        assert len(o1) == 3 and len(o2) == 3
        # Outcomes must be byte-identical (exclude timestamps by comparing the
        # deterministic fields only).
        fields = ["episode", "outcome", "fail_reason", "success", "steps", "violations", "recoveries"]
        for a, b in zip(o1, o2):
            assert {k: a[k] for k in fields} == {k: b[k] for k in fields}

        rep1, rep2 = _report_dict(d1), _report_dict(d2)
        for key in ["task_counts", "safety_violations", "recovery_events", "recommendation"]:
            assert rep1[key] == rep2[key], f"report field {key} differs between runs"

    def test_random_policy_is_fail_and_scripted_is_success(self, tmp_path: Path) -> None:
        """Sanity on the POC story: random triggers safety, scripted completes."""
        out = tmp_path / "out"
        r = _run_pai(
            ["eval", "--model", "scripted", "--episodes", "3", "--seed", "7", "--output-dir", str(out)]
        )
        assert r.returncode == 0, r.stderr
        d = sorted(p for p in out.iterdir() if p.is_dir())[0]
        outcomes = _episode_outcomes(d)
        assert any(o["success"] for o in outcomes), "scripted policy should succeed"
        rep = _report_dict(d)
        assert rep["safety_violations"] == 0, "scripted policy should not violate limits"