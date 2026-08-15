from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The scripted policy must be robust across these seeds: per seed the cube is
# placed at a different spot (deterministic jitter), so a grasp that only works
# for one geometry is a flaky baseline.
SEEDS = [1, 7, 42, 123, 2024]
EPISODES_PER_SEED = 5
REQUIRED_SUCCESS_RATE = 0.9  # >= 90% success across the whole sweep


def _run_pai(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "aegis.cli", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _episode_outcomes(report_dir: Path) -> list[dict]:
    lines = (report_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    return [
        json.loads(line)
        for line in lines
        if json.loads(line)["event"] == "episode_end"
    ]


class TestScriptedSeedSweep:
    """Scripted baseline robustness gate before it becomes the Phase 2 baseline.

    Asserts >=90% success across {1,7,42,123,2024} x 5 episodes and zero safety
    violations. A flaky baseline would make Phase 2 safety-layer behaviour
    impossible to attribute.
    """

    def test_scripted_robust_across_seeds(self, tmp_path: Path) -> None:
        total_success = 0
        total_violations = 0
        per_seed: dict[int, dict] = {}

        for seed in SEEDS:
            out = tmp_path / f"seed-{seed}"
            r = _run_pai(
                [
                    "eval",
                    "--model",
                    "scripted",
                    "--episodes",
                    str(EPISODES_PER_SEED),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(out),
                ]
            )
            assert r.returncode == 0, r.stderr
            d = sorted(p for p in out.iterdir() if p.is_dir())[0]
            outcomes = _episode_outcomes(d)
            success = sum(1 for o in outcomes if o["success"])
            violations = sum(o["violations"] for o in outcomes)
            total_success += success
            total_violations += violations
            per_seed[seed] = {"success": success, "violations": violations}

        n_episodes = len(SEEDS) * EPISODES_PER_SEED
        assert (
            total_success >= n_episodes * REQUIRED_SUCCESS_RATE
        ), f"success {total_success}/{n_episodes}, below 90%: {per_seed}"
        assert total_violations == 0, f"violations across seeds: {per_seed}"