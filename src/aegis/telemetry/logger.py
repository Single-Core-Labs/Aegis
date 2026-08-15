from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _clean(value: Any) -> Any:
    """Convert numpy scalars/arrays and Paths into JSON-serializable values."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class RunLogger:
    """NDJSON structured logger for one eval run.

    Layout under output_dir/<run_id>/:
      run.json         full validated config + start time
      episodes.jsonl   one line per episode
      steps.jsonl      one line per step (per episode)
      trajectory.jsonl per-step state snapshots
      report.json      final summary (written by report module)
    """

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self._run_dir = output_dir / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_fh = (self.run_dir / "episodes.jsonl").open("w", encoding="utf-8")
        self._steps_fh = (self.run_dir / "steps.jsonl").open("w", encoding="utf-8")
        self._traj_fh = (self.run_dir / "trajectory.jsonl").open("w", encoding="utf-8")
        self._step_episode: int | None = None
        self._closed = False

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def write_run(self, config: dict[str, Any]) -> None:
        payload = {
            "event": "run_start",
            "run_id": self.run_dir.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": _clean(config),
        }
        (self.run_dir / "run.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def episode_start(self, episode: int, seed: int) -> None:
        self._step_episode = episode
        self._episodes_fh.write(
            json.dumps({"event": "episode_start", "episode": episode, "seed": seed})
            + "\n"
        )
        self._episodes_fh.flush()

    def step(
        self,
        episode: int,
        step: int,
        *,
        source: str,
        violated: bool,
        violation: dict[str, Any] | None,
        inference_s: float,
        gateway_s: float,
        env_s: float,
        info: dict[str, Any],
    ) -> None:
        payload = {
            "event": "step",
            "episode": episode,
            "step": step,
            "source": source,
            "violated": violated,
            "violation": violation,
            "inference_s": round(inference_s, 9),
            "gateway_s": round(gateway_s, 9),
            "env_s": round(env_s, 9),
            "info": _clean(info),
        }
        self._steps_fh.write(json.dumps(payload) + "\n")

    def trajectory(self, episode: int, step: int, state: dict[str, Any]) -> None:
        payload = {"episode": episode, "step": step, **_clean(state)}
        self._traj_fh.write(json.dumps(payload) + "\n")

    def episode_end(
        self,
        episode: int,
        *,
        outcome: str,
        fail_reason: str | None,
        steps: int,
        duration_s: float,
        violations: int,
        recoveries: int,
        success: bool,
    ) -> None:
        payload = {
            "event": "episode_end",
            "episode": episode,
            "outcome": outcome,
            "fail_reason": fail_reason,
            "success": success,
            "steps": steps,
            "duration_s": round(duration_s, 6),
            "violations": violations,
            "recoveries": recoveries,
        }
        self._episodes_fh.write(json.dumps(payload) + "\n")
        self._episodes_fh.flush()

    def write_report(self, report: dict[str, Any]) -> None:
        (self.run_dir / "report.json").write_text(
            json.dumps(_clean(report), indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        if not self._closed:
            self._episodes_fh.close()
            self._steps_fh.close()
            self._traj_fh.close()
            self._closed = True