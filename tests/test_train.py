from __future__ import annotations

import json
from pathlib import Path

from aegis.data.export import export_run
from aegis.train.bc_trainer import load_dataset, train_bc, train_ridge


def _fake_dataset(tmp: Path) -> Path:
    run = tmp / "run-x"
    run.mkdir()
    steps, trajs = [], []
    for i in range(1, 11):
        steps.append({"event": "step", "episode": 0, "step": i, "source": "policy",
                      "violated": False, "inference_s": 0.001, "gateway_s": 0.0, "env_s": 0.001})
        trajs.append({"episode": 0, "step": i, "qpos": [0.01 * i] * 7,
                      "action": [0.02 * i] * 8, "source": "policy"})
    (run / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps), encoding="utf-8")
    (run / "trajectory.jsonl").write_text("\n".join(json.dumps(t) for t in trajs), encoding="utf-8")
    meta = export_run(run)
    assert meta["episodes"] == [0]
    return run / "datasets"


class TestBCTrainer:
    def test_load_and_ridge(self, tmp_path) -> None:
        ds = _fake_dataset(tmp_path)
        X, Y = load_dataset(ds)
        assert X.shape == (10, 7) and Y.shape == (10, 8)
        res = train_ridge(X, Y)
        assert res["mse"] >= 0.0 and res["weights"].shape == (8, 8)

    def test_train_bc_writes_artifacts(self, tmp_path) -> None:
        ds = _fake_dataset(tmp_path)
        out = tmp_path / "bc"
        metrics = train_bc(ds, out, epochs=2)
        assert metrics["n"] == 10
        assert (out / "metrics.json").exists()
        assert (out / "bc_policy.npz").exists() or (out / "bc_policy.pt").exists()


class TestRLEnvSafety:
    def test_gateway_always_on(self) -> None:
        from aegis.train.rl_env import AegisRLEnv

        class _Gated:
            def __init__(self) -> None:
                self.calls = 0

            def filter(self, action, state):
                import numpy as np  # type: ignore

                self.calls += 1

                class _G:
                    command = np.zeros(8)
                    source = "fallback"
                    violated = True

                return _G()

        class _Env:
            def reset(self, seed):
                return {"arm_qpos": [0.0] * 7}

            def state_snapshot(self):
                return {"qpos": [0.0] * 7}

            def step(self, action):
                return ({"arm_qpos": [0.0] * 7}, False, False, {"success": False})

        gw = _Gated()
        rl = AegisRLEnv(_Env(), gw, seed=1)
        rl.reset(1)
        _, _, _, _, info = rl.step([9.0] * 8)
        assert gw.calls == 1 and info["violated"] is True and rl.violation_count == 1
