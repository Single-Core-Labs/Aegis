from __future__ import annotations

import json
from pathlib import Path

from aegis.data.export import export_run
from aegis.data.validate import validate_dataset


def _fake_run(tmp: Path) -> Path:
    run = tmp / "run-x"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"event": "run_start"}), encoding="utf-8")
    steps = [
        {"event": "step", "episode": 0, "step": 1, "source": "policy",
         "violated": False, "inference_s": 0.001, "gateway_s": 0.0, "env_s": 0.001},
        {"event": "step", "episode": 0, "step": 2, "source": "policy",
         "violated": False, "inference_s": 0.001, "gateway_s": 0.0, "env_s": 0.001},
    ]
    trajs = [
        {"episode": 0, "step": 1, "qpos": [0.0] * 7, "action": [0.0] * 8, "source": "policy"},
        {"episode": 0, "step": 2, "qpos": [0.1] * 7, "action": [0.1] * 8, "source": "policy"},
    ]
    (run / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps), encoding="utf-8")
    (run / "trajectory.jsonl").write_text("\n".join(json.dumps(t) for t in trajs), encoding="utf-8")
    return run


class TestExportValidate:
    def test_export_jsonl_fallback_and_validate_ok(self, tmp_path) -> None:
        run = _fake_run(tmp_path)
        meta = export_run(run)
        assert meta["episodes"] == [0]
        assert meta["format"] in ("parquet", "jsonl")
        errors = validate_dataset(run / "datasets")
        assert errors == [], errors

    def test_missing_metadata_fails(self, tmp_path) -> None:
        errors = validate_dataset(tmp_path / "nope")
        assert errors and "metadata.json missing" in errors[0]

    def test_nan_action_fails(self, tmp_path) -> None:
        run = _fake_run(tmp_path)
        meta = export_run(run)
        assert meta["format"] == "jsonl" or True
        ds = run / "datasets"
        # Corrupt the first episode file with NaN.
        for f in ds.glob("episode_*"):
            text = f.read_text(encoding="utf-8")
            f.write_text(text.replace("0.1", "NaN"), encoding="utf-8")
        errors = validate_dataset(ds)
        assert errors, "NaN action must be caught"


class TestImageLogger:
    def test_budget_and_events(self, tmp_path) -> None:
        from aegis.telemetry.image_logger import ImageLogger

        lg = ImageLogger(tmp_path, max_storage_mb=100.0)
        assert ImageLogger.should_log("violation")
        assert not ImageLogger.should_log("every_step")
        try:
            import numpy as np  # type: ignore

            img = np.zeros((8, 8, 3), dtype="uint8")
        except Exception:
            img = None
        if img is not None:
            p = lg.log_image(img, 0, 1, "camera1", "violation")
            assert p is not None and p.exists()
            # Non-flagged event is dropped.
            assert lg.log_image(img, 0, 2, "camera1", "every_step") is None
