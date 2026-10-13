from __future__ import annotations

from aegis.eval.batch import build_batch_report, combo_dir_name, combo_seed


class TestComboSeeds:
    def test_stride_deterministic(self) -> None:
        assert combo_seed(42, 0) == 42
        assert combo_seed(42, 1) == 1042
        assert combo_seed(42, 2) == 2042
        # Re-computing gives the same value regardless of order.
        assert combo_seed(7, 3) == 7 + 3 * 1000

    def test_dir_name_stable(self) -> None:
        assert combo_dir_name("franka", "scripted", "pick-place", 42) == (
            "franka-scripted-pick-place-seed42"
        )


def _rep(task="pick-place", robot="franka", model="scripted", success=3, fail=0,
         viol=0, rec=0, steps=750, mode="cpu"):
    return {
        "task_counts": {task: {"success": success, "fail": fail}},
        "robot": robot,
        "model": model,
        "safety_violations": viol,
        "recovery_events": rec,
        "total_steps": steps,
        "inference_mode": mode,
    }


class TestBuildBatchReport:
    def test_sequential_mode(self) -> None:
        combos = [{"robot": "franka", "model": "scripted", "task": "pick-place"}]
        reps = [_rep()]
        out = build_batch_report("batch-x", reps, combos, parallel=1)
        assert out["mode"] == "sequential"
        assert out["parallel"] == 1
        assert out["summary"] == {
            "combos": 1, "success": 3, "fail": 0,
            "safety_violations": 0, "recovery_events": 0, "total_steps": 750,
        }
        assert out["task_counts"]["pick-place"]["franka"]["scripted"] == {
            "success": 3, "fail": 0,
        }

    def test_parallel_mode_order_independent(self) -> None:
        combos = [
            {"robot": "franka", "model": "scripted", "task": "pick-place"},
            {"robot": "franka", "model": "random", "task": "pick-place"},
        ]
        reps = [_rep(model="scripted", success=3, fail=0), _rep(model="random", success=0, fail=3, viol=4, rec=2)]
        seq = build_batch_report("b", reps, combos, parallel=1)
        par = build_batch_report("b", reps, combos, parallel=4)
        assert par["mode"] == "parallel-4"
        assert par["parallel"] == 4
        # Same totals regardless of mode flag.
        assert par["summary"]["success"] == seq["summary"]["success"] == 3
        assert par["summary"]["fail"] == 3
        assert par["summary"]["safety_violations"] == 4
        # Reversed input order aggregates to the same totals.
        rev = build_batch_report("b", list(reversed(reps)), list(reversed(combos)), parallel=4)
        assert rev["summary"] == par["summary"]

    def test_cuda_parallel_warns(self) -> None:
        combos = [{"robot": "franka", "model": "smolvla_libero", "task": "pick-place"}]
        reps = [_rep(model="smolvla_libero", success=0, fail=3, viol=99, rec=99, mode="cuda")]
        out = build_batch_report("b", reps, combos, parallel=4)
        assert any("6GB" in w for w in out["warnings"])
        seq = build_batch_report("b", reps, combos, parallel=1)
        assert not any("6GB" in w for w in seq["warnings"])
