"""Multi-combo batching (Beyond Phase 3, MuJoCo-safe slice).

Runs the Cartesian product ``robots x models x tasks`` — one isolated
env + gateway per combo (no shared ``MjData``) — and writes an aggregated
batch ``report.json``. ``--parallel N`` runs combos in separate processes
(ProcessPoolExecutor, per-combo isolation); Isaac ``num_envs`` is still
deferred until real USD exists (see ``docs/batching.md``).

Determinism: combo ``i`` runs with ``seed = base_seed + i*1000``; inside a
combo, episode ``e`` uses ``seed + e`` (standard runner rule). Re-running a
batch with the same base seed reproduces every combo.
"""

from __future__ import annotations

from typing import Any

COMBO_SEED_STRIDE = 1000


def combo_seed(base_seed: int, combo_index: int) -> int:
    return int(base_seed) + int(combo_index) * COMBO_SEED_STRIDE


def combo_dir_name(robot: str, model: str, task: str, seed: int) -> str:
    return f"{robot}-{model}-{task}-seed{seed}"


def build_batch_report(
    batch_id: str,
    combo_reports: list[dict[str, Any]],
    combos: list[dict[str, Any]],
    parallel: int = 1,
) -> dict[str, Any]:
    """Aggregate per-combo ``report.json`` dicts into one batch report.

    ``task_counts`` becomes nested ``task_counts[task][robot][model]`` plus a
    ``summary`` rollup. Single-combo runs keep their flat per-combo reports
    untouched on disk.

    ``parallel`` records how combos ran: 1 keeps legacy ``mode="sequential"``,
    >1 records ``mode="parallel-N"``. Aggregation is order-independent so
    parallel runs reproduce sequential results given the same base seed.
    """
    nested: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    total_success = total_fail = 0
    total_violations = total_recoveries = 0
    total_steps = 0
    for rep, combo in zip(combo_reports, combos):
        task = str(combo.get("task", rep.get("task_counts") and next(iter(rep["task_counts"]))))
        robot = str(combo.get("robot", rep.get("robot", "?")))
        model = str(combo.get("model", rep.get("model", "?")))
        counts = rep.get("task_counts", {}).get(task, {"success": 0, "fail": 0})
        nested.setdefault(task, {}).setdefault(robot, {})[model] = {
            "success": int(counts.get("success", 0)),
            "fail": int(counts.get("fail", 0)),
        }
        total_success += int(counts.get("success", 0))
        total_fail += int(counts.get("fail", 0))
        total_violations += int(rep.get("safety_violations", 0))
        total_recoveries += int(rep.get("recovery_events", 0))
        total_steps += int(rep.get("total_steps", 0))
    mode = "sequential" if int(parallel) <= 1 else f"parallel-{int(parallel)}"
    return {
        "batch_id": batch_id,
        "mode": mode,
        "parallel": int(parallel),
        "combos": combos,
        "task_counts": nested,
        "summary": {
            "combos": len(combos),
            "success": total_success,
            "fail": total_fail,
            "safety_violations": total_violations,
            "recovery_events": total_recoveries,
            "total_steps": total_steps,
        },
        "warnings": [
            (
                "batch mode is sequential (one isolated env per combo); parallel episodes not implemented"
                if int(parallel) <= 1
                else f"batch mode is {mode} (one isolated env per worker; no shared MjData; order-independent aggregation)"
            ),
            "recommendation lines inside per-combo reports are rule-based (heuristic_v1), not learned",
        ]
        + (
            [
                "cuda inference with parallel>1 shares no GPU context per worker; on 6GB cards prefer --parallel 1 for smolvla/groot cuda"
            ]
            if any(
                str((rep.get("inference_mode") or "")) == "cuda" for rep in combo_reports
            ) and int(parallel) > 1
            else []
        ),
    }
