"""Sequential multi-combo batching (Beyond Phase 3, MuJoCo-safe slice).

Runs the Cartesian product ``robots x models x tasks`` sequentially — one
isolated env + gateway per combo (no shared ``MjData``) — and writes an
aggregated batch ``report.json``. Parallel episodes and Isaac ``num_envs``
are explicitly NOT implemented here (deferred until real USD exists, per
``docs/batching.md``).

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
) -> dict[str, Any]:
    """Aggregate per-combo ``report.json`` dicts into one batch report.

    ``task_counts`` becomes nested ``task_counts[task][robot][model]`` plus a
    ``summary`` rollup. Single-combo runs keep their flat per-combo reports
    untouched on disk.
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
    return {
        "batch_id": batch_id,
        "mode": "sequential",
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
            "batch mode is sequential (one isolated env per combo); parallel episodes not implemented",
            "recommendation lines inside per-combo reports are rule-based (heuristic_v1), not learned",
        ],
    }
