from __future__ import annotations

import json
from typing import Any

from aegis.config.models import PhysicalAIYaml
from aegis.eval.metrics import summarize
from aegis.eval.runner import EpisodeResult


def _recommendation(m: dict) -> str:
    total = m["episodes"]
    if total == 0:
        return "no episodes ran — harness error, do not trust output"
    if m["successes"] == total:
        return "all episodes succeeded — harness ready to evaluate real policies"
    if m["safety_violations"] > 0 and m["recovery_events"] > 0 and m["successes"] == 0:
        return (
            f"policy repeatedly violated safety limits ({m['safety_violations']} "
            f"violations, {m['recovery_events']} recoveries); tighten safety limits or "
            "fix the policy before hardware exposure"
        )
    if m["truncated"] > total / 2:
        return "most failures are truncations — raise max_steps_per_episode"
    return "harness validated — stand-in policy expected to fail; evaluate a real policy next"


def build_report(
    cfg: PhysicalAIYaml, results: list[EpisodeResult], run_id: str
) -> dict[str, Any]:
    m = summarize(results)
    task_name = cfg.task.name
    warnings = [
        "safety limits are uniform per-joint; real robots need per-joint limits",
        "recommendation line is rule-based (4 rules), not learned",
    ]
    if m["episodes"] == 0:
        warnings.append("no episodes completed — check logs for errors")
    report = {
        "run_id": run_id,
        "model": cfg.model.name,
        "robot": cfg.robot.name,
        "sim": cfg.environment.sim,
        "task_counts": {task_name: {"success": m["successes"], "fail": m["fails"]}},
        "latency_p50_ms": m["inference_latency_p50_ms"],
        "latency_p95_ms": m["inference_latency_p95_ms"],
        "inference_mode": cfg.eval.inference_mode,
        "inference_budget_ms": cfg.eval.inference_budget_ms,
        "inference_budget_violations": m["inference_budget_violations"],
        "model_errors": m["model_errors"],
        "safety_violations": m["safety_violations"],
        "recovery_events": m["recovery_events"],
        "gpu_hours": _gpu_hours(cfg, m),
        "gpu_hours_note": (
            "wall-clock proxy for cuda runs (no per-kernel accounting)"
            if cfg.eval.inference_mode == "cuda"
            else "stub — no GPU used in this run"
        ),
        "total_steps": m["total_steps"],
        "total_duration_s": m["total_duration_s"],
        "recommendation": _recommendation(m),
        "warnings": warnings,
    }
    return report


def _gpu_hours(cfg: PhysicalAIYaml, m: dict) -> float:
    if cfg.eval.inference_mode != "cuda":
        return 0.0
    return round(m["total_duration_s"] / 3600.0, 6)


def print_summary(report: dict[str, Any]) -> str:
    task, counts = next(iter(report["task_counts"].items()))
    lines = [
        f"task          : {task}",
        f"episodes      : {sum(counts.values())}   "
        f"(success {counts['success']} / fail {counts['fail']})",
        f"inference     : p50 {report['latency_p50_ms']} ms  p95 {report['latency_p95_ms']} ms   (per-step, {report['inference_mode']})",
        f"safety        : violations {report['safety_violations']}  recoveries {report['recovery_events']}"
        f"  (budget {report['inference_budget_violations']}, model errors {report['model_errors']})",
        f"gpu hours     : {report['gpu_hours']} ({report['gpu_hours_note']})",
        f"recommendation: {report['recommendation']}",
    ]
    text = "\n".join(lines)
    for w in report.get("warnings", []):
        text += f"\nwarning       : {w}"
    return text


def write_report_json(path, report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)