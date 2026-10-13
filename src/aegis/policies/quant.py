"""Quantization harness (int8 vs none) for VLA policies.

Measures accuracy-vs-latency per quant level through the same safety gateway:
runs the eval reports you already have and attaches a quant comparison block.
No new inference backend here — SmolVLA/GR00T ``quantize`` flags already exist
in config; this module standardizes the A/B so ``--quantize int8`` claims are
measured, not asserted. Distillation (teacher->student) reuses
``train/bc_trainer.py`` on teacher rollouts; see docs/perf-tuning.md.
"""

from __future__ import annotations

from typing import Any


def compare_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare per-quant-level reports: {level: report.json dict}."""
    out: dict[str, Any] = {"levels": {}, "note": "p50/p95 per-step inference latency + success counts per quant level"}
    for level, rep in reports.items():
        tc = rep.get("task_counts", {})
        succ = sum(v.get("success", 0) for v in tc.values()) if tc else 0
        fail = sum(v.get("fail", 0) for v in tc.values()) if tc else 0
        out["levels"][level] = {
            "success": succ,
            "fail": fail,
            "latency_p50_ms": rep.get("latency_p50_ms"),
            "latency_p95_ms": rep.get("latency_p95_ms"),
            "safety_violations": rep.get("safety_violations", 0),
            "gpu_hours": rep.get("gpu_hours", 0.0),
        }
    return out
