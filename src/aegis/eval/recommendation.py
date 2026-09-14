"""Learned-recommendation interface (Beyond Phase 3, incremental slice).

Current model is ``heuristic_v1`` — the same 4 deterministic rules as
``aegis.eval.report._recommendation``. This module adds:

- ``extract_features()``: per-run feature dict that a future ``learned_v1``
  classifier can train on (spec: ``docs/learned-recommendation.md``).
- ``recommend()``: structured ``{label, confidence, evidence}`` alongside the
  legacy human-readable string (which stays the backward-compatible field).

Honesty contract: ``report.json`` always carries
``recommendation_model: "heuristic_v1"`` until a real learned model ships;
grasp/ROS features that need per-step aggregation are ``None`` (not zero).
"""

from __future__ import annotations

from typing import Any

RECOMMENDATION_MODEL = "heuristic_v1"


def extract_features(
    summary: dict[str, Any],
    *,
    gpu_hours: float = 0.0,
    dr_enabled: bool = False,
) -> dict[str, Any]:
    episodes = int(summary.get("episodes", 0) or 0)
    successes = int(summary.get("successes", 0) or 0)
    fails = int(summary.get("fails", 0) or 0)
    violations = int(summary.get("safety_violations", 0) or 0)
    recoveries = int(summary.get("recovery_events", 0) or 0)
    return {
        "success_rate": (successes / episodes) if episodes else 0.0,
        "fail_rate": (fails / episodes) if episodes else 0.0,
        "violations_per_ep": (violations / episodes) if episodes else 0.0,
        "recoveries_per_ep": (recoveries / episodes) if episodes else 0.0,
        "budget_violations": int(summary.get("inference_budget_violations", 0) or 0),
        "model_errors": int(summary.get("model_errors", 0) or 0),
        "truncated": int(summary.get("truncated", 0) or 0),
        "truncated_rate": (float(summary.get("truncated", 0) or 0) / episodes) if episodes else 0.0,
        "timeouts": int(summary.get("timeouts", 0) or 0),
        "latency_p50_ms": float(summary.get("inference_latency_p50_ms", 0.0) or 0.0),
        "latency_p95_ms": float(summary.get("inference_latency_p95_ms", 0.0) or 0.0),
        "total_inference_s": float(summary.get("total_inference_s", 0.0) or 0.0),
        "total_steps": int(summary.get("total_steps", 0) or 0),
        "gpu_hours": float(gpu_hours),
        "dr_enabled": bool(dr_enabled),
        # Per-step aggregations for learned_v1 — not yet computed from
        # steps.jsonl, so None (unknown) rather than a fabricated 0.0.
        "grasp_contact_rate": None,
        "grasp_contact_force_p50": None,
        "dist_to_target_min": None,
        "ros_latency_p95_ms": None,
    }


def recommend(summary: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Map the 4 heuristic rules to (label, confidence, evidence).

    Confidence is 1.0: the rules are deterministic, and the report flags the
    model as ``heuristic_v1`` so nobody mistakes it for a learned classifier.
    """
    total = int(summary.get("episodes", 0) or 0)
    successes = int(summary.get("successes", 0) or 0)
    violations = int(summary.get("safety_violations", 0) or 0)
    recoveries = int(summary.get("recovery_events", 0) or 0)
    truncated = int(summary.get("truncated", 0) or 0)
    if total == 0:
        return {"label": "error", "confidence": 1.0, "evidence": ["no episodes ran"]}
    if successes == total:
        return {"label": "ready", "confidence": 1.0, "evidence": [f"{successes}/{total} success"]}
    if violations > 0 and recoveries > 0 and successes == 0:
        return {
            "label": "tighten_limits_or_fix_policy",
            "confidence": 1.0,
            "evidence": [
                f"violations {violations} ({features.get('violations_per_ep', 0):.2f}/ep)",
                f"recoveries {recoveries}",
            ],
        }
    if truncated > total / 2:
        return {
            "label": "raise_max_steps",
            "confidence": 1.0,
            "evidence": [f"truncated {truncated}/{total}"],
        }
    return {
        "label": "standin_expected_fail",
        "confidence": 1.0,
        "evidence": [f"success {successes}/{total}"],
    }
