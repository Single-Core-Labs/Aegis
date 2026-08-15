from __future__ import annotations

import numpy as np

from aegis.eval.runner import EpisodeResult


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), q))


def summarize(results: list[EpisodeResult]) -> dict:
    latencies = [v for r in results for v in r.inference_latencies_s]
    successes = sum(1 for r in results if r.success)
    fails = sum(1 for r in results if not r.success)
    timeouts = sum(1 for r in results if r.fail_reason == "timeout")
    truncated = sum(1 for r in results if r.fail_reason == "truncated")
    violations = sum(r.violations for r in results)
    recoveries = sum(r.recoveries for r in results)
    budget_violations = sum(r.budget_violations for r in results)
    model_errors = sum(r.model_errors for r in results)
    total_steps = sum(r.steps for r in results)
    total_duration_s = sum(r.duration_s for r in results)

    return {
        "episodes": len(results),
        "successes": successes,
        "fails": fails,
        "timeouts": timeouts,
        "truncated": truncated,
        "total_steps": total_steps,
        "total_duration_s": round(total_duration_s, 3),
        "inference_latency_p50_ms": round(percentile(latencies, 50) * 1e3, 3),
        "inference_latency_p95_ms": round(percentile(latencies, 95) * 1e3, 3),
        "safety_violations": violations,
        "recovery_events": recoveries,
        "inference_budget_violations": budget_violations,
        "model_errors": model_errors,
    }