# Learned Recommendation — Spec (P3)

**Current:** 4-rule heuristic `src/aegis/eval/report.py:11` — deterministic, flagged as `warnings: ["recommendation line is rule-based (4 rules), not learned"]`.

```python
def _recommendation(m):
    if total == 0: return "no episodes ran — harness error"
    if successes == total: return "all succeeded — ready to evaluate real policies"
    if violations and recoveries and successes == 0: return "tighten safety limits or fix policy"
    if truncated > total/2: return "raise max_steps_per_episode"
    return "harness validated — stand-in expected to fail"
```

## Future: Learned Model

**Goal:** Replace heuristic with a calibrated classifier that predicts `{ready, tighten_limits, fix_policy, raise_steps, investigate_gap}` with confidence and evidence pointers.

## Features (per run)

- `success_rate`, `violations_per_ep`, `recoveries_per_ep`, `budget_violations`, `model_errors`
- `latency_p50/p95`, `total_inference_s`, `gpu_hours` (per-kernel)
- `grasp_has_contacts` rate, `grasp_contact_force` p50, `dist_to_target` min
- `ros_latency_p95` (when available)

## Training Data

- Synthetic: sweep `scripted` vs `random` vs `slow_policy` vs `smolvla` across limit configs (`franka.yaml` vs `franka_uniform.yaml` vs `franka_diag.yaml`) and seeds — reports are the dataset.
- Labels: hand-labeled by harness engineer (not learned from prod).

## Interface

```python
# Future: src/aegis/eval/recommendation.py
def recommend(report: dict) -> dict:
    # model.pkl loaded from assets/model/recommendation_v1.pkl if present,
    # else fall back to heuristic
    return {"label": "tighten_limits", "confidence": 0.87, "evidence": ["violations 33/ep", "fallback 1/3"]}
```

Report gains `recommendation_model: "heuristic_v1" | "learned_v1"` + `confidence` — honest about which path was used.

## Not Building Now

Heuristic is correct and auditable; learned model waits for enough real Isaac + ROS2 runs to train on.
