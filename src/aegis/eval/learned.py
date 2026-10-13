"""learned_v1 recommendation (data-driven, honest fallback).

Small calibrated scorer over the same ``extract_features()`` dict that
``heuristic_v1`` uses. Model file: ``assets/model/recommendation_v1.json``
(thresholds + weights learned offline from synthetic sweeps: scripted vs
random vs slow vs smolvla across franka / franka_uniform / franka_diag).

Rule: if the model file is missing or unparsable, fall back to
``heuristic_v1`` and say so in ``recommendation_model``. Reports therefore
carry ``heuristic_v1`` until a real trained file ships — never a fake
``learned_v1`` label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEARNED_MODEL = "learned_v1"
MODEL_PATH = Path("assets/model/recommendation_v1.json")


def _load_model(path: str | Path = MODEL_PATH) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def recommend_learned(
    summary: dict[str, Any],
    features: dict[str, Any],
    model: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Score with the learned model. Returns None when unavailable."""
    mdl = model if model is not None else _load_model()
    if not mdl or not mdl.get("trained", False):
        return None
    try:
        w = mdl.get("weights", {})
        score = float(mdl.get("bias", 0.0))
        for k, weight in w.items():
            v = features.get(k)
            if isinstance(v, (int, float)):
                score += float(weight) * float(v)
        labels: list[str] = mdl.get("labels", ["standin_expected_fail", "tighten_limits_or_fix_policy", "ready"])
        # 3-way head on score thresholds (calibrated offline; see assets doc).
        t = mdl.get("thresholds", [-0.5, 0.5])
        idx = 0 if score < t[0] else (1 if score < t[1] else 2)
        idx = max(0, min(idx, len(labels) - 1))
        # Sigmoid-ish confidence from margin to nearest threshold.
        margin = min(abs(score - th) for th in t) if t else abs(score)
        conf = min(0.99, max(0.51, 0.5 + margin))
        ev = [
            f"score {score:.2f}",
            f"success_rate {features.get('success_rate', 0):.2f}",
            f"violations/ep {features.get('violations_per_ep', 0):.2f}",
        ]
        return {"label": labels[idx], "confidence": round(conf, 3), "evidence": ev}
    except Exception:
        return None


def recommend_auto(
    summary: dict[str, Any], features: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Try learned_v1, else heuristic. Returns (structured, model_name)."""
    from aegis.eval.recommendation import RECOMMENDATION_MODEL, recommend

    learned = recommend_learned(summary, features)
    if learned is not None:
        return learned, LEARNED_MODEL
    return recommend(summary, features), RECOMMENDATION_MODEL
