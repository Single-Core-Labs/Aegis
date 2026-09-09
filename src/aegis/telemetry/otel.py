from __future__ import annotations

"""OTel / Prometheus exporter for Aegis NDJSON telemetry.

Reads outputs/<run_id>/steps.jsonl + episodes.jsonl + report.json and
exposes:
  - Prometheus text exposition at /metrics (if prometheus_client is installed)
  - OTel-compatible JSON for external collectors
  - Grafana dashboard JSON scaffold

No hard dependency: if prometheus_client / opentelemetry are missing, the
module still provides pure-Python JSON export and a static dashboard file.
"""

import json
from pathlib import Path
from typing import Any


def parse_run(run_dir: Path) -> dict[str, Any]:
    """Load report + aggregates from NDJSON."""
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8")) if (run_dir / "report.json").exists() else {}
    episodes = []
    steps = []
    if (run_dir / "episodes.jsonl").exists():
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("event") == "episode_end":
                episodes.append(j)
    if (run_dir / "steps.jsonl").exists():
        for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            steps.append(json.loads(line))
    return {"report": report, "episodes": episodes, "steps": steps, "run_id": run_dir.name}


def to_otel_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Convert NDJSON to OTel-like metric points (vendor-neutral JSON)."""
    data = parse_run(run_dir)
    report = data["report"]
    points: list[dict[str, Any]] = []
    # Gauge: success rate
    if report:
        total = sum(report.get("task_counts", {}).get("pick-place", {}).values()) if report.get("task_counts") else report.get("episodes", 0)
        # episodes/success from report
        points.append({"name": "aegis_episodes_success", "value": report.get("task_counts", {}).get("pick-place", {}).get("success", 0), "attrs": {"run_id": data["run_id"], "sim": report.get("sim", "mujoco")}})
        points.append({"name": "aegis_safety_violations", "value": report.get("safety_violations", 0), "attrs": {"run_id": data["run_id"]}})
        points.append({"name": "aegis_recovery_events", "value": report.get("recovery_events", 0), "attrs": {"run_id": data["run_id"]}})
        points.append({"name": "aegis_latency_p50_ms", "value": report.get("latency_p50_ms", 0), "attrs": {"run_id": data["run_id"]}})
        points.append({"name": "aegis_latency_p95_ms", "value": report.get("latency_p95_ms", 0), "attrs": {"run_id": data["run_id"]}})
        points.append({"name": "aegis_gpu_hours", "value": report.get("gpu_hours", 0), "attrs": {"run_id": data["run_id"], "inference_mode": report.get("inference_mode", "cpu")}})
    # Per-step histograms (sampled)
    for s in data["steps"][:1000]:  # cap for exporter
        if s.get("violated"):
            points.append({"name": "aegis_step_violation", "value": 1, "attrs": {"episode": s.get("episode"), "step": s.get("step"), "type": (s.get("violation") or {}).get("type", "unknown")}})
    return points


def prometheus_exposition(run_dir: Path) -> str:
    """Render Prometheus text format (no client lib required)."""
    metrics = to_otel_metrics(run_dir)
    lines: list[str] = []
    # Aggregate by metric name
    from collections import defaultdict

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        grouped[m["name"]].append(m)
    for name, pts in grouped.items():
        # Use first point's value for gauge, sum for counters
        if name in ("aegis_safety_violations", "aegis_recovery_events", "aegis_episodes_success"):
            val = sum(p["value"] for p in pts) if len(pts) > 1 else pts[0]["value"]
        elif name == "aegis_step_violation":
            val = len(pts)
        else:
            val = pts[0]["value"] if pts else 0
        lines.append(f"# HELP {name} Aegis harness metric")
        lines.append(f"# TYPE {name} gauge")
        # add label for run_id if present
        attrs = pts[0].get("attrs", {}) if pts else {}
        if attrs:
            label_str = ",".join(f'{k}=\"{v}\"' for k, v in attrs.items() if k in ("run_id", "sim", "inference_mode"))
            if label_str:
                lines.append(f"{name}{{{label_str}}} {val}")
            else:
                lines.append(f"{name} {val}")
        else:
            lines.append(f"{name} {val}")
    return "\n".join(lines) + "\n"


def grafana_dashboard_json() -> dict[str, Any]:
    """Return a Grafana dashboard JSON scaffold for Aegis runs."""
    return {
        "title": "Aegis — Physical AI Harness",
        "tags": ["aegis", "physical-ai", "isaac", "ros2"],
        "timezone": "browser",
        "panels": [
            {"title": "Success Rate", "type": "stat", "targets": [{"expr": "aegis_episodes_success"}]},
            {"title": "Safety Violations", "type": "stat", "targets": [{"expr": "aegis_safety_violations"}]},
            {"title": "Recovery Events", "type": "stat", "targets": [{"expr": "aegis_recovery_events"}]},
            {"title": "Inference Latency p50/p95 (ms)", "type": "timeseries", "targets": [{"expr": "aegis_latency_p50_ms"}, {"expr": "aegis_latency_p95_ms"}]},
            {"title": "GPU Hours (per-kernel)", "type": "stat", "targets": [{"expr": "aegis_gpu_hours"}]},
            {"title": "Step Violations by Type", "type": "barchart", "targets": [{"expr": "aegis_step_violation"}]},
        ],
        "templating": {"list": [{"name": "run_id", "type": "query", "query": "label_values(aegis_episodes_success, run_id)"}]},
    }


def export_dashboard(path: Path = Path("grafana/aegis_dashboard.json")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(grafana_dashboard_json(), indent=2), encoding="utf-8")
    return path
