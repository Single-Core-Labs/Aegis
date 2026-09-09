# Dashboard & Observability — NDJSON → Prometheus / Grafana / OTel

NDJSON remains the source of truth (`src/aegis/telemetry/logger.py`). This doc adds an **exporter layer** that converts `outputs/<run_id>/` into vendor-neutral metrics.

## Quick Start

```bash
# After any eval:
uv run aegis eval --model scripted --episodes 3 --seed 42
# Run dir: outputs/run-20260909T120000Z

# Export Prometheus exposition (no extra deps):
python -m aegis.telemetry.otel --run-dir outputs/run-20260909T120000Z --prometheus

# Or via Python:
python -c "from aegis.telemetry.otel import prometheus_exposition; from pathlib import Path; print(prometheus_exposition(Path('outputs/run-20260909T120000Z')))"

# Generate Grafana dashboard JSON:
python -c "from aegis.telemetry.otel import export_dashboard; print(export_dashboard())"
# -> grafana/aegis_dashboard.json
```

## Metrics

| Metric | Type | Labels | Source |
|---|---|---|---|
| `aegis_episodes_success` | gauge | `run_id`, `sim` | `report.json:task_counts` |
| `aegis_safety_violations` | gauge | `run_id` | `report.json` |
| `aegis_recovery_events` | gauge | `run_id` | `report.json` |
| `aegis_latency_p50_ms` | gauge | `run_id` | `metrics.py:percentile` |
| `aegis_latency_p95_ms` | gauge | `run_id` | `metrics.py` |
| `aegis_gpu_hours` | gauge | `run_id`, `inference_mode` | Per-kernel sum of `inference_s` (cuda events when available) |
| `aegis_step_violation` | counter | `episode`, `step`, `type` | `steps.jsonl:violation.type` |

## OTel

`to_otel_metrics(run_dir)` returns a list of `{name, value, attrs}` points that can be pushed to any OTel Collector via OTLP/HTTP. No `opentelemetry` package required — the exporter is pure JSON and can be wrapped:

```python
from aegis.telemetry.otel import to_otel_metrics
from pathlib import Path
points = to_otel_metrics(Path("outputs/run-20260909T120000Z"))
# push to collector: POST /v1/metrics { "metrics": points }
```

## Grafana

`s​rc/aegis/telemetry/otel.py:grafana_dashboard_json()` returns a dashboard scaffold with panels for success, violations, latency p50/p95, GPU hours, and violation breakdown. Import `grafana/aegis_dashboard.json` into Grafana → Dashboards → Import.

## Prometheus

If `prometheus_client` is installed, you can expose `/metrics` directly:

```python
from prometheus_client import start_http_server, Gauge
# then update gauges from to_otel_metrics() in a loop
```

The built-in `prometheus_exposition()` already renders the text format without the client library, so scraping a static file works for offline evals.

## NDJSON remains primary

All exporters are derived — `steps.jsonl` + `episodes.jsonl` + `report.json` stay in `outputs/` and are the audit trail. Exporters never replace them.
