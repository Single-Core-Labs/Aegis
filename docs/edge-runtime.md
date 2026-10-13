# Edge Runtime — Architecture (docs-first)

**Status:** Design only. No `src/aegis_edge/` code yet. The sim gateway
(`src/aegis/safety/`) is the reference implementation; the edge runtime is a
port, not a rewrite.

## Goals

- Same checks (NaN -> effort -> measured vel -> measured force), same
  `recovery_steps`/`recovery_mode` semantics, microsecond latency on Jetson Orin.
- Safety config as versioned artifact: deploy `policy v3 + franka.yaml v3`
  together; roll back both. OTA limit updates without touching the policy.
- Policy server / sidecar: policy process sends actions over localhost/ROS 2;
  gateway cuts off over-budget inference (`inference_budget_ms`) and falls back.
- Shadow mode: new policy sees real sensors, gateway logs but never executes;
  promote after N violation-free episodes. Canary per robot.
- Continuous validation: every real action logged; violation-rate spike pages
  before it becomes an accident. Fleet stream -> anomaly detection -> drift
  (sim profile vs real profile) -> compliance report (healthcare/manufacturing).

## Migration path

1. Keep Python for dev/eval (today).
2. Rust/C++ sidecar gateway (`ndarray`, `r2r` ROS 2, `cudarc` interop) — see
   `DOMAINS_AND_TECH_STACK.md` Part 3 for sketches.
3. Jetson Orin deploy: Isaac + ROS 2 + gateway on device.
4. Fleet dashboard (Prometheus/Grafana) + compliance export.

## Acceptance (when built)

- Versioned `franka.yaml` deploy/rollback demo on 1 robot.
- Shadow vs active A/B report with `sim: hardware-shadow`.
- p95 gateway latency < 0.1ms on Orin (CUDA checks).
