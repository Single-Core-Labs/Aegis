# Aegis Documentation

Welcome to the Aegis Physical AI Harness documentation.

## For NVIDIA Stack Users — Start Here

| Document | Who It's For | What You'll Learn |
|---|---|---|
| **[NVIDIA Stack Manual](nvidia-stack-manual.md)** | Robotics engineers integrating with Isaac + ROS 2 + real Franka | End-to-end: WSL2 setup → Isaac Sim/Lab → ROS 2 bridge → USD authoring → hardware dry run |
| [Architecture](architecture.md) | Engineers extending the harness | Safety gateway, eval loop, telemetry schema, module map |
| [Safety Gateway](safety-gateway.md) | Safety / controls engineers | Per-joint limits, fallback tuning, adding a new robot |
| [Isaac Lab Guide](isaac-lab.md) | Sim engineers | USD authoring, PhysX setup, sim-to-real gap measurement |
| [ROS 2 Bridge](ros2-bridge.md) | Systems / deployment engineers | Topics, latency benchmarking, real-robot wiring |
| [Dashboard](dashboard.md) | Observability engineers | NDJSON → Prometheus / Grafana / OTel (`src/aegis/telemetry/otel.py`) |
| [Perf Tuning](perf-tuning.md) | Sim / deployment engineers | Isaac USD VRAM-safe for RTX 4050 6GB |
| [Batching](batching.md) | Platform engineers | Multi-robot/task spec (P3) |
| [Learned Reco](learned-recommendation.md) | ML engineers | Heuristic → learned recommendation spec (P3) |
| [Camera Calib](camera-calibration.md) | Perception engineers | Per-model calibration + domain randomization (P3) |

## Reference

| Document | Description |
|---|---|
| [`/README.md`](../README.md) | Project overview, quick start, CLI reference, benchmarks |
| [`/CONTEXT.md`](../CONTEXT.md) | Current phase gate, what’s done, what’s pending, NVIDIA partnership asks |
| [`/DESIGN.md`](../DESIGN.md) | Phase 1 design doc — schema, gateway, eval loop |
| [`/HARDWARE_CHECKLIST.md`](../HARDWARE_CHECKLIST.md) | Go/No-Go gates for real Franka exposure |
| [`/STEP6_ISAAC_VS_MUJOCO.md`](../STEP6_ISAAC_VS_MUJOCO.md) | Scaffold comparison — per-joint, Isaac fallback, rosbench |
| [`/PHASE_1_SUMMARY.md`](../PHASE_1_SUMMARY.md) · [`/PHASE_2_SUMMARY.md`](../PHASE_2_SUMMARY.md) | Phase completion evidence |

## Documentation Conventions

- **Honest by default** — if a backend is stubbed/mocked, the doc and the report say so explicitly
- **File references** use `path:line` (e.g., `src/aegis/safety/gateway.py:28`) for direct navigation
- Commands assume `uv` and are run from the repository root
