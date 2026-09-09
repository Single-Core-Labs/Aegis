# Aegis — Physical AI Safety Harness & Evaluation Framework

<div align="center">

**Safety-gated evaluation for Vision-Language-Action (VLA) policies on physical robots**

*Built for the NVIDIA Physical AI Stack — Isaac Sim · Isaac Lab · ROS 2 · CUDA*

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Isaac Sim](https://img.shields.io/badge/Isaac_Sim-6.0.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac/sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.x-76B900)](https://isaac-sim.github.io/IsaacLab)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-20_passed-brightgreen)](#testing)

[Overview](#overview) · [Quick Start](#quick-start) · [Documentation](docs/README.md) · [NVIDIA Stack Guide](docs/nvidia-stack-manual.md) · [Architecture](#architecture) · [Benchmarks](#benchmarks)

</div>

---

## Overview

**Aegis** is a safety and evaluation harness that sits between *any* robot AI policy and *any* robot — simulated or real. Every action is gated through a hardware-aware safety layer before it reaches actuators, and every episode produces a reproducible, auditable report.

```
┌─────────────┐      8-D joint-vel + gripper     ┌─────────────────────────┐      safe action     ┌──────────────────────┐
│  AI Policy  │ ───────────────────────────────▶ │  AEGIS Safety Gateway  │ ───────────────────▶ │  Robot               │
│ VLA / Classic│                                 │  + Eval Harness        │                      │  MuJoCo / Isaac Lab  │
│ SmolVLA etc │ ◀─────────────────────────────── │  Fallback · Telemetry  │ ◀─────────────────── │  / Real Franka       │
└─────────────┘      observation + images        └─────────────────────────┘      state + sensors  └──────────────────────┘
```

### Why Aegis?

| Challenge Today | How Aegis Solves It |
|---|---|
| No standard safety layer — VLA can command excessive velocity/force and damage hardware | **Safety Gateway** blocks violations, engages PID-to-home fallback for 50 steps, then resumes. No bypass path exists by construction. |
| Every team builds bespoke eval scripts — results are incomparable | **Unified eval harness** — `aegis eval` runs N episodes, emits `report.json` + NDJSON traces with identical schema across sims |
| Sim-to-real gap is unmeasured — models that work in training sim fail silently elsewhere | **Cross-sim evaluation** — same policy, same limits, MuJoCo vs. Isaac Lab. Gap is quantified, not guessed |

> **Design principle — Honest by default.** If Isaac Sim is not installed, the report says `isaaclab-fallback-mujoco` with a `RuntimeWarning`. If a value is stubbed, the report's `warnings` field says so. No fabricated numbers, ever.

---

## Key Features

- **Hardware-aware Safety Gateway** — Per-joint velocity `[2.175 .. 2.61] rad/s` and effort `[87 .. 12] Nm` limits (Franka Panda), NaN/Inf rejection, measured `qvel` + `qfrc` checks, configurable `recovery_steps` / `recovery_mode`
- **Deterministic Evaluation** — Seeded episodes (`seed + episode_id`), reproducible trajectories, p50/p95 latency instrumentation on every step
- **Multi-Sim Backend** — `MuJoCo 3.x` (Menagerie Franka) for lightweight CI + `Isaac Lab / Isaac Sim 6.0.1` (PhysX, USD) for photorealistic VLA evaluation
- **VLA-Ready** — Native `SmolVLA-450M` (`lerobot/smolvla_libero`) integration: checkpoint tokenizer/normalizer applied verbatim, 7-D cartesian-delta → 8-D joint-vel via DLS resolved-rate, chunk-50 inference with budget enforcement
- **ROS 2 Bridge** — `aegis ↔ ROS 2` topics (`/aegis/action`, `/aegis/state`, `/aegis/safety`) with `aegis rosbench` latency profiling (mock + real `rclpy`)
- **Auditable Outputs** — Every run writes `outputs/<run_id>/{report.json, run.json, episodes.jsonl, steps.jsonl, trajectory.jsonl}`

---

## System Requirements

| Component | Requirement | Notes |
|---|---|---|
| **OS** | Ubuntu 22.04 (native or WSL2) · Windows 11 + WSL2 | Fully-inside-WSL2 recommended for ROS 2 + Isaac |
| **Python** | `3.12` | Managed via `uv` |
| **GPU** | NVIDIA RTX with CUDA 12.8, ≥6 GB VRAM | 16 GB recommended for Isaac Sim 6.0.1 VRAM-safe scenes |
| **CUDA** | `12.8` + `torch 2.11+cu128` | Installed via `uv` from `pytorch-cu128` index |
| **Sim (lightweight)** | MuJoCo `≥3.0` | No GPU required, runs in CI |
| **Sim (photoreal)** | Isaac Sim `6.0.1` + Isaac Lab `2.x` | Requires VRAM-safe USD (see [NVIDIA Stack Guide](docs/nvidia-stack-manual.md)) |
| **Middleware** | ROS 2 Humble | `rclpy` — auto-mocks if absent |

---

## Installation

### 1 — Clone and create environment

```bash
git clone https://github.com/your-org/physical-ai-harness.git
cd physical-ai-harness

# uv will create a Python 3.12 venv and install all deps
uv sync
```

> **Windows + WSL2 users:** Follow [docs/nvidia-stack-manual.md — WSL2 Setup](docs/nvidia-stack-manual.md#1-wsl2--ubuntu-2204-setup) for GPU passthrough verification (`nvidia-smi` inside WSL2) before installing Isaac.

### 2 — Verify installation

```bash
uv run aegis validate
# => config OK

uv run aegis validate --robot franka_uniform   # legacy uniform limits A/B
```

### 3 — Optional: Isaac Sim + ROS 2

See the full **[NVIDIA Stack Manual](docs/nvidia-stack-manual.md)** for:

- Installing Isaac Sim 6.0.1 + Isaac Lab inside WSL2
- Installing ROS 2 Humble (`apt` vs `Docker` path)
- Authoring the pick-place USD scene
- VRAM-safe scene defaults for 6 GB GPUs

---

## Quick Start

### Evaluate the scripted baseline (should succeed)

```bash
uv run aegis eval --model scripted --episodes 3 --seed 42
```

Expected output (per-joint limits, post Phase 3.3):

```
task          : pick-place
episodes      : 3   (success 3 / fail 0)
inference     : p50 0.12 ms  p95 0.45 ms   (per-step, cpu)
safety        : violations 0  recoveries 0  (budget 0, model errors 0)
gpu hours     : 0.0 (stub — no GPU used in this run)
recommendation: all episodes succeeded — harness ready to evaluate real policies
warning       : recommendation line is rule-based (4 rules), not learned
report        : outputs/run-20260909T120000Z/report.json
```

### Evaluate the random policy (negative control — should fail safely)

```bash
uv run aegis eval --model random --episodes 3 --seed 7
# => 0/3 success, ~4 violations / 2 recoveries (per-joint) — fallback engaged
```

### Cross-sim evaluation (Isaac Lab scaffold)

```bash
# Without Isaac Sim installed: falls back to MuJoCo with explicit warning + info["sim"]="isaaclab-fallback-mujoco"
uv run aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42

# With Isaac Sim 6.0.1: runs real PhysX + USD cameras (see docs)
uv run aegis eval --sim isaaclab --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600
```

### ROS 2 bridge latency benchmark

```bash
uv run aegis rosbench --mock --n 20    # in-memory, no ROS 2 required
# ros bridge: mock  p50 0.000 ms  p95 0.002 ms

# After ROS 2 Humble install (inside WSL2):
uv run aegis rosbench --real --n 100
```

### Real VLA evaluation (requires GPU + checkpoint download)

```bash
uv run aegis eval --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600
```

---

## CLI Reference

```
aegis eval      Run a safety-gated evaluation and emit a report
aegis validate  Validate config and asset paths without running
aegis rosbench  Benchmark ROS 2 bridge publish latency (mock vs real)
```

| Flag | Description | Default |
|---|---|---|
| `--model` | Model config in `configs/models/<name>.yaml` (`random` / `scripted` / `smolvla_libero`) | `random` |
| `--robot` | Robot config in `configs/robots/<name>.yaml` (`franka` / `franka_uniform`) | `franka` |
| `--sim` | Simulator backend (`mujoco` / `isaaclab`) | `mujoco` |
| `--tasks` | Task (`pick-place`) | `pick-place` |
| `--episodes` | Number of episodes `1..1000` | `10` (from YAML) |
| `--seed` | Base seed (episode seed = `seed + episode_id`) | `42` |
| `--max-steps` | Max steps per episode | `2500` |
| `--inference-mode` | `cpu` / `cuda` | `cpu` |
| `--inference-budget-ms` | Per-step budget; over-budget → fallback | `2000.0` |
| `--config` | Root config file | `physical-ai.yaml` |
| `--output-dir` | Output directory | `./outputs` |
| `--run-id` | Override auto-generated `run-YYYYMMDDTHHMMSSZ` | auto |
| `--verbose` | Debug logging to stderr | `false` |

Exit codes: `0` success · `2` config error · `3` internal error (partial logs preserved).

---

## Architecture

```
configs/{robots,models,tasks}/     Declarative YAML — no hardcoded params in code
        │
        ▼
   physical-ai.yaml  ──►  Pydantic validation (extra="forbid")  ──►  aegis validate (exit 0/2)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  EvalRunner  (src/aegis/eval/runner.py)                         │
│    policy.act(obs)  ──timed──►  SafetyGateway.filter()           │
│                                    │                             │
│                              ┌──────┴──────┐                     │
│                              │  Checks     │  1. NaN/Inf         │
│                              │  (checks.py)│  2. Effort clamp    │
│                              │             │  3. Measured qvel   │
│                              │             │  4. Measured qfrc   │
│                              └──────┬──────┘                     │
│                                     │ violation?                 │
│                          ┌──────────┴──────────┐                 │
│                          │  Fallback           │  PID-to-home     │
│                          │  (fallback.py)      │  50 steps,       │
│                          │  bounded 0.3 rad/s  │  resume mode     │
│                          └──────────┬──────────┘                 │
│                                     ▼                             │
│                              env.step(safe_action)                │
│                                     │                             │
│                              RunLogger  ──►  outputs/<run_id>/    │
│                                NDJSON + report.json               │
└─────────────────────────────────────────────────────────────────┘
```

### Module Map

| Module | Path | Responsibility |
|---|---|---|
| CLI | `src/aegis/cli.py` | Typer entrypoint, backend selection, error handling |
| Config | `src/aegis/config/models.py` · `loader.py` | Pydantic schema, YAML loading, per-joint `float\|list[7]` limits |
| Env — MuJoCo | `src/aegis/envs/mujoco_pick_place.py` | Franka Panda (Menagerie) + cube + target, 3 cameras `256×256` |
| Env — Isaac | `src/aegis/envs/isaac_pick_place.py` | Isaac Lab scaffold — real USD/PhysX when available, fallback otherwise |
| Policies | `src/aegis/policies/` | `random`, `scripted` (6-phase DLS IK), `smolvla` (450M, chunk-50) |
| Safety | `src/aegis/safety/` | `checks.py` · `gateway.py` · `fallback.py` |
| Telemetry | `src/aegis/telemetry/` | `logger.py` (NDJSON) · `timing.py` (`perf_counter_ns`) |
| ROS 2 | `src/aegis/ros2/bridge.py` | `RosBridge` — auto-mock / real `rclpy`, `benchmark_latency()` |

---

## Configuration

All tuning is declarative YAML. No robot/model constants are hardcoded.

```yaml
# physical-ai.yaml (root)
model_name: random
robot_name: franka          # -> configs/robots/franka.yaml
task_name: pick-place       # -> configs/tasks/pick-place.yaml

eval:
  episodes: 10
  seed: 42
  max_steps_per_episode: 2500
  time_step: 0.02
  inference_mode: cpu       # cpu | cuda
  inference_budget_ms: 2000.0

environment:
  sim: mujoco               # mujoco | isaaclab
  scene_mjcf: assets/scenes/pick_place.xml
  control_mode: joint_velocity
```

Per-joint safety limits (`configs/robots/franka.yaml`):

```yaml
safety:
  max_velocity: [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]  # rad/s j1..j7
  max_force: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]        # Nm j1..j7
  max_effort_action: 1.0
  reject_nan_actions: true
  action_clamp: reject
  recovery_steps: 50
  recovery_mode: resume
```

---

## Benchmarks

Reproducible on `seed 42` / `seed 7` (deterministic, `seed + episode_id`):

| Simulator | Policy | Episodes | Success | Violations | Recoveries | Notes |
|---|---|---|---|---|---|---|
| MuJoCo | `scripted` | 3 | **3/3** | 0 | 0 | Smoke-test baseline, per-joint limits |
| MuJoCo | `random` | 3 | 0/3 | 4 | 2 | Negative control — fallback engaged (was 145 with legacy uniform `1.0 rad/s`) |
| Isaac Lab (fallback) | `scripted` | 3 | **3/3** | 0 | 0 | `RuntimeWarning` + `info["sim"]="isaaclab-fallback-mujoco"` — honest |
| MuJoCo | `smolvla_libero` (cuda) | 3 | 0/3 | 33/ep | 33/ep | 1 budget + 32 velocity on `0.6 rad/s` uniform; `0/3` even at `5.0 rad/s` — vision domain gap |

> SmolVLA zero-shot mis-localizes in MuJoCo (hand `0.16m → 0.6m` from cube, gripper closes in mid-air). Re-evaluation in Isaac Lab with LIBERO-matched cameras/lighting is the remaining Phase 3 milestone.

---

## Testing

```bash
# All tests (20)
python -m pytest -q

# Subsets
python -m pytest tests/test_eval.py -q          # config + determinism
python -m pytest tests/test_policy_model.py -q  # budget, adapter math, SmolVLA config
python -m pytest tests/test_seed_sweep.py -q    # scripted robustness: 5 seeds × 5 eps, ≥90% success
```

---

## Industries

Harness is **industry-agnostic** — any domain that puts a learned policy on a physical robot. Value is highest where failure cost is high.

| Tier | Industries | Why Aegis | Example Task |
|---|---|---|---|
| **P0 — Healthcare** | Hospitals, labs, elder-care, pharma, surgical assist, rehab | Human proximity, sterile contact, 0-tolerance for force/velocity violation; audit trail `report.json` for compliance | Lab vial/tray pick-place, assistive feeding, instrument handover — `contact-force >0.5N` + `lift` proves gentle grasp |
| **P0 — Manufacturing** | Automotive, electronics/PCB, precision assembly | High-cost gear damage; needs per-joint `[2.175..2.61] rad/s, [87..12] Nm` enforcement + deterministic re-run | PCB component place, bin picking |
| **P1 — Logistics / Warehousing** | E-commerce, 3PL, fulfillment | High throughput, sim-to-real camera/lighting gap kills success | Parcel pick-place to tote |
| **P1 — Food / Agriculture** | Food processing, harvesting | Deformable objects, hygiene, variable lighting — needs domain randomization | Produce handling |
| **P2 — Retail / Hospitality** | Stores, kitchens, hotels | Human-collaborative, front-of-house | Shelf restocking |
| **P2 — Construction / Field** | Inspection, material handling | Harsh env, fallback-to-home critical | Block placement |

> **Positioning:** Healthcare is the strongest NVIDIA Physical AI narrative (safety + auditability), but the same `Policy → Gateway → MuJoCo/Isaac/Real` (`PRD.md:5`) serves all above — only `configs/robots/` + `configs/tasks/` + `cameras` change. For healthcare, **start with lab automation** (lowest regulatory barrier) before surgical.

## Project Status & Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | MuJoCo POC — safety gateway, scripted/random policies, eval harness, reports | ✅ Complete |
| **Phase 2** | Real model — SmolVLA-450M integration, inference budget, DLS adapter | ✅ Complete |
| **Phase 3 scaffold** | Per-joint limits, ROS 2 mock bridge, Isaac Lab fallback, hardware checklist | ✅ Complete |
| **Phase 3 real runtime** | ROS 2 Humble (real `rclpy`), Isaac Sim USD + PhysX, hardware dry run | 🔲 Pending — requires WSL2 ROS 2 install + 16 GB VRAM or NVIDIA VRAM-safe guidance |

See [docs/nvidia-stack-manual.md](docs/nvidia-stack-manual.md) for the NVIDIA stack integration path.

**Beyond Phase 3:** Multi-robot batching · Learned recommendation · Prometheus/Grafana observability · RL training & dataset collection · Contact-force grasp detection · Domain randomization

---

## Documentation

| Document | Description |
|---|---|
| [NVIDIA Stack Manual](docs/nvidia-stack-manual.md) | **Start here for NVIDIA users** — WSL2, Isaac Sim/Lab, ROS 2, USD authoring, hardware bring-up |
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/architecture.md](docs/architecture.md) | Deep dive: gateway, eval loop, telemetry schema |
| [docs/safety-gateway.md](docs/safety-gateway.md) | Safety limits, fallback tuning, adding a new robot |
| [docs/isaac-lab.md](docs/isaac-lab.md) | Isaac Lab USD authoring and sim-to-real gap measurement |
| [docs/ros2-bridge.md](docs/ros2-bridge.md) | ROS 2 topics, latency benchmarking, real-robot wiring |
| [docs/dashboard.md](docs/dashboard.md) | Observability: NDJSON → Prometheus / Grafana / OTel |
| [docs/perf-tuning.md](docs/perf-tuning.md) | Isaac USD VRAM-safe tuning for 6 GB |

---

## Honest Implementation Notes

1. **Velocity control is torque control.** The 8-D action (`7× joint vel + gripper`, rad/s) is integrated to a position target and tracked by `tau = qfrc_bias + 40·err + 5·vel_err` — gravity-compensated PD, not a perfect velocity servo.
2. **Per-joint limits are now default** (`franka.yaml`); legacy uniform `1.0 rad/s` kept as `franka_uniform.yaml` for A/B. Reports include `warnings` when uniform is active.
3. **Recommendation is rule-based** (4 hand-written rules), not learned.
4. **GPU hours** is `0.0` on `cpu` and per-kernel sum of inference time (`total_inference_s/3600`, CUDA events when available) on `cuda`.
5. **Success is `grasped_ever && dist≤0.05m`** — grasp = `lifted (z>table+0.025)` **AND** finger-object contacts with `mj_contactForce>0.5N` (sticky, logged as `grasp_has_contacts/force`).
6. **Determinism** via `seed + episode_id` and deterministic policies (`tests/test_eval.py`).
7. **Gripper** is a tendon motor `0..255` (open→close) remapped from the Menagerie model.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

- Franka Panda MJCF from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (MIT)
- [LeRobot](https://github.com/huggingface/lerobot) + `lerobot/smolvla_libero` checkpoint
- Built for the [NVIDIA Isaac](https://developer.nvidia.com/isaac) Physical AI ecosystem

---

<div align="center">

**Aegis** — *Because every action that reaches hardware must be safe, measured, and honest.*

`aegis eval --model <policy> --sim <mujoco|isaaclab> --episodes N` → `outputs/<run_id>/report.json`

</div>
