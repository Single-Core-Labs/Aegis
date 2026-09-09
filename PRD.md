# PRD — Aegis: Physical AI Safety Harness & Evaluation Framework

> **Version:** 1.0 — September 2026 | **Status:** Phase 1 & 2 Complete, Phase 3 Scaffold Complete, Phase 3 Real Runtime Pending | **License:** Apache-2.0 | **Entry:** `aegis eval --model <policy> --sim <mujoco|isaaclab> --episodes N`

---

## 1. Executive Summary

**Aegis** is infrastructure that sits between **any** robot AI policy (VLA like SmolVLA/RT-2/π0, or classical) and **any** robot (MuJoCo, Isaac Lab, real Franka via ROS 2). Every action **must** traverse a hardware-aware **Safety Gateway**; every episode produces a **deterministic, auditable report** (`report.json` + NDJSON). The harness is **honest by default** — if a backend is mocked or fallback is active, the report says so.

One-liner: `Policy —action→ [ AEGIS Safety Gateway + Eval Harness ] —safe action→ Robot`

Built for the **NVIDIA Physical AI Stack**: Isaac Sim 6.0.1 • Isaac Lab 2.x • ROS 2 Humble • CUDA 12.8.

---

## 2. Problem

VLA models are powerful but unsafe by default — they can command infeasible velocities, excessive forces, or hallucinated trajectories that damage hardware or endanger operators.

| Gap Today | Impact |
|---|---|
| No standard safety layer between policy and actuator | Broken gears, unsafe contacts, no attribution |
| No standard evaluation — bespoke scripts per team | Results incomparable, failures not attributable |
| Sim-to-real gap unmeasured — training sim ≠ deployment sim | Model that works in LIBERO fails silently in new sim |

Aegis solves all three with one harness, not another model.

---

## 3. Goals / Non-Goals

### Goals
1. **Mandatory safety gating** — per-joint velocity/force/NaN/inference-budget checks + PID-to-home fallback with no bypass path.
2. **Unified deterministic evaluation** — `aegis eval` with seeded episodes, p50/p95 latency, violation/recovery accounting, unified `report.json` across sims.
3. **Honest sim-to-real measurement** — same policy, same limits, MuJoCo vs Isaac Lab vs hardware; relaxed-limit A/B to separate limit interruption from model mis-localization.
4. **NVIDIA-stack ready** — Isaac Sim/Lab USD + PhysX, ROS 2 topics, CUDA per-kernel `gpu_hours`, 6GB VRAM-safe path.

### Non-Goals (this PRD)
- No new VLA model / training
- No learned safety filter (checks are interpretable, auditable)
- No GPU orchestration (local CUDA only)
- No multi-robot fleet management (P3 batching is spec-only)

---

## 4. Personas

| Persona | Need | Aegis Command |
|---|---|---|
| **Robotics Engineer** (NVIDIA stack) | Prove VLA is safe in Isaac before hardware | `aegis eval --sim isaaclab --model smolvla_libero --inference-mode cuda` |
| **Safety / Controls Engineer** | Set per-joint limits, prove fallback, pass Go/No-Go | `aegis validate --robot franka` + `HARDWARE_CHECKLIST.md` Gates |
| **ML Researcher** | Attribute failure: limits vs vision gap | Relaxed A/B `physical-ai-diag.yaml` + visual `camera1..3.png` |
| **Platform Engineer** | Observability, reproducibility | `outputs/<run_id>/report.json` + `steps.jsonl` → Prometheus/Grafana/OTel |

---

## 5. User Stories (acceptance)

1. **As an engineer, I can** `aegis validate` **and get** exit `0` or `2` with a readable error if YAML or asset path is wrong.
2. **As an engineer, I can** `aegis eval --model scripted --episodes 3 --seed 42` **and get** `3/3 success, 0 violations` deterministically (same seed → same outcome).
3. **As a safety engineer, I can** configure `max_velocity: [2.175..2.61]` per-joint in `configs/robots/franka.yaml` and the gateway enforces it on **measured** `qvel`/`qfrc` (not just command).
4. **As a safety engineer, I can** run `aegis eval --model random --seed 7` and see `0/3` with violations/recoveries — proving fallback engages (PID-to-home 50 steps, `resume`).
5. **As an ML researcher, I can** run SmolVLA `aegis eval --model smolvla_libero --inference-mode cuda` and get honest `0/3` with `33 violations (1 budget + 32 vel)` vs `0 violations` at `5.0 rad/s` — proving which failures are limit-induced vs vision gap (0.16m→0.60m, mid-air grips).
6. **As a platform engineer, I can** benchmark `aegis rosbench --mock` → `p50 0.000ms p95 0.002ms` and `aegis rosbench --real` inside WSL2 after ROS 2 install.
7. **As an engineer on 6GB laptop, I can** `aegis eval --sim isaaclab` **without Isaac Sim** and get honest fallback `RuntimeWarning` + `info["sim"]="isaaclab-fallback-mujoco"` (no silent mock).

---

## 6. Functional Requirements

### 6.1 Safety Gateway (`src/aegis/safety/`)
- **Checks in order:** `1. NaN/Inf` → `2. Effort` (`max_effort_action`, clamp/reject) → `3. Measured velocity` (`|qvel|>max_velocity`) → `4. Measured force` (`|torque|>max_force`). Per-joint `float|list[7]` schema (`config/models.py:68`).
- **On violation:** increment `violation_count`, emit `safety_violation` event, switch to fallback for `recovery_steps=50`, mode `resume|hold`. Fallback action is re-checked.
- **Fallback:** PID-to-home `tau = qfrc_bias + 40·err + 5·vel_err`, bounded `0.3 rad/s` (`fallback.py`), gravity-compensated.
- **Budget/model-error:** Over `inference_budget_ms` or `PolicyModelError` → same violation path. Cold CUDA start 5.09s at `2000ms` budget must trip fallback on step 1 (proven).

### 6.2 Evaluation Harness (`src/aegis/eval/`)
- **Loop:** `env.reset(seed+episode)` → `policy.act(obs)` (timed) → `gateway.filter` → `env.step(gated.command)` (timed) → `logger.step` (NDJSON). `perf_counter_ns` throughout.
- **Success:** `grasped_ever` **AND** `dist<=0.05m`. Grasp = `lifted (z>table+0.025)` **AND** finger-object contacts with `mj_contactForce>0.5N` (`mujoco_pick_place.py:68`). Sticky once achieved. Lift+contact logged as `grasp_lifted/has_contacts/force`.
- **Termination:** `terminated` (success) | `truncated` (max_steps) | `timeout` (wall-clock `episode_timeout_sec`).

### 6.3 Config (`physical-ai.yaml` + `configs/`)
- **Root `PhysicalAIYaml`:** `schema_version, eval, model, robot, environment, task, output`. `extra="forbid"` — typos fail.
- **EvalSpec:** `episodes 1..1000, seed, max_steps 1..100000, time_step>0, episode_timeout_sec>0, inference_mode cpu|cuda, inference_budget_ms>0`.
- **ModelSpec:** `name, kind random|scripted|smolvla, policy (discriminator), endpoint (required for smolvla)`.
- **RobotSpec/SafetyLimits:** `name, mjcf_path (must exist), safety {max_velocity, max_force: float|list[7]>0, max_effort_action, reject_nan_actions, action_clamp, recovery_steps, recovery_mode}`.
- **EnvSpec:** `sim mujoco|isaaclab, scene_mjcf, robot_name, control_mode joint_velocity, render_cameras`.
- **TaskSpec:** `name pick-place, success_threshold_m>0, object_name, target_name`.
- **Override:** CLI flags `> YAML > defaults`. `load_run_config` validates asset paths exist; exit `2` on failure.

### 6.4 CLI (`src/aegis/cli.py` — Typer `aegis`)
- `aegis eval --model --robot --sim --tasks --episodes --seed --max-steps --inference-mode --inference-budget-ms --config --output-dir --run-id --verbose` — backend selection `mujoco|isaaclab`, writes `outputs/<run_id>/` + stdout summary.
- `aegis validate --config --model --robot --tasks` — validates without running.
- `aegis rosbench --n --mock/--real` — publishes `n` 8-D actions, returns `LatencyStats p50/p95/mean/min/max`.
- **Exits:** `0` honest run (even `0/3`), `2` config error, `3` internal error (partial logs preserved).

### 6.5 Sim Backends (`src/aegis/envs/`)
- **MuJoCo:** Menagerie Franka Panda `panda_vel.xml`, 7 velocity + tendon gripper `0..255`, `mujoco_pick_place.py` with torque-PD `tau=qfrc_bias+40*err+5*(vel-qd)`, 3 cameras `256×256` at chunk boundaries only, `Env` protocol `reset/step/observe/render_images/state_snapshot`.
- **Isaac Lab:** `isaac_pick_place.py` same `Env` API; without Isaac Sim 6.0.1 → fallback to MuJoCo with `RuntimeWarning` + `info["sim"]="isaaclab-fallback-mujoco"`; with Isaac Sim → PhysX + USD (author via `workflows/agentic/arena/run.sh --create-env pick_place`, replace `NotImplementedError:44`).

### 6.6 Policies (`src/aegis/policies/`)
- `random` — uniform joint velocities (negative control).
- `scripted` — 6-phase DLS IK state machine (approach→grasp→lift→pre-place→place→retreat), deterministic, `3/3` at seed 42.
- `smolvla` — `lerobot/smolvla_libero` 450M, Apache-2.0, checkpoint pipelines verbatim, 8-D state `[eef_pos(3), axis-angle(3), gripper(2)]`, 7-D cartesian-delta, 3 cameras, chunk 50, DLS adapter `0.5 rad/s` clamp, budget-enforced.

### 6.7 Telemetry & Reports
- **NDJSON under `outputs/<run_id>/`:** `run.json` (validated config+git), `episodes.jsonl` (one per episode), `steps.jsonl` (per-step source/violation/latencies/info), `trajectory.jsonl` (qpos/qvel).
- **report.json:** `{run_id, model, robot, sim, task_counts, latency_p50/p95, inference_mode/budget/violations/model_errors, safety_violations, recovery_events, gpu_hours, gpu_hours_note, total_steps, total_duration_s, total_inference_s, recommendation, warnings}`. `gpu_hours` = per-kernel sum of `inference_s` when `cuda` (CUDA events when available) else `0`.
- **OTel/Prometheus/Grafana:** `src/aegis/telemetry/otel.py` → `aegis_episodes_success, safety_violations, recovery_events, latency_p50/p95, gpu_hours, step_violation` + `grafana/aegis_dashboard.json`.

### 6.8 ROS 2 Bridge (`src/aegis/ros2/bridge.py`)
- `RosBridge` auto-detects `rclpy`: topics `/aegis/action` (Float64MultiArray 8-D), `/aegis/state` (JointState), `/aegis/safety` (JSON). Mock in-memory when `rclpy` absent. `publish_action/get_state/publish_safety/spin_once/latency_stats`. `benchmark_latency(n)` used by `rosbench`.

---

## 7. Non-Functional Requirements

| NFR | Requirement |
|---|---|
| **Safety** | No bypass path — every action through gateway by construction; fallback drives to home on any violation. |
| **Honesty** | Stubbed/mocked backends flagged in `warnings` and `info["sim"]`; no fabricated numbers (`agent.md:16`). |
| **Determinism** | Same `seed+episode` → same placement, same policy, same report; verified across 5 seeds × 5 episodes ≥90% success. |
| **Timing** | All latency-sensitive paths instrumented from day one (`perf_counter_ns`, CUDA events when `cuda`). |
| **Config** | Declarative YAML only, no hardcoded robot/model params in code. |
| **Portability** | Python 3.12, `uv`, `src/` layout; Windows + WSL2 (fully-inside-WSL2 recommended). |
| **Test** | `python -m pytest -q` → 20 tests: config validation (exit 2), determinism, random-fails/scripted-succeeds, budget/adapter/SmolVLA config, seed sweep. |

---

## 8. Architecture

```
configs/{robots,models,tasks}/  →  physical-ai.yaml  →  Pydantic validate  →  aegis validate
        │
        ▼
┌─────────────────────────────────────────────────┐
│ EvalRunner                                      │
│  policy.act(obs) ──timed──▶ SafetyGateway.filter│
│                        │                        │
│              ┌─────────┴─────────┐               │
│              │ Checks 1-4 + budget│ → violation? │
│              └─────────┬─────────┘               │
│                 fallback 50 steps               │
│                        ▼                        │
│                  env.step(safe_action)          │
│                        │                        │
│                  RunLogger → outputs/<run_id>/  │
└─────────────────────────────────────────────────┘
```

Module map: `cli.py` (Typer) | `config/` (Pydantic loader) | `envs/` (MuJoCo + Isaac fallback) | `policies/` (random/scripted/smolvla) | `safety/` (checks/gateway/fallback) | `eval/` (runner/metrics/report) | `ros2/` (RosBridge) | `telemetry/` (logger/timing/otel).

---

## 9. Report Schema (contract)

```json
{
  "run_id": "run-20260909T120000Z",
  "model": "scripted", "robot": "franka", "sim": "mujoco",
  "task_counts": {"pick-place": {"success": 3, "fail": 0}},
  "latency_p50_ms": 0.012, "latency_p95_ms": 0.034,
  "inference_mode": "cpu", "inference_budget_ms": 2000.0,
  "inference_budget_violations": 0, "model_errors": 0,
  "safety_violations": 0, "recovery_events": 0,
  "gpu_hours": 0.0, "gpu_hours_note": "stub — no GPU used",
  "total_steps": 750, "total_duration_s": 12.3, "total_inference_s": 0.09,
  "recommendation": "all episodes succeeded — harness ready",
  "warnings": ["recommendation line is rule-based (4 rules), not learned"]
}
```

Recommendation rules (4): `no episodes` → error; `all success` → ready; `violations+recoveries & 0 success` → tighten limits; `truncated > half` → raise max_steps; else → stand-in expected to fail.

---

## 10. Benchmarks (must reproduce on seed 42/7)

| Sim | Policy | Episodes | Success | Violations | Recoveries | Note |
|---|---|---|---|---|---|---|
| MuJoCo | scripted | 3 | 3/3 | 0 | 0 | smoke-test, per-joint [2.175..2.61] |
| MuJoCo | random | 3 | 0/3 | 4 | 2 | was 145 with uniform 1.0 — realistic envelope |
| MuJoCo | smolvla_libero (cuda) | 3 | 0/3 | 33/ep (1 budget + 32 vel) | 33/ep | 0/3 even at 5.0 rad/s — vision domain gap (0.16m→0.60m, mid-air grips) |
| Isaac fallback | scripted | 3 | 3/3 | 0 | 0 | `RuntimeWarning`, `isaaclab-fallback-mujoco` |

Latency: scripted `p50 0.012 p95 0.034` ms (cpu), SmolVLA `p50 1.29 p95 1.87` ms amortized, chunk `0.35-0.41s`, cold `5.09s` (budget violation). ROS mock `p50 0.000 p95 0.002` ms. `gpu_hours` = per-kernel sum /3600 when `cuda`.

---

## 11. Roadmap

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | MuJoCo POC — gateway, scripted/random, eval, reports | ✅ Done |
| Phase 2 | SmolVLA-450M integration, budget, DLS adapter | ✅ Done |
| Phase 3 scaffold | Per-joint limits, ROS2 mock, Isaac fallback, checklist | ✅ Done |
| Phase 3 real | ROS2 Humble (`rclpy`), Isaac USD + PhysX (6GB VRAM-safe), hardware dry-run | 🔲 Pending — WSL2 `curl -k` apt fix + 16→6GB guidance |
| Beyond Phase 3 | Contact-force grasp (done), per-kernel gpu_hours (done), dashboard/OTel (done), Isaac 6GB perf, multi-robot batching (spec), learned recommendation (spec), camera calibration + DR (spec), RL/dataset (backlog) | 🔲 Spec/partial |

---

## 12. Hardware Readiness (Go/No-Go)

Gates: **0 Safety in sim** (`scripted 3/3`, per-joint, `random 0/3`) ✅ | **1 E-stop + physical** (cuts power, workspace barriers) ⏳ | **2 Software bridge** (`ros2 topic list`, `rosbench --real p95 < budget`, USD for 6GB) ⏳ | **3 Dry-run** (gravity-comp, empty hand, inject NaN → fallback home, `sim:hardware` report) ⏳ | **4 Go/No-Go** (all gates + logs, signed) ⏳ — **Decision until then: No-go, no fabricated hardware report.**

---

## 13. Open Questions for NVIDIA Partnership

1. VRAM-safe Isaac Sim 6.0.1 + LIBERO camera placement for RTX 4050 6GB (16GB min official)
2. ROS2 Humble on WSL2 Ubuntu 22.04 — preferred install (apt `curl -k` vs Docker `osrf/ros:humble-desktop`)
3. Expected `report.json` fields / latency SLOs / Isaac Sim vs Lab preference
4. Isaac Franka assets / office-hours for gap triage

---

## 14. Files (canonical after cleanup)

```
PRD.md                  ← this file
README.md               quick start + benchmarks
agent.md                engineering standards (no mocks as real, no bypass)
physical-ai.yaml        root config defaults
configs/robots/franka.yaml              per-joint [2.175..2.61], [87..12]
configs/models/{random,scripted,smolvla_libero}.yaml
configs/tasks/pick-place.yaml
assets/scenes/pick_place.xml + menagerie/franka_emika_panda/panda_vel.xml
src/aegis/{cli,config,envs,policies,safety,eval,ros2,telemetry}/
docs/{nvidia-stack-manual,architecture,safety-gateway,isaac-lab,ros2-bridge,dashboard,perf-tuning,batching,camera-calibration,learned-recommendation,README}.md
whitepaper/{aegis_whitepaper.tex, aegis_research_paper.tex (main.tex)}
tests/{test_eval,test_policy_model,test_seed_sweep}.py (20 tests)
```

---

*This PRD is the single source for requirements. Implementation follows it verbatim; deviations are honest, flagged, and tested.*
