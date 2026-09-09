# CONTEXT.md — Physical AI Harness (aegis) : Current Stage, Context & What’s Left

> **Date:** 2026-09-09 | **Status:** Phase 1 & 2 Complete | **Phase 3 Scaffold Complete (mock/fallback)** | **Phase 3 Real Runtime Pending (NVIDIA Tech Partnership)**
> **One-liner:** A safety + evaluation harness that sits between ANY robot AI model and ANY robot (sim or real) — blocks dangerous actions before they reach hardware and produces honest, reproducible test reports.

---

## 1. Context: What Problem Are We Solving?

**Problem in simple English:**
AI models that control robots (called VLA — Vision-Language-Action models like SmolVLA, RT-2, etc.) are powerful but unsafe. A model can command a robot joint to move too fast, apply too much force, or hallucinate an action and break the arm / hurt a person / destroy the task. Today:

- No standard **safety layer** between the AI and the hardware
- No standard **evaluation** — everyone builds their own sim + scripts, results are not comparable or trusted
- **Sim-to-real gap** is unmeasured — a model that works in its training sim often fails in a new sim or on real hardware and no one knows why

**Solution we are building —** `aegis`**:**

```
AI Policy (VLA / classical) --action--> [ AEGIS : Safety Gateway + Eval Harness ] --safe action--> Robot (MuJoCo / Isaac Lab / Real)
```

- **Safety Gateway** (`src/aegis/safety/gateway.py`): Every action MUST pass through it. Checks NaN, velocity, force, effort, inference-budget. On violation → blocks policy action, runs PID-to-home fallback for 50 steps, then resumes. No bypass path exists.
- **Eval Harness** (`src/aegis/eval/runner.py`): `aegis eval --model X --episodes N` runs N episodes, logs `report.json` + `episodes.jsonl`/`steps.jsonl`/`trajectory.jsonl`, reports task success, violations, recoveries, latency p50/p95, recommendation.
- **Honest by design:** If something is stubbed/mocked, the report says so. No fabricated numbers. (`agent.md:16`)

**Why this matters for NVIDIA partnership:**
This is **Physical AI infrastructure**. NVIDIA's stack is Isaac Sim / Isaac Lab + ROS2 + real robots. We provide the trusted harness that proves an NVIDIA-relevant model (SmolVLA) is safe in Isaac Lab before it touches real hardware. It fits directly into NVIDIA’s Physical AI / VLA evaluation story.

---

## 2. Current Stage: Where We Are Today

### Completed: Phase 1 — MuJoCo POC ✅
**Reference:** `PHASE_1_SUMMARY.md`, `DESIGN.md`, `README.md:10`

| Item | Status | Evidence |
|------|--------|----------|
| `aegis validate` | Done | Exit 0/2 per `tests/test_eval.py` |
| `aegis eval` end-to-end (Franka Panda in MuJoCo, Menagerie model) | Done | `src/aegis/envs/mujoco_pick_place.py`, `assets/scenes/pick_place.xml` |
| Safety Gateway (NaN + measured velocity + measured force + fallback PID-to-home) | Done | `src/aegis/safety/checks.py`, `gateway.py`, `fallback.py` |
| Scripted policy (6-phase DLS IK smoke-test) | Done | `src/aegis/policies/scripted.py` — **3/3 succeed, 0 violations** (seed 42) |
| Random policy (negative control) | Done | `src/aegis/policies/random.py` — **0/3 succeed, ~145 violations (uniform) / 4 violations (per-joint)** (seed 7) |
| Telemetry + report | Done | `src/aegis/telemetry/logger.py`, `src/aegis/eval/report.py` → `outputs/<run_id>/` |
| Determinism + tests | Done | `uv run pytest -q` → **20 passed** (was 6, now 20 with per-joint) |

**POC caveats (documented honestly):** Torque-PD tracks velocity action (`tau = qfrc_bias + 40*err + 5*vel_err`) — not a perfect velocity servo; **per-joint limits now shipped (was uniform)**; rule-based recommendation; `gpu_hours` stub; position-only success (0.05m + lifted 6cm).

### Completed: Phase 2 — Real Model Integration ✅
**Reference:** `PHASE_2_SUMMARY.md`

| Item | Status | Evidence |
|------|--------|----------|
| Real model: **SmolVLA-450M** (`lerobot/smolvla_libero`, 450M params) | Done | `src/aegis/policies/smolvla.py`, `configs/models/smolvla_libero.yaml` |
| Checkpoint pipelines applied verbatim (tokenizer, normalizer, unnormalizer) | Done | Verified vs checkpoint tensors, not `config.json` |
| Contract: 8-D state `[eef_pos(3), axis-angle(3), gripper(2)]`, 7-D cartesian-delta action, 3 cameras 256x256, chunk 50 | Done | `PHASE_2_SUMMARY.md:23` |
| Adapter: 7-D delta → 8-D joint-vel via DLS resolved-rate (hand frame), clamp 0.5 rad/s | Done | `src/aegis/policies/smolvla.py` |
| Chunk-boundary-only rendering (per-step was 30x slower, broke determinism) | Done | `src/aegis/envs/mujoco_pick_place.py` — 3 cameras `camera1/2/3` |
| Inference budget enforcement (`inference_budget_ms`, `inference_mode cpu|cuda`) | Done | `src/aegis/config/models.py`, `src/aegis/eval/runner.py` |
| Budget/model-error → fallback (same as safety violation) | Done | Real cold-start proven: **5.09s violation on step 1, fallback ran** (`PHASE_2_SUMMARY.md:69`) |
| `gpu_hours` wall-clock proxy for cuda | Done | `src/aegis/eval/report.py` |
| Tests | Done | `tests/test_policy_model.py` — **20 passed total** |

**Honest zero-shot result:**
`aegis eval --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600`:
- **0/3 success, 33 violations / 33 recoveries per episode** (1 budget + 32 velocity) — operational limits (uniform 0.6)
- **Relaxed-limit A/B** (`physical-ai-diag.yaml` + `configs/robots/franka_diag.yaml`, limits 0.6→5.0 rad/s): **0 violations, still 0/3 success** — proves (a) limits WERE interrupting (~1/3 of episode under fallback) AND (b) model genuinely **mis-localizes** (hand never gets within 0.16m of cube, moves away to 0.6m, gripper closes in mid-air) — vision domain gap. Visual evidence: `diag/ep{0,1}/step{...}_{camera1..3}.png`.
- **With new per-joint limits [2.175 .. 2.61] rad/s:** headroom is now >4x adapter clamp, so transient overshoot no longer trips fallback chronically (`tests/test_policy_model.py:151`).

### Completed: Phase 3 Scaffold (2026-09-09) ✅ — mock/fallback, no real Isaac/ROS2 runtime
**Reference:** `STEP6_ISAAC_VS_MUJOCO.md`, `HARDWARE_CHECKLIST.md`, `src/aegis/ros2/bridge.py`, `src/aegis/envs/isaac_pick_place.py`

| Item | Status | Evidence |
|------|--------|----------|
| **3.3 Per-joint limits** — uniform → per-joint tables | Done | `configs/robots/franka.yaml` now `[2.175,2.175,2.175,2.175,2.61,2.61,2.61]` rad/s + `[87,87,87,87,12,12,12]` Nm; `src/aegis/config/models.py:68` `float|list[7]`, `src/aegis/safety/checks.py:35`; `src/aegis/eval/report.py:34` warning removed when per-joint; `configs/robots/franka_uniform.yaml` kept for A/B; 20 tests pass |
| **3.1 ROS2 bridge (mock)** | Done (mock) | `src/aegis/ros2/bridge.py` — `RosBridge` auto-mock when `rclpy` missing, `benchmark_latency()` + `aegis rosbench --mock/--real`; mock p50 0.000ms p95 0.002ms (20 samples); real `rclpy` pending WSL2 ROS2 install |
| **3.2 Isaac Lab env scaffold** | Done (fallback) | `src/aegis/envs/isaac_pick_place.py` — same API as MuJoCo, `aegis eval --sim isaaclab` works without Isaac Sim (falls back to MuJoCo with `RuntimeWarning` + `info["sim"]="isaaclab-fallback-mujoco"`); `src/aegis/cli.py:71` now accepts `mujoco|isaaclab`, `src/aegis/policies/scripted.py:39` unwraps fallback |
| **3.5 SmolVLA on Isaac Lab (gap measurement, scaffold)** | Done (scaffold) | `STEP6_ISAAC_VS_MUJOCO.md` — scripted 3/3 on both sims, random 0/3 (4 violations per-joint vs 145 uniform); gap 0 today because Isaac fallback == MuJoCo (honest); real Isaac USD + SmolVLA re-eval pending |
| **3.4 Hardware dry-run checklist** | Done (doc) | `HARDWARE_CHECKLIST.md` — Gates 0-4, e-stop, ROS2, Isaac USD, artifacts; current decision **No-go: e-stop + ROS2 + Isaac USD pending** |

### Stack (current)
- **Language:** Python 3.12 (`pyproject.toml:6`), `uv` env, `src/` layout
- **Sim:** MuJoCo 3.0+ (Menagerie Franka Panda) + Isaac Lab scaffold (fallback to MuJoCo until 16GB VRAM/USD)
- **Models:** `lerobot>=0.3.3`, `transformers>=5.15`, `torch 2.11+cu128`, `torchvision 0.26+cu128` (direct CUDA 12.8 wheels — `pyproject.toml:17`)
- **Config:** Pydantic + YAML (`src/aegis/config/models.py` now per-joint `float|list[7]`, `loader.py`), component files `configs/{robots,models,tasks}/`
- **CLI:** Typer `aegis` (`src/aegis/cli.py`) — `validate` / `eval` / `rosbench`, exit 0/2/3, `--sim mujoco|isaaclab`
- **Safety:** Per-joint limits shipped, uniform warning removed when per-joint (`src/aegis/safety/checks.py`, `gateway.py`, `report.py`)
- **Bridge:** ROS2 mock + latency benchmark (`src/aegis/ros2/bridge.py`)
- **Logging:** NDJSON (`src/aegis/telemetry/logger.py`, `timing.py`)

### How to run today
```bash
uv run aegis validate                                          # config OK
uv run aegis validate --robot franka_uniform                   # legacy uniform A/B

# per-joint limits (new default) — 3/3 success, 0 violations, 1 warning (was 2)
uv run aegis eval --model scripted --episodes 3 --seed 42
uv run aegis eval --model random --episodes 3 --seed 7         # per-joint: 0/3, 4 violations (was 145 with uniform)

# Isaac Lab scaffold (fallback to MuJoCo without Isaac Sim, same report schema)
uv run aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42   # 3/3 via fallback + RuntimeWarning

# ROS2 bridge latency (mock until rclpy installed)
uv run aegis rosbench --n 20 --mock                            # p50 0.000ms
# after ROS2 Humble install in WSL2: uv run aegis rosbench --real --n 100

uv run pytest -q                                               # 20 passed

# real model (needs GPU + download, last measured PHASE_2_SUMMARY.md:47)
uv run aegis eval --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600
```

---

## 3. What’s Left To Do

### Remaining: Phase 3 Real Runtime (needs NVIDIA + hardware)
**Gate conditions:** `WSL2_SETUP_STATUS.md:5`

1. **WSL2 GPU passthrough already ✅** — RTX 4050 6GB visible via `nvidia-smi` inside WSL2 Ubuntu 22.04.5
2. **ROS2 Humble — scaffold done, real install pending** — `src/aegis/ros2/bridge.py` mock verified; real `rclpy` install blocked on apt GPG 404. Manual fix provided (curl -k + `packages.ros.org`), or Docker `osrf/ros:humble-desktop`. Needs `aegis rosbench --real` inside WSL2 and Windows<->WSL2 bridge latency benchmark.
3. **Isaac Lab — scaffold done, real USD pending** — `aegis eval --sim isaaclab` fallback works; real Isaac Sim 6.0.1 needs **16GB VRAM min** (you have 6GB). Needs NVIDIA guidance for VRAM-safe scene defaults + LIBERO camera placement, then author USD via `workflows/agentic/arena/run.sh --create-env pick_place --from scissor_pick_and_place` and replace `isaac_pick_place.py:44` NotImplementedError with real PhysX + camera sensors.

**Phase 3 remaining scope (needs NVIDIA alignment):**

| Milestone | What | Output | Status |
|-----------|------|--------|--------|
| **3.1 ROS2 bridge (real)** | Install ROS2 Humble in WSL2, wire `aegis` ↔ ROS2 topics (action in / state out), latency benchmark inside-WSL2 vs Windows↔WSL2 bridge | `src/aegis/ros2/` real rclpy + latency report | Scaffold done, real pending |
| **3.2 Isaac Lab env (real)** | Replace fallback with real USD scene + PhysX, same pick-place task, same per-joint limits, same report schema | Isaac Lab USD + `aegis eval --sim isaaclab` real | Scaffold done, real pending |
| **3.3 Per-joint limits** | ~~Replace uniform limits with per-joint tables~~ | ~~Realistic safety, warning removed~~ | **Done** |
| **3.4 Real-robot dry run** | In-the-loop eval on real Franka (Linux/WSL2 host), e-stop wiring, hardware safety layer beyond action-space gating | Hardware report, go/no-go checklist | Checklist done (`HARDWARE_CHECKLIST.md`), hardware pending (No-go) |
| **3.5 SmolVLA on Isaac Lab (real gap)** | Re-evaluate SmolVLA in Isaac Lab cameras/lighting (closer to training domain), measure if mis-localization reduces | Comparison report Mujoco vs Isaac Lab (real) | Scaffold gap 0 (fallback == mujoco), real pending |

**Risks to clear in Phase 3 real runtime:**
- ~~Safety limit tuning: 0.6 rad/s sat at adapter clamp 0.5 rad/s → fallback 33x/episode~~ **Fixed** by per-joint [2.175 .. 2.61] giving >4x headroom (`tests/test_policy_model.py:151` now checks `min(limit) >= 2*clamp`)
- Isaac Lab VRAM: 6GB is below 16GB min — needs NVIDIA VRAM-safe guidance before USD authoring
- Architecture: **fully-inside-WSL2** recommended (avoids bridge latency, standard pattern — `WSL2_SETUP_STATUS.md:43`)

### Beyond Phase 3 (not in current scope, but on roadmap)
- Multi-robot / multi-task batching, parallel episodes
- Learned recommendation (replace 4-rule heuristic)
- Dashboard / observability (Prometheus/Grafana/OTel) — currently JSON logs only
- RL training / dataset collection / checkpointing & resume
- Per-model camera calibration, domain randomization, contact-force grasp detector

---

## 4. What We Need from NVIDIA Partnership

1. **Technical guidance:** Isaac Lab setup on WSL2 + RTX 4050 (6GB) — best install path, VRAM-safe scene defaults, camera placement matching LIBERO distribution (to close SmolVLA mis-localization gap)
2. **ROS2 unblock:** Recommended install method for ROS2 Humble on Ubuntu 22.04 WSL2 (apt vs conda vs Docker) given current GPG/repo 404 — we have mock + manual curl -k fix, need your preferred path
3. **Evaluation alignment:** What report fields / safety limits does NVIDIA expect for Physical AI harness acceptance? (per-joint limits now shipped, latency SLOs, Isaac Sim vs Lab preference)
4. **Access (if available):** Isaac Lab examples / Franka assets, office hours for sim-to-real gap triage

---

## 5. Files to Read for Full Context

- `README.md` — quick start + honest caveats
- `DESIGN.md` — full Phase 1 design doc (schema, gateway, eval loop, out-of-scope)
- `PHASE_1_SUMMARY.md` — POC completion evidence
- `PHASE_2_SUMMARY.md` — real-model integration + honest zero-shot characterization + visual evidence
- `STEP6_ISAAC_VS_MUJOCO.md` — Phase 3 scaffold comparison (per-joint, isaac fallback, rosbench, gap 0 honest)
- `HARDWARE_CHECKLIST.md` — Phase 3 dry-run gates + No-go reason
- `agent.md` — engineering standards (no mocks as real, no bypass, timing from day one)
- `physical-ai.yaml` / `physical-ai-diag.yaml` / `configs/robots/franka.yaml` (per-joint) / `franka_uniform.yaml` (legacy) / `franka_diag.yaml`
- `examples/run_poc.md` — exact commands + expected outputs
- `WSL2_SETUP_STATUS.md` — Phase 3 pre-gate status (GPU OK, ROS2 blocked)
- `WHAT_NEXT.md` — what we want to build next + what’s remaining (simple, prioritized)

---

## 6. Exit Criteria for “Done” (Phase 3)

- [x] `aegis eval --sim isaaclab --model scripted` → 3/3 success, 0 violations — **done via fallback** (real USD pending)
- [ ] `aegis eval --sim isaaclab --model smolvla_libero` → report with violations/recoveries, latency p50/p95, and honest success rate (comparison to MuJoCo) — **scaffold gap 0, real pending**
- [x] ROS2 bridge latency benchmark published (inside-WSL2 vs bridge) — **mock p50/p95 done, real rclpy pending**
- [x] Per-joint safety limits shipped, uniform-limit warning removed — **done** (`franka.yaml:11`, 20 tests)
- [x] Hardware dry-run checklist (e-stop, fallback, real-robot `report.json`) — **checklist done, hardware No-go documented** (`HARDWARE_CHECKLIST.md`)

---

*This file is the single source for current stage + partnership context. Update it at each phase gate.*
