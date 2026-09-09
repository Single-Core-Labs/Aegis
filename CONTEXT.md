# CONTEXT.md — Physical AI Harness (aegis) : Current Stage, Context & What’s Left

> **Date:** 2026-09-09 | **Status:** Phase 1 & 2 Complete | **Phase 3:** Next (NVIDIA Tech Partnership)
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


| Item                                                                             | Status | Evidence                                                                          |
| -------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------- |
| `aegis validate`                                                                 | Done   | Exit 0/2 per `tests/test_eval.py`                                                 |
| `aegis eval` end-to-end (Franka Panda in MuJoCo, Menagerie model)                | Done   | `src/aegis/envs/mujoco_pick_place.py`, `assets/scenes/pick_place.xml`             |
| Safety Gateway (NaN + measured velocity + measured force + fallback PID-to-home) | Done   | `src/aegis/safety/checks.py`, `gateway.py`, `fallback.py`                         |
| Scripted policy (6-phase DLS IK smoke-test)                                      | Done   | `src/aegis/policies/scripted.py` — **3/3 succeed, 0 violations** (seed 42)        |
| Random policy (negative control)                                                 | Done   | `src/aegis/policies/random.py` — **0/3 succeed, ~145 violations** (seed 7)        |
| Telemetry + report                                                               | Done   | `src/aegis/telemetry/logger.py`, `src/aegis/eval/report.py` → `outputs/<run_id>/` |
| Determinism + tests                                                              | Done   | `uv run pytest tests/test_eval.py -q` → **6 passed**                              |


**POC caveats (documented honestly):** Torque-PD tracks velocity action (`tau = qfrc_bias + 40*err + 5*vel_err`) — not a perfect velocity servo; uniform per-joint limits; rule-based recommendation; `gpu_hours` stub; position-only success (0.05m + lifted 6cm).

### Completed: Phase 2 — Real Model Integration ✅

**Reference:** `PHASE_2_SUMMARY.md`


| Item                                                                                                                   | Status | Evidence                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------- |
| Real model: **SmolVLA-450M** (`lerobot/smolvla_libero`, 450M params)                                                   | Done   | `src/aegis/policies/smolvla.py`, `configs/models/smolvla_libero.yaml`                         |
| Checkpoint pipelines applied verbatim (tokenizer, normalizer, unnormalizer)                                            | Done   | Verified vs checkpoint tensors, not `config.json`                                             |
| Contract: 8-D state `[eef_pos(3), axis-angle(3), gripper(2)]`, 7-D cartesian-delta action, 3 cameras 256x256, chunk 50 | Done   | `PHASE_2_SUMMARY.md:23`                                                                       |
| Adapter: 7-D delta → 8-D joint-vel via DLS resolved-rate (hand frame), clamp 0.5 rad/s                                 | Done   | `src/aegis/policies/smolvla.py`                                                               |
| Chunk-boundary-only rendering (per-step was 30x slower, broke determinism)                                             | Done   | `src/aegis/envs/mujoco_pick_place.py` — 3 cameras `camera1/2/3`                               |
| Inference budget enforcement (`inference_budget_ms`, `inference_mode cpu                                               | cuda`) | Done                                                                                          |
| Budget/model-error → fallback (same as safety violation)                                                               | Done   | Real cold-start proven: **5.09s violation on step 1, fallback ran** (`PHASE_2_SUMMARY.md:69`) |
| `gpu_hours` wall-clock proxy for cuda                                                                                  | Done   | `src/aegis/eval/report.py`                                                                    |
| Tests                                                                                                                  | Done   | `tests/test_policy_model.py` — **18 passed total (~35s)**                                     |


**Honest zero-shot result:**
`aegis eval --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600`:

- **0/3 success, 33 violations / 33 recoveries per episode** (1 budget + 32 velocity) — operational limits
- **Relaxed-limit A/B** (`physical-ai-diag.yaml` + `configs/robots/franka_diag.yaml`, limits 0.6→5.0 rad/s): **0 violations, still 0/3 success** — proves (a) limits WERE interrupting (~1/3 of episode under fallback) AND (b) model genuinely **mis-localizes** (hand never gets within 0.16m of cube, moves away to 0.6m, gripper closes in mid-air) — vision domain gap (fixed third-person cameras vs LIBERO training views). Visual evidence saved: `diag/ep{0,1}/step{...}_{camera1..3}.png`.



### Stack (current)

- **Language:** Python 3.12 (`pyproject.toml:6`), `uv` env, `src/` layout
- **Sim:** MuJoCo 3.0+ (Menagerie Franka Panda), `mujoco` + torque PD
- **Models:** `lerobot>=0.3.3`, `transformers>=5.15`, `torch 2.11+cu128`, `torchvision 0.26+cu128` (direct CUDA 12.8 wheels — `pyproject.toml:17`)
- **Config:** Pydantic + YAML (`src/aegis/config/models.py`, `loader.py`), component files `configs/{robots,models,tasks}/`
- **CLI:** Typer `aegis` (`src/aegis/cli.py`) — `validate` / `eval`, exit 0/2/3
- **Logging:** NDJSON (`src/aegis/telemetry/logger.py`, `timing.py`)



### How to run today

```bash
uv run aegis validate
uv run aegis eval --model scripted --episodes 3 --seed 42          # should: 3/3 success, 0 violations
uv run aegis eval --model random --episodes 3 --seed 7             # should: 0/3, ~145 violations
uv run aegis eval --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600  # real model
uv run pytest -q                                                    # 18 passed
```

---



## 3. What’s Left To Do



### Immediate: Phase 3 — ROS2 + Isaac Lab + Hardware Readiness (NVIDIA partnership core)

**Gate conditions (must pass before committing):** `WSL2_SETUP_STATUS.md:5`

1. **WSL2 GPU passthrough already ✅** — RTX 4050 visible via `nvidia-smi` inside WSL2 Ubuntu 22.04.5
2. **ROS2 Humble install — BLOCKED** — apt/GPG key 404 / malformed entry (`WSL2_SETUP_STATUS.md:10`). Needs resolution.
3. **Isaac Lab install — pending** — after ROS2, then smoke test (load simple scene)

**Phase 3 scope (proposed, needs NVIDIA alignment):**


| Milestone                    | What                                                                                                                                       | Output                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| **3.1 ROS2 bridge**          | ROS2 Humble in WSL2, `aegis` ↔ ROS2 topic bridge (action in / state out), latency benchmark (inside-WSL2 vs Windows↔WSL2 bridge)           | `src/aegis/ros2/` bridge, latency report      |
| **3.2 Isaac Lab env**        | Port `mujoco_pick_place` to Isaac Lab (`src/aegis/envs/isaac_pick_place.py`), same pick-place task, same safety limits, same report schema | Isaac Lab scene + `aegis eval --sim isaaclab` |
| **3.3 Per-joint limits**     | Replace uniform limits with per-joint tables (`configs/robots/franka.yaml` → 7x velocity + 7x force), update `checks.py`                   | Realistic safety, warning removed             |
| **3.4 Real-robot dry run**   | In-the-loop eval on real Franka (Linux/WSL2 host), e-stop wiring, hardware safety layer beyond action-space gating                         | Hardware report, go/no-go checklist           |
| **3.5 SmolVLA on Isaac Lab** | Re-evaluate SmolVLA in Isaac Lab cameras/lighting (closer to training domain), measure if mis-localization reduces                         | Comparison report Mujoco vs Isaac Lab         |


**Risks to clear in Phase 3:**

- Safety limit tuning: operational 0.6 rad/s sits at adapter clamp 0.5 rad/s → transient overshoot trips fallback constantly. Need headroom (≥2x clamp) or make adapter clamp the bound (`PHASE_2_SUMMARY.md:179`).
- Isaac Lab VRAM: 6 GB RTX 4050 Ti is tight — monitor.
- Architecture: **fully-inside-WSL2** recommended (avoids bridge latency, standard pattern — `WSL2_SETUP_STATUS.md:43`).



### Beyond Phase 3 (not in current scope, but on roadmap)

- Multi-robot / multi-task batching, parallel episodes
- Learned recommendation (replace 4-rule heuristic)
- Dashboard / observability (Prometheus/Grafana/OTel) — currently JSON logs only
- RL training / dataset collection / checkpointing & resume
- Per-model camera calibration, domain randomization, contact-force grasp detector

---



## 4. What We Need from NVIDIA Partnership

1. **Technical guidance:** Isaac Lab setup on WSL2 + RTX 4050 (6 GB) — best install path, VRAM-safe scene defaults, camera placement matching LIBERO distribution
2. **ROS2 unblock:** Recommended install method for ROS2 Humble on Ubuntu 22.04 WSL2 (apt vs conda vs Docker) given current GPG/repo 404
3. **Evaluation alignment:** What report fields / safety limits does NVIDIA expect for Physical AI harness acceptance? (per-joint limits, latency SLOs, Isaac Sim vs Lab preference)
4. **Access (if available):** Isaac Lab examples / Franka assets, office hours for sim-to-real gap triage

---



## 5. Files to Read for Full Context

- `README.md` — quick start + honest caveats
- `DESIGN.md` — full Phase 1 design doc (schema, gateway, eval loop, out-of-scope)
- `PHASE_1_SUMMARY.md` — POC completion evidence
- `PHASE_2_SUMMARY.md` — real-model integration + honest zero-shot characterization + visual evidence
- `agent.md` — engineering standards (no mocks as real, no bypass, timing from day one)
- `physical-ai.yaml` / `physical-ai-diag.yaml` — operational vs diagnostic configs
- `examples/run_poc.md` — exact commands + expected outputs
- `WSL2_SETUP_STATUS.md` — Phase 3 pre-gate status (GPU OK, ROS2 blocked)

---



## 6. Exit Criteria for “Done” (Phase 3)

- [ ] `aegis eval --sim isaaclab --model scripted` → 3/3 success, 0 violations
- [ ] `aegis eval --sim isaaclab --model smolvla_libero` → report with violations/recoveries, latency p50/p95, and honest success rate (comparison to MuJoCo)
- [ ] ROS2 bridge latency benchmark published (inside-WSL2 vs bridge)
- [ ] Per-joint safety limits shipped, uniform-limit warning removed
- [ ] Hardware dry-run checklist (e-stop, fallback, real-robot `report.json`) — or documented “no-go + reason”

---

*This file is the single source for current stage + partnership context. Update it at each phase gate.*