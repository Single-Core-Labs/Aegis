# CURRENT STAGE — Aegis Physical AI Safety Harness

> **Date:** 2026-09-14 | **Repo:** `A:\physical-ai-harness` | **Version:** 0.1.0 (`pyproject.toml`)
> **PRD:** `PRD.md v1.0` — Phase 1 & 2 Complete, Phase 3 Scaffold Complete, Phase 3 Real Runtime Pending
> **Entry:** `aegis eval --model <policy> --sim <mujoco|isaaclab> --episodes N` (`src/aegis/cli.py`)

---

## 1. Current stage in one paragraph

**Aegis is a working, tested sim-only safety harness.** The full loop
`Policy → SafetyGateway → MuJoCo → report.json + NDJSON` runs deterministically
today for 3 policies (`random`, `scripted`, `smolvla_libero`), with per-joint
Franka limits, PID-to-home fallback, inference-budget enforcement, ROS 2 mock
bridge, Isaac Lab honest-fallback scaffold, and OTel/Prometheus/Grafana telemetry.
Beyond-Phase-3 incremental slices are in: sequential `aegis eval-batch`
(robots×models×tasks, one isolated env per combo), seeded MuJoCo domain
randomization (`--dr`: light/friction/camera jitter), and structured
recommendation logging (`heuristic_v1` label + `features` dict per report for the
future `learned_v1` classifier). CLI `--help` crash fixed (typer 0.15.4→0.27.1,
floor pinned `typer>=0.16` in `pyproject.toml`).
What is **not** real yet: Isaac Sim PhysX/USD execution (falls back to MuJoCo with
`RuntimeWarning`), real `rclpy` ROS 2 (mock only on this machine), and any hardware
dry-run — Go/No-Go decision is explicitly **No-go** until Gates 1–4 pass.

---

## 2. Problem we are solving

VLA models (SmolVLA / RT-2 / π0 class) are powerful but unsafe by default:

| Gap today | Impact | Aegis answer |
|---|---|---|
| No standard safety layer between policy and actuator | Broken gears, unsafe contacts, no attribution | **Mandatory Safety Gateway** — every action filtered, no bypass by construction (`src/aegis/safety/gateway.py:28`) |
| Bespoke eval scripts per team | Results incomparable, failures unattributable | **Unified deterministic eval** — `aegis eval`, seeded `seed+episode`, same `report.json` schema across sims (`src/aegis/eval/runner.py:80`, `report.py`) |
| Sim-to-real gap unmeasured (train sim ≠ deploy sim) | LIBERO-trained model fails silently in new sim | **Honest cross-sim A/B** — same policy/limits on MuJoCo vs Isaac Lab + relaxed-limit diag config to separate limit-interruption from vision gap (`physical-ai-diag.yaml`, `configs/robots/franka_diag.yaml`) |

Design principle (from `agent.md:18` + `README.md:42`): **honest by default**.
Mock/fallback/stub is always labeled in `warnings` / `info["sim"]`. No fabricated numbers.

---

## 3. Solution we are building

```
Policy —action→ [ AEGIS Safety Gateway + Eval Harness ] —safe action→ Robot
                  checks 1-4 + budget → fallback 50 steps → env.step → NDJSON
```

`PRD.md §8` / `README.md §Architecture`. Concrete flow in code:

1. `aegis eval` (`src/aegis/cli.py:48`) loads + validates YAML (`src/aegis/config/loader.py:load_run_config`), selects backend (`mujoco` | `isaaclab`), policy (`random` | `scripted` | `smolvla`), builds `PidToHomeFallback` + `SafetyGateway`, runs `EvalRunner.run()`.
2. `EvalRunner._run_episode` (`src/aegis/eval/runner.py:88`): `env.reset(seed+ep)` → `policy.act(obs)` timed (CUDA events when `cuda`, else `perf_counter_ns`) → budget check → `gateway.filter()` → `env.step(gated.command)` timed → `RunLogger.step/trajectory` (NDJSON).
3. Success = `grasped_ever AND dist<=0.05m`; grasp = `lifted (z>table+0.025)` AND finger-object contacts with `mj_contactForce>0.5N`, sticky (`src/aegis/envs/mujoco_pick_place.py:68` per PRD).
4. Outputs `outputs/<run_id>/{run.json, episodes.jsonl, steps.jsonl, trajectory.jsonl, report.json}` (`src/aegis/telemetry/logger.py`), stdout summary (`src/aegis/eval/report.py:build_report/print_summary`).
5. Exits: `0` honest run (even `0/3`), `2` config error, `3` internal (partial logs preserved).

---

## 4. Codebase walkthrough (what each part does)

| Module | Path | What it does | Key detail |
|---|---|---|---|
| CLI | `src/aegis/cli.py` | Typer `aegis` with `eval`, `eval-batch`, `validate`, `rosbench` | Backend select `mujoco\|isaaclab`, overrides CLI>YAML>defaults, VRAM-safe `--headless --quantize`, DR `--dr/--no-dr`, exits 0/2/3; `typer>=0.16` floor (click-8.2 `--help` fix) |
| Config schema | `src/aegis/config/models.py` | Pydantic `PhysicalAIYaml`: `eval, model, robot, environment, task, output`, `extra="forbid"` | Per-joint `float\|list[7]` limits (`:68` per PRD); `ModelSpec.kind random\|scripted\|smolvla`; `EvalSpec episodes 1..1000, inference_mode cpu\|cuda, budget>0` |
| Config loader | `src/aegis/config/loader.py` | `load_run_config()` merges root + `configs/{models,robots,tasks}/`, validates asset paths exist | Typo/ missing MJCF → exit `2` |
| Safety checks | `src/aegis/safety/checks.py` | Pure functions `check_finite, check_velocities, check_forces` → `Violation` | Operate on **measured** `qvel/qfrc`, not just command |
| Safety gateway | `src/aegis/safety/gateway.py` | `filter()` order: 1.NaN/Inf → 2.effort clamp/reject → 3.measured vel → 4.measured force; `budget_violation()`, `model_error()` same path | Violation → fallback `recovery_steps=50`, mode `resume\|hold`; fallback action re-checked (`:136-164`) |
| Fallback | `src/aegis/safety/fallback.py` | `PidToHomeFallback: tau = qfrc_bias + 40·err + 5·vel_err`, bounded `0.3 rad/s`, gravity-compensated | CLI caps at `min(0.3, min(limits))` (`cli.py:166`) |
| Eval runner | `src/aegis/eval/runner.py` | Episode loop, per-kernel GPU timing, timeout/truncate/terminate handling | `seed+ep` determinism; budget-over → gateway path, never silent drop (`:146-151`) |
| Metrics/report | `src/aegis/eval/metrics.py`, `report.py`, `recommendation.py`, `batch.py` | p50/p95 latency, `report.json` contract (run_id, task_counts, violations, recoveries, gpu_hours, `recommendation` string + structured `recommendation_label/evidence/features`, `recommendation_model: heuristic_v1`, DR flag, warnings) | `gpu_hours` = per-kernel `inference_s` sum when `cuda` else `0`; batch aggregates nested `task_counts[task][robot][model]` + `summary` |
| MuJoCo env | `src/aegis/envs/mujoco_pick_place.py` | Menagerie Franka `panda_vel.xml`, 7 vel + tendon gripper `0..255`, torque-PD, 3 cams `256×256` at chunk boundaries, seeded DR (light ±0.1m / friction ±0.002 / camera ±0.02m, nominals restored per reset, deltas in `episodes.jsonl` + step `info`), `Env` protocol `reset/step/observe/render_images/state_snapshot` | Contact-force grasp logic; DR measurably stresses scripted (seed 42: 1/1 → 1/2), byte-identical across repeats |
| Isaac env | `src/aegis/envs/isaac_pick_place.py` | Same `Env` API; without Isaac Sim 6.0.1 → fallback MuJoCo + `RuntimeWarning` + `info["sim"]="isaaclab-fallback-mujoco"`; with Sim → PhysX+USD (`NotImplementedError:44` slot for `workflows/agentic/arena/run.sh --create-env`) | ~6.1 KB scaffold, not real runtime |
| Policies | `src/aegis/policies/` | `random.py` uniform vel (negative control); `scripted.py` 6-phase DLS-IK state machine (~11 KB); `smolvla.py` 450M `lerobot/smolvla_libero`, 8-D state, 7-D cartesian-delta, 3 cams, chunk-50, DLS adapter `0.5 rad/s` clamp, budget-enforced (~8 KB) | Checkpoint pipelines applied verbatim |
| ROS 2 bridge | `src/aegis/ros2/bridge.py` | `RosBridge` auto-detects `rclpy`; topics `/aegis/action` (Float64MultiArray 8-D), `/aegis/state`, `/aegis/safety` (JSON); mock in-memory when absent; `benchmark_latency(n)` | `rosbench --mock p50 0.000 p95 0.002ms`; `--real` needs WSL2 install |
| Telemetry | `src/aegis/telemetry/{logger,timing,otel}.py` | NDJSON logger, `perf_counter_ns` timing, OTel metrics `aegis_episodes_success, safety_violations, recovery_events, latency_p50/p95, gpu_hours, step_violation` + `grafana/aegis_dashboard.json` | Structured JSON from day one per `agent.md` |
| Configs | `physical-ai.yaml`, `physical-ai-diag.yaml`, `configs/robots/{franka,franka_uniform,franka_diag}.yaml`, `configs/models/{random,scripted,smolvla_libero}.yaml`, `configs/tasks/pick-place.yaml` | Declarative only — no hardcoded robot/model params in code | Default per-joint vel `[2.175×4, 2.61×3]`, force `[87×4, 12×3]`; diag relaxes limits + `budget 10000ms` for A/B |
| Tests | `tests/test_eval.py`, `test_policy_model.py`, `test_seed_sweep.py` (20 tests per PRD/README) | Config exit-2, determinism, random-fails/scripted-succeeds, budget/adapter/SmolVLA config, 5 seeds×5 eps ≥90% | **20/20 pass** (2026-09-14, system Python 3.13 editable install; see §6) |
| Docs | `docs/{nvidia-stack-manual,architecture,safety-gateway,isaac-lab,ros2-bridge,dashboard,perf-tuning,batching,camera-calibration,learned-recommendation,README}.md`, `whitepaper/`, `examples/`, `assets/scenes/pick_place.xml + menagerie/` | NVIDIA-stack manual is entry for Isaac/ROS2/WSL2/USD/VRAM-safe; `perf-tuning.md` = 6 GB checklist (applied in `b2f2941`) | Phase docs removed for maintainability (`53157ae`) |

---

## 5. Config & data at a glance

- Root defaults `physical-ai.yaml`: `model random, robot franka, task pick-place, eval.episodes 10 seed 42 max_steps 2500 dt 0.02 timeout 60s cpu/2000ms, env mujoco, output ./outputs`.
- Diag A/B `physical-ai-diag.yaml`: `smolvla_libero + franka_diag, 2 eps, 600 steps, cuda, budget 10000ms` — isolates limit vs vision failure.
- DR defaults in `configs/tasks/pick-place.yaml`: `domain_randomization: false` + ranges (`dr_light_jitter_m 0.1`, `dr_friction_delta 0.002`, `dr_camera_jitter_m 0.02`); override with `--dr/--no-dr`.
- Override chain: CLI flags > YAML > defaults; `extra="forbid"` catches typos.
- `outputs/` already has ~50 prior runs (`run-20260815T*`, `run-20260909T*`, `step6-{isaac,mujoco,random}`, `test-{isaac,mujoco}`) — evidence of repeated benchmarking.

---

## 6. Proven behavior (benchmarks — reproduce on seed 42/7)

| Sim | Policy | Result | Meaning |
|---|---|---|---|
| MuJoCo | `scripted` 3 eps seed 42 | **3/3, 0 violations** | Smoke baseline, per-joint limits |
| MuJoCo | `random` 3 eps seed 7 | **0/3, ~4 viol / 2 rec** (was 145 on legacy uniform 1.0) | Negative control — fallback engages |
| MuJoCo | `smolvla_libero` cuda 3 eps | **0/3, ~33 viol/ep (1 budget + 32 vel)**; still 0/3 at 5.0 rad/s | Vision domain gap (hand 0.16m→0.60m, mid-air grips), not just limits |
| Isaac fallback | `scripted` 3 eps | **3/3 + `RuntimeWarning`, `isaaclab-fallback-mujoco`** | Honest scaffold, no silent mock |

Latency: scripted `p50 ~0.012 p95 ~0.034ms` cpu; SmolVLA `p50 1.29 p95 1.87ms` amortized, chunk `0.35-0.41s`, cold `5.09s` (trips 2000ms budget on step 1 — proven); ROS mock `p50 0.000 p95 0.002ms`.

New slices (2026-09-14, verified live): `eval-batch` 2 combos (scripted+random, seed 7) → `success 1/fail 1`, combo seeds 7/1007, nested `task_counts` + `summary` ✓; `--dr` scripted 2 eps seed 42 → 1/2 with DR warning, byte-identical across repeats ✓.

Test + env status (2026-09-14): **20/20 pass** on system Python 3.13 (editable install pointing at `src/`). `--help` crash fixed by upgrading typer 0.15.4→0.27.1 (matches `uv.lock`); `pyproject.toml` floor pinned `typer>=0.16` so pip users can't land on the broken old-typer/new-click combo. Known pre-existing env drift (not touched): `pillow 12.2.0 vs required 12.3.0`, `transformers 4.57.6 vs 5.15.0` — runtime + tests green regardless. `uv run` unusable from Windows PowerShell here (`.venv` is WSL-style with `bin/`, and `uv` tries to recreate it but can't remove the `lib64` symlink — Access denied); housekeeping item: recreate the venv from inside WSL2 or set `UV_PROJECT_ENVIRONMENT`. `outputs/verify-batch/` artifacts are git-ignored. `git status`: 12 modified + 3 new files (`CURRENT_STAGE.md`, `eval/batch.py`, `eval/recommendation.py`).

---

## 7. Done vs scaffold vs pending

| Phase | Status |
|---|---|
| Phase 1 MuJoCo POC (gateway, scripted/random, eval, reports) | ✅ Done |
| Phase 2 SmolVLA-450M + budget + DLS adapter | ✅ Done |
| Phase 3 scaffold (per-joint limits, ROS2 mock, Isaac fallback, checklist, contact-force grasp, per-kernel gpu_hours, dashboard/OTel) | ✅ Done |
| Phase 3 **real runtime**: real `rclpy` ROS 2 Humble, Isaac USD+PhysX VRAM-safe, hardware dry-run | 🔲 Pending |
| Beyond Phase 3: Isaac 6 GB perf validation, multi-robot batching (spec `docs/batching.md`), learned recommendation (spec, current 4 rules), camera calibration + DR (spec), RL/dataset (backlog) | 🔲 Spec/partial |
| Beyond Phase 3 incremental slices (2026-09-14, MuJoCo-only) | ✅ Done: sequential `aegis eval-batch` (`src/aegis/eval/batch.py`), MuJoCo DR `--dr` (`MujocoPickPlaceEnv`), recommendation `features` + `heuristic_v1` label (`src/aegis/eval/recommendation.py`). Deferred: parallel batching, P3a USD calibration, `learned_v1` training, RL |
| GR00T N1.7 Phase A (2026-09-14) | ✅ Done: `GrootPolicySpec` + `kind: groot`, `configs/models/groot_n17.yaml` (`LIBERO_PANDA`), `policies/groot.py` honest stub (exit 2 without checkpoint), CLI factory arm, 4 new tests — suite 24/24 green. Phase B (in-process loading) pending |

---

## 8. What's left to build (ordered)

1. **Phase 3 real — ROS 2 Humble (`rclpy`).** WSL2 Ubuntu 22.04 fully-inside-WSL2, fix `curl -k` apt issue, verify `ros2 topic list`, `aegis rosbench --real p95 < budget`. Docs: `docs/nvidia-stack-manual.md`, `docs/ros2-bridge.md`.
2. **Phase 3 real — Isaac Sim 6.0.1 + Lab 2.x USD + PhysX.** Author `pick_place` USD (`workflows/agentic/arena/run.sh --create-env`), replace `NotImplementedError:44` in `isaac_pick_place.py`, apply `docs/perf-tuning.md` VRAM-safe checklist (headless, RTX bounces=1, int8 SmolVLA `--quantize int8`), re-run SmolVLA A/B with LIBERO-matched cameras/lighting to close vision gap. Open Q for NVIDIA: 6 GB (RTX 4050) vs 16 GB official min.
3. **Hardware dry-run + Go/No-Go Gates 1–4** (`PRD.md §12`): E-stop+barriers, software bridge, gravity-comp empty-hand + NaN-injection → fallback-home with `sim:hardware` report, signed Go/No-Go. Until then: **No-go, no fabricated hardware report.**
4. **Beyond Phase 3 remaining (spec → build):** parallel batching (`--parallel N`, Isaac `num_envs`), learned `learned_v1` classifier (features are now logged — needs Isaac+ROS2 runs to train), P3a USD camera calibration, RL training/dataset collection, Isaac 6 GB perf numbers, Prometheus/Grafana live validation.
5. **Housekeeping:** recreate WSL-style `.venv` from inside WSL2 (or set `UV_PROJECT_ENVIRONMENT`) so `uv run pytest` works again; refresh `pytest_output.txt`; resolve `torch cu128` Windows-vs-WSL2 env split + pillow/transformers drift; keep `PRD.md` as single source (update on any deviation).

Open questions for NVIDIA partnership (`PRD.md §13`): VRAM-safe Isaac on 6 GB, ROS 2 WSL2 install preference (apt vs `osrf/ros:humble-desktop`), expected `report.json`/latency SLOs, Isaac Sim vs Lab preference, Franka assets/office-hours.

---

## 9. Honest limitations (do not present as done)

- Velocity control is torque control (`tau = qfrc_bias + 40·err + 5·vel_err`) — gravity-compensated PD, not perfect servo (`README.md §Honest Notes`).
- Recommendation is 4 hand-written rules, exposed as structured `heuristic_v1` (`label/confidence/evidence` + `features`) — not learned; grasp/ROS features are `None` (unknown), never fabricated.
- `gpu_hours` is `0.0` on cpu; per-kernel sum only on cuda.
- Gripper tendon `0..255` remap; success needs both lift AND `>0.5N` contact.
- Legacy `franka_uniform.yaml` kept only for A/B; reports warn when active.

---

## 10. How to verify from here

```bash
uv sync
uv run aegis validate
uv run aegis eval --model scripted --episodes 3 --seed 42        # expect 3/3
uv run aegis eval --model random --episodes 3 --seed 7           # expect 0/3 + violations
uv run aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42  # expect fallback warning
uv run aegis eval-batch --models scripted,random --robots franka --episodes 1 --seed 7  # expect 2 combos, 1/1
uv run aegis eval --model scripted --episodes 2 --seed 42 --dr   # expect DR warning + stressed baseline
uv run aegis rosbench --mock --n 20
python -m pytest -q   # via uv venv, expect 20 passed
```

---

## 11. Canonical file index

`PRD.md`, `README.md`, `agent.md`, `physical-ai.yaml`, `physical-ai-diag.yaml`,
`configs/robots/franka.yaml`, `configs/models/*.yaml`, `configs/tasks/pick-place.yaml`,
`assets/scenes/pick_place.xml`, `src/aegis/{cli,config,envs,policies,safety,eval,ros2,telemetry}/`
(new: `src/aegis/eval/{batch,recommendation}.py`),
`docs/*.md` (11 files), `whitepaper/`, `tests/*.py` (3 files, 20 tests), `outputs/<run_id>/`.
