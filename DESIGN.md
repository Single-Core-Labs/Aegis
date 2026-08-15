# PAI — Phase 1 (POC) Design Doc

Status: **DRAFT — awaiting approval.** No implementation code written yet.

Single command to prove: `aegis eval --model <model> --robot <robot> --sim mujoco --tasks pick-place`

---

## 1. Repo / Module Structure

```
Physical Harnesss/
├── pyproject.toml                 # uv project, entrypoint `aegis`
├── uv.lock
├── README.md                      # what works / what's stubbed (honesty contract)
├── physical-ai.yaml               # example POC config (checked in)
├── configs/
│   ├── robots/franka.yaml         # robot + safety limits
│   ├── tasks/pick-place.yaml      # task spec + success condition
│   └── models/random.yaml         # policy spec (random stand-in)
├── src/aegis/
│   ├── __init__.py
│   ├── cli.py                     # Typer app, `aegis eval` / `aegis validate`
│   ├── config/
│   │   ├── models.py              # Pydantic schema (section 2)
│   │   └── loader.py              # YAML → validated models
│   ├── envs/
│   │   ├── base.py                # Env protocol: reset(), step(action) → (obs, reward, terminated, truncated, info)
│   │   └── mujoco_pick_place.py   # MuJoCo wrapper (menagerie franka + box)
│   ├── policies/
│   │   ├── base.py                # Policy protocol: act(obs) → action, name
│   │   ├── random.py              # random policy (POC stand-in)
│   │   └── scripted.py            # scripted pick-place policy (used for smoke test)
│   ├── safety/
│   │   ├── gateway.py             # SafetyGateway (section 3)
│   │   ├── checks.py              # velocity / force / NaN checks
│   │   └── fallback.py            # PID-to-home fallback controller
│   ├── eval/
│   │   ├── runner.py              # episode loop (section 4)
│   │   ├── metrics.py             # p50/p95 latency, counts
│   │   └── report.py              # report.json + stdout summary
│   └── telemetry/
│       ├── logger.py              # structured NDJSON logger
│       └── timing.py              # monotonic timing context manager
└── examples/
    └── run_poc.md                 # exact commands + expected outputs
```

Python 3.11+, `uv` for env management, Typer for CLI. Package name `aegis`, module layout `src/`.

---

## 2. `physical-ai.yaml` Schema (Pydantic)

`config/models.py` defines these models (`model_config = ConfigDict(extra="forbid")` — typos fail loudly). Fields below with types and constraints.

### 2.1 Top level — `PhysicalAIYaml`

| Field | Type | Constraint | Default |
|---|---|---|---|
| `schema_version` | `int` | `ge=1` | `1` |
| `eval` | `EvalSpec` | required | — |
| `model` | `ModelSpec` | required | — |
| `robot` | `RobotSpec` | required | — |
| `environment` | `EnvSpec` | required | — |
| `task` | `TaskSpec` | required | — |
| `output` | `OutputSpec` | required | — |

CLI flags override YAML values (CLI > YAML > defaults).

### 2.2 `EvalSpec`

| Field | Type | Constraint | Default |
|---|---|---|---|
| `episodes` | `int` | `1..1000` | `10` |
| `seed` | `int` | any | `42` |
| `max_steps_per_episode` | `int` | `1..100000` | `500` |
| `time_step` | `float` | `gt=0` | `0.02` (sim dt) |
| `episode_timeout_sec` | `float` | `gt=0` | `10.0` (wall clock per episode) |
| `inference_mode` | `Literal["cpu", "cuda"]` | — | `"cpu"` |
| `inference_budget_ms` | `float` | `gt=0` | `200.0` — per-step inference budget; an over-budget step is counted like a safety violation and the fallback runs for that step |

### 2.3 `ModelSpec`

| Field | Type | Constraint | Default |
|---|---|---|---|
| `name` | `str` | non-empty | — |
| `kind` | `Literal["random", "scripted", "smolvla"]` | — | — |
| `policy` | `RandomPolicySpec \| ScriptedPolicySpec \| SmolVLAPolicySpec` | discriminator `kind` | — |
| `endpoint` | `str` | required when `kind="smolvla"`, forbidden otherwise | `""` |
| `load_params` | `dict` | free-form loader params | `{}` |

`RandomPolicySpec`: `{seed: int}`. `ScriptedPolicySpec`: `{home_position: list[float] | null, grasp_steps: int=40, ...}` — placeholder fields, finalized when the scripted smoke-test policy is written. `SmolVLAPolicySpec`: `{instruction: str, cameras: list[str]=["camera1","camera2","camera3"]}` — LeRobot SmolVLA policy (endpoint is an HF repo id, e.g. `lerobot/smolvla_libero`); the harness applies the checkpoint's bundled pre/post processor pipelines verbatim and converts its 7-D cartesian-delta actions to joint velocities with a damped least squares resolved-rate controller.

### 2.4 `RobotSpec` / `SafetyLimits`

| Field | Type | Constraint |
|---|---|---|
| `name` | `str` | non-empty |
| `mjcf_path` | `Path` | must exist (menagerie `panda` model for POC) |
| `safety` | `SafetyLimits` | required — **no config, no run** |
| `safety.max_velocity` | `float` | `gt=0`, rad/s per joint (applied to all joints in POC) |
| `safety.max_force` | `float` | `gt=0`, N·m per joint (applied to all joints in POC) |
| `safety.max_effort_action` | `float` | `gt=0`, clamp on commanded torque, `inf` allowed |
| `safety.reject_nan_actions` | `bool` | — default `true` |
| `safety.action_clamp` | `Literal["clamp", "reject"]` | default `"reject"` — see section 3 |

### 2.5 `EnvSpec`

| Field | Type | Constraint |
|---|---|---|
| `sim` | `Literal["mujoco"]` | — |
| `scene_mjcf` | `Path` | must exist |
| `robot_name` | `str` | matches robot in scene |
| `control_mode` | `Literal["position_delta", "joint_velocity"]` | — POC uses `joint_velocity` |
| `render_cameras` | `list[str]` | empty = no per-step images | — cameras rendered at 256x256 into `obs["images"]` each step when set; vision policies (SmolVLA) instead render lazily at chunk boundaries via `env.render_images()` |

### 2.6 `TaskSpec`

| Field | Type | Constraint |
|---|---|---|
| `name` | `Literal["pick-place"]` | — |
| `success_threshold_m` | `float` | `gt=0`, default `0.05` (object center within X m of target center) |
| `object_name`, `target_name` | `str` | must exist in scene |
| `max_grasp_attempts` | `int` | `1..10`, default `3` |

### 2.7 `OutputSpec`

| Field | Type | Constraint |
|---|---|---|
| `dir` | `Path` | default `./outputs` |
| `run_id` | `str` | auto `YYYYMMDDTHHMMSSZ` |
| `report_format` | `Literal["json"]` | POC: JSON only |

**Validation rule (loader):** `safety` limits must be finite and positive; `mjcf_path` and `scene_mjcf` must exist at load time. Fail fast, exit code 2 with a readable message.

---

## 3. Safety Gateway

Sits between policy output and `env.step()`. **There is no code path where an action reaches the sim without passing through the gateway.** Enforced by construction: the eval runner only calls `env.step(SafetyGateway.filter(...))`.

### Interface (protocol shape)

```
SafetyGateway(limits: SafetyLimits)
  filter(action, state) -> GatedAction
```

`GatedAction = (command: np.ndarray, source: Literal["policy", "fallback"], violated: bool, violation: Violation | None)`

### Checks (in order)

1. **NaN/Inf check** — if any non-finite value in action → immediate violation (severity: fatal).
2. **Commanded force/effort clamp** — if `action` exceeds `max_effort_action` and `action_clamp="clamp"`, clamp in place and record a `CLAMP` event (counted as a violation, action proceeds). If `"reject"`, treat as violation.
3. **Measured velocity check** — from `state` (sim qvel). Any joint `> max_velocity` → violation.
4. **Measured force check** — from `state` (actuator torque). Any joint `> max_force` → violation.

The gateway checks **measured** sim quantities (velocity/torque) plus **commanded** quantities (action bounds). This keeps it meaningful even for a random policy that would otherwise produce junk commands.

### On violation

- Increment `violation_count`, emit structured log event:
  `{event: "safety_violation", type, joint, value, limit, step, episode, ts}`
- If severity allows recovery (`clamp` only): apply clamp, continue policy.
- Otherwise: **switch source to fallback** for `recovery_steps` (constant, `= 50`, not yet configurable — flag in risks), then:
  - `recovery_mode="resume"` → policy resumes afterwards
  - `recovery_mode="hold"` → fallback holds until episode end (default for POC: `"resume"`)
- Each switch event increments `recovery_event_count`.

### Fallback controller

Trivial **PID-to-home** controller: joints driven to the robot's home pose (from MJCF) with bounded velocity, using only measured state. It is intentionally dumb — its job is *demonstrable recovery*, not task success. If the fallback is active at episode end, the episode is a fail (task not completed).

---

## 4. Eval Loop Architecture

### Per-episode flow (`EvalRunner.run_episode`)

```
env.reset(seed)                        # log episode_start
obs = env.observe()
for step in 1..max_steps_per_episode:
    t0 = now()
    raw_action = policy.act(obs)                  # timed: inference latency
    gated = gateway.filter(raw_action, obs)       # timed: gateway latency
    if gated.source == "fallback":                # fallback action computed here
        gated = gateway.filter(fallback.act(obs), obs)
    obs, terminated, truncated, info = env.step(gated.command)   # timed: env step
    logger.step(step, gated, info, latencies)     # NDJSON line per step
    if terminated or truncated or wall_clock > episode_timeout_sec:
        break
```

- `terminated` → success iff `info["success"]` (object within `success_threshold_m` of target at end, after a valid grasp).
- `truncated` (timeout / max steps) → fail with reason `"truncated"`.
- Wall-clock timeout per episode → fail with reason `"timeout"`.

### What gets logged (all NDJSON, under `output/<run_id>/`)

| File | Contents |
|---|---|
| `run.json` | full validated config + git/version info + `started_at` |
| `episodes.jsonl` | one line per episode: id, seed, outcome (`success/fail/timeout`), steps, duration, violations, recoveries |
| `steps.jsonl` | per-step: step, action source, violation flag, latencies (inference/gateway/env), joint velocities/torques (for later replay) |
| `trajectory.jsonl` | per-step full state snapshot (qpos/qvel) — POC keeps it minimal |
| `report.json` | section 5 summary (also printed to stdout) |

Timing via `time.perf_counter_ns`; latency recorded per step, aggregated in `metrics.py` (p50/p95 via sorted samples, no external deps).

### Honesty rule (from agent.md)

If the sim or a policy isn't wired, the report **says so explicitly** in a `warnings` field — no fabricated numbers, ever.

---

## 5. CLI Surface

```
aegis eval --model <model> --robot <robot> --sim mujoco --tasks pick-place [options]

Options (all optional; defaults from physical-ai.yaml):
  --model str            model name (configs/models/<name>.yaml); default: random
  --robot str            robot name (configs/robots/<name>.yaml); default: franka
  --sim str              must be "mujoco" in POC; default: mujoco
  --tasks str            comma-separated; must be "pick-place" in POC; default: pick-place
  --episodes int         override YAML; default from config (10)
  --seed int             override YAML; default from config (42)
  --max-steps int        override YAML
  --config Path          alternate physical-ai.yaml; default: ./physical-ai.yaml
  --output-dir Path      override YAML; default: ./outputs
  --run-id str           override auto-generated run id
  --verbose              debug logging to stderr
```

Secondary command: `aegis validate --config <path>` — load + validate config, exit 0/2. (No `list-*` commands in POC; models/robots/tasks are documented in README.)

### Exit codes

| Code | Meaning |
|---|---|
| 0 | run completed (even if all episodes failed — a completed honest run is not an error) |
| 2 | config/validation error |
| 3 | internal error (crash mid-run, partial logs preserved) |

### Stdout summary (human readable) + `report.json`

```
task          : pick-place
episodes      : 10   (success 6 / fail 3 / timeout 1)
inference     : p50 4.2 ms  p95 8.9 ms   (per-step, cpu)
safety        : violations 12  recoveries 4
gpu hours     : 0.0 (stub — no GPU used in POC)
recommendation: <line>
```

`report.json` fields (matches product notes): `task_counts`, `latency_p50_ms`, `latency_p95_ms`, `safety_violations`, `recovery_events`, `gpu_hours` (always `0.0` + `"note": "stub"` in POC), `recommendation` (string, rules: all-success → "ready to evaluate real policies"; failures dominated by truncation → "raise max steps"; violations high → "tighten safety limits / policy unusable"; mixed → "random policy expected to fail — harness validated"), `warnings: []`.

---

## 6. Explicitly OUT OF SCOPE (POC)

- **No ROS2** — nothing robot-message-flavored; sim-only loop.
- **No real VLA/model loading** — only `random` / `scripted` stand-in policies. No HF hub, no checkpoints, no batching.
- **No GPU orchestration** — `gpu_hours` is a hard-coded stub `0.0` with a note; no CUDA, no `torch`, no Isaac.
- **No Isaac Lab** (Phase 3+ target).
- **No dashboard / UI / web service** — JSON logs + stdout only.
- **No real robot hardware, no e-stop hardware layer** — safety gating is action-space only.
- **No multi-robot, multi-task batching, or parallel episodes.**
- **No RL training / reward learning / dataset collection.**
- **No checkpointing or resume of interrupted runs** (crash → exit 3, logs preserved).
- **No observability stack** (Prometheus/Grafana/OTel) — structured logs are the contract.
- **No config discovery (`aegis list-models`), no plugin system.**
- **No gripper force/tactile modeling, no object randomization beyond fixed seed.**
- **No report formats other than JSON.**

---

## 7. Open Risks / Assumptions — need your confirmation

1. **Action space**: I assume `joint_velocity` control (7-DOF arm velocities + binary/velocity gripper command). If you intend position-delta control, the gateway force check changes meaning — confirm.
2. **Safety limits semantics**: `max_velocity`/`max_force` applied **per-joint, uniformly** to all 7 joints in POC. Real robots have per-joint limits — acceptable for POC?
3. **Fallback behavior**: trivial PID-to-home, `recovery_steps=50`, then policy resumes (`recovery_mode="resume"`). Fine? Or should a violated episode just end?
4. **Success criterion**: object center within `0.05 m` of target center at episode end after a grasp — no orientation check. OK?
5. **Grasp detection**: object lifted off table (`z > threshold`) counts as "grasped"; no contact-force modeling. OK?
6. **Sim model**: use MuJoCo Menagerie `franka_panda` (MIT license) + a box in a scene I compose. Acceptable, or do you have a scene/model already?
7. **MuJoCo physics dt**: fixed `0.02 s`, control at same rate (no control decimation). Acceptable for POC?
8. **"Recommendation" line**: generated by the 4 simple rules in section 5 — fine, or do you want it to be a fixed template?
9. **`action_clamp="reject"` default**: clamping-by-default (more permissive) vs reject-by-default (stricter). I chose strict — confirm.
10. **Windows dev machine**: all paths/mkdir logic must be Windows-safe (we're on `A:\Physical Harnesss`). MuJoCo wheels support Windows — any constraint on your side (e.g., Linux-only tooling later)?

---

*End of design doc. Awaiting review/approval before implementation.*