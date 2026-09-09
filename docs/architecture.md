# Architecture — Aegis Internals

This document is the engineering deep dive for contributors and integrators.

## Eval Loop

```
env.reset(seed)                          # seed + episode_id => deterministic placement
obs = env.observe()
for step in 1..max_steps:
    t0 = perf_counter_ns()
    raw_action = policy.act(obs)                  # timed: inference
    gated = gateway.filter(raw_action, obs)       # timed: gateway
    if gated.source == "fallback":
        gated = gateway.filter(fallback.act(obs), obs)
    obs, terminated, truncated, info = env.step(gated.command)  # timed: env
    logger.step(step, gated, info, latencies)     # NDJSON
    if terminated or truncated or wall_clock > timeout:
        break
```

- `terminated` → success iff `info["success"]` (object within `success_threshold_m` after grasp)
- `truncated` → fail (`max_steps` reached)
- Wall-clock timeout → fail (`episode_timeout_sec`)

See `src/aegis/eval/runner.py`, `src/aegis/telemetry/timing.py`.

## Safety Gateway

**Invariant:** No action reaches `env.step()` without passing `SafetyGateway.filter()` — enforced by construction in `EvalRunner`.

### Checks (in order, `src/aegis/safety/checks.py`)

1. **NaN/Inf** — any non-finite in action → `Violation("nan", ...)`
2. **Effort clamp** — `max_effort_action` with `action_clamp: clamp|reject`
3. **Measured velocity** — `|qvel[j]| > max_velocity[j]` → `Violation("velocity", ...)`
4. **Measured force** — `|torque[j]| > max_force[j]` → `Violation("force", ...)`

Per-joint limits accept `float | list[7]` (`src/aegis/config/models.py:68`). Single float = uniform (legacy, warns).

### On violation

- Increment `violation_count`, emit `safety_violation` event
- If `clamp` severity → clamp and continue
- Else → switch to **fallback** for `recovery_steps=50`, then `recovery_mode: resume|hold`

### Fallback

`src/aegis/safety/fallback.py` — PID-to-home, bounded `0.3 rad/s`, gravity-compensated. Intentionally dumb: demonstrable recovery, not task success.

## Env Protocol

`src/aegis/envs/base.py` — `Env` protocol: `reset(seed)`, `step(action)`, `observe()`, `render_images()`, `state_snapshot()`.

| Env | File | Backend | Cameras |
|---|---|---|---|
| `MujocoPickPlaceEnv` | `mujoco_pick_place.py` | MuJoCo 3.x, Menagerie Franka, `tau = qfrc_bias + 40*err + 5*vel_err` | 3 × 256×256 |
| `IsaacPickPlaceEnv` | `isaac_pick_place.py` | Isaac Sim PhysX + USD (or fallback to MuJoCo with `RuntimeWarning`) | Same API |

## Telemetry

`src/aegis/telemetry/logger.py` writes NDJSON under `outputs/<run_id>/`:

| File | Content |
|---|---|
| `run.json` | Full validated config + git/version + `started_at` |
| `episodes.jsonl` | `episode_start` / `episode_end` per episode |
| `steps.jsonl` | Per-step: action source, violation flag, latencies, qvel/torque |
| `trajectory.jsonl` | Per-step `qpos`/`qvel` snapshot |
| `report.json` | Aggregated summary (`src/aegis/eval/report.py`) |

Latency via `time.perf_counter_ns`, aggregated in `src/aegis/eval/metrics.py` (sorted p50/p95, no external deps).

## Config Validation

`src/aegis/config/models.py` — Pydantic `extra="forbid"` (typos fail), `src/aegis/config/loader.py` — YAML → validated `PhysicalAIYaml`. CLI flags override YAML (`CLI > YAML > defaults`). Exit `2` on validation error.

## Honesty Contract

If a sim/policy is not wired, `report.json: warnings` and `info["sim"]` say so explicitly. No fabricated numbers. See `agent.md:16`.
