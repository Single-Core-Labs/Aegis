# Phase 1 Summary — MuJoCo POC (`aegis eval`)

Status: **COMPLETE.** Stop-and-review point; next phase not started.

## What works

- `aegis validate` — config load + validation, exit 0 (exit 2 on bad config).
- `aegis eval --model <model> --robot franka --sim mujoco --tasks pick-place`
  end-to-end on a Franka Emika Panda in MuJoCo (Menagerie model, motor
  actuators + gravity-compensated torque controller):
  - `scripted` policy: **3/3 episodes succeed, 0 safety violations**
    (`--seed 42`).
  - `random` policy: 3/3 fail with ~145 violations / recoveries and the
    correct "tighten safety limits" recommendation (`--seed 7`).
- Safety Gateway in the action path only: NaN check, measured velocity
  check, measured force check (effort above gravity feedforward), fallback
  = PID-to-home with `recovery_steps=50`, `recovery_mode=resume`.
- NDJSON telemetry (`episodes.jsonl`, `steps.jsonl`, `trajectory.jsonl`) +
  `report.json` (task counts, p50/p95 latency, violations, recoveries,
  stub GPU hours, rule-based recommendation, warnings) per run under
  `outputs/`.
- Determinism: same seed → byte-identical outcomes and report fields,
  verified by tests for both policies.
- Tests: `uv run pytest tests/test_eval.py -q` → **6 passed** (config
  validation exit codes, determinism x2, random-fails/scripted-succeeds).

## How the "scripted" policy works (brief)

Phase machine over 6 waypoints (approach → grasp → lift → pre-place →
place → retreat) computed per episode with damped least-squares IK on the
hand frame (z-down). Descent and lift are slowed (0.15 / 0.08 rad/s) so
fingertips don't shove the cube and the grip holds; the gripper stays
closed from grasp until retreat. This is a smoke-test policy, not a
general solution.

## What's stubbed / honest caveats

1. **"Velocity control" is torque control** — action is integrated into a
   joint-position target and tracked by `tau = qfrc_bias + Kp*err +
   Kd*vel_err` (Kp=40, Kd=5). Ideal gravity feedforward; a policy assuming
   a perfect velocity servo may still fail.
2. **Safety limits are uniform per-joint** (single velocity + force limit
   for all 7 joints) — flagged as a warning in every report.
3. **Recommendation is rule-based** (4 rules), not learned — flagged.
4. **GPU hours is a stub** (0.0, no GPU used) — flagged.
5. **Success is position-only** (object within 0.05 m of target XY after
   being lifted ≥ 6 cm off the table); no contact-force grasp detector,
   no orientation check.
6. **No ROS2, no real VLA/model loading, no Isaac Lab, no dashboard, no
   hardware, no RL training** — all per design-doc scope.
7. `run.json` (full config + git info) from the design doc was replaced by
   config being logged in the telemetry logger instead — minor deviation;
   `episodes.jsonl`/`steps.jsonl`/`trajectory.jsonl`/`report.json` match
   the design.

## Deviations from the design doc

- Top-level config keys are `model_name`/`robot_name`/`task_name` with
  component YAMLs in `configs/{models,robots,tasks}/` (design draft showed
  a different shape); the loader validates per-component files.
- Episode seeds are `seed + episode` (0,1,2 for seed 42); jitter is
  deterministic per seed.
- Gripper actuator in `panda_vel.xml` is a tendon motor remapped from the
  Menagerie `0..255` ctrl range with strengthened bias so the grip can
  hold the cube — the stock model's gripper is too weak to lift 0.125 kg.

## Go / no-go for Phase 2

- **Go.** The harness shape (gateway, fallback, eval loop, telemetry,
  report) is proven end-to-end with two policies, and the acceptance
  tests pass. Phase 2 can build hard per-joint limit layers and real
  limit tables on this skeleton without rework.
- Risks to carry forward: uniform-limit assumption must be replaced by
  per-joint tables; success/grasp detectors should gain contact-force
  grounding; torque-control caveat must be documented for any Phase 2
  policy contract.
- Open question (Phase 3, not blocking): ROS2 + real-robot eval requires
  a Linux/WSL2 host — confirm that environment before scoping hardware
  work.

## Example output (scripted, seed 42)

```
task          : pick-place
episodes      : 3   (success 3 / fail 0)
inference     : p50 0.01 ms  p95 0.021 ms   (per-step, cpu)
safety        : violations 0  recoveries 0
gpu hours     : 0.0 (stub — no GPU used in POC)
recommendation: all episodes succeeded — harness ready to evaluate real policies
report      : outputs\run-<timestamp>\report.json
```

Run it yourself: see `examples/run_poc.md`.