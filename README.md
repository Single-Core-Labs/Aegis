# Physical AI Harness (aegis)

A minimal, honest harness for evaluating robot-control policies against
physical safety constraints. Phase 1 POC: one task (pick-and-place on a
Franka Emika Panda in MuJoCo), a safety gateway between the policy and the
simulated robot, an eval CLI that emits a report, and an end-to-end example.

## What you can run

```bash
# validate the config
uv run aegis validate

# evaluate the scripted (smoke-test) policy: should succeed, 0 violations
uv run aegis eval --model scripted --episodes 3 --seed 42

# evaluate a random policy: should fail, with safety violations + recoveries
uv run aegis eval --model random --episodes 3 --seed 7
```

Each eval writes a timestamped run directory under `outputs/` with:

- `report.json` — task counts, latency p50/p95, safety violations,
  recoveries, stub GPU hours, rule-based recommendation;
- `episodes.jsonl` / `steps.jsonl` / `trajectory.jsonl` — event logs.

Exit codes: `0` OK, `2` config error, `3` internal error.

## Architecture

```
policy  --8-dim action (7 joint velocities + gripper)-->  safety gateway
                                                            | (checks + fallback)
env (MuJoCo pick-place)  <--  clamped/torque action <-------+
```

- `src/aegis/policies/` — `scripted` (deterministic phase machine + DLS IK)
  and `random` (smoke-test policy that trips safety).
- `src/aegis/safety/` — `checks` (velocity/force limits), `fallback`
  (PID-to-home), `gateway` (rejects violations, resumes policy after
  recovery).
- `src/aegis/envs/mujoco_pick_place.py` — the Franka + cube + target scene.

## Honest implementation notes (POC caveats)

1. **"Velocity control" is torque control.** The Franka velocity-mode
   action (rad/s) is integrated into a joint-position target and tracked
   by a gravity-compensated torque PD (`tau = qfrc_bias + Kp*err + Kd*vel_err`,
   Kp=40, Kd=5) — the same idea as a real Franka controller, but simulated
   with ideal gravity feedforward. Success rate is not guaranteed for
   policies that assume a perfect velocity servo.
2. **Safety limits are uniform per-joint** (one velocity, one force limit
   for all 7 arm joints) — a POC simplification. Real robots need per-joint
   limits; this is flagged as a warning in every report.
3. **The recommendation line is rule-based** (4 hand-written rules), not
   learned.
4. **GPU hours is a stub** (0.0) — no GPU is used in this POC.
5. **Success is position-only** within 0.05 m of the target marker, plus a
   "was once lifted" grasp check (object center 6 cm above the table).
   There is no contact-force grasp detector; the "grasp" is a z-height
   threshold.
6. **Determinism** is guaranteed by fixed seeds (one per episode, `seed +
   episode`) and a deterministic policy; see `tests/test_eval.py`.
7. The gripper actuator in `panda_vel.xml` is a tendon motor remapped from
   the Menagerie model (`0..255` ctrl = open..close) with a strong bias so
   the grip can actually hold the cube; the arm joints are torque motors.

## Tests

```bash
uv run pytest tests/test_eval.py -q
```

Covers config validation exit codes, seed determinism for both policies,
and the POC acceptance check (scripted succeeds with zero safety
violations; random fails with violations).

## Roadmap / phases

- Phase 1 (this repo): MuJoCo-only POC, as described above.
- Phase 2: hard-limit layered safety (joint position/velocity/torque
  limits), real per-joint limits, recovery to the fallback pose.
- Phase 3: ROS2 bridge and hardware (in-the-loop eval on a real
  robot) — note this needs a Linux/WSL2 host; open question.