# Safety Gateway — Limits, Fallback & Adding Robots

## Per-Joint Limits

`configs/robots/franka.yaml` (default, from Franka datasheet):

```yaml
safety:
  max_velocity: [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]  # rad/s j1..j7
  max_force: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]        # Nm j1..j7
```

- `src/aegis/config/models.py:68` accepts `float | list[7]` — uniform float is legacy (warns in report)
- `src/aegis/safety/checks.py:35` iterates per-joint, skips `lim <= 0`
- `tests/test_policy_model.py:151` asserts `min(limit) >= 2 * adapter_clamp (0.5 rad/s)` — ensures `>4×` headroom

Legacy A/B: `configs/robots/franka_uniform.yaml` (`1.0 rad/s` uniform) vs `franka_diag.yaml` (`5.0 rad/s` relaxed).

## Fallback Controller

`src/aegis/safety/fallback.py` — PID-to-home:

```python
fallback = PidToHomeFallback(home_qpos=env.home_qpos, max_velocity=min(0.3, min(limit)))
gateway = SafetyGateway(limits=cfg.robot.safety, joint_names=env.joint_names, fallback=fallback)
```

- Bounded `0.3 rad/s` (or `min(limit)` if tighter)
- `recovery_steps: 50` (`configs/robots/franka.yaml:11`) then `recovery_mode: resume` (policy resumes) or `hold` (holds to end)
- Tested: `tests/test_policy_model.py::TestBudgetEnforcement` — slow/crashing/garbage policies all engage fallback

## Adding a New Robot

```bash
cp configs/robots/franka.yaml configs/robots/my_robot.yaml
# Edit:
#   name: my_robot
#   mjcf_path: ../../assets/menagerie/my_robot/model.xml
#   safety.max_velocity: [ ... ]  # 7 values or single float
#   safety.max_force: [ ... ]

uv run aegis validate --robot my_robot
uv run aegis eval --robot my_robot --model scripted --episodes 3 --seed 42
```

No code change needed — `src/aegis/envs/mujoco_pick_place.py` reads `mjcf_path` from config, `SafetyGateway` reads limits from `RobotSpec`.

## Tuning Guidance

| Symptom | Action |
|---|---|
| `violations 33/ep` on valid policy | Limits too tight — raise `max_velocity` or switch to per-joint |
| `violations 0` but policy is unsafe | Limits too loose — tighten to datasheet, add Cartesian limits (roadmap) |
| Fallback oscillates | Lower `fallback.max_velocity` or increase `Kp/Kd` in `mujoco_pick_place.py` |
| Inference budget trips on cold start | Increase `inference_budget_ms` or warm up model before episode 1 |
