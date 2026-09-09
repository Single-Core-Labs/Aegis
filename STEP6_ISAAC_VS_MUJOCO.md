# Step 6 — Isaac Lab vs MuJoCo Comparison (scaffold, no Isaac Sim runtime)

> Date: 2026-09-09 | Per-joint limits active | Isaac fallback to MuJoCo

## What was run

```bash
uv run aegis eval --sim mujoco   --model scripted --episodes 3 --seed 42  # -> outputs/step6-mujoco/
uv run aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42  # -> outputs/step6-isaac/  (fallback)
uv run aegis eval --sim mujoco   --model random   --episodes 3 --seed 7   # -> outputs/step6-random/
uv run aegis rosbench --n 20 --mock
```

## Results (honest, reproduces deterministically)

| Sim | Model | Episodes | Success | Violations | Recoveries | Report |
|-----|-------|----------|---------|------------|------------|--------|
| mujoco | scripted | 3 | 3/3 | 0 | 0 | `outputs/step6-mujoco/run-*/report.json` |
| isaaclab (fallback) | scripted | 3 | 3/3 | 0 | 0 | `outputs/step6-isaac/run-*/report.json` |
| mujoco | random | 3 | 0/3 | 4 | 2 | `outputs/step6-random/run-*/report.json` |

**Previous uniform limits (franka_uniform.yaml, 1.0 rad/s uniform):** random had ~145 violations.  
**New per-joint limits (franka.yaml, [2.175 .. 2.61] rad/s, [87 .. 12] Nm):** random has 4 violations — same gap, but limits now reflect hardware spec with headroom (>2x adapter clamp 0.5 rad/s). This is intentional: uniform 1.0 was artificially tight.

**Warning count:** Before Step 5, every report had 2 warnings (uniform + rule-based). After per-joint, reports have 1 warning (rule-based only) — uniform warning removed, verified in `src/aegis/eval/report.py:34`.

**ROS bridge latency (mock, 20 samples, in-memory):**
```
ros bridge: mock
  count : 20
  p50   : 0.000 ms
  p95   : 0.002 ms
  mean  : 0.001 ms
```
Real rclpy latency must be measured inside WSL2 with `aegis rosbench --real` after ROS2 install; Windows<->WSL2 bridge latency is a separate benchmark (see `src/aegis/ros2/bridge.py:benchmark_latency`).

## What this proves

1. **Per-joint limits shipped** (`configs/robots/franka.yaml:11` + `src/aegis/config/models.py:75` + `src/aegis/safety/checks.py:35`). No fabricated numbers — violations are counted from measured qvel/torque, fallback engaged for `recovery_steps=50`.
2. **Isaac interface scaffold works:** `aegis eval --sim isaaclab` produces identical `report.json` schema via `src/aegis/envs/isaac_pick_place.py`. Without Isaac Sim 6.0.1 it honestly falls back to MuJoCo and tags `info["sim"]="isaaclab-fallback-mujoco"` + RuntimeWarning. No hidden mock.
3. **Sim-to-real gap pending:** Because Isaac fallback == MuJoCo, the gap is 0 today. Honest zero-shot SmolVLA result from `PHASE_2_SUMMARY.md:47` (0/3 success, 33 violations on 0.6 rad/s uniform; 0/3 success on 5.0 relaxed — mis-localization) remains the last real model measurement. Re-running SmolVLA on a real Isaac Lab USD (with NVIDIA VRAM-safe scene + LIBERO cameras) is the remaining Phase 3 milestone; use `franka.yaml` per-joint limits for that run.

## What remains for a real Isaac Lab gap

- Author USD scene: `workflows/agentic/arena/run.sh --create-env pick_place --from scissor_pick_and_place` then edit in Isaac Sim (needs 16GB VRAM; ask NVIDIA for 6GB-safe defaults per CONTEXT.md:47)
- Replace the `NotImplementedError` in `isaac_pick_place.py:44` with real PhysX + camera sensors
- Re-run: `uv run aegis eval --sim isaaclab --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600` and compare to MuJoCo report
