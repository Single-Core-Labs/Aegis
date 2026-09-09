# Isaac Lab Guide — USD Authoring & Gap Measurement

## Current State

`src/aegis/envs/isaac_pick_place.py` is a **scaffold** — same `Env` protocol as MuJoCo, honest fallback when Isaac Sim is absent.

```
--sim isaaclab + isaacsim missing  →  RuntimeWarning + MujocoPickPlaceEnv  (info["sim"]="isaaclab-fallback-mujoco")
--sim isaaclab + isaacsim present  →  NotImplementedError until USD is authored (isaac_pick_place.py:88)
```

## Authoring the USD Scene

### Prerequisites

- Isaac Sim 6.0.1 + Isaac Lab 2.x installed in WSL2 (see `nvidia-stack-manual.md`)
- 16 GB VRAM or NVIDIA VRAM-safe guidance for 6 GB

### Steps

```bash
# 1. Scaffold the Isaac Lab env from the Franka template
workflows/agentic/arena/run.sh --create-env pick_place --from scissor_pick_and_place

# 2. In Isaac Sim: File → Open USD, edit:
#    - Import Franka Panda (same MJCF geometry as assets/menagerie/franka_emika_panda/)
#    - Add cube (0.04m) + target marker (match assets/scenes/pick_place.xml)
#    - Add 3 cameras: camera1 (wrist), camera2 (front), camera3 (overhead) — see nvidia-stack-manual.md §5.3
#    - PhysX: dt 0.02, gravity -9.81, contact, friction matching MuJoCo
#    - Save: assets/usd/pick_place.usd

# 3. Wire it
# Edit src/aegis/envs/isaac_pick_place.py:44 — replace NotImplementedError with:
#   - USD stage loading
#   - PhysX scene + articulation
#   - Camera sensors (256×256 RGB)
#   - Env protocol: reset/step/observe/render_images

# 4. Validate
uv run aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42
# => 3/3 success, 0 violations, info["sim"]="isaaclab"
```

## Measuring Sim-to-Real Gap

```bash
uv run aegis eval --sim mujoco   --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600 --output-dir outputs/gap-mujoco
uv run aegis eval --sim isaaclab --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600 --output-dir outputs/gap-isaac

# Compare reports
diff <(jq .task_counts outputs/gap-mujoco/run-*/report.json) <(jq .task_counts outputs/gap-isaac/run-*/report.json)
diff <(jq .safety_violations outputs/gap-mujoco/run-*/report.json) <(jq .safety_violations outputs/gap-isaac/run-*/report.json)
```

Gap `0` today (fallback) is honest. Real gap (Isaac photoreal vs MuJoCo) will show vision transfer for SmolVLA.

## Troubleshooting

| Issue | Fix |
|---|---|
| `NotImplementedError: USD scene not yet authored` | Author USD per steps above, or pass `scene_mjcf` to use fallback |
| Textures black / VRAM OOM | Reduce texture res, `Max Bounces 1`, disable RTX — see nvidia-stack-manual.md §5.4 |
| Cameras return black | Check camera prim paths, ensure `render_images()` is called at chunk boundaries |
