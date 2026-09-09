# Per-Model Camera Calibration + Domain Randomization — Spec (P3)

**Problem:** SmolVLA `0/3` on MuJoCo fixed cameras (0.16m→0.60m, mid-air grips) is a vision domain gap vs LIBERO training distribution. Isaac Lab with photoreal cameras is the first fix; this spec is the systematic follow-on.

## P3a: Per-Model Camera Calibration

Each `configs/models/*.yaml` can declare its expected camera extrinsics/intrinsics:

```yaml
# configs/models/smolvla_libero.yaml (future)
cameras:
  camera1: { pos: [0.95, 0.05, 0.75], zaxis: [0.894, 0, 0.447], fovy: 50 }
  camera2: { pos: [0.55, 0.45, 0.85], zaxis: [0.172, 0.772, 0.600], fovy: 50 }
  camera3: { pos: [0.25, -0.6, 0.6], zaxis: [-0.351, -0.912, 0.210], fovy: 50 }
calibration:
  source: "LIBERO"   # or "isaac_lab"
  # If Isaac USD provides camera prims, these override YAML for that run
```

Env (`mujoco_pick_place.py` or `isaac_pick_place.py`) loads these at `reset()` and sets `Renderer` / USD camera prims accordingly. Mismatch is logged as `warnings: ["camera calibration mismatch: ..."]`.

## P3b: Domain Randomization

For robustness, randomize per episode (seeded):

| Parameter | Range | Seed |
|---|---|---|
| `light.pos` | ±0.1m jitter | `seed+episode` |
| `object` friction | `0.005±0.002` | `seed+episode` |
| `camera` pos jitter | ±0.02m | `seed+episode` |
| Table texture brightness | ±10% | `seed+episode` |

Disabled by default; enabled via `task.domain_randomization: true` in `configs/tasks/pick-place.yaml`.

## Evaluation

- Ablate: `calibration: none` vs `LIBERO-matched` vs `LIBERO+randomization` — report success/violation delta.
- Goal: close vision gap without touching policy weights.

## Not Building Now

Requires real Isaac USD cameras first. Spec kept here for roadmap.
