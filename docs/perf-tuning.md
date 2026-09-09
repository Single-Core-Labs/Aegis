# Isaac USD Performance Tuning — 6GB VRAM Safe

Official Isaac Sim 6.0.1 minimum is 16GB VRAM. On RTX 4050 6GB, the Franka + 3 cameras USD will OOM with default settings. This doc gives VRAM-safe defaults to request from NVIDIA and apply locally.

## Target: Franka Pick-Place USD on 6GB

Scene: Franka Panda (Menagerie) + 0.025m cube + target marker + table + 3× 256×256 cameras + PhysX. No RTX path tracing required for Aegis — raster is enough for VLA.

## VRAM Budget

| Asset | Default VRAM | Tuned (6GB) |
|---|---|---|
| Franka meshes (collision + visual) | ~1.2 GB | ~1.2 GB (no change) |
| Textures (4K groundplane, etc.) | ~1.5 GB | ~0.3 GB (downscale to 1K) |
| RTX — bounce count / denoiser | ~8 GB | Disabled (use RTX-RealTime or Raster) |
| 3 cameras 256×256 + buffers | ~0.5 GB | ~0.5 GB |
| PhysX scratch | ~0.5 GB | ~0.5 GB |
| **Total** | **~12 GB** | **~2.5 GB** |

## Settings to Apply in Isaac Sim (VRAM-Safe)

```python
# In Isaac Sim UI: Render → Settings
# Or via USD: /Render/Settings

# 1. Disable RTX path tracing
render_mode = "RTX - Real-Time"   # not "RTX - Path Traced"
max_bounces = 1
denoiser = False
dlss = False

# 2. Downscale textures
texture_resolution = 1024   # from 4096
anisotropic_filtering = False

# 3. Camera: keep 256×256 (required by SmolVLA), no supersampling
camera_resolution = (256, 256)
camera_aa = "off"

# 4. PhysX: keep dt=0.02, reduce solver iterations if needed
physx_solver_iterations = 4   # from 8 (minor accuracy tradeoff)

# 5. No need for RTX shadows on 6GB
shadows = False
```

## USD Authoring Checklist

- [ ] Import Franka from `assets/menagerie/franka_emika_panda/panda_vel.xml` — keep collision meshes, drop high-res visual LOD if present
- [ ] Table + cube + target: simple box/cylinder prims (no textures) — material `table` / `object` / `target` only
- [ ] 3 cameras: `camera1` (wrist-like fixed), `camera2` (front), `camera3` (overhead) — `fovy 50`, look at workspace center `0.55 0 0.465`
- [ ] Lighting: single directional + headlight (no HDRI, no dome light)
- [ ] Save as `assets/usd/pick_place.usd` — keep under 50 MB

## Verification

```bash
# Inside WSL2 with Isaac Sim 6.0.1:
nvidia-smi --query-gpu=memory.used --format=csv   # should stay < 5000 MiB
# Run Aegis on real USD:
uv run aegis eval --sim isaaclab --model scripted --episodes 1 --seed 42
# Expect: 1/1 success, info["sim"]="isaaclab" (no fallback), VRAM stable
```

## When to Request NVIDIA Guidance

Ask for: “VRAM-safe Isaac Sim 6.0.1 scene defaults for RTX 4050 6GB + LIBERO camera extrinsics/intrinsics to match SmolVLA training distribution.” Reference `CONTEXT.md:59` and `WHAT_NEXT.md:P1`.

## Fallback Remains Honest

If VRAM still OOM, Aegis falls back to MuJoCo with `RuntimeWarning` + `info["sim"]="isaaclab-fallback-mujoco"` (`src/aegis/envs/isaac_pick_place.py:70`). No silent degrade, no fabricated report.
