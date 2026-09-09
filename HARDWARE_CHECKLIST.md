# Hardware Dry-Run Checklist (Step 7) — Go/No-Go for Real Franka

> Status: **No-go documentation path** until e-stop + ROS2 + Isaac Lab are wired. This file is the checklist; fill it before any hardware exposure.

## Gate 0 — Safety Gateway (must be proven in sim first)
- [ ] `aegis eval --sim mujoco --model scripted --episodes 3 --seed 42` → 3/3 success, 0 violations (`src/aegis/safety/gateway.py:28`)
- [ ] `aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42` → 3/3 success, 0 violations (via `src/aegis/envs/isaac_pick_place.py`)
- [ ] Per-joint limits active: `configs/robots/franka.yaml:11` shows `[2.175 .. 2.61]` rad/s and `[87 .. 12]` Nm, report warnings == 1 (no uniform warning)
- [ ] Random negative control: 0/3 success with violations observed (per-joint 4 violations, uniform 145 — both prove fallback engages)

## Gate 1 — E-Stop and Physical Safety
- [ ] Physical e-stop button wired, tested, reachable by operator and spotter
- [ ] E-stop cuts power to Franka, not just software — tested with `robot is in e-stop`
- [ ] Workspace limits: no humans in reach envelope during eval, barriers/marked floor
- [ ] Maximum Cartesian velocity and force limits validated against Franka datasheet (per-joint limits above map to Cartesian limits at home pose)
- [ ] Operator trained on `aegis` fallback: `recovery_steps=50` PID-to-home, `recovery_mode=resume` — understands that fallback is not a guarantee of safety beyond action-space gating

## Gate 2 — Software Bridge
- [ ] ROS2 Humble installed and `ros2 topic list` succeeds (fix per manual instructions in previous message, or Docker `osrf/ros:humble-desktop`)
- [ ] `aegis rosbench --real --n 100` inside WSL2 measured and recorded (p50/p95 < inference_budget_ms)
- [ ] Windows<->WSL2 bridge latency benchmarked if running split; otherwise fully-inside-WSL2 architecture confirmed (`WSL2_SETUP_STATUS.md:46`)
- [ ] `aegis eval --sim isaaclab` USD scene authored for 6GB VRAM (ask NVIDIA per `CONTEXT.md:47`)

## Gate 3 — Dry Run (no object, no grasp)
- [ ] `aegis eval --sim hardware --model scripted --episodes 1 --seed 42` with robot in gravity-compensation mode, hand empty, verified that fallback drives to home on injected violation (publish NaN action via `src/aegis/ros2/bridge.py:publish_action`)
- [ ] Measured joint velocities and torques stay under `configs/robots/franka.yaml` limits for 60s idle + 60s motion
- [ ] Logs written to `outputs/` with `report.json` containing `sim: hardware`, latency p50/p95, violations/recoveries

## Gate 4 — Go/No-Go Decision
- [ ] All Gate 0-3 boxes checked, logs attached to `outputs/hardware-dry-run-<timestamp>/`
- [ ] If any box unchecked → **No-go** — document reason in `report.json` warnings and do not proceed to object pick-place on hardware (`CONTEXT.md:61` exit criteria: "or documented no-go + reason")

## Artifacts required for a Go
- `outputs/hardware-dry-run-*/report.json` (hardware sim)
- `outputs/hardware-dry-run-*/episodes.jsonl` + `steps.jsonl`
- `aegis rosbench` p50/p95 log
- Signed checklist (this file) with operator + spotter names and date

## Current status (2026-09-09)
- Sim gates: **pass** (scripted 3/3, per-joint active, isaac fallback tagged)
- E-stop: **not wired** — no hardware exposure until wired
- ROS2: **blocked (apt GPG), mock bridge verified** — `src/aegis/ros2/bridge.py` + `aegis rosbench --mock` pass
- Isaac Lab: **scaffold only, no USD** — needs 16GB or NVIDIA 6GB-safe scene guidance
- **Decision: No-go, reason: e-stop + ROS2 + Isaac USD pending** — harness is honest, no fabricated hardware numbers
