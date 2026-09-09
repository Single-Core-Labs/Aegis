# WHAT_NEXT.md — What We Want To Build Next & What’s Remaining

> **In simple English** — what's left after Phase 1+2 + Phase 3 scaffold (per-joint + ROS2 mock + Isaac fallback are done today). No mocks as real — if it says scaffold, the real runtime is still pending.

---

## 1. What We Want To Build Next (in priority order)

### P1 — Real Isaac Lab USD (needs NVIDIA + 6GB guidance) ⭐ Partnership blocker
**Why:** Today `aegis eval --sim isaaclab` honestly falls back to MuJoCo (`src/aegis/envs/isaac_pick_place.py:1`, `RuntimeWarning` + `info["sim"]="isaaclab-fallback-mujoco"`). The true sim-to-real gap can't be measured until Isaac runs for real.

**What to build:**
- Ask NVIDIA: VRAM-safe Isaac Sim 6.0.1 scene defaults for RTX 4050 6GB (your `i4h-workflows/README.md:27` needs 16GB min) + LIBERO camera placement to close SmolVLA mis-localization (hand never closer than 0.16m in `PHASE_2_SUMMARY.md:161`)
- Author USD: `workflows/agentic/arena/run.sh --create-env pick_place --from scissor_pick_and_place`, add Franka + cube + target + 3x 256px cameras + table + PhysX
- Replace `isaac_pick_place.py:44` `NotImplementedError` with real Isaac Lab `DirectRLEnv` + camera sensors
- Keep report schema identical so `STEP6_ISAAC_VS_MUJOCO.md` can show honest `mujoco vs isaaclab` table

**Done when:** `aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42` → 3/3 success on **real** PhysX (no fallback), and `aegis eval --sim isaaclab --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600` emits a real comparison report (vs `PHASE_2_SUMMARY.md:47` 0/3 on MuJoCo)

**Owner:** You + NVIDIA office hours — blocking on their guidance

---

### P2 — Real ROS2 rclpy bridge (needs 1 WSL2 apt fix)
**Why:** `src/aegis/ros2/bridge.py:1` mock is verified (`aegis rosbench --mock` p50 0.000ms), but real robot needs real topics `/aegis/action` + `/aegis/state` + latency SLOs.

**What to build:**
- Manual fix you can run inside WSL2 Ubuntu 22.04 (from previous message):
  ```bash
  wsl -u root bash -c 'curl -k -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list'
  sudo apt update && sudo apt install ros-humble-desktop -y
  source /opt/ros/humble/setup.bash && ros2 --help
  ```
  Or Docker `osrf/ros:humble-desktop` if apt still 404 (per `WSL2_SETUP_STATUS.md:10`)
- Wire `RosBridge` (`ros2/bridge.py:60`) to real `rclpy` publishers/subscribers, then benchmark:
  ```bash
  aegis rosbench --real --n 100   # inside WSL2
  # vs Windows<->WSL2 bridge: run same from Windows via `wsl ros2 topic` and compare p50/p95
  ```
- Add `inference_budget_ms` vs `ros latency p95` to `report.json` so the recommendation can say "tighten budget or fix bridge"

**Done when:** `aegis rosbench --real` prints real p50/p95 (not mock) and `aegis eval` can optionally route actions through ROS2 topics without bypassing `SafetyGateway` (`src/aegis/safety/gateway.py:28`)

**Owner:** You — 30 min WSL2 apt fix, then I wire the topic types

---

### P3 — Real SmolVLA gap re-measurement (after P1, unlocks the story)
**Why:** Phase 2 proved `franka.yaml:11` per-joint limits fix the chronic 33 violations/episode (`PHASE_2_SUMMARY.md:69` 5.09s cold start, relaxed A/B proved mis-localization). Now limits have >4x headroom, so the next measurement will be clean: is SmolVLA still 0/3 or does Isaac's cameras let it localize?

**What to build:**
- Re-run after P1 USD is real: both sims, same per-joint limits, same 3 cameras 256x256
  ```bash
  aegis eval --sim mujoco   --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600
  aegis eval --sim isaaclab --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600
  ```
- Produce updated `STEP6_ISAAC_VS_MUJOCO.md` with real numbers: success, violations, latency p50/p95, min hand-cube distance, gripper closures at distance
- Visual evidence: `diag/ep{0,1}/step{...}_{camera1..3}.png` per chunk (as in `PHASE_2_SUMMARY.md:170`)

**Done when:** Comparison report published with honest "still 0/3 in Isaac (vision gap persists) vs X/3" and a clear next tuning hypothesis (camera calibration, domain randomization)

**Owner:** You — needs Isaac USD + GPU, no code beyond the eval runner

---

### P4 — Hardware dry-run Go (after P1+P2, needs Franka + e-stop)
**Why:** `HARDWARE_CHECKLIST.md:1` is done as a doc, but hardware exposure is still **No-go** — e-stop not wired, no real `report.json`.

**What to build:**
- Wire e-stop to Franka (cuts power, not just software), barriers, operator + spotter
- Fully-inside-WSL2 arch per `WSL2_SETUP_STATUS.md:43` (avoids Windows<->WSL2 bridge latency)
- Dry run (no object, no grasp) with injected NaN action via `ros2/bridge.py:publish_action` to prove fallback `recovery_steps=50` drives to home
- Log `outputs/hardware-dry-run-*/report.json` with `sim: hardware`, latency, violations, signed checklist

**Done when:** `HARDWARE_CHECKLIST.md` Gates 0-4 checked, logs attached, Go/No-go signed

**Owner:** You + hardware — blocking on access

---

## 2. What's Remaining (full list — what P1-P4 don't cover is still out of scope until Phase 3 real is green)

**Must do for Phase 3 exit (`CONTEXT.md:59`):**

| # | Remaining | Blocker | File to change |
|---|-----------|---------|----------------|
| 1 | Isaac Lab real USD + PhysX (6GB-safe) | NVIDIA guidance | `src/aegis/envs/isaac_pick_place.py:44` |
| 2 | ROS2 real `rclpy` install + `rosbench --real` | WSL2 apt GPG fix | `src/aegis/ros2/bridge.py:60` |
| 3 | SmolVLA real Isaac vs MuJoCo comparison | #1 done | `STEP6_ISAAC_VS_MUJOCO.md` |
| 4 | Hardware dry-run `report.json` with e-stop | Franka access | `HARDWARE_CHECKLIST.md` |

**Nice to have (do after Phase 3 real is green):**

| Item | Why | Effort |
|------|-----|--------|
| `gpu_hours` per-kernel accounting (not wall-clock proxy) | NVIDIA acceptance cares about true GPU hours (`src/aegis/eval/report.py:67`) | Small |
| Isaac Lab USD performance tuning for 6GB (`omniverse-usd-performance-tuning` skill) | Avoid OOM on Franka + 3 cameras | Medium — needs NVIDIA scene defaults |
| Contact-force grasp detector (replace z-height 6cm threshold `src/aegis/envs/mujoco_pick_place.py:131`) | More honest success signal | Medium |
| Dashboard / Prometheus/OTel (currently NDJSON only `src/aegis/telemetry/logger.py:1`) | Observability beyond `report.json` | Large |

**Explicitly not building now (per `DESIGN.md:262` and `agent.md:30` scope discipline):**

- No multi-robot/multi-task batching, no parallel episodes
- No RL training / dataset collection / checkpointing
- No learned recommendation (still 4-rule heuristic `src/aegis/eval/report.py:11`)
- No `aegis` dashboard / web service
- No real robot hardware, no e-stop hardware layer beyond action-space gating (until Gate 1 checked)

---

## 3. How to pick what to do next (1 command each)

```bash
# If you have 30 min and WSL2 access:
# → Do P2 real ROS2: run the curl -k fix above, then `aegis rosbench --real`

# If you have 1 hour and NVIDIA office hours:
# → Do P1 Isaac USD: ask for 6GB-safe defaults, then `arena/run.sh --create-env pick_place`

# If you have a GPU and want a number today (no Isaac needed):
# → Do P3 MuJoCo only: re-run SmolVLA with new per-joint limits to prove headroom fix
uv run aegis eval --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600

# If you have hardware access:
# → Do P4 dry-run: follow HARDWARE_CHECKLIST.md Gate 3 with empty hand
```

**My recommendation next:** Since P1 and P2 both need your WSL2 shell, do **P2 first (30 min apt fix)** — it unblocks real latency numbers for the NVIDIA partnership update, while P1 needs their VRAM guidance anyway. I'm ready to wire the ROS2 topic types as soon as `ros2 --help` succeeds in WSL2.
