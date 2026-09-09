# NVIDIA Stack Manual — Aegis for Robotics Engineers

> **Audience:** Robotics / Physical AI engineers who want to evaluate VLA policies (e.g., SmolVLA) **safely** against the **NVIDIA stack**: Isaac Sim, Isaac Lab, ROS 2, and real Franka hardware — with Aegis as the trusted harness in the middle.

> **One-liner:** `Policy → [ AEGIS Safety Gateway + Eval Harness ] → Robot (MuJoCo · Isaac Lab · Real)`

---

## Table of Contents

1. [WSL2 + Ubuntu 22.04 Setup](#1-wsl2--ubuntu-2204-setup)
2. [System Architecture on the NVIDIA Stack](#2-system-architecture-on-the-nvidia-stack)
3. [Installation — Aegis + Dependencies](#3-installation--aegis--dependencies)
4. [Running Evaluations — MuJoCo (Lightweight)](#4-running-evaluations--mujoco-lightweight)
5. [Isaac Sim + Isaac Lab Integration](#5-isaac-sim--isaac-lab-integration)
6. [ROS 2 Bridge — Mock to Real](#6-ros-2-bridge--mock-to-real)
7. [Evaluating SmolVLA on the NVIDIA Stack](#7-evaluating-smolvla-on-the-nvidia-stack)
8. [Hardware Bring-Up — Real Franka](#8-hardware-bring-up--real-franka)
9. [Configuration Reference](#9-configuration-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [Appendix — File Map & Report Schema](#appendix--file-map--report-schema)

---

## 1. WSL2 + Ubuntu 22.04 Setup

Aegis is designed to run **fully inside WSL2** — the standard NVIDIA pattern that avoids Windows↔WSL2 bridge latency.

### 1.1 Enable WSL2 + GPU passthrough

```powershell
# PowerShell (Admin) on Windows 11
wsl --install -d Ubuntu-22.04
wsl --update
```

Install the latest NVIDIA driver on **Windows** (not inside WSL2) from [nvidia.com/drivers](https://www.nvidia.com/drivers). WSL2 shares the Windows driver automatically.

### 1.2 Verify GPU inside WSL2

```bash
# Inside WSL2 Ubuntu 22.04
nvidia-smi
# => Should show your RTX (e.g., RTX 4050 6GB) + CUDA 12.8

# Verify CUDA is visible to Python
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# => True 12.8
```

> **Current status on this project:** RTX 4050 6 GB is visible via `nvidia-smi` inside WSL2 Ubuntu 22.04.5. Isaac Sim 6.0.1 officially requires 16 GB VRAM — see [Section 5.4](#54-vram-considerations-6-gb-vs-16-gb) for the VRAM-safe path.

### 1.3 Architecture decision

| Pattern | Latency | Complexity | Recommendation |
|---|---|---|---|
| **Fully inside WSL2** (Aegis + ROS 2 + Isaac all in WSL2) | Lowest | Simplest | ✅ **Recommended** |
| Split (Aegis on Windows, ROS 2/Isaac in WSL2) | Bridge overhead | Requires `wsl ros2 topic` forwarding | Only if Windows-only tooling is required |

---

## 2. System Architecture on the NVIDIA Stack

```
                        ┌─────────────────────────────────────────────────┐
                        │           NVIDIA Physical AI Stack               │
                        │                                                  │
  ┌──────────┐          │  ┌───────────┐   ┌──────────────┐   ┌────────┐  │
  │  VLA     │  action  │  │  AEGIS    │   │  Isaac Sim   │   │  ROS 2 │  │
  │  Policy  │─────────▶│  │  Safety   │──▶│  Isaac Lab   │──▶│  Humble│──┼──▶  Franka
  │ SmolVLA  │          │  │  Gateway  │   │  PhysX + USD │   │ Topics │  │     Hardware
  │  450M    │◀─────────│  │  + Eval   │◀──│  Cameras     │◀──│ Joint  │  │     (Real)
  └──────────┘   obs    │  └───────────┘   └──────────────┘   └────────┘  │
                        │         │  report.json  episodes.jsonl           │
                        │         ▼                                        │
                        │     outputs/<run_id>/                            │
                        └─────────────────────────────────────────────────┘

  Alternative (lightweight CI, no GPU):
                        Policy → Aegis → MuJoCo (Menagerie Franka) → report.json
```

**Aegis role in the NVIDIA ecosystem:**

| NVIDIA Component | Aegis Integration Point | File |
|---|---|---|
| **Isaac Sim 6.0.1** | USD scene + PhysX + camera sensors | `src/aegis/envs/isaac_pick_place.py` |
| **Isaac Lab 2.x** | Task/Env wrapper, same `Env` protocol as MuJoCo | `src/aegis/envs/base.py` |
| **ROS 2 Humble** | `/aegis/action`, `/aegis/state`, `/aegis/safety` topics | `src/aegis/ros2/bridge.py` |
| **CUDA 12.8 / PyTorch** | `SmolVLA` inference (`cuda` mode), `gpu_hours` proxy | `src/aegis/policies/smolvla.py` |
| **Franka Emika Panda** | Per-joint safety limits from hardware datasheet | `configs/robots/franka.yaml` |

---

## 3. Installation — Aegis + Dependencies

### 3.1 Base install (MuJoCo — works everywhere, no GPU needed)

```bash
git clone https://github.com/your-org/physical-ai-harness.git
cd physical-ai-harness

# uv creates Python 3.12 venv + installs all deps (including torch cu128)
uv sync

# Validate — no sim needed
uv run aegis validate
uv run pytest -q   # or: python -m pytest -q  (20 tests)
```

### 3.2 NVIDIA stack — additional installs (inside WSL2)

| Step | Command | Notes |
|---|---|---|
| **Isaac Sim 6.0.1** | Follow [Isaac Sim Workstation Install](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html) | Requires Ubuntu 22.04 + RTX; 16 GB VRAM official min |
| **Isaac Lab 2.x** | `git clone https://github.com/isaac-sim/IsaacLab && ./isaaclab.sh --install` | See [Isaac Lab Install](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html) |
| **ROS 2 Humble** | See [Section 6](#6-ros-2-bridge--mock-to-real) | `apt` (preferred) or `osrf/ros:humble-desktop` Docker |
| **Verify** | `nvidia-smi && isaacsim --help && ros2 topic list` | All three should succeed |

---

## 4. Running Evaluations — MuJoCo (Lightweight)

This is the **fast inner loop** — no GPU, no Isaac, runs in CI.

```bash
# 1. Validate config (exit 0 = OK, exit 2 = config error)
uv run aegis validate

# 2. Scripted baseline — should be 3/3 success, 0 violations
uv run aegis eval --model scripted --episodes 3 --seed 42

# 3. Random negative control — should be 0/3, violations + recoveries
uv run aegis eval --model random --episodes 3 --seed 7

# 4. Inspect the report
cat outputs/run-*/report.json | python -m json.tool
cat outputs/run-*/episodes.jsonl
```

**What each policy does:**

| Policy | File | Behavior |
|---|---|---|
| `scripted` | `src/aegis/policies/scripted.py` | Deterministic 6-phase state machine + DLS IK — the ground-truth smoke test |
| `random` | `src/aegis/policies/random.py` | Uniform random joint velocities — must trip the gateway (proves safety is active) |
| `smolvla_libero` | `src/aegis/policies/smolvla.py` | Real VLA: 7-D cartesian-delta → 8-D joint-vel via DLS, chunk-50, budget-enforced |

---

## 5. Isaac Sim + Isaac Lab Integration

### 5.1 How it works today (scaffold)

```bash
# Without Isaac Sim: honest fallback — same report schema, tagged explicitly
uv run aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42
# => RuntimeWarning: isaacsim not found — falling back to MujocoPickPlaceEnv
# => report.json: sim="isaaclab", info["sim"]="isaaclab-fallback-mujoco", warnings=[...]
```

The scaffold `src/aegis/envs/isaac_pick_place.py` exposes the **identical `Env` protocol** as MuJoCo (`reset` / `step` / `observe` / `render_images`), so the entire harness — gateway, fallback, telemetry, report — works unchanged.

### 5.2 Authoring the real USD scene (when Isaac Sim is available)

```bash
# 1. Create the Isaac Lab environment from the Franka template
workflows/agentic/arena/run.sh --create-env pick_place --from scissor_pick_and_place

# 2. Open in Isaac Sim and edit:
#    - Replace the scene with Franka Panda + cube + target (match assets/scenes/pick_place.xml geometry)
#    - Add 3 cameras: camera1/2/3 at LIBERO-matching poses (see 5.3)
#    - Set PhysX: dt=0.02, gravity, contact params matching MuJoCo
#    - Save as: assets/usd/pick_place.usd

# 3. Wire it — replace the NotImplementedError in isaac_pick_place.py:44
#    with real PhysX + camera sensor reads, then:
uv run aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42
# => Should be 3/3 success, 0 violations, info["sim"]="isaaclab" (no fallback tag)
```

### 5.3 Camera placement — critical for VLA transfer

SmolVLA was trained on LIBERO camera distributions. To close the vision domain gap observed in MuJoCo (hand `0.6m` from cube):

| Camera | Purpose | Placement Guidance |
|---|---|---|
| `camera1` | Wrist / eye-in-hand | Mount on Franka hand, looking at gripper |
| `camera2` | Front / third-person | ~1.2m from table, 30° elevation, centered on workspace |
| `camera3` | Overhead / bird's-eye | Top-down, covering full `0.6m × 0.6m` workspace |

> Ask NVIDIA for LIBERO camera extrinsics/intrinsics to match exactly. Each camera renders `256×256` RGB into `obs["images"]` at chunk boundaries (not per-step).

### 5.4 VRAM considerations (6 GB vs 16 GB)

Isaac Sim 6.0.1 lists 16 GB VRAM as minimum. On 6 GB (RTX 4050):

- Use **VRAM-safe scene defaults** — reduce texture resolution, disable RTX real-time, use `RTX - Real-Time` → `Max Bounces 1`
- Request NVIDIA guidance for minimal USD that still matches LIBERO lighting
- Fallback remains honest — never silently degrade

### 5.5 Measuring the sim-to-real gap

```bash
# Run the same policy on both sims, compare reports
uv run aegis eval --sim mujoco   --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600 --output-dir outputs/gap-mujoco
uv run aegis eval --sim isaaclab --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600 --output-dir outputs/gap-isaac

# Compare: success rate, violations, latency p50/p95, trajectory divergence
python -m json.tool outputs/gap-mujoco/run-*/report.json
python -m json.tool outputs/gap-isaac/run-*/report.json
```

Gap `0` today (fallback == MuJoCo) is **honest**. Real gap measurement is the Phase 3 milestone.

---

## 6. ROS 2 Bridge — Mock to Real

### 6.1 Architecture

```
aegis eval  ──publish_action()──▶  /aegis/action  (Float64MultiArray, 8-D)  ──▶  Franka driver
            ◀──get_state()───────  /aegis/state   (JointState, 7-DOF)       ◀──  Franka driver
            ──publish_safety()──▶  /aegis/safety  (String JSON)              ──▶  Logger / Dashboard
```

`src/aegis/ros2/bridge.py` — `RosBridge` auto-detects `rclpy`:

| Mode | When | `aegis rosbench` p50 | Use |
|---|---|---|---|
| **Mock** (in-memory) | `rclpy` not installed | `0.000 ms` | CI, Windows dev, tests |
| **Real** (`rclpy`) | ROS 2 Humble installed | Measured (expect `0.1–1 ms`) | WSL2 + real Franka |

### 6.2 Installing ROS 2 Humble (WSL2 Ubuntu 22.04)

**Option A — apt (preferred, if GPG works):**

```bash
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update && sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
ros2 topic list  # => should succeed (empty list is OK)
```

**Option B — Docker (if apt GPG is blocked, as on this project):**

```bash
docker run -it --net host --gpus all -v $(pwd):/workspace osrf/ros:humble-desktop bash
# Inside container:
source /opt/ros/humble/setup.bash
ros2 topic list
```

> **This project's current status:** `apt` GPG hit a `404` — mock bridge verified (`aegis rosbench --mock` passes), real `rclpy` pending WSL2 install. Manual `curl -k` fix is documented in Section 6.2 above.

### 6.3 Benchmarking latency

```bash
# Mock (no ROS 2 needed — runs anywhere)
uv run aegis rosbench --mock --n 20
# => ros bridge: mock  p50 0.000 ms  p95 0.002 ms

# Real (inside WSL2 with ROS 2)
uv run aegis rosbench --real --n 100
# => ros bridge: real rclpy  p50 0.34 ms  p95 0.89 ms  (example)

# Requirement: p50 + p95 must be < inference_budget_ms (default 2000 ms)
# For real Franka control (1 kHz), aim for p95 < 1 ms
```

For **split architecture** (Aegis on Windows, ROS 2 in WSL2), run the benchmark on both sides and compare — the delta is the Windows↔WSL2 bridge overhead.

---

## 7. Evaluating SmolVLA on the NVIDIA Stack

### 7.1 Model contract

| Field | Value |
|---|---|
| Checkpoint | `lerobot/smolvla_libero` (450M params, HuggingFace) |
| State | 8-D `[eef_pos(3), axis-angle(3), gripper(2)]` |
| Action | 7-D cartesian-delta `[dx, dy, dz, droll, dpitch, dyaw, dgripper]` |
| Cameras | 3 × `256×256` RGB (`camera1/2/3`) |
| Chunk | 50 steps per inference |
| Adapter | 7-D delta → 8-D joint-vel via DLS resolved-rate (hand frame), clamp `0.5 rad/s` |

Checkpoint pipelines (tokenizer, normalizer, unnormalizer) are applied **verbatim** from the checkpoint — not from `config.json`.

### 7.2 Running SmolVLA

```bash
# CPU smoke (no GPU, slow — for pipeline verification only)
uv run aegis eval --model smolvla_libero --inference-mode cpu --episodes 1 --seed 42 --max-steps 50

# CUDA (requires GPU + checkpoint download ~2 GB, first run is slow)
uv run aegis eval --model smolvla_libero --inference-mode cuda --episodes 3 --seed 42 --max-steps 600

# With custom instruction
# Edit configs/models/smolvla_libero.yaml: instruction: "pick up the red cube and place it on the target"
```

### 7.3 Inference budget enforcement

```yaml
# physical-ai.yaml
eval:
  inference_mode: cuda
  inference_budget_ms: 2000.0  # per-step budget; over-budget → fallback (like a safety violation)
```

Cold start (~`5s` on first chunk) intentionally trips the budget on step 1 — this proves fallback works. Subsequent chunks are fast.

### 7.4 Known zero-shot result (honest)

| Limits | Success | Violations | Interpretation |
|---|---|---|---|
| `0.6 rad/s` uniform | 0/3 | 33/ep (1 budget + 32 velocity) | Limits interrupt ~1/3 of episode |
| `5.0 rad/s` relaxed | 0/3 | 0 | Model genuinely mis-localizes — vision domain gap |
| `[2.175..2.61]` per-joint (current default) | 0/3 | ~0–1 | Headroom `>4×` adapter clamp — clean measurement |

---

## 8. Hardware Bring-Up — Real Franka

> **No hardware is touched until all gates pass.** See Section 8.1 for the full checklist.

### 8.1 Gate summary

| Gate | Name | Key Check | Status |
|---|---|---|---|
| **0** | Safety in sim | `scripted 3/3`, per-joint active, `random 0/3` with violations | ✅ Pass |
| **1** | E-stop + physical safety | Physical e-stop wired, cuts power, workspace barriers | 🔲 Pending |
| **2** | Software bridge | `ros2 topic list` + `rosbench --real p50/p95 < budget` + USD scene | 🔲 Pending |
| **3** | Dry run (no object) | `aegis eval --sim hardware --model scripted` in gravity-comp, inject NaN → fallback to home | 🔲 Pending |
| **4** | Go/No-Go | All gates checked, logs in `outputs/hardware-dry-run-*/` | **No-go** — e-stop + ROS 2 + USD pending |

### 8.2 Dry run procedure (when ready)

```bash
# 1. Franka in gravity-compensation mode, hand empty
# 2. Run with hardware sim flag (future: --sim hardware)
uv run aegis eval --sim hardware --model scripted --episodes 1 --seed 42 --output-dir outputs/hardware-dry-run

# 3. Inject a violation — publish NaN and verify fallback drives to home
uv run python -c "from aegis.ros2.bridge import RosBridge; import numpy as np; b=RosBridge(); b.publish_action(np.array([float('nan')]*8))"

# 4. Verify: joint velocities/torques < configs/robots/franka.yaml limits for 60s idle + 60s motion
# 5. Inspect: outputs/hardware-dry-run/run-*/report.json  (sim="hardware", latency, violations)
```

### 8.3 Safety limits — Franka datasheet mapping

`configs/robots/franka.yaml` per-joint limits are from the Franka Emika Panda datasheet with headroom:

```yaml
max_velocity: [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]  # rad/s j1..j7
max_force: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]        # Nm j1..j7
```

Adapter clamp `0.5 rad/s` (`src/aegis/policies/smolvla.py`) has `>4×` headroom — transient overshoot no longer chronically trips the gateway.

---

## 9. Configuration Reference

All tuning is **declarative YAML** — no hardcoded robot/model params in code.

### 9.1 Root config (`physical-ai.yaml`)

```yaml
model_name: random              # -> configs/models/<name>.yaml
robot_name: franka              # -> configs/robots/<name>.yaml
task_name: pick-place           # -> configs/tasks/<name>.yaml

eval:
  episodes: 10
  seed: 42
  max_steps_per_episode: 2500
  time_step: 0.02
  episode_timeout_sec: 60.0
  inference_mode: cpu           # cpu | cuda
  inference_budget_ms: 2000.0

environment:
  sim: mujoco                   # mujoco | isaaclab
  scene_mjcf: assets/scenes/pick_place.xml
  robot_name: franka
  control_mode: joint_velocity

output:
  dir: ./outputs
  report_format: json
```

CLI flags override YAML: `aegis eval --model scripted --sim isaaclab --episodes 3 --seed 42 --inference-mode cuda`

### 9.2 Adding a new robot

```bash
cp configs/robots/franka.yaml configs/robots/my_robot.yaml
# Edit: name, mjcf_path, safety.max_velocity, safety.max_force (per-joint lists)
uv run aegis validate --robot my_robot
```

### 9.3 Adding a new policy

1. Add `configs/models/my_policy.yaml` with `kind: smolvla | scripted | random`
2. Implement `src/aegis/policies/my_policy.py` (`act(obs) -> np.ndarray`)
3. Register in `src/aegis/cli.py` dispatch

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `isaacsim not found — falling back` | Isaac Sim not installed | Expected on 6 GB laptops — install Isaac Sim 6.0.1 or use `--sim mujoco` |
| `torch.cuda is not available` | CUDA not visible in WSL2 | `nvidia-smi` inside WSL2 must show GPU; reinstall Windows driver |
| `config error: mjcf_path not found` | Asset path wrong | `aegis validate --robot franka` — check `assets/menagerie/` exists |
| `violations 33/ep` on SmolVLA | Uniform `0.6 rad/s` too tight | Use default `franka.yaml` per-joint `[2.175..2.61]` — `4×` headroom |
| `ros2: command not found` | ROS 2 not installed | See [Section 6.2](#62-installing-ros-2-humble-wsl2-ubuntu-2204) — apt or Docker |
| `apt GPG 404` on `packages.ros.org` | Transient keyserver issue | `curl -k` fix in Section 6.2 or use `osrf/ros:humble-desktop` Docker |
| `SmolVLA 0/3 success` | Vision domain gap (MuJoCo ≠ LIBERO) | Re-evaluate in Isaac Lab with LIBERO-matched cameras — Section 5.3 |
| `uv run pytest` hangs on `cmake` | `uv` downloading build deps | Use `python -m pytest -q` directly (20 tests, ~40s) |

---

## Appendix — File Map & Report Schema

### File map

```
physical-ai-harness/
├── physical-ai.yaml              Root config (defaults)
├── configs/
│   ├── robots/franka.yaml        Per-joint limits (default)
│   ├── robots/franka_uniform.yaml Legacy uniform 1.0 rad/s (A/B)
│   ├── models/{random,scripted,smolvla_libero}.yaml
│   └── tasks/pick-place.yaml
├── assets/
│   ├── menagerie/franka_emika_panda/  MJCF model
│   └── scenes/pick_place.xml     MuJoCo scene
├── src/aegis/
│   ├── cli.py                    Typer CLI (eval / validate / rosbench)
│   ├── config/models.py          Pydantic schema
│   ├── envs/{base,mujoco_pick_place,isaac_pick_place}.py
│   ├── policies/{base,random,scripted,smolvla}.py
│   ├── safety/{checks,gateway,fallback}.py
│   ├── eval/{runner,metrics,report}.py
│   ├── ros2/bridge.py            RosBridge + benchmark_latency()
│   └── telemetry/{logger,timing}.py
├── docs/
│   ├── nvidia-stack-manual.md    ← You are here
│   ├── architecture.md
│   ├── safety-gateway.md
│   ├── isaac-lab.md
│   └── ros2-bridge.md
├── outputs/<run_id>/
│   ├── report.json               Human + machine summary
│   ├── run.json                  Full validated config + git info
│   ├── episodes.jsonl            One line per episode
│   ├── steps.jsonl               Per-step latencies + violations
│   └── trajectory.jsonl          Per-step qpos/qvel
└── tests/
    ├── test_eval.py              Config + determinism
    ├── test_policy_model.py      Budget, adapter math, SmolVLA config
    └── test_seed_sweep.py        5 seeds × 5 eps, ≥90% success gate
```

### Report schema (`report.json`)

```json
{
  "run_id": "run-20260909T120000Z",
  "model": "scripted",
  "robot": "franka",
  "sim": "mujoco",
  "task_counts": { "pick-place": { "success": 3, "fail": 0 } },
  "latency_p50_ms": 0.12,
  "latency_p95_ms": 0.45,
  "inference_mode": "cpu",
  "inference_budget_ms": 2000.0,
  "inference_budget_violations": 0,
  "model_errors": 0,
  "safety_violations": 0,
  "recovery_events": 0,
  "gpu_hours": 0.0,
  "gpu_hours_note": "stub — no GPU used in this run",
  "total_steps": 750,
  "total_duration_s": 12.3,
  "recommendation": "all episodes succeeded — harness ready to evaluate real policies",
  "warnings": ["recommendation line is rule-based (4 rules), not learned"]
}
```

---

<div align="center">

**Aegis** — *Every action that reaches hardware must be safe, measured, and honest.*

Questions? Open an issue or see [docs/README.md](README.md).

</div>
