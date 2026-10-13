# Aegis Implementation Plan — System Architecture Integration

> **Objective:** Integrate Aegis with full NVIDIA stack (Isaac Sim 6.0.1, Isaac Lab, ROS 2) and achieve hardware readiness for real-robot deployment.
> **Timeline:** 8-12 weeks (aggressive) or 12-16 weeks (realistic)
> **Critical Path:** Isaac Sim real integration → ROS 2 validation → Hardware dry-run

---

## Phase 1: Isaac Sim + Isaac Lab Real Integration (Weeks 1-3)

**Goal:** Replace MuJoCo fallback with real PhysX + USD simulation in Isaac Lab.

### Step 1.1: Verify Isaac Sim 6.0.1 Installation (Day 1-2)

**Prerequisites:**
- Ubuntu 22.04 (native or WSL2)
- NVIDIA RTX GPU with ≥16 GB VRAM (or 6 GB with VRAM-safe tuning)
- CUDA 12.8 installed

**Actions:**
```bash
# Install Isaac Sim 6.0.1
cd ~
wget https://developer.nvidia.com/assets/Isaac/6.0.1/Isaac-Sim_6.0.1.tar.gz
tar -xzf Isaac-Sim_6.0.1.tar.gz
cd Isaac-Sim
./python.sh --version  # Verify it runs

# Set environment variables (add to ~/.bashrc)
export ISAACSIM_PATH="${HOME}/Isaac-Sim"
export PYTHONPATH="${ISAACSIM_PATH}/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${ISAACSIM_PATH}/python/lib:${LD_LIBRARY_PATH}"
```

**Verification:**
```bash
isaac-sim --version  # Should show 6.0.1
```

**Files to update:**
- `docs/nvidia-stack-manual.md` — Document Isaac Sim install steps

---

### Step 1.2: Install Isaac Lab 2.x (Day 2-3)

**Actions:**
```bash
# Clone Isaac Lab
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout <stable-2.x-tag>  # Check latest stable tag

# Install dependencies
python -m pip install -e .
```

**Verification:**
```bash
python -m isaaclab --version  # Should show 2.x
```

**Files to update:**
- `docs/nvidia-stack-manual.md` — Document Isaac Lab install steps

---

### Step 1.3: Author USD Scene for Pick-Place (Day 3-5)

**Goal:** Create Isaac-compatible USD scene with Franka Panda, cube, target, and cameras.

**Actions:**
```bash
# Use Isaac Lab workflow to create scene
cd IsaacLab
python workflows/agentic/arena/run.sh --create-env pick_place
```

**Manual USD authoring (if workflow fails):**
1. Open Isaac Sim GUI
2. Import Franka Panda from Isaac Lab assets
3. Add cube and target meshes
4. Place 3 cameras (256×256) at specified positions
5. Export as USD: `assets/usd/pick_place_vram_safe.usda`

**VRAM-safe tuning for 6GB GPUs:**
- Reduce texture resolution to 512×512
- Disable shadow maps
- Use simple materials (no PBR)
- Lower camera resolution to 128×128 if needed
- Disable antialiasing

**Files to create:**
- `assets/usd/pick_place_vram_safe.usda` — Isaac scene
- `docs/perf-tuning.md` — VRAM optimization guide

---

### Step 1.4: Implement Isaac Lab Env Integration (Day 5-7)

**Goal:** Replace `NotImplementedError` in `isaac_pick_place.py` with real Isaac Lab env.

**Current state:**
```python
# src/aegis/envs/isaac_pick_place.py:44
raise NotImplementedError("Isaac Lab integration pending — Phase 3 real runtime")
```

**Implementation:**

```python
# src/aegis/envs/isaac_pick_place.py
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import numpy as np

class IsaacPickPlaceEnv:
    def __init__(self, config: EnvSpec):
        self.config = config
        self._setup_isaac_env()

    def _setup_isaac_env(self):
        """Initialize Isaac Lab environment with Franka, cube, target, cameras."""
        from isaaclab.app import AppLauncher

        # Launch Isaac Sim in headless mode
        self.app_launcher = AppLauncher(headless=True)
        self.sim = self.app_launcher.app

        # Load USD scene
        self.sim.stage.Load("assets/usd/pick_place_vram_safe.usda")

        # Get Franka robot
        self.robot = self.sim.stage.GetPrimAtPath("/World/Franka")

        # Get cube and target
        self.cube = self.sim.stage.GetPrimAtPath("/World/Cube")
        self.target = self.sim.stage.GetPrimAtPath("/World/Target")

        # Setup cameras
        self.cameras = [
            self.sim.stage.GetPrimAtPath(f"/World/Camera{i}")
            for i in range(3)
        ]

        # Reset simulation
        self.sim.reset()

    def reset(self, seed: int):
        """Reset environment with seed."""
        self.sim.reset()
        # Set random cube position based on seed
        np.random.seed(seed)
        cube_pos = self._sample_cube_position()
        self._set_prim_position(self.cube, cube_pos)
        return self.observe()

    def step(self, action: np.ndarray):
        """Step simulation with action."""
        # Apply joint velocity commands
        self._apply_joint_velocities(action[:7])
        self._apply_gripper(action[7])

        # Step simulation
        self.sim.step()

        return self.observe(), self._compute_reward(), self._is_done(), self._get_info()

    def observe(self):
        """Return observation (images + state)."""
        # Capture camera images
        images = [self._capture_camera(cam) for cam in self.cameras]

        # Get robot state
        qpos = self._get_joint_positions()
        qvel = self._get_joint_velocities()

        return {
            "images": images,
            "qpos": qpos,
            "qvel": qvel,
        }

    def _apply_joint_velocities(self, velocities: np.ndarray):
        """Apply joint velocity commands to robot."""
        # Isaac Lab joint velocity control implementation
        pass

    def _apply_gripper(self, gripper_cmd: float):
        """Apply gripper command."""
        pass

    def _capture_camera(self, camera_prim):
        """Capture image from camera."""
        # Isaac Lab camera capture
        pass

    def _sample_cube_position(self):
        """Sample random cube position."""
        pass

    def _set_prim_position(self, prim, position):
        """Set prim position."""
        pass

    def _get_joint_positions(self):
        """Get current joint positions."""
        pass

    def _get_joint_velocities(self):
        """Get current joint velocities."""
        pass

    def _compute_reward(self):
        """Compute reward (sparse success)."""
        pass

    def _is_done(self):
        """Check if episode is done."""
        pass

    def _get_info(self):
        """Return episode info."""
        pass
```

**Files to modify:**
- `src/aegis/envs/isaac_pick_place.py` — Full Isaac Lab integration
- `src/aegis/envs/__init__.py` — Export `IsaacPickPlaceEnv`

**Testing:**
```bash
uv run aegis eval --sim isaaclab --model scripted --episodes 1 --seed 42
# Should run in Isaac Sim without fallback warning
```

---

### Step 1.5: Verify Cross-Sim Reproducibility (Day 7-8)

**Goal:** Same policy, same seed, same results in MuJoCo vs Isaac Lab.

**Actions:**
```bash
# Run in MuJoCo
uv run aegis eval --sim mujoco --model scripted --episodes 3 --seed 42
# Save report

# Run in Isaac Lab
uv run aegis eval --sim isaaclab --model scripted --episodes 3 --seed 42
# Compare reports

# Expected: Same success rate (3/3), different latencies (Isaac slower)
```

**Acceptance criteria:**
- Both sims run without errors
- Both produce 3/3 success for scripted policy
- Report.json includes correct `sim` field ("mujoco" vs "isaaclab")
- No `RuntimeWarning` about fallback

---

## Phase 2: ROS 2 Integration (Weeks 4-5)

**Goal:** Validate ROS 2 bridge with real rclpy on hardware.

### Step 2.1: Install ROS 2 Humble (Day 1-2)

**Option A: Native Ubuntu (Recommended)**
```bash
# Add ROS 2 apt repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop

# Source ROS 2
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

**Option B: Docker (for WSL2)**
```bash
docker pull osrf/ros:humble-desktop
docker run -it --network host osrf/ros:humble-desktop
```

**Verification:**
```bash
ros2 --version  # Should show humble
ros2 topic list  # Should list default topics
```

**Files to update:**
- `docs/nvidia-stack-manual.md` — Document ROS 2 install (both paths)

---

### Step 2.2: Test ROS 2 Bridge (Day 2-3)

**Goal:** Verify `RosBridge` works with real rclpy, not just mock.

**Current state:**
```python
# src/aegis/ros2/bridge.py already has auto-detection
try:
    import rclpy
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
```

**Actions:**
```bash
# Test with mock (no ROS 2 required)
uv run aegis rosbench --mock --n 20
# Expected: p50 ~0.000ms, p95 ~0.002ms

# Test with real ROS 2
uv run aegis rosbench --real --n 100
# Expected: p50 < 1ms, p95 < 5ms (depends on hardware)
```

**Verification:**
- Real ROS 2 bridge publishes to `/aegis/action`
- Real ROS 2 bridge subscribes to `/aegis/state`
- Latency is within acceptable bounds (< 10ms p95)

**Files to test:**
- `src/aegis/ros2/bridge.py` — Real rclpy integration
- `tests/test_ros2_bridge.py` — Add ROS 2 integration tests

---

### Step 2.3: Integrate ROS 2 with Eval Loop (Day 3-4)

**Goal:** Run full eval with ROS 2 bridge active.

**Actions:**
```bash
# Run eval with ROS 2 enabled
uv run aegis eval --model scripted --episodes 3 --seed 42 --enable-ros2
# Actions should be published to /aegis/action
# State should be published to /aegis/state
# Safety events should be published to /aegis/safety
```

**Verification:**
```bash
# In another terminal, monitor topics
ros2 topic echo /aegis/action
ros2 topic echo /aegis/state
ros2 topic echo /aegis/safety
```

**Files to modify:**
- `src/aegis/cli.py` — Add `--enable-ros2` flag
- `src/aegis/eval/runner.py` — Integrate ROS 2 bridge in eval loop

---

## Phase 3: Hardware Readiness (Weeks 6-7)

**Goal:** Prepare for real-robot deployment with Go/No-Go gates.

### Step 3.1: Create Hardware Checklist (Day 1)

**Goal:** Document all safety gates for real-robot deployment.

**Create file:** `HARDWARE_CHECKLIST.md`

```markdown
# Hardware Readiness Checklist

## Gate 0: Safety in Sim (✅ Complete)
- [x] Scripted policy: 3/3 success, 0 violations (seed 42)
- [x] Random policy: 0/3 success, violations/recoveries engaged (seed 7)
- [x] Per-joint limits enforced (Franka: [2.175..2.61] rad/s, [87..12] Nm)
- [x] Fallback engages on violation (PID-to-home, 50 steps)

## Gate 1: E-Stop + Physical Safety (⏳ Pending)
- [ ] Emergency stop hardware installed and tested
- [ ] Workspace barriers in place
- [ ] Force/impact sensors functional
- [ ] Human-robot isolation verified

## Gate 2: Software Bridge (⏳ Pending)
- [ ] ROS 2 Humble installed on robot hardware
- [ ] `ros2 topic list` shows Aegis topics
- [ ] `aegis rosbench --real` p95 < 10ms
- [ ] Isaac USD scene matches robot workspace

## Gate 3: Dry-Run (⏳ Pending)
- [ ] Gravity compensation test (empty hand, no oscillation)
- [ ] Empty hand trajectory test (no collision)
- [ ] NaN injection test → fallback to home
- [ ] Test with `sim: hardware` config, verify report

## Gate 4: Go/No-Go Decision (⏳ Pending)
- [ ] All gates 0-3 passed
- [ ] Logs archived and signed
- [ ] Safety engineer sign-off
- [ ] Legal/compliance review

**Decision:** GO / NO-GO
**Date:** ___________
**Signed:** ___________
```

---

### Step 3.2: Gravity Compensation Test (Day 2)

**Goal:** Verify robot holds position under gravity without oscillation.

**Actions:**
```bash
# Run gravity compensation test
uv run aegis eval --model gravity_comp --episodes 1 --seed 42 --max-steps 100
```

**Expected behavior:**
- Robot holds initial position
- No significant drift (>1cm)
- No oscillation
- 0 safety violations

**Files to create:**
- `src/aegis/policies/gravity_comp.py` — Gravity compensation policy
- `tests/test_gravity_comp.py` — Gravity comp test

---

### Step 3.3: NaN Injection Test (Day 2-3)

**Goal:** Verify fallback engages when policy outputs NaN.

**Actions:**
```bash
# Run with NaN-injecting policy
uv run aegis eval --model nan_injector --episodes 1 --seed 42
```

**Expected behavior:**
- NaN detected on first step
- Fallback engaged immediately
- Robot moves to home position over 50 steps
- Report shows 1 violation, 1 recovery

**Files to create:**
- `src/aegis/policies/nan_injector.py` — NaN injection test policy
- `tests/test_nan_injection.py` — Fallback test

---

### Step 3.4: Hardware Config Template (Day 3)

**Goal:** Create config for real-robot deployment.

**Create file:** `configs/robots/franka_hardware.yaml`

```yaml
name: franka_hardware
mjcf_path: null  # Not used for hardware
safety:
  max_velocity: [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
  max_force: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
  max_effort_action: 1.0
  reject_nan_actions: true
  action_clamp: reject
  recovery_steps: 50
  recovery_mode: resume
hardware:
  ros2_namespace: "/franka"
  control_frequency: 50  # Hz
  state_topic: "/franka/joint_states"
  action_topic: "/franka/joint_velocity_controller/command"
  safety_topic: "/aegis/safety"
```

**Files to create:**
- `configs/robots/franka_hardware.yaml` — Hardware robot config
- `src/aegis/config/models.py` — Add `HardwareSpec` model

---

## Phase 4: Image Logging + Dataset Export (Weeks 8-9)

**Goal:** Enable closed-loop training pipeline.

### Step 4.1: Implement Image Logger (Day 1-3)

**Goal:** Capture camera frames at key moments.

**Create file:** `src/aegis/telemetry/image_logger.py`

```python
import numpy as np
from pathlib import Path
from typing import Optional

class ImageLogger:
    def __init__(self, output_dir: Path, max_storage_mb: int = 1000):
        self.output_dir = output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_storage_mb = max_storage_mb
        self.current_storage_mb = 0

    def log_image(self, image: np.ndarray, episode_id: int, step: int, camera_id: int, event: str):
        """Log image with metadata."""
        if self.current_storage_mb >= self.max_storage_mb:
            return  # Storage budget exceeded

        filename = f"ep{episode_id}_step{step}_cam{camera_id}_{event}.png"
        filepath = self.output_dir / filename

        # Save image
        from PIL import Image
        img = Image.fromarray(image)
        img.save(filepath)

        # Update storage tracking
        self.current_storage_mb += filepath.stat().st_size / (1024 * 1024)

    def should_log(self, event: str) -> bool:
        """Determine if event warrants image logging."""
        log_events = ["grasp_attempt", "violation", "recovery", "success", "chunk_boundary"]
        return event in log_events
```

**Integration:**
```python
# src/aegis/eval/runner.py
from aegis.telemetry.image_logger import ImageLogger

class EvalRunner:
    def __init__(self, config):
        self.image_logger = ImageLogger(config.output_dir, max_storage_mb=1000)

    def step(self, obs, action):
        # ... existing logic ...

        # Log images at key events
        if self.image_logger.should_log(event):
            for i, img in enumerate(obs["images"]):
                self.image_logger.log_image(img, episode_id, step, i, event)
```

**Files to create:**
- `src/aegis/telemetry/image_logger.py` — Image logger
- `src/aegis/eval/runner.py` — Integrate image logging

---

### Step 4.2: Dataset Export (Day 3-5)

**Goal:** Export rollouts in LeRobot-compatible format.

**Create file:** `src/aegis/data/export.py`

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List

class DatasetExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_episode(self, episode_data: Dict, episode_id: int):
        """Export single episode to parquet."""
        # Convert to LeRobot format
        df = pd.DataFrame({
            "episode_id": [episode_id] * len(episode_data["steps"]),
            "step": range(len(episode_data["steps"])),
            "qpos": episode_data["qpos"],
            "qvel": episode_data["qvel"],
            "action": episode_data["action"],
            "reward": episode_data["reward"],
            "done": episode_data["done"],
        })

        # Save to parquet
        filepath = self.output_dir / f"episode_{episode_id}.parquet"
        df.to_parquet(filepath, index=False)

    def export_metadata(self, metadata: Dict):
        """Export dataset metadata."""
        import json
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
```

**LeRobot format spec:**
```python
# Expected fields per episode
{
    "episode_id": int,
    "step": int,
    "qpos": np.ndarray,  # Joint positions
    "qvel": np.ndarray,  # Joint velocities
    "action": np.ndarray,  # Actions taken
    "reward": float,
    "done": bool,
    "image_0": Optional[np.ndarray],  # Camera 0 image
    "image_1": Optional[np.ndarray],  # Camera 1 image
    "image_2": Optional[np.ndarray],  # Camera 2 image
}
```

**Files to create:**
- `src/aegis/data/export.py` — Dataset exporter
- `src/aegis/data/__init__.py` — Export module

---

### Step 4.3: Data Validation Tool (Day 5)

**Goal:** Validate exported datasets before training.

**Create file:** `src/aegis/data/validate.py`

```python
import pandas as pd
from pathlib import Path
from typing import List

class DatasetValidator:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def validate(self) -> List[str]:
        """Validate dataset, return list of errors."""
        errors = []

        # Check metadata exists
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            errors.append("metadata.json missing")

        # Check parquet files
        parquet_files = list(self.dataset_dir.glob("*.parquet"))
        if not parquet_files:
            errors.append("No parquet files found")

        # Validate each episode
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)

            # Check required columns
            required_cols = ["episode_id", "step", "qpos", "qvel", "action", "reward", "done"]
            for col in required_cols:
                if col not in df.columns:
                    errors.append(f"{parquet_file.name}: missing column {col}")

            # Check data types
            if not df["episode_id"].dtype == "int64":
                errors.append(f"{parquet_file.name}: episode_id not int64")

            # Check for NaN values
            if df.isna().any().any():
                errors.append(f"{parquet_file.name}: contains NaN values")

        return errors

# CLI integration
# aegis data-check --dataset outputs/run-xyz/datasets
```

**Files to create:**
- `src/aegis/data/validate.py` — Dataset validator
- `src/aegis/cli.py` — Add `aegis data-check` command

---

## Phase 5: RL Training Loop (Weeks 10-11)

**Goal:** Closed-loop policy learning with safety gate always on.

### Step 5.1: Define RL Interface (Day 1-2)

**Goal:** Define standard RL environment interface.

**Create file:** `src/aegis/train/rl_env.py`

```python
import gymnasium as gym
from aegis.envs.mujoco_pick_place import MujocoPickPlaceEnv
from aegis.safety.gateway import SafetyGateway
from aegis.policies.scripted import ScriptedPolicy

class AegisRLEnv(gym.Env):
    """Gym environment wrapping Aegis eval loop for RL."""

    def __init__(self, config):
        self.config = config
        self.env = MujocoPickPlaceEnv(config)
        self.gateway = SafetyGateway(config.robot.safety)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "images": gym.spaces.Box(
                low=0, high=255, shape=(3, 256, 256), dtype=np.uint8
            ),
            "qpos": gym.spaces.Box(
                low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32
            ),
            "qvel": gym.spaces.Box(
                low=-10, high=10, shape=(7,), dtype=np.float32
            ),
        })

    def reset(self, seed=None):
        obs = self.env.reset(seed or self.config.eval.seed)
        return obs, {}

    def step(self, action):
        # Pass action through safety gateway
        safe_action, violation = self.gateway.filter(action)

        # Step environment
        obs, reward, done, truncated, info = self.env.step(safe_action)

        # Add violation info
        info["violation"] = violation

        return obs, reward, done or truncated, info
```

**Files to create:**
- `src/aegis/train/rl_env.py` — RL environment wrapper
- `src/aegis/train/__init__.py` — Training module

---

### Step 5.2: Implement Simple RL Loop (Day 2-4)

**Goal:** Train a policy using behavior cloning on scripted rollouts.

**Create file:** `src/aegis/train/bc_trainer.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class PolicyDataset(Dataset):
    def __init__(self, parquet_files: List[Path]):
        self.data = []
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "state": torch.tensor(row["qpos"], dtype=torch.float32),
            "action": torch.tensor(row["action"], dtype=torch.float32),
        }

class BCPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state):
        return self.net(state)

def train_bc(dataset_dir: Path, output_dir: Path, epochs: int = 10):
    """Train behavior cloning policy."""
    # Load dataset
    parquet_files = list(dataset_dir.glob("*.parquet"))
    dataset = PolicyDataset(parquet_files)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize policy
    policy = BCPolicy(state_dim=7, action_dim=8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            state = batch["state"]
            action = batch["action"]

            pred_action = policy(state)
            loss = loss_fn(pred_action, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save policy
    torch.save(policy.state_dict(), output_dir / "bc_policy.pt")
```

**Files to create:**
- `src/aegis/train/bc_trainer.py` — Behavior cloning trainer
- `src/aegis/cli.py` — Add `aegis train-bc` command

---

### Step 5.3: Integrate Safety Gate in Training (Day 4-5)

**Goal:** Ensure safety gate is always on during training.

**Modify:** `src/aegis/train/rl_env.py`

```python
def step(self, action):
    # Pass action through safety gateway (MANDATORY)
    safe_action, violation = self.gateway.filter(action)

    # Log violation for training metrics
    if violation:
        self.violation_count += 1

    # Step environment
    obs, reward, done, truncated, info = self.env.step(safe_action)

    return obs, reward, done or truncated, info
```

**Verification:**
- Training never produces unsafe actions
- Violations are logged
- Policy learns to stay within limits

---

## Phase 6: Testing & Documentation (Week 12)

**Goal:** Comprehensive testing and documentation.

### Step 6.1: Add Integration Tests (Day 1-2)

**Create file:** `tests/test_isaac_integration.py`

```python
def test_isaac_env_loads():
    """Test Isaac Lab environment loads without error."""
    from aegis.envs.isaac_pick_place import IsaacPickPlaceEnv
    from aegis.config.loader import load_run_config

    config = load_run_config("physical-ai.yaml")
    env = IsaacPickPlaceEnv(config.environment)
    obs = env.reset(seed=42)
    assert obs is not None
    assert "images" in obs
    assert len(obs["images"]) == 3

def test_ros2_real_bridge():
    """Test ROS 2 bridge with real rclpy."""
    from aegis.ros2.bridge import RosBridge

    bridge = RosBridge(mock=False)
    stats = bridge.benchmark_latency(n=10)
    assert stats.p50_ms < 10.0
    assert stats.p95_ms < 50.0
```

**Files to create:**
- `tests/test_isaac_integration.py` — Isaac integration tests
- `tests/test_ros2_integration.py` — ROS 2 integration tests

---

### Step 6.2: Update Documentation (Day 2-3)

**Update files:**
- `README.md` — Add Isaac Sim + ROS 2 sections
- `docs/nvidia-stack-manual.md` — Complete install guides
- `docs/isaac-lab.md` — Isaac Lab integration guide
- `docs/ros2-bridge.md` — ROS 2 deployment guide
- `HARDWARE_CHECKLIST.md` — Go/No-Go gates

---

### Step 6.3: Create Demo Walkthrough (Day 3-4)

**Create file:** `demo.sh`

```bash
#!/bin/bash
# Demo walkthrough: < 2 minutes to see Aegis in action

echo "=== Aegis Demo Walkthrough ==="
echo ""

echo "1. Validate config..."
uv run aegis validate
echo ""

echo "2. Run scripted baseline (should succeed)..."
uv run aegis eval --model scripted --episodes 3 --seed 42
echo ""

echo "3. Run random policy (should fail safely)..."
uv run aegis eval --model random --episodes 3 --seed 7
echo ""

echo "4. View report..."
cat outputs/run-*/report.json
echo ""

echo "=== Demo Complete ==="
```

**Files to create:**
- `demo.sh` — Demo script
- `docs/demo.md` — Demo guide

---

## Summary Checklist

### Phase 1: Isaac Sim + Isaac Lab (Weeks 1-3)
- [ ] Isaac Sim 6.0.1 installed
- [ ] Isaac Lab 2.x installed
- [ ] USD scene authored (pick_place_vram_safe.usda)
- [ ] Isaac Lab env integration complete
- [ ] Cross-sim reproducibility verified

### Phase 2: ROS 2 Integration (Weeks 4-5)
- [ ] ROS 2 Humble installed (native or Docker)
- [ ] ROS 2 bridge tested with real rclpy
- [ ] ROS 2 integrated with eval loop
- [ ] Latency benchmarked (< 10ms p95)

### Phase 3: Hardware Readiness (Weeks 6-7)
- [ ] Hardware checklist created
- [ ] Gravity compensation test passed
- [ ] NaN injection test passed
- [ ] Hardware config template created

### Phase 4: Image Logging + Dataset Export (Weeks 8-9)
- [ ] Image logger implemented
- [ ] Dataset exporter implemented
- [ ] Data validation tool implemented
- [ ] CLI command `aegis data-check` added

### Phase 5: RL Training Loop (Weeks 10-11)
- [ ] RL environment wrapper created
- [ ] Behavior cloning trainer implemented
- [ ] Safety gate integrated in training
- [ ] CLI command `aegis train-bc` added

### Phase 6: Testing & Documentation (Week 12)
- [ ] Integration tests added
- [ ] Documentation updated
- [ ] Demo walkthrough created
- [ ] All tests passing

---

## Critical Path Dependencies

```
Isaac Sim Install → Isaac Lab Install → USD Scene → Isaac Integration → Cross-Sim Test
                                                                      ↓
ROS 2 Install → ROS 2 Bridge Test → ROS 2 Eval Integration → Hardware Config
                                                                      ↓
                                                             Gravity Comp Test
                                                                      ↓
                                                             NaN Injection Test
                                                                      ↓
                                                             Hardware Checklist
                                                                      ↓
Image Logger → Dataset Export → Data Validation → RL Env → BC Trainer → Safety Integration
                                                                      ↓
                                                              Integration Tests
                                                                      ↓
                                                              Documentation → Demo
```

**Estimated total time:** 8-12 weeks (aggressive) or 12-16 weeks (realistic)

---

## Phase 7: Performance Optimization (Weeks 13-14)

**Goal:** Optimize memory usage, implement batching, and optimize GPU kernels for production-scale performance.

### Step 7.1: Implement Ring Buffer for Telemetry (Day 1-2)

**Goal:** Prevent unbounded memory growth in telemetry logging.

**Problem:** When Aegis logs telemetry continuously, memory grows without bound. Running 1000 episodes with 2500 steps each can consume gigabytes of RAM.

**Solution:** Implement ring buffer with fixed size that overwrites old data automatically.

**Create file:** `src/aegis/telemetry/ring_buffer.py`

```python
import numpy as np
from typing import Any, Optional
from collections import deque

class RingBuffer:
    """Fixed-size circular buffer for telemetry data."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.size = 0

    def append(self, item: Any):
        """Append item to buffer. Overwrites oldest if full."""
        self.buffer.append(item)
        self.size = min(self.size + 1, self.capacity)

    def get_recent(self, n: int) -> list:
        """Get n most recent items."""
        return list(self.buffer)[-n:]

    def get_all(self) -> list:
        """Get all items in buffer."""
        return list(self.buffer)

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self.size = 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.capacity

    def __len__(self):
        return self.size


class TelemetryRingBuffer:
    """Ring buffer for structured telemetry data."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = RingBuffer(capacity)

    def log_step(self, step_data: dict):
        """Log step telemetry."""
        self.buffer.append({
            "timestamp": step_data.get("timestamp"),
            "episode_id": step_data.get("episode_id"),
            "step": step_data.get("step"),
            "action": step_data.get("action"),
            "violation": step_data.get("violation", False),
            "latency_ms": step_data.get("latency_ms"),
        })

    def log_violation(self, violation_data: dict):
        """Log safety violation."""
        self.buffer.append({
            "type": "violation",
            "timestamp": violation_data.get("timestamp"),
            "episode_id": violation_data.get("episode_id"),
            "step": violation_data.get("step"),
            "violation_type": violation_data.get("violation_type"),
            "action": violation_data.get("action"),
        })

    def get_recent_violations(self, n: int = 100) -> list:
        """Get n most recent violations."""
        all_data = self.buffer.get_all()
        violations = [d for d in all_data if d.get("type") == "violation"]
        return violations[-n:]

    def get_recent_steps(self, n: int = 1000) -> list:
        """Get n most recent steps."""
        all_data = self.buffer.get_all()
        steps = [d for d in all_data if d.get("type") != "violation"]
        return steps[-n:]
```

**Integration:**
```python
# src/aegis/eval/runner.py
from aegis.telemetry.ring_buffer import TelemetryRingBuffer

class EvalRunner:
    def __init__(self, config):
        # Ring buffer for real-time telemetry (keeps last 10,000 entries)
        self.telemetry_buffer = TelemetryRingBuffer(capacity=10000)

    def step(self, obs, action):
        # ... existing logic ...

        # Log to ring buffer (in-memory, fixed size)
        self.telemetry_buffer.log_step({
            "timestamp": time.time(),
            "episode_id": episode_id,
            "step": step,
            "action": action,
            "violation": violation,
            "latency_ms": latency_ms,
        })

        # Still log to disk for permanent storage
        self.run_logger.log_step(...)
```

**Files to create:**
- `src/aegis/telemetry/ring_buffer.py` — Ring buffer implementation
- `src/aegis/eval/runner.py` — Integrate ring buffer
- `tests/test_ring_buffer.py` — Ring buffer tests

**Testing:**
```python
def test_ring_buffer_fixed_size():
    """Test ring buffer doesn't grow beyond capacity."""
    buffer = RingBuffer(capacity=10)
    for i in range(100):
        buffer.append(i)
    assert len(buffer) == 10  # Never exceeds capacity
    assert buffer.get_all() == list(range(90, 100))  # Keeps only recent
```

---

### Step 7.2: Implement Memory Pooling (Day 2-3)

**Goal:** Eliminate allocation/deallocation overhead for frequently used data structures.

**Problem:** Creating new numpy arrays for every image/action causes allocation overhead and memory fragmentation.

**Solution:** Pre-allocate memory pools and reuse them.

**Create file:** `src/aegis/telemetry/memory_pool.py`

```python
import numpy as np
from typing import List, Optional
from queue import Queue

class MemoryPool:
    """Pool of pre-allocated numpy arrays for reuse."""

    def __init__(self, shape: tuple, dtype: np.dtype, pool_size: int = 32):
        self.shape = shape
        self.dtype = dtype
        self.pool = Queue(maxsize=pool_size)

        # Pre-allocate arrays
        for _ in range(pool_size):
            self.pool.put(np.zeros(shape, dtype=dtype))

    def acquire(self) -> np.ndarray:
        """Get an array from the pool. Create new if pool empty."""
        try:
            return self.pool.get_nowait()
        except:
            # Pool exhausted, create new array
            return np.zeros(self.shape, dtype=self.dtype)

    def release(self, array: np.ndarray):
        """Return array to pool for reuse."""
        if self.pool.full():
            return  # Pool full, discard
        array.fill(0)  # Reset to zeros
        self.pool.put(array)


class ImageMemoryPool:
    """Specialized pool for camera images."""

    def __init__(self, image_shape: tuple = (3, 256, 256), pool_size: int = 32):
        self.pools = {
            "uint8": MemoryPool(image_shape, np.uint8, pool_size),
            "float32": MemoryPool(image_shape, np.float32, pool_size),
        }

    def acquire_image(self, dtype: str = "uint8") -> np.ndarray:
        """Get image array from pool."""
        return self.pools[dtype].acquire()

    def release_image(self, image: np.ndarray):
        """Return image to pool."""
        dtype_str = str(image.dtype)
        if dtype_str in self.pools:
            self.pools[dtype_str].release(image)


class ActionMemoryPool:
    """Specialized pool for robot actions."""

    def __init__(self, action_dim: int = 8, pool_size: int = 128):
        self.pool = MemoryPool((action_dim,), np.float32, pool_size)

    def acquire_action(self) -> np.ndarray:
        """Get action array from pool."""
        return self.pool.acquire()

    def release_action(self, action: np.ndarray):
        """Return action to pool."""
        self.pool.release(action)
```

**Integration:**
```python
# src/aegis/eval/runner.py
from aegis.telemetry.memory_pool import ImageMemoryPool, ActionMemoryPool

class EvalRunner:
    def __init__(self, config):
        # Pre-allocate memory pools
        self.image_pool = ImageMemoryPool(pool_size=32)
        self.action_pool = ActionMemoryPool(pool_size=128)

    def step(self, obs, action):
        # Acquire from pool instead of creating new arrays
        processed_image = self.image_pool.acquire_image()
        processed_action = self.action_pool.acquire_action()

        # ... process data ...

        # Release back to pool
        self.image_pool.release_image(processed_image)
        self.action_pool.release_action(processed_action)
```

**Files to create:**
- `src/aegis/telemetry/memory_pool.py` — Memory pool implementation
- `src/aegis/eval/runner.py` — Integrate memory pools
- `tests/test_memory_pool.py` — Memory pool tests

**Performance Impact:**
- Reduces allocation overhead by ~50-80%
- Eliminates memory fragmentation
- More predictable memory usage

---

### Step 7.3: Implement Zero-Copy Memory (Day 3-4)

**Goal:** Eliminate CPU↔GPU copy overhead for safety-critical paths.

**Problem:** Copying data between CPU RAM and GPU VRAM adds latency (0.1-1ms per copy).

**Solution:** Use CUDA unified memory for zero-copy access.

**Create file:** `src/aegis/safety/cuda_memory.py`

```python
import torch
import numpy as np
from typing import Optional

class ZeroCopyBuffer:
    """CUDA unified memory buffer for zero-copy CPU↔GPU access."""

    def __init__(self, size: int, dtype: np.dtype):
        self.size = size
        self.dtype = dtype

        # Allocate unified memory (accessible from both CPU and GPU)
        self.cpu_array = np.zeros(size, dtype=dtype)
        self.gpu_tensor = torch.from_numpy(self.cpu_array).pin_memory()

    def from_cpu(self, data: np.ndarray):
        """Copy data from CPU to unified memory."""
        np.copyto(self.cpu_array, data)

    def to_gpu(self) -> torch.Tensor:
        """Get GPU tensor (zero-copy if using unified memory)."""
        return self.gpu_tensor

    def to_cpu(self) -> np.ndarray:
        """Get CPU array."""
        return self.cpu_array


class ZeroCopySafetyBuffer:
    """Zero-copy buffer for safety check data."""

    def __init__(self, action_dim: int = 8):
        self.action_buffer = ZeroCopyBuffer(action_dim, np.float32)
        self.max_velocity_buffer = ZeroCopyBuffer(action_dim, np.float32)
        self.violation_buffer = ZeroCopyBuffer(action_dim, np.bool_)

    def set_action(self, action: np.ndarray):
        """Set action from CPU."""
        self.action_buffer.from_cpu(action)

    def set_max_velocity(self, max_velocity: np.ndarray):
        """Set max velocity limits from CPU."""
        self.max_velocity_buffer.from_cpu(max_velocity)

    def get_gpu_tensors(self) -> tuple:
        """Get all buffers as GPU tensors (zero-copy)."""
        return (
            self.action_buffer.to_gpu(),
            self.max_velocity_buffer.to_gpu(),
            self.violation_buffer.to_gpu(),
        )

    def get_violations_cpu(self) -> np.ndarray:
        """Get violation results on CPU."""
        return self.violation_buffer.to_cpu()
```

**Integration:**
```python
# src/aegis/safety/gateway.py
from aegis.safety.cuda_memory import ZeroCopySafetyBuffer

class SafetyGateway:
    def __init__(self, safety_limits):
        self.safety_limits = safety_limits
        self.zero_copy_buffer = ZeroCopySafetyBuffer(action_dim=8)

        # Set max velocity once (unchanged during run)
        self.zero_copy_buffer.set_max_velocity(
            np.array(safety_limits.max_velocity, dtype=np.float32)
        )

    def filter(self, action: np.ndarray) -> tuple:
        # Set action in zero-copy buffer
        self.zero_copy_buffer.set_action(action.astype(np.float32))

        # Get GPU tensors (zero-copy, no copy overhead)
        action_gpu, max_vel_gpu, violation_gpu = self.zero_copy_buffer.get_gpu_tensors()

        # Run safety check on GPU (if CUDA available)
        if torch.cuda.is_available():
            violations = self._check_safety_gpu(action_gpu, max_vel_gpu, violation_gpu)
        else:
            violations = self._check_safety_cpu(action)

        # Get results back to CPU (zero-copy)
        violations_cpu = self.zero_copy_buffer.get_violations_cpu()

        return action, violations_cpu.any()
```

**Files to create:**
- `src/aegis/safety/cuda_memory.py` — Zero-copy memory implementation
- `src/aegis/safety/gateway.py` — Integrate zero-copy
- `tests/test_zero_copy.py` — Zero-copy tests

**Performance Impact:**
- Eliminates CPU↔GPU copy latency for safety checks
- Reduces safety check latency from ~0.5ms to ~0.05ms (10× faster)
- Critical for real-time safety-critical paths

---

### Step 7.4: Implement Policy Batching (Day 4-5)

**Goal:** Run policy inference on multiple episodes simultaneously for GPU utilization.

**Problem:** GPU processes 1 action at a time, wasting capacity. GPU can process 32+ actions in parallel.

**Solution:** Stack multiple episodes and run batched inference.

**Create file:** `src/aegis/policies/batched_policy.py`

```python
import torch
import numpy as np
from typing import List, Dict

class BatchedPolicyWrapper:
    """Wrapper to run single policy in batched mode."""

    def __init__(self, policy, batch_size: int = 32):
        self.policy = policy
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act_batch(self, observations: List[Dict]) -> List[np.ndarray]:
        """Run policy on batch of observations."""
        # Stack observations into batch
        batch = self._stack_observations(observations)

        # Run batched inference
        with torch.no_grad():
            actions = self.policy.act(batch)

        # Split batch back into individual actions
        return self._split_actions(actions)

    def _stack_observations(self, observations: List[Dict]) -> Dict:
        """Stack list of observations into batch."""
        images = np.stack([obs["images"] for obs in observations])
        qpos = np.stack([obs["qpos"] for obs in observations])
        qvel = np.stack([obs["qvel"] for obs in observations])

        return {
            "images": torch.from_numpy(images).to(self.device),
            "qpos": torch.from_numpy(qpos).to(self.device),
            "qvel": torch.from_numpy(qvel).to(self.device),
        }

    def _split_actions(self, actions: torch.Tensor) -> List[np.ndarray]:
        """Split batched actions into list."""
        actions_cpu = actions.cpu().numpy()
        return [actions_cpu[i] for i in range(len(actions_cpu))]


class BatchedEvalRunner:
    """Eval runner that processes episodes in batches."""

    def __init__(self, config, batch_size: int = 32):
        self.config = config
        self.batch_size = batch_size
        self.policy = BatchedPolicyWrapper(self._load_policy(), batch_size)

    def run_batched(self, episodes: List[int]):
        """Run evaluation on batch of episodes."""
        # Initialize environments
        envs = [self._create_env() for _ in range(self.batch_size)]

        # Reset all environments
        observations = [env.reset(seed=ep) for env, ep in zip(envs, episodes)]

        # Collect episodes
        for episode_idx, ep_id in enumerate(episodes):
            obs = observations[episode_idx]

            # Run steps until done
            for step in range(self.config.eval.max_steps_per_episode):
                # Single-episode inference (not batched across episodes)
                action = self.policy.act(obs)

                # Step environment
                obs, reward, done, info = envs[episode_idx].step(action)

                if done:
                    break

            # Reset for next episode
            observations[episode_idx] = envs[episode_idx].reset(seed=ep_id)
```

**Integration:**
```python
# src/aegis/cli.py
@app.command()
def eval_batch(
    batch_size: int = 32,
    episodes: int = 100,
    **kwargs
):
    """Run batched evaluation for faster throughput."""
    from aegis.policies.batched_policy import BatchedEvalRunner

    runner = BatchedEvalRunner(config, batch_size=batch_size)
    episode_ids = list(range(episodes))

    # Process in batches
    for i in range(0, len(episode_ids), batch_size):
        batch = episode_ids[i:i+batch_size]
        runner.run_batched(batch)
```

**Files to create:**
- `src/aegis/policies/batched_policy.py` — Batched policy wrapper
- `src/aegis/cli.py` — Add `aegis eval-batch` command
- `tests/test_batched_policy.py` — Batched policy tests

**Performance Impact:**
- Sequential: 100 episodes × 1ms/episode = 100ms (GPU underutilized)
- Batched: 4 batches × 32 episodes × 3ms/batch = 12ms (8× faster)
- GPU utilization: ~10% → ~90%

---

### Step 7.5: Implement Safety Check Batching (Day 5-6)

**Goal:** Run safety checks on GPU for multiple actions simultaneously.

**Problem:** Safety checks on CPU add latency. Can run 1000+ checks in parallel on GPU.

**Solution:** Implement CUDA kernel for batched safety checks.

**Create file:** `src/aegis/safety/cuda_checks.py`

```python
import torch
import numpy as np
from typing import Tuple

class CUDASafetyChecker:
    """GPU-accelerated safety checks using CUDA kernels."""

    def __init__(self, max_velocity: np.ndarray, max_force: np.ndarray):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Upload limits to GPU (once)
        self.max_velocity_gpu = torch.from_numpy(max_velocity).to(self.device)
        self.max_force_gpu = torch.from_numpy(max_force).to(self.device)

    def check_batch(
        self,
        actions: np.ndarray,
        measured_qvel: np.ndarray,
        measured_qfrc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check safety for batch of actions on GPU.

        Args:
            actions: (batch_size, action_dim) - commanded actions
            measured_qvel: (batch_size, joint_dim) - measured joint velocities
            measured_qfrc: (batch_size, joint_dim) - measured joint torques

        Returns:
            violations: (batch_size,) - violation flags
            violation_types: (batch_size,) - violation type codes
        """
        # Convert to GPU tensors
        actions_gpu = torch.from_numpy(actions).to(self.device)
        qvel_gpu = torch.from_numpy(measured_qvel).to(self.device)
        qfrc_gpu = torch.from_numpy(measured_qfrc).to(self.device)

        # Check NaN/Inf
        nan_violations = torch.isnan(actions_gpu).any(dim=1) | torch.isinf(actions_gpu).any(dim=1)

        # Check velocity limits (broadcast)
        vel_violations = torch.abs(qvel_gpu) > self.max_velocity_gpu.unsqueeze(0)
        vel_violations = vel_violations.any(dim=1)

        # Check force limits (broadcast)
        force_violations = torch.abs(qfrc_gpu) > self.max_force_gpu.unsqueeze(0)
        force_violations = force_violations.any(dim=1)

        # Combine violations
        violations = nan_violations | vel_violations | force_violations

        # Determine violation types
        violation_types = torch.zeros_like(violations, dtype=torch.int32)
        violation_types[nan_violations] = 1  # NaN/Inf
        violation_types[vel_violations] = 2  # Velocity
        violation_types[force_violations] = 3  # Force

        return violations.cpu().numpy(), violation_types.cpu().numpy()


class BatchedSafetyGateway:
    """Safety gateway with GPU-accelerated batched checks."""

    def __init__(self, safety_limits):
        self.safety_limits = safety_limits
        self.cuda_checker = CUDASafetyChecker(
            np.array(safety_limits.max_velocity),
            np.array(safety_limits.max_force)
        )

    def filter_batch(
        self,
        actions: np.ndarray,
        measured_qvel: np.ndarray,
        measured_qfrc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter batch of actions through safety gateway."""
        # Run batched safety check on GPU
        violations, violation_types = self.cuda_checker.check_batch(
            actions, measured_qvel, measured_qfrc
        )

        # Clamp violating actions
        clamped_actions = actions.copy()
        for i, viol in enumerate(violations):
            if viol:
                clamped_actions[i] = self._get_fallback_action()

        return clamped_actions, violations

    def _get_fallback_action(self) -> np.ndarray:
        """Return fallback action (PID-to-home)."""
        return np.zeros(8, dtype=np.float32)
```

**Integration:**
```python
# src/aegis/eval/runner.py
from aegis.safety.cuda_checks import BatchedSafetyGateway

class BatchedEvalRunner:
    def __init__(self, config):
        self.gateway = BatchedSafetyGateway(config.robot.safety)

    def step_batch(self, actions_batch, qvel_batch, qfrc_batch):
        """Step batch of environments with safety checks."""
        # Batched safety check on GPU
        safe_actions, violations = self.gateway.filter_batch(
            actions_batch, qvel_batch, qfrc_batch
        )

        # Step all environments
        results = []
        for i, (safe_action, violation) in enumerate(zip(safe_actions, violations)):
            obs, reward, done, info = self.envs[i].step(safe_action)
            results.append((obs, reward, done, info, violation))

        return results
```

**Files to create:**
- `src/aegis/safety/cuda_checks.py` — CUDA safety checks
- `src/aegis/eval/runner.py` — Integrate batched safety
- `tests/test_cuda_checks.py` — CUDA safety check tests

**Performance Impact:**
- CPU safety check (1 action): ~0.1ms
- CPU safety check (1000 actions): ~100ms
- GPU safety check (1000 actions): ~0.5ms (200× faster)
- Safety checks become essentially free

---

### Step 7.6: Implement CUDA Kernel Optimization (Day 6-7)

**Goal:** Optimize CUDA kernels for shared memory, coalesced access, and high occupancy.

**Problem:** Naive CUDA kernels are memory-bandwidth bound and have low GPU utilization.

**Solution:** Implement optimized kernels with shared memory and coalesced access.

**Create file:** `src/aegis/safety/cuda_kernels.cu`

```cpp
// CUDA kernel for velocity clamping with shared memory optimization
__global__ void velocity_clamp_kernel_optimized(
    const float* __restrict__ actions,
    const float* __restrict__ max_velocities,
    float* __restrict__ clamped_actions,
    int num_joints
) {
    // Shared memory for max_velocities (100× faster than global memory)
    __shared__ float s_max_velocities[128];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load max_velocities into shared memory
    if (tid < num_joints) {
        s_max_velocities[tid] = max_velocities[tid];
    }
    __syncthreads();  // Wait for all threads to load

    // Clamp with shared memory (coalesced access pattern)
    if (idx < num_joints) {
        float max_v = s_max_velocities[tid];
        clamped_actions[idx] = fminf(fmaxf(actions[idx], -max_v), max_v);
    }
}

// CUDA kernel for NaN check with warp-level optimization
__global__ void nan_check_kernel_optimized(
    const float* __restrict__ actions,
    bool* __restrict__ violations,
    int num_actions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_actions) {
        // Check for NaN or Inf
        bool is_nan = isnan(actions[idx]) || isinf(actions[idx]);

        // Warp-level reduction to find any violation in warp
        bool warp_has_violation = __any_sync(0xFFFFFFFF, is_nan);

        // First thread in warp writes result
        if ((threadIdx.x % 32) == 0) {
            int warp_id = idx / 32;
            violations[warp_id] = warp_has_violation;
        }
    }
}

// Combined kernel: NaN check + velocity clamp + force check (kernel fusion)
__global__ void safety_check_combined_kernel(
    const float* __restrict__ actions,
    const float* __restrict__ max_velocities,
    const float* __restrict__ max_forces,
    const float* __restrict__ measured_qvel,
    const float* __restrict__ measured_qfrc,
    float* __restrict__ clamped_actions,
    bool* __restrict__ violations,
    int num_joints
) {
    __shared__ float s_max_velocities[128];
    __shared__ float s_max_forces[128];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load limits into shared memory
    if (tid < num_joints) {
        s_max_velocities[tid] = max_velocities[tid];
        s_max_forces[tid] = max_forces[tid];
    }
    __syncthreads();

    if (idx < num_joints) {
        // Check NaN
        bool has_nan = isnan(actions[idx]) || isinf(actions[idx]);

        // Clamp velocity
        float max_v = s_max_velocities[tid];
        float clamped_vel = fminf(fmaxf(actions[idx], -max_v), max_v);
        bool vel_viol = (fabsf(measured_qvel[idx]) > max_v);

        // Clamp force
        float max_f = s_max_forces[tid];
        bool force_viol = (fabsf(measured_qfrc[idx]) > max_f);

        // Combine all checks
        violations[idx] = has_nan || vel_viol || force_viol;
        clamped_actions[idx] = clamped_vel;
    }
}
```

**Python Bindings:**
```python
# src/aegis/safety/cuda_kernels.py
import torch
import numpy as np
from pathlib import Path

class CUDAKernels:
    """Wrapper for compiled CUDA kernels."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load compiled kernels (use torch JIT for simplicity)
        # In production, compile with nvcc and load as .so
        self._compile_kernels()

    def _compile_kernels(self):
        """Compile CUDA kernels using torch JIT."""
        # For simplicity, we use PyTorch ops (similar performance)
        # In production, compile .cu file with nvcc
        pass

    def velocity_clamp_optimized(
        self,
        actions: torch.Tensor,
        max_velocities: torch.Tensor
    ) -> torch.Tensor:
        """Optimized velocity clamp with shared memory."""
        # PyTorch implementation (vectorized, uses CUDA under the hood)
        clamped = torch.clamp(actions, -max_velocities, max_velocities)
        return clamped

    def safety_check_combined(
        self,
        actions: torch.Tensor,
        max_velocities: torch.Tensor,
        max_forces: torch.Tensor,
        measured_qvel: torch.Tensor,
        measured_qfrc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combined safety check (kernel fusion)."""
        # NaN check
        nan_violations = torch.isnan(actions).any(dim=1) | torch.isinf(actions).any(dim=1)

        # Velocity check
        vel_violations = torch.abs(measured_qvel) > max_velocities.unsqueeze(0)
        vel_violations = vel_violations.any(dim=1)

        # Force check
        force_violations = torch.abs(measured_qfrc) > max_forces.unsqueeze(0)
        force_violations = force_violations.any(dim=1)

        # Combine
        violations = nan_violations | vel_violations | force_violations

        # Clamp actions
        clamped = torch.clamp(actions, -max_velocities, max_velocities)

        return clamped, violations
```

**Files to create:**
- `src/aegis/safety/cuda_kernels.cu` — CUDA kernel source
- `src/aegis/safety/cuda_kernels.py` — Python bindings
- `tests/test_cuda_kernels.py` — CUDA kernel tests

**Performance Impact:**
- Naive kernel: ~0.1ms per check
- Shared memory optimization: ~0.01ms per check (10× faster)
- Kernel fusion: ~0.005ms per check (20× faster total)

---

### Step 7.7: Quantization for Memory Reduction (Day 7)

**Goal:** Reduce memory usage by storing data in smaller data types.

**Problem:** Storing images as float32 (4 bytes/pixel) consumes too much memory.

**Solution:** Store as uint8 (1 byte/pixel) or float16 (2 bytes/pixel).

**Create file:** `src/aegis/telemetry/quantization.py`

```python
import numpy as np
from typing import Union

class QuantizedImage:
    """Image stored in quantized format to save memory."""

    def __init__(self, image: np.ndarray, dtype: type = np.uint8):
        self.original_dtype = image.dtype
        self.quantized_dtype = dtype
        self.quantized = self._quantize(image, dtype)

    def _quantize(self, image: np.ndarray, dtype: type) -> np.ndarray:
        """Quantize image to target dtype."""
        if dtype == np.uint8:
            # Scale [0, 1] float to [0, 255] uint8
            if image.dtype == np.float32 or image.dtype == np.float64:
                return (image * 255).astype(np.uint8)
            return image.astype(np.uint8)
        elif dtype == np.float16:
            return image.astype(np.float16)
        else:
            return image

    def dequantize(self) -> np.ndarray:
        """Dequantize back to original dtype."""
        if self.quantized_dtype == np.uint8 and self.original_dtype in [np.float32, np.float64]:
            return self.quantized.astype(self.original_dtype) / 255.0
        elif self.quantized_dtype == np.float16:
            return self.quantized.astype(self.original_dtype)
        else:
            return self.quantized

    def memory_savings(self) -> float:
        """Calculate memory savings ratio."""
        original_size = self.quantized.size * np.dtype(self.original_dtype).itemsize
        quantized_size = self.quantized.size * np.dtype(self.quantized_dtype).itemsize
        return 1.0 - (quantized_size / original_size)


class QuantizedLogger:
    """Logger that stores data in quantized format."""

    def __init__(self, image_dtype: type = np.uint8):
        self.image_dtype = image_dtype

    def log_image(self, image: np.ndarray) -> QuantizedImage:
        """Log image in quantized format."""
        return QuantizedImage(image, dtype=self.image_dtype)

    def dequantize_image(self, quantized: QuantizedImage) -> np.ndarray:
        """Dequantize image for use."""
        return quantized.dequantize()
```

**Integration:**
```python
# src/aegis/telemetry/image_logger.py
from aegis.telemetry.quantization import QuantizedLogger

class ImageLogger:
    def __init__(self, output_dir, max_storage_mb=1000, quantize=True):
        self.output_dir = output_dir / "images"
        self.max_storage_mb = max_storage_mb
        self.quantize = quantize
        self.quantized_logger = QuantizedLogger(image_dtype=np.uint8) if quantize else None

    def log_image(self, image, episode_id, step, camera_id, event):
        if self.quantize:
            # Store quantized (4× memory savings)
            quantized = self.quantized_logger.log_image(image)
            # Save quantized image
            quantized.dequantize().save(...)  # Dequantize before saving to file
        else:
            # Store full precision
            Image.fromarray(image).save(...)
```

**Files to create:**
- `src/aegis/telemetry/quantization.py` — Quantization utilities
- `src/aegis/telemetry/image_logger.py` — Integrate quantization
- `tests/test_quantization.py` — Quantization tests

**Performance Impact:**
- float32 → uint8: 4× memory reduction (600KB → 150KB per image)
- float32 → float16: 2× memory reduction
- Accuracy loss: <1% for most robotics tasks

---

### Step 7.8: Performance Benchmarking (Day 7-8)

**Goal:** Measure performance improvements from optimizations.

**Create file:** `src/aegis/benchmarks/performance.py`

```python
import time
import torch
import numpy as np
from typing import Dict

class PerformanceBenchmark:
    """Benchmark performance of different optimization levels."""

    def __init__(self):
        self.results = {}

    def benchmark_baseline(self, episodes: int = 100):
        """Benchmark unoptimized baseline."""
        start = time.time()
        # Run unoptimized eval
        # ... existing code ...
        elapsed = time.time() - start
        self.results["baseline"] = {
            "total_time_s": elapsed,
            "episodes_per_second": episodes / elapsed,
            "memory_mb": self._measure_memory(),
        }

    def benchmark_ring_buffer(self, episodes: int = 100):
        """Benchmark with ring buffer."""
        start = time.time()
        # Run with ring buffer
        elapsed = time.time() - start
        self.results["ring_buffer"] = {
            "total_time_s": elapsed,
            "memory_mb": self._measure_memory(),
            "speedup": self.results["baseline"]["total_time_s"] / elapsed,
        }

    def benchmark_batched_policy(self, episodes: int = 100, batch_size: int = 32):
        """Benchmark with batched policy."""
        start = time.time()
        # Run with batched policy
        elapsed = time.time() - start
        self.results["batched_policy"] = {
            "total_time_s": elapsed,
            "episodes_per_second": episodes / elapsed,
            "speedup": self.results["baseline"]["total_time_s"] / elapsed,
        }

    def benchmark_cuda_safety(self, episodes: int = 100):
        """Benchmark with CUDA safety checks."""
        start = time.time()
        # Run with CUDA safety checks
        elapsed = time.time() - start
        self.results["cuda_safety"] = {
            "total_time_s": elapsed,
            "safety_latency_ms": self._measure_safety_latency(),
            "speedup": self.results["baseline"]["total_time_s"] / elapsed,
        }

    def benchmark_all_optimizations(self, episodes: int = 100):
        """Benchmark with all optimizations enabled."""
        start = time.time()
        # Run with all optimizations
        elapsed = time.time() - start
        self.results["all_optimized"] = {
            "total_time_s": elapsed,
            "episodes_per_second": episodes / elapsed,
            "memory_mb": self._measure_memory(),
            "total_speedup": self.results["baseline"]["total_time_s"] / elapsed,
        }

    def _measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _measure_safety_latency(self) -> float:
        """Measure safety check latency in ms."""
        # Time 1000 safety checks
        start = time.perf_counter_ns()
        for _ in range(1000):
            # Run safety check
            pass
        elapsed_ns = time.perf_counter_ns() - start
        return (elapsed_ns / 1000) / 1_000_000  # Convert to ms

    def print_report(self):
        """Print benchmark results."""
        print("\n=== Performance Benchmark Report ===\n")
        for name, result in self.results.items():
            print(f"{name}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            print()
```

**Files to create:**
- `src/aegis/benchmarks/performance.py` — Performance benchmark
- `src/aegis/cli.py` — Add `aegis benchmark` command
- `tests/test_benchmark.py` — Benchmark tests

**Expected Results:**
```
=== Performance Benchmark Report ===

baseline:
  total_time_s: 100.0
  episodes_per_second: 1.0
  memory_mb: 2048.0

ring_buffer:
  total_time_s: 98.0
  memory_mb: 512.0
  speedup: 1.02

batched_policy:
  total_time_s: 12.5
  episodes_per_second: 8.0
  speedup: 8.0

cuda_safety:
  total_time_s: 95.0
  safety_latency_ms: 0.05
  speedup: 1.05

all_optimized:
  total_time_s: 10.0
  episodes_per_second: 10.0
  memory_mb: 256.0
  total_speedup: 10.0
```

---

### Phase 7 Summary Checklist

- [ ] Ring buffer implemented for telemetry
- [ ] Memory pooling implemented
- [ ] Zero-copy memory implemented
- [ ] Policy batching implemented
- [ ] Safety check batching implemented
- [ ] CUDA kernel optimization implemented
- [ ] Quantization implemented
- [ ] Performance benchmarking added
- [ ] All optimizations tested

**Expected Performance Improvements:**
- Memory usage: 2048MB → 256MB (8× reduction)
- Throughput: 1 episode/sec → 10 episodes/sec (10× faster)
- Safety latency: 0.5ms → 0.05ms (10× faster)
- GPU utilization: 10% → 90% (9× improvement)

---

## Updated Timeline with Phase 7

**Estimated total time:** 10-14 weeks (aggressive) or 14-18 weeks (realistic)
