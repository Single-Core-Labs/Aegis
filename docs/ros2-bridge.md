# ROS 2 Bridge — Topics, Latency & Real-Robot Wiring

## Topics

`src/aegis/ros2/bridge.py` — `RosBridge`:

| Topic | Type | Direction | Content |
|---|---|---|---|
| `/aegis/action` | `Float64MultiArray` (8-D) | Aegis → Robot | 7 joint velocities + gripper |
| `/aegis/state` | `JointState` (7-DOF) | Robot → Aegis | `position[7]`, `velocity[7]` → `obs["arm_qpos"]`, `obs["arm_qvel"]` |
| `/aegis/safety` | `Float64MultiArray` (JSON) | Aegis → Logger | Violation events |

## Modes

| Mode | Condition | Latency | Use |
|---|---|---|---|
| **Mock** | `rclpy` not importable | `p50 0.000 ms` (in-memory) | CI, Windows dev, `aegis eval --sim mujoco` |
| **Real** | `rclpy` available, `rclpy.ok()` | Measured (`0.1–1 ms` typical) | WSL2 + Franka hardware |

`RosBridge(mock=None)` auto-detects. Force with `RosBridge(mock=True|False)`.

## Latency Benchmark

```bash
# Mock — runs anywhere, no ROS 2
uv run aegis rosbench --mock --n 20

# Real — inside WSL2 with ROS 2 Humble
uv run aegis rosbench --real --n 100
# => ros bridge: real rclpy
#      count : 100
#      p50   : 0.34 ms
#      p95   : 0.89 ms
#      mean  : 0.41 ms

# Requirement: p50 + p95 < inference_budget_ms (default 2000 ms)
# For 1 kHz Franka control, target p95 < 1 ms
```

`benchmark_latency()` in `src/aegis/ros2/bridge.py:189` — publishes `n` random 8-D actions, measures `perf_counter` per publish, returns `LatencyStats` (p50/p95/mean/min/max).

For **split** (Windows + WSL2), run `rosbench` on both sides — delta is bridge overhead.

## Real-Robot Wiring (Future: `--sim hardware`)

```python
from aegis.ros2.bridge import RosBridge
import numpy as np

bridge = RosBridge(mock=False)  # real rclpy

# Publish — called by SafetyGateway after filtering
latency_ms = bridge.publish_action(safe_action, step=42)

# Subscribe — called by EvalRunner to get state
bridge.spin_once(timeout_sec=0.01)
obs = bridge.get_state()  # {"arm_qpos": ..., "arm_qvel": ...}

# Safety events
bridge.publish_safety({"type": "velocity", "joint": "joint_3", "value": 2.8, "limit": 2.61})
```

See `HARDWARE_CHECKLIST.md` Gates 1–3 for the full bring-up sequence (e-stop, dry run, Go/No-Go).

## API Reference

```python
RosBridge(mock: bool | None = None)
    .publish_action(action: np.ndarray, step: int | None = None) -> float  # latency ms
    .get_state() -> dict[str, np.ndarray] | None
    .set_mock_state(obs: dict)  # test helper
    .publish_safety(event: dict) -> None
    .spin_once(timeout_sec: float = 0.01) -> None
    .latency_stats() -> LatencyStats
    .close() -> None

benchmark_latency(bridge: RosBridge | None = None, n: int = 100, action_dim: int = 8) -> LatencyStats
```
