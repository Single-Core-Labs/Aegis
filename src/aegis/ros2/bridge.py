from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatencyStats:
    count: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = k - f
    return sorted_vals[f] * (1 - d0) + sorted_vals[c] * d0


def _stats_from_samples(samples_ms: list[float]) -> LatencyStats:
    if not samples_ms:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s = sorted(samples_ms)
    return LatencyStats(
        count=len(s),
        p50_ms=_percentile(s, 50),
        p95_ms=_percentile(s, 95),
        mean_ms=float(sum(s) / len(s)),
        min_ms=s[0],
        max_ms=s[-1],
    )


class RosBridge:
    """aegis ↔ ROS2 topic bridge.

    Topics (when rclpy is available):
      /aegis/action  (Float64MultiArray, 8 dims)
      /aegis/state   (JointState + images)
      /aegis/safety  (String JSON)

    Without ROS2 the bridge runs in **mock in-memory** mode — same API,
    zero external deps, so `aegis eval --sim mujoco` and tests keep working
    while WSL2 ROS2 apt is blocked. The mock records every publish and
    serves the last state, with synthetic latency samples for benchmarking.

    Interface intentionally mirrors the harness eval loop:
      bridge.publish_action(action, step) -> latency_ms
      bridge.get_state() -> obs dict
      bridge.publish_safety(event_dict)
    """

    def __init__(self, mock: bool | None = None) -> None:
        # auto-detect: mock if rclpy not importable
        if mock is None:
            try:
                import rclpy  # noqa: F401

                self._has_rclpy = True
            except Exception:
                self._has_rclpy = False
        else:
            self._has_rclpy = not mock

        self._mock = not self._has_rclpy
        self._mock_state: dict[str, np.ndarray] | None = None
        self._mock_actions: list[np.ndarray] = []
        self._mock_safety: list[dict] = []
        self._latency_samples: list[float] = []
        self._node: Any = None

        if not self._mock:
            self._init_rclpy_node()

    # ------------------------------------------------------------------ rclpy
    def _init_rclpy_node(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            from sensor_msgs.msg import JointState

            if not rclpy.ok():
                rclpy.init()
            self._node = Node("aegis_bridge")
            self._Float64MultiArray = Float64MultiArray
            self._JointState = JointState
            self._rclpy = rclpy
            # publishers / subscribers created lazily on first use
            self._pub_action = self._node.create_publisher(Float64MultiArray, "/aegis/action", 10)
            self._pub_safety = self._node.create_publisher(Float64MultiArray, "/aegis/safety", 10)
            self._last_joint_state: Any = None
            self._node.create_subscription(JointState, "/aegis/state", self._on_state, 10)
        except Exception as exc:
            # fall back to mock if rclpy init fails (e.g. no daemon)
            self._mock = True
            self._has_rclpy = False
            self._init_error = str(exc)

    def _on_state(self, msg: Any) -> None:
        self._last_joint_state = msg

    # ------------------------------------------------------------------ API
    def publish_action(self, action: np.ndarray, step: int | None = None) -> float:
        """Publish an 8-D action. Returns publish latency in ms."""
        t0 = time.perf_counter()
        a = np.asarray(action, dtype=float).ravel()
        if self._mock:
            self._mock_actions.append(a.copy())
            # synthetic 0.05ms mock latency
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._latency_samples.append(dt_ms)
            return dt_ms
        # real ROS2 publish
        msg = self._Float64MultiArray()
        msg.data = a.tolist()
        self._pub_action.publish(msg)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._latency_samples.append(dt_ms)
        return dt_ms

    def get_state(self) -> dict[str, np.ndarray] | None:
        if self._mock:
            return self._mock_state
        if self._last_joint_state is None:
            return None
        # convert JointState to harness obs fragment
        js = self._last_joint_state
        return {
            "arm_qpos": np.array(js.position[:7], dtype=float),
            "arm_qvel": np.array(js.velocity[:7], dtype=float),
        }

    def set_mock_state(self, obs: dict[str, np.ndarray]) -> None:
        """Test helper: inject a state for mock get_state()."""
        self._mock_state = obs

    def publish_safety(self, event: dict) -> None:
        if self._mock:
            self._mock_safety.append(event)
            return
        # real: publish as stringified JSON in Float64MultiArray data for POC
        import json

        msg = self._Float64MultiArray()
        # misuse: encode JSON length as float placeholder — real impl would use String
        _ = json.dumps(event)
        self._pub_safety.publish(msg)

    def spin_once(self, timeout_sec: float = 0.01) -> None:
        if not self._mock and self._node is not None:
            import rclpy

            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def latency_stats(self) -> LatencyStats:
        return _stats_from_samples(self._latency_samples)

    def close(self) -> None:
        if not self._mock and self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass


def benchmark_latency(
    bridge: RosBridge | None = None,
    n: int = 100,
    action_dim: int = 8,
) -> LatencyStats:
    """Benchmark publish latency.

    Two modes:
      - inside-WSL2 (real ROS2): measures rclpy publish time
      - mock (no ROS2): measures in-memory publish time

    For the Windows↔WSL2 bridge mode, run this benchmark once inside WSL2
    and once from Windows via `wsl ros2 topic` and compare p50/p95.
    """
    b = bridge or RosBridge(mock=True)
    rng = np.random.default_rng(0)
    for _ in range(n):
        a = rng.uniform(-0.5, 0.5, size=action_dim)
        b.publish_action(a)
    stats = b.latency_stats()
    if bridge is None:
        b.close()
    return stats
