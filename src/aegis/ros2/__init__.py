"""ROS2 bridge package — aegis ↔ ROS2 topics (Phase 3, Step 3).

Real transport uses rclpy when available; otherwise an in-memory mock
lets the harness and tests run without a ROS2 install (WSL2 apt is
blocked per WSL2_SETUP_STATUS.md, 6GB VRAM limits Isaac Sim).
"""

from aegis.ros2.bridge import RosBridge, LatencyStats, benchmark_latency

__all__ = ["RosBridge", "LatencyStats", "benchmark_latency"]
