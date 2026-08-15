from __future__ import annotations

import numpy as np


class PidToHomeFallback:
    """Trivial PID-to-home controller used as the safety fallback.

    Drives joints toward the robot's home pose with bounded velocity and
    opens the gripper. Intentionally dumb: its job is demonstrable recovery,
    not task success.
    """

    name = "pid-to-home"

    def __init__(self, home_qpos: np.ndarray, max_velocity: float = 0.3) -> None:
        self._home = np.asarray(home_qpos, dtype=float)
        self._max_velocity = float(max_velocity)

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        q = np.asarray(state["arm_qpos"], dtype=float).ravel()
        err = self._home - q
        vel = np.clip(1.5 * err, -self._max_velocity, self._max_velocity)
        return np.concatenate([vel, [1.0]])  # gripper open