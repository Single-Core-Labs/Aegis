from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Env(ABC):
    """Environment protocol shared by all aegis envs (sim POC: MuJoCo)."""

    name: str = "base"

    @abstractmethod
    def reset(self, seed: int) -> dict[str, np.ndarray]:
        """Reset the episode. Returns an observation dict."""

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], bool, bool, dict[str, Any]]:
        """Apply a gated action. Returns (obs, terminated, truncated, info)."""

    @abstractmethod
    def observe(self) -> dict[str, np.ndarray]:
        """Current observation (used before the first action too)."""

    @abstractmethod
    def state_snapshot(self) -> dict[str, np.ndarray]:
        """Full state the Safety Gateway inspects (velocities, forces, etc.)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def dt(self) -> float:
        ...

    @property
    @abstractmethod
    def joint_names(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def home_qpos(self) -> np.ndarray:
        """Arm joint positions at the scene's home keyframe (excl. gripper)."""

    @property
    @abstractmethod
    def gripper_open_ctrl(self) -> float:
        """Gripper command (in action units) that opens the gripper."""

    @property
    @abstractmethod
    def gripper_closed_ctrl(self) -> float:
        """Gripper command (in action units) that closes the gripper."""