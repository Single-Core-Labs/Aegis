from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from aegis.envs.base import Env


class Policy(ABC):
    """Policy interface. `act` must be deterministic given the episode seed."""

    name: str = "base"

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Called at episode start; resets any internal state/RNG."""

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return a raw (ungated) action: [7 velocities, 1 gripper]."""