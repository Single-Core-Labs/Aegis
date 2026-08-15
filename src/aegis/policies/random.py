from __future__ import annotations

import numpy as np

from aegis.config.models import RandomPolicySpec
from aegis.policies.base import Policy


class RandomPolicy(Policy):
    """Seeded uniform-random stand-in policy (proves the harness, not the policy).

    Velocities in [-0.9, 0.9] rad/s (deliberately beyond the 0.6 rad/s safety
    limit so the gateway is exercised), gripper open/close each step with p=0.5.
    """

    name = "random"

    def __init__(self, spec: RandomPolicySpec) -> None:
        self._seed = spec.seed
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(self._seed + seed)

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        assert self._rng is not None, "act() before reset()"
        vel = self._rng.uniform(-0.9, 0.9, size=7)
        gripper = float(self._rng.integers(0, 2))
        return np.concatenate([vel, [gripper]])