from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aegis.config.loader import ConfigError
from aegis.config.models import GrootPolicySpec
from aegis.envs.base import Env
from aegis.policies.base import Policy
from aegis.policies.smolvla import PolicyModelError


class Gr00tPolicy(Policy):
    """NVIDIA Isaac GR00T N1.7 adapter — Phase A: config + honest stub.

    N1.7 is a 3B-parameter dual-system VLA (Cosmos-Reason2/Qwen3-VL System 2 +
    diffusion-transformer System 1) emitting action chunks of relative joint
    motions. Code: Apache 2.0. Weights: NVIDIA Open Model License
    (commercially usable, NOT Apache-2.0).

    Phase A implements endpoint resolution with honest failures; in-process
    checkpoint loading lands in Phase B, PolicyServer mode in Phase D.
    The gateway contract is unchanged: whatever this policy eventually
    returns still passes through `SafetyGateway.filter()`.
    """

    name = "groot"

    # Pinned checkpoint — never `latest` (upstream moves N1.5->N1.6->N1.7 fast).
    CHECKPOINT_PIN = "nvidia/GR00T-N1.7"

    def __init__(
        self,
        endpoint: str,
        spec: GrootPolicySpec,
        env: Env,
        device: str = "cpu",
    ) -> None:
        self._endpoint = endpoint
        self._spec = spec
        self._env = env
        self._device = device
        self._rng: np.random.Generator | None = None
        if spec.server_url:
            raise ConfigError(
                "GR00T PolicyServer mode (server_url="
                f"{spec.server_url!r}) lands in Phase D — not implemented. "
                "Use in-process checkpoint mode (Phase B): leave server_url empty "
                "and point endpoint at a local checkpoint dir."
            )
        local = Path(endpoint).expanduser() if endpoint else None
        if local is not None and local.is_dir():
            raise ConfigError(
                f"GR00T N1.7 checkpoint found at {local} but in-process loading "
                "lands in Phase B — not implemented yet. No actions were produced; "
                "nothing was evaluated."
            )
        raise ConfigError(
            "GR00T N1.7 checkpoint not available "
            f"(endpoint={endpoint!r} is not a local checkpoint dir). "
            f"Pinned checkpoint: {self.CHECKPOINT_PIN} "
            "(weights: NVIDIA Open Model License). "
            "In-process loading lands in Phase B — not implemented yet. "
            "See docs/groot-integration-plan.md."
        )

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        # Backstop: __init__ always raises in Phase A, so reaching here means
        # a future phase regressed. Count it like a model crash, never fake it.
        raise PolicyModelError(
            "Gr00tPolicy.act called before Phase B implements checkpoint loading"
        )

    @property
    def embodiment_tag(self) -> str:
        return self._spec.embodiment_tag

    @property
    def checkpoint_pin(self) -> str:
        return self.CHECKPOINT_PIN
