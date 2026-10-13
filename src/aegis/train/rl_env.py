"""Gym-style wrapper around the Aegis eval loop (MuJoCo-first).

No hard ``gymnasium`` dependency: exposes the familiar
``reset(seed) -> obs`` / ``step(action) -> (obs, reward, terminated, truncated, info)``
tuple so existing RL code plugs in, and provides ``as_gym()`` when gymnasium
is installed. The ``SafetyGateway`` is mandatory inside ``step()`` — there is
no path that writes an unfiltered action to the env.
"""

from __future__ import annotations

from typing import Any


class AegisRLEnv:
    def __init__(self, env: Any, gateway: Any, seed: int = 42) -> None:
        self.env = env
        self.gateway = gateway
        self.seed = int(seed)
        self.violation_count = 0
        self.action_dim = 8

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        s = int(seed) if seed is not None else self.seed
        return self.env.reset(s)

    def step(self, action: Any):
        import numpy as np  # type: ignore

        state = self.env.state_snapshot()
        gated = self.gateway.filter(np.asarray(action, dtype=float), state)
        if getattr(gated, "violated", False):
            self.violation_count += 1
        obs, terminated, truncated, info = self.env.step(gated.command)
        info = dict(info)
        info["gated_source"] = getattr(gated, "source", "policy")
        info["violated"] = bool(getattr(gated, "violated", False))
        # Sparse task reward: 1 on success-termination, else 0. Dense shaping
        # lives in the trainer, not here, so eval and training share semantics.
        reward = 1.0 if terminated and bool(info.get("success", False)) else 0.0
        done = bool(terminated or truncated)
        return obs, reward, bool(terminated), bool(truncated), info

    @property
    def observation_space(self) -> dict[str, tuple[int, ...]]:
        return {"arm_qpos": (7,), "arm_qvel": (7,), "object_pos": (3,)}

    @property
    def action_space(self) -> tuple[int, ...]:
        return (8,)

    def as_gym(self):  # pragma: no cover - thin adapter, needs gymnasium
        try:
            import gymnasium as gym  # type: ignore
        except Exception as exc:
            raise ImportError("gymnasium is not installed") from exc

        outer = self

        class _Gym(gym.Env):  # type: ignore
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                import numpy as np  # type: ignore

                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
                self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                obs = outer.reset(seed)
                import numpy as np  # type: ignore

                return np.asarray(obs.get("arm_qpos", []), dtype=np.float32), {}

            def step(self, action):
                obs, reward, terminated, truncated, info = outer.step(action)
                import numpy as np  # type: ignore

                return np.asarray(obs.get("arm_qpos", []), dtype=np.float32), reward, terminated, truncated, info

        return _Gym()
