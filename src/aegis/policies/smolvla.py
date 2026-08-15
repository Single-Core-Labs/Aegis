from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from aegis.config.models import SmolVLAPolicySpec
from aegis.envs.base import Env
from aegis.envs.mujoco_pick_place import MujocoPickPlaceEnv
from aegis.policies.base import Policy

MAX_JOINT_VEL = 0.5  # rad/s clamp on resolved-rate output (gateway enforces measured)
DLS_LAMBDA = 1e-4


class PolicyModelError(Exception):
    """The policy model crashed or produced a non-finite action."""


def axis_angle_from_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    if angle < 1e-8:
        return np.zeros(3)
    s = np.sin(angle)
    if abs(s) > 1e-8:
        axis = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        ) / (2.0 * s)
    else:
        # angle near pi: R = 2 u u^T - I is symmetric; recover |u| from the
        # diagonal and resolve signs from the off-diagonals.
        u = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, 1.0))
        if R[2, 1] < 0.0:
            u[0] = -u[0]
        if R[0, 2] < 0.0:
            u[1] = -u[1]
        if R[1, 0] < 0.0:
            u[2] = -u[2]
        axis = u
    return axis * angle


def resolved_rate_velocity(
    jacobian: np.ndarray,
    cart_delta: np.ndarray,
    dt: float,
    max_vel: float = MAX_JOINT_VEL,
    lam: float = DLS_LAMBDA,
) -> np.ndarray:
    """Damped least squares resolved-rate: joint velocities for a 6-D cart delta."""
    J = np.asarray(jacobian, dtype=float).reshape(6, 7)
    err = np.asarray(cart_delta, dtype=float).reshape(6) / dt
    qd = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
    return np.clip(qd, -max_vel, max_vel)


class SmolVLAPolicy(Policy):
    """LeRobot SmolVLA-450M policy fine-tuned on LIBERO (Franka Panda).

    Observation contract (from the checkpoint): 3 images at 256x256, an 8-dim
    state [eef_pos(3), axis-angle(3), gripper_qpos(2)], and a fixed language
    instruction. The checkpoint's bundled pre/post processor pipelines are
    applied verbatim. Actions are 7-dim EEF cartesian deltas + gripper; they
    are converted to joint velocities with a damped least squares resolved-rate
    controller so the harness's 8-dim joint-velocity action space can consume
    them.
    """

    name = "smolvla"

    def __init__(
        self,
        endpoint: str,
        spec: SmolVLAPolicySpec,
        env: Env,
        device: str = "cpu",
    ) -> None:
        assert isinstance(env, MujocoPickPlaceEnv), "smolvla policy needs the MuJoCo env"
        import torch  # deferred: heavy import, only needed for model policies
        from lerobot.policies.smolvla import SmolVLAPolicy as LeRobotPolicy
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        self._env = env
        self._device = device
        self._instruction = spec.instruction
        self._cameras = list(spec.cameras)

        self._policy = LeRobotPolicy.from_pretrained(endpoint)
        self._policy.to(device)
        self._pre = PolicyProcessorPipeline.from_pretrained(
            endpoint, config_filename="policy_preprocessor.json"
        )
        self._post = PolicyProcessorPipeline.from_pretrained(
            endpoint, config_filename="policy_postprocessor.json"
        )
        self._torch = torch
        self._chunk_remaining = 0
        self._batch: dict[str, Any] | None = None

    def reset(self, seed: int) -> None:
        # LeRobot's select_action keeps an internal 50-action chunk queue that
        # survives episodes; stale chunks must not leak across resets.
        for queue in self._policy._queues.values():
            queue.clear()
        self._chunk_remaining = 0
        self._batch = None

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        from lerobot.processor.pipeline import TransitionKey, create_transition

        try:
            if self._chunk_remaining == 0:
                # Chunk exhausted: fresh images + instruction at inference time.
                images = self._env.render_images(self._cameras)
                self._last_images = images
                transition = create_transition(
                    observation=self._build_observation(obs, images),
                    complementary_data={"task": self._instruction},
                )
                batch = self._pre._forward(transition)[TransitionKey.OBSERVATION]
                self._batch = {
                    k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    action = self._policy.select_action(self._batch)
                    unnorm = self._post._forward(
                        create_transition(action=action)
                    )[TransitionKey.ACTION]
                self._chunk_remaining = int(self._policy.config.n_action_steps) - 1
            else:
                # Queue pop: no inference, no images needed.
                with torch.no_grad():
                    action = self._policy.select_action(self._batch)
                    unnorm = self._post._forward(
                        create_transition(action=action)
                    )[TransitionKey.ACTION]
                self._chunk_remaining -= 1
        except Exception as exc:
            raise PolicyModelError(f"model inference failed: {exc!r}") from exc

        a = unnorm.detach().cpu().numpy().reshape(-1)
        if not np.isfinite(a).all():
            raise PolicyModelError(f"non-finite action from {self.name}: {a}")
        return self._adapt(a)

    # ------------------------------------------------------------------ internals

    def _build_observation(
        self, obs: dict[str, np.ndarray], images: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        state = np.concatenate(
            [obs["hand_pos"].ravel(), axis_angle_from_matrix(obs["hand_xmat"]),
             obs["finger_qpos"].ravel()]
        ).astype(np.float32)
        batch: dict[str, Any] = {"observation.state": state[None]}
        for cam in self._cameras:
            img = np.asarray(images[cam])
            batch[f"observation.images.{cam}"] = (
                img.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
            )
        return batch

    def _adapt(self, action: np.ndarray) -> np.ndarray:
        env = self._env
        hand_id = env.model.body("hand").id
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jacBody(env.model, env.data, jacp, jacr, hand_id)
        J = np.vstack([jacp, jacr])[:, env.arm_qvel_ids].reshape(6, 7)
        vel = resolved_rate_velocity(J, action[0:6], env.dt)
        gripper = float(np.clip(action[6], 0.0, 1.0))
        return np.concatenate([vel, [gripper]])