from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from aegis.config.loader import ConfigError
from aegis.config.models import GrootPolicySpec
from aegis.envs.base import Env
from aegis.policies.base import Policy
from aegis.policies.smolvla import (
    MAX_JOINT_VEL,
    PolicyModelError,
    axis_angle_from_matrix,
    resolved_rate_velocity,
)

# Assumptions (calibration risks, see docs/groot-integration-plan.md §8):
# - RPY convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). If LIBERO uses a
#   different euler order, orientation deltas degrade but stay bounded and
#   gateway-gated; roundtrip-tested (mat_to_rpy/rpy_to_mat are inverses).
# - Gripper: our MuJoCo finger_qpos mean (Menagerie range 0..0.04 m) linearly
#   mapped to LIBERO's 1-D gripper in [0, 1], 1 = open. If the checkpoint's
#   statistics.json shows a different raw range, this constant is wrong and
#   must be recalibrated — the failure mode is visible (gripper never closes).
FINGER_QPOS_RANGE_M = 0.04

# libero_sim action horizon in the N1.7-LIBERO checkpoints (16-step chunks).
# Read back from the vendor modality config at load; this is the fallback.
DEFAULT_ACTION_HORIZON = 16


def mat_to_rpy(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (roll, pitch, yaw) with R = Rz(yaw)Ry(pitch)Rx(roll)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(abs(pitch) - np.pi / 2) < 1e-6:
        roll, yaw = 0.0, float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw])


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) -> rotation matrix, inverse of mat_to_rpy."""
    roll, pitch, yaw = (float(v) for v in np.asarray(rpy, dtype=float).ravel()[:3])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _resolve_gr00t_repo(repo_path: str) -> Path:
    """Locate the Isaac-GR00T checkout that provides the `gr00t` package."""
    candidates: list[Path] = []
    if repo_path:
        candidates.append(Path(repo_path).expanduser())
    env = os.environ.get("GR00T_REPO", "")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path(__file__).resolve().parents[3] / "third_party" / "Isaac-GR00T")
    for cand in candidates:
        if (cand / "gr00t" / "policy" / "gr00t_policy.py").is_file():
            return cand
    tried = ", ".join(str(c) for c in candidates)
    raise ConfigError(
        "Isaac-GR00T repo not found. Set policy.repo_path in "
        f"configs/models/groot_n17.yaml or $GR00T_REPO (tried: {tried}). "
        "Expected layout: <repo>/gr00t/policy/gr00t_policy.py."
    )


class Gr00tPolicy(Policy):
    """NVIDIA Isaac GR00T N1.7 adapter — Phase B: in-process loading.

    Loads a post-trained LIBERO checkpoint (e.g. `nvidia/GR00T-N1.7-LIBERO`
    suite dirs) via the Isaac-GR00T `Gr00tPolicy`, feeds it
    `image` + `wrist_image` + 7-D eef state + instruction, buffers the
    absolute-gripper action chunk, and converts each step to 8-D joint
    velocities with the shared DLS resolved-rate adapter. Base
    `nvidia/GR00T-N1.7-3B` does NOT support `libero_sim` (POSTTRAIN tag) —
    the vendor constructor says so explicitly, and we surface that verbatim.

    Code: Apache 2.0. Weights: NVIDIA Open Model License (not Apache-2.0).
    """

    name = "groot"

    # Pinned post-trained suites (one dir each). Never `latest`.
    CHECKPOINT_PIN = "nvidia/GR00T-N1.7-LIBERO (libero_object/ or libero_spatial/ subdir)"

    # libero_sim modality keys (verified against the published checkpoint).
    VIDEO_KEYS = ("image", "wrist_image")
    STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")

    def __init__(
        self,
        endpoint: str,
        spec: GrootPolicySpec,
        env: Env,
        device: str = "cpu",
    ) -> None:
        self._spec = spec
        self._env = env
        self._device = device
        self._seed = 0
        self._chunks_used = 0
        self._chunk: np.ndarray | None = None  # (T, 7) absolute eef+gripper
        self._cameras = list(spec.cameras)
        if len(self._cameras) < 2:
            raise ConfigError(
                "GR00T libero_sim needs 2 camera views (image + wrist_image); "
                f"policy.cameras has {len(self._cameras)}."
            )
        if spec.server_url:
            raise ConfigError(
                "GR00T PolicyServer mode (server_url="
                f"{spec.server_url!r}) lands in Phase D — not implemented. "
                "Use in-process checkpoint mode: leave server_url empty."
            )
        ckpt = Path(endpoint).expanduser() if endpoint else None
        if ckpt is None or not ckpt.is_dir():
            raise ConfigError(
                "GR00T N1.7 LIBERO checkpoint not available "
                f"(endpoint={endpoint!r} is not a local checkpoint dir). "
                "Phase B needs one post-trained suite subdir of nvidia/GR00T-N1.7-LIBERO "
                "(libero_object/ recommended for pick-place) pointed at endpoint. "
                "Base nvidia/GR00T-N1.7-3B will NOT work (libero_sim is POSTTRAIN-only). "
                "See docs/groot-integration-plan.md."
            )
        if not (ckpt / "processor_config.json").is_file():
            raise ConfigError(
                f"GR00T checkpoint dir {ckpt} lacks processor_config.json — "
                "point endpoint at a suite subdir (e.g. .../GR00T-N1.7-LIBERO/libero_object)."
            )
        repo = _resolve_gr00t_repo(spec.repo_path)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy.gr00t_policy import Gr00tPolicy as VendorPolicy
        except ImportError as exc:
            raise ConfigError(
                f"Could not import the `gr00t` package from {repo}: {exc!r}. "
                "Install the Isaac-GR00T dependencies (torch/transformers/numpy) "
                "in this environment."
            ) from exc
        vendor_device = "cuda:0" if device == "cuda" else device
        try:
            tag = EmbodimentTag.resolve(spec.embodiment_tag)
        except ValueError as exc:
            raise ConfigError(f"Unknown GR00T embodiment tag: {exc}") from exc
        try:
            self._vendor = VendorPolicy(
                embodiment_tag=tag, model_path=str(ckpt), device=vendor_device
            )
        except ValueError as exc:
            # Vendor names the supported tags verbatim — surface it, don't paraphrase.
            raise ConfigError(f"GR00T checkpoint rejected the embodiment: {exc}") from exc
        except Exception as exc:
            raise ConfigError(f"GR00T checkpoint failed to load from {ckpt}: {exc!r}") from exc
        got_video = tuple(self._vendor.modality_configs["video"].modality_keys)
        got_state = tuple(self._vendor.modality_configs["state"].modality_keys)
        if got_video != self.VIDEO_KEYS or got_state != self.STATE_KEYS:
            raise ConfigError(
                "GR00T checkpoint modality contract differs from libero_sim "
                f"(video={got_video}, state={got_state}); adapter mapping "
                "assumes image/wrist_image + x,y,z,roll,pitch,yaw,gripper."
            )
        self._language_key: str = self._vendor.modality_configs["language"].modality_keys[0]
        self._horizon: int = len(self._vendor.modality_configs["action"].delta_indices) or DEFAULT_ACTION_HORIZON

    def reset(self, seed: int) -> None:
        self._seed = int(seed)
        self._chunks_used = 0
        self._chunk = None

    @property
    def embodiment_tag(self) -> str:
        return self._spec.embodiment_tag

    @property
    def checkpoint_pin(self) -> str:
        return self.CHECKPOINT_PIN

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if self._chunk is None:
            self._chunk = self._infer_chunk(obs)
        row = self._chunk[0]
        self._chunk = self._chunk[1:]
        if len(self._chunk) == 0:
            self._chunk = None
        return self._adapt(obs, row)

    # ------------------------------------------------------------------ internals

    def _infer_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        import torch

        # Deterministic diffusion sampling: same episode seed + chunk index
        # always draws the same noise. Without this, eval determinism breaks.
        torch.manual_seed((self._seed * 100003 + self._chunks_used) % (2**32))
        self._chunks_used += 1
        images = self._env.render_images(self._cameras)
        try:
            img0 = np.asarray(images[self._cameras[0]], dtype=np.uint8).reshape(1, 1, 256, 256, 3)
            img1 = np.asarray(images[self._cameras[1]], dtype=np.uint8).reshape(1, 1, 256, 256, 3)
        except KeyError as exc:
            raise PolicyModelError(f"GR00T camera missing from renderer: {exc!r}") from exc
        hand_pos = np.asarray(obs["hand_pos"], dtype=np.float32).ravel()[:3]
        rpy = mat_to_rpy(np.asarray(obs["hand_xmat"], dtype=float))
        grip = float(np.clip(np.mean(np.asarray(obs["finger_qpos"], dtype=float)) / FINGER_QPOS_RANGE_M, 0.0, 1.0))
        state_vals = np.concatenate([hand_pos, rpy, [grip]]).astype(np.float32)
        vendor_obs = {
            "video": {
                "image": img0,
                "wrist_image": img1,
            },
            "state": {k: state_vals[i : i + 1].reshape(1, 1, 1) for i, k in enumerate(self.STATE_KEYS)},
            "language": {self._language_key: [[self._spec.instruction]]},
        }
        try:
            action, _ = self._vendor.get_action(vendor_obs)
        except Exception as exc:
            raise PolicyModelError(f"GR00T inference failed: {exc!r}") from exc
        cols = []
        for k in self.STATE_KEYS:
            if k not in action:
                raise PolicyModelError(f"GR00T action missing key {k!r}: {sorted(action)}")
            cols.append(np.asarray(action[k], dtype=np.float32).reshape(-1, 1))
        chunk = np.concatenate(cols, axis=1)  # (T, 7) absolute eef+gripper
        if chunk.shape[0] == 0 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
            raise PolicyModelError(f"non-finite/misshaped GR00T chunk: {chunk.shape}")
        return chunk

    def _adapt(self, obs: dict[str, np.ndarray], row: np.ndarray) -> np.ndarray:
        env = self._env
        hand_id = env.model.body("hand").id
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        mujoco.mj_jacBody(env.model, env.data, jacp, jacr, hand_id)
        J = np.vstack([jacp, jacr])[:, env.arm_qvel_ids].reshape(6, 7)
        # Absolute eef target -> cartesian delta from current pose, then DLS.
        dpos = np.asarray(row[0:3], dtype=float) - np.asarray(obs["hand_pos"], dtype=float).ravel()[:3]
        R_err = rpy_to_mat(row[3:6]) @ np.asarray(obs["hand_xmat"], dtype=float).reshape(3, 3).T
        cart6 = np.concatenate([dpos, axis_angle_from_matrix(R_err)])
        vel = resolved_rate_velocity(J, cart6, env.dt, max_vel=MAX_JOINT_VEL)
        gripper = float(np.clip(row[6], 0.0, 1.0))
        out = np.concatenate([vel, [gripper]])
        if not np.isfinite(out).all():
            raise PolicyModelError(f"non-finite adapted GR00T action: {out}")
        return out
