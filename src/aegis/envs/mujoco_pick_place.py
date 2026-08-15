from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from aegis.config.models import TaskSpec
from aegis.envs.base import Env


class MujocoPickPlaceEnv(Env):
    """Joint-velocity-controlled Franka pick-place scene.

    Action space (8 dims): [v_joint1..v_joint7 (rad/s), gripper (0=closed, 1=open)].
    Velocities feed `<velocity>` actuators (ctrl = desired velocity, rad/s).
    Gripper maps to the tendon actuator (ctrl 0..255, 255 = fully open).

    Success: the object was once lifted off the table (grasped) and ends the
    episode within `task.success_threshold_m` of the target marker.
    """

    name = "mujoco-pick-place"

    ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    def __init__(
        self,
        scene_mjcf: str,
        task: TaskSpec,
        time_step: float,
        render_cameras: list[str] | None = None,
    ) -> None:
        self._task = task
        self._model = mujoco.MjModel.from_xml_path(str(Path(scene_mjcf)))
        self._model.opt.timestep = time_step
        self._data = mujoco.MjData(self._model)

        self._arm_joint_ids = np.array(
            [self._model.joint(j).id for j in self.ARM_JOINTS], dtype=int
        )
        self._arm_qvel_ids = np.array(
            [self._model.joint(j).dofadr for j in self.ARM_JOINTS], dtype=int
        )
        self._arm_qpos_ids = np.array(
            [self._model.joint(j).qposadr for j in self.ARM_JOINTS], dtype=int
        )
        self._finger_qpos_ids = np.array(
            [self._model.joint(f"finger_joint{i}").qposadr for i in (1, 2)], dtype=int
        )
        self._gripper_actuator_id = self._model.actuator("actuator8").id

        self._object_body = self._model.body(task.object_name)
        self._target_body = self._model.body(task.target_name)
        self._hand_body = self._model.body("hand")
        table = self._model.body("table")
        top = self._model.geom("table_top")
        self._table_top_z = float(table.pos[2] + top.size[2])

        self._home_arm_qpos = np.asarray(self._model.key_qpos)[0, self._arm_qpos_ids].ravel()
        self._home_gripper_open = 0.04
        self._nominal_object_pos = self._object_body.pos.copy()
        # Gravity-compensated torque tracking gains (velocity mode).
        self._Kp = 40.0
        self._Kd = 5.0

        self._rng: np.random.Generator | None = None
        self._grasped_ever = False
        self._step_count = 0
        self._q_target = self._home_arm_qpos.copy()
        self._render_cameras = list(render_cameras or [])
        self._renderer: mujoco.Renderer | None = None

    # ------------------------------------------------------------------ Env API

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        self._rng = np.random.default_rng(seed)
        self._data.qpos[:] = self._model.key_qpos[0]
        self._data.qvel[:] = 0.0
        self._q_target = self._home_arm_qpos.copy()
        # Small deterministic object placement jitter around its nominal spot.
        jitter = self._rng.uniform(-0.02, 0.02, size=2)
        pos = self._nominal_object_pos.copy()
        pos[:2] += jitter
        obj_adr = self._model.body(self._task.object_name).jntadr[0]
        self._data.qpos[obj_adr : obj_adr + 3] = pos
        self._data.qpos[obj_adr + 3 : obj_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self._data.ctrl[self._gripper_actuator_id] = 255.0
        mujoco.mj_forward(self._model, self._data)
        # Hold the home pose with gravity-compensating torque so the first
        # state snapshot reads ~zero effort above gravity (else the force
        # check flags -qfrc_bias as a violation at step 1).
        self._data.ctrl[:7] = self._data.qfrc_bias[self._arm_qvel_ids].ravel()
        self._grasped_ever = False
        self._step_count = 0
        return self.observe()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], bool, bool, dict[str, Any]]:
        if self._rng is None:
            raise RuntimeError("step() called before reset()")
        vel = np.clip(np.asarray(action[:7], dtype=float), -1.0, 1.0)
        gripper = float(np.clip(action[7], 0.0, 1.0))
        # Velocity control like a real Franka controller: integrate the commanded
        # velocity into a position target, then track it with a gravity-compensated
        # torque controller (feedforward gravity + P on target error + D on
        # velocity error).
        self._q_target += vel * self.dt
        self._q_target = np.clip(
            self._q_target,
            self._model.jnt_range[self._arm_joint_ids, 0],
            self._model.jnt_range[self._arm_joint_ids, 1],
        )
        q = self._data.qpos[self._arm_qpos_ids].ravel()
        qd = self._data.qvel[self._arm_qvel_ids].ravel()
        err = self._q_target - q
        tau = self._data.qfrc_bias[self._arm_qvel_ids].ravel()
        tau = tau + self._Kp * err + self._Kd * (vel - qd)
        self._data.ctrl[:7] = tau
        self._data.ctrl[self._gripper_actuator_id] = 255.0 * gripper
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        object_pos = self._data.xpos[self._object_body.id]
        target_pos = self._data.xpos[self._target_body.id]
        # Lifted: object center above table top by more than its rest height
        # (half cube = 0.025) plus a margin, so resting on the table never
        # counts as grasped.
        if object_pos[2] > self._table_top_z + 0.06:
            self._grasped_ever = True

        dist = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
        success = self._grasped_ever and dist <= self._task.success_threshold_m
        info: dict[str, Any] = {
            "success": bool(success),
            "object_pos": object_pos.copy(),
            "target_pos": target_pos.copy(),
            "grasped": bool(self._grasped_ever),
            "object_to_target_dist": round(dist, 5),
        }
        return self.observe(), bool(success), False, info

    def observe(self) -> dict[str, np.ndarray]:
        obs = {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "arm_qpos": self._data.qpos[self._arm_qpos_ids].copy(),
            "arm_qvel": self._data.qvel[self._arm_qvel_ids].copy(),
            "finger_qpos": self._data.qpos[self._finger_qpos_ids].copy(),
            "object_pos": self._data.xpos[self._object_body.id].copy(),
            "target_pos": self._data.xpos[self._target_body.id].copy(),
            "hand_pos": self._data.xpos[self._hand_body.id].copy(),
            "hand_xmat": self._data.xmat[self._hand_body.id].reshape(3, 3).copy(),
        }
        if self._render_cameras:
            obs["images"] = self.render_images(self._render_cameras)
        return obs

    def render_images(self, cameras: list[str] | None = None) -> dict[str, np.ndarray]:
        """Render the given cameras at 256x256. Used by vision policies at
        inference time (chunk boundaries) so per-step observation stays cheap."""
        names = list(cameras or self._render_cameras)
        if not names:
            return {}
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=256, width=256)
        images: dict[str, np.ndarray] = {}
        for cam in names:
            self._renderer.update_scene(self._data, camera=cam)
            images[cam] = self._renderer.render().copy()
        return images

    def state_snapshot(self) -> dict[str, np.ndarray]:
        arm = slice(0, 7)
        return {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "arm_qpos": self._data.qpos[self._arm_qpos_ids].ravel().copy(),
            "arm_qvel": self._data.qvel[self._arm_qvel_ids].ravel().copy(),
            # Effort above gravity compensation: the controller's feedback term.
            # Raw qfrc_actuator includes the gravity feedforward (tens of Nm),
            # which would trip any meaningful limit even at rest.
            "arm_torque": (
                self._data.qfrc_actuator[self._arm_qvel_ids].ravel()
                - self._data.qfrc_bias[self._arm_qvel_ids].ravel()
            ).copy(),
            "gripper_ctrl": np.array([self._data.ctrl[self._gripper_actuator_id]]),
            "object_pos": self._data.xpos[self._object_body.id].copy(),
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------ Env properties

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def dt(self) -> float:
        return float(self._model.opt.timestep)

    @property
    def joint_names(self) -> list[str]:
        return list(self.ARM_JOINTS)

    @property
    def home_qpos(self) -> np.ndarray:
        return self._home_arm_qpos.copy()

    @property
    def gripper_open_ctrl(self) -> float:
        return 1.0

    @property
    def gripper_closed_ctrl(self) -> float:
        return 0.0

    @property
    def table_top_z(self) -> float:
        return self._table_top_z

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def arm_qvel_ids(self) -> np.ndarray:
        return self._arm_qvel_ids

    @property
    def arm_qpos_ids(self) -> np.ndarray:
        return self._arm_qpos_ids