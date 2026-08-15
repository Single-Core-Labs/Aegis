from __future__ import annotations

import mujoco
import numpy as np

from aegis.config.models import ScriptedPolicySpec
from aegis.envs.base import Env
from aegis.envs.mujoco_pick_place import MujocoPickPlaceEnv
from aegis.policies.base import Policy

GRASP_Z_OFFSET = 0.095  # hand origin above object center: pads grab the cube's upper half
APPROACH_Z_OFFSET = 0.16
LIFT_Z_OFFSET = 0.25
MOVE_GAIN = 1.2
PHASE_TOL = 0.015  # joint-space convergence tolerance (rad)
CART_TOL = 0.012  # hand-position convergence tolerance (m)
GRASP_CENTER_TOL = 0.002  # hand must be centered over the cube before closing (m)
HOLD_STEPS = 60  # let the fingers settle around the cube before lifting
LIFT_VEL = 0.08  # slow lift so the grip holds without jerking the cube out
GRASP_DESCENT_VEL = 0.15  # slow descent so fingertips don't shove the cube
MAX_GRASP_ATTEMPTS = 3  # matches TaskSpec.max_grasp_attempts default


def _rot_err(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    skew = 0.5 * (R_des @ R_cur.T - R_cur @ R_des.T)
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


class ScriptedPolicy(Policy):
    """Deterministic pick-place state machine (smoke-test policy).

    Phase machine over waypoints computed per-episode with damped least
    squares IK on the hand frame (fingers pointing down). It proves the full
    pipeline can succeed; it is NOT a general-purpose policy.
    """

    name = "scripted"

    def __init__(self, spec: ScriptedPolicySpec, env: Env) -> None:
        assert isinstance(env, MujocoPickPlaceEnv), "scripted policy needs the MuJoCo env"
        self._spec = spec
        self._env = env
        self._vel_limit = spec.velocity_limit
        self._home = env.home_qpos if spec.home_qpos is None else np.asarray(spec.home_qpos)
        self._phase = 0
        self._waypoints: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._hold = 0
        self._planned_obj = np.zeros(3)
        self._grasp_attempts = 0

    # ---------------------------------------------------------------- policy API

    def reset(self, seed: int) -> None:
        self._phase = 0
        self._hold = 0
        self._grasp_attempts = 0
        # Planning mutates env.data.qpos for FK; _plan_waypoints restores the
        # sim state afterwards, so the episode picks up where it left off.
        self._plan_waypoints()

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        q = np.asarray(obs["arm_qpos"]).ravel()
        hand_pos = np.asarray(obs["hand_pos"]).ravel()

        while self._phase < len(self._waypoints):
            target = self._waypoints[self._phase]
            cart = self._targets[self._phase]
            err = target - q

            if self._phase == 1:
                return self._grasp_step(q, hand_pos, obs)

            # Advance on hand position (contact can stall exact joint convergence).
            if np.linalg.norm(hand_pos - cart) <= CART_TOL:
                self._phase += 1
                self._hold = HOLD_STEPS if self._phase != 1 else 0
                continue

            vel = np.clip(MOVE_GAIN * err, -self._vel_limit, self._vel_limit)
            if self._phase == 2:
                vel = np.clip(MOVE_GAIN * err, -LIFT_VEL, LIFT_VEL)
            return self._action(vel, self._gripper_for_phase(self._phase))

        return self._action(np.zeros(7), 1.0)

    # ------------------------------------------------------------------ internals

    def _action(self, vel: np.ndarray, gripper: float) -> np.ndarray:
        return np.concatenate([vel, [gripper]])

    def _gripper_for_phase(self, phase: int) -> float:
        # 0=approach, 1=grasp, 2=lift, 3=pre_place, 4=place, 5=retreat
        # Close while grasping/lifting/carrying/placing; open only on retreat.
        if phase in (2, 3, 4):
            return 0.0
        return 1.0

    def _grasp_step(
        self, q: np.ndarray, hand_pos: np.ndarray, obs: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Descend with fingers open, close centered over the cube, verify the
        grip by finger blockage, and retry (re-planning from the cube's
        current position) if the grasp missed. Failing the grip stops the
        retries and the episode simply ends without success -- an honest
        failure."""
        obj_pos = np.asarray(obs["object_pos"]).ravel()
        finger_qpos = np.asarray(obs["finger_qpos"]).ravel()
        if self._hold > 0:
            self._hold -= 1
            if self._hold == 0:
                self._finish_grasp(finger_qpos, hand_pos, obj_pos)
            # Two-stage close: gentle pinch first so the cube settles centered
            # between the pads instead of being squeezed out, then full grip.
            gripper = 0.0 if self._hold <= HOLD_STEPS // 2 else 0.5
            return self._action(np.zeros(7), gripper)

        cart = self._targets[1]
        err = self._waypoints[1] - q

        if np.linalg.norm(hand_pos - cart) <= CART_TOL:
            if np.linalg.norm(hand_pos[:2] - obj_pos[:2]) <= GRASP_CENTER_TOL:
                self._hold = HOLD_STEPS
                return self._action(np.zeros(7), 0.0)
            # Off-center at the grasp point: if the cube drifted since the
            # last plan (fingertip graze shoves it), re-target at its live
            # position, then keep descending toward it.
            if np.linalg.norm(self._planned_obj[:2] - obj_pos[:2]) > 0.001:
                self._plan_waypoints()
            cart = self._targets[1]
            err = self._waypoints[1] - q
            if np.linalg.norm(err) < PHASE_TOL:
                # Arm converged but the cube is still out of center reach
                # (shoved near the workspace edge): count a failed attempt.
                self._grasp_attempts += 1
                if self._grasp_attempts >= MAX_GRASP_ATTEMPTS:
                    self._phase += 1
                    self._hold = HOLD_STEPS
                    return self._action(np.zeros(7), 1.0)
                return self._action(np.zeros(7), 1.0)
            # Fall through and keep descending toward the re-targeted cube.

        vel = np.clip(MOVE_GAIN * err, -GRASP_DESCENT_VEL, GRASP_DESCENT_VEL)
        return self._action(vel, 1.0)  # fingers open while descending

    def _finish_grasp(
        self, finger_qpos: np.ndarray, hand_pos: np.ndarray, obj_pos: np.ndarray
    ) -> None:
        # Grasp is real when the pads are blocked by the cube (finger joints
        # stop ~0.025 m, half the cube width) instead of closing fully on air
        # (~0.003 m), AND the cube is still centered in the grip (a shove
        # during the close leaves an off-center pinch that the lift pops out).
        # The object's z never rises here -- it's still on the table -- so
        # finger blockage is the reliable signal.
        grasped = float(finger_qpos[0]) > 0.015
        centered = np.linalg.norm(hand_pos[:2] - obj_pos[:2]) <= 0.005
        if grasped and centered:
            # Grasped: re-plan lift/carry from the cube's current position.
            self._plan_waypoints()
            self._phase += 1
            self._hold = HOLD_STEPS
            return
        self._grasp_attempts += 1
        if self._grasp_attempts >= MAX_GRASP_ATTEMPTS:
            # Give up: finish the motion anyway; success simply won't fire.
            self._phase += 1
            self._hold = HOLD_STEPS
            return
        # Missed: open the fingers, re-plan from the cube's current position,
        # and descend again.
        self._plan_waypoints()
        self._hold = 0

    def _plan_waypoints(self) -> None:
        env = self._env
        saved_qpos = env.data.qpos.copy()
        saved_qvel = env.data.qvel.copy()
        saved_ctrl = env.data.ctrl.copy()
        try:
            obj = env.data.xpos[env.model.body("object").id].copy()
            tgt = env.data.xpos[env.model.body("target").id].copy()
            self._planned_obj = obj.copy()

            approach = obj + np.array([0.0, 0.0, APPROACH_Z_OFFSET])
            grasp = obj + np.array([0.0, 0.0, GRASP_Z_OFFSET])
            lift = obj + np.array([0.0, 0.0, LIFT_Z_OFFSET])
            pre_place = tgt + np.array([0.0, 0.0, APPROACH_Z_OFFSET])
            place = tgt + np.array([0.0, 0.0, GRASP_Z_OFFSET])
            retreat = tgt + np.array([0.0, 0.0, LIFT_Z_OFFSET])

            self._targets = [approach, grasp, lift, pre_place, place, retreat]
            self._waypoints = [self._ik(self._home, p) for p in self._targets]
        finally:
            env.data.qpos[:] = saved_qpos
            env.data.qvel[:] = saved_qvel
            env.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(env.model, env.data)

    def _ik(self, q_init: np.ndarray, target_pos: np.ndarray, iters: int = 60) -> np.ndarray:
        """Damped least squares IK: hand position + z-down orientation."""
        import mujoco

        model, data = self._env.model, self._env.data
        hand_id = model.body("hand").id
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        q = q_init.copy()
        # Fingers pointing down: rotate +z to -z about x (proper rotation).
        R_des = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        lam = 1e-4
        arm_ids = env_arm_qpos_ids(model)
        arm_dof_ids = np.array(
            [model.joint(f"joint{i}").dofadr for i in range(1, 8)]
        )

        for _ in range(iters):
            data.qpos[arm_ids] = q.reshape(-1, 1)
            mujoco.mj_forward(model, data)
            R_cur = data.xmat[hand_id].reshape(3, 3)
            pos = data.xpos[hand_id]
            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
            e_pos = (target_pos - pos).ravel()
            e_rot = _rot_err(R_des, R_cur)
            err = np.concatenate([e_rot.ravel(), e_pos])
            jac = np.vstack([jacr, jacp])[:, arm_dof_ids].reshape(6, 7)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(6), err)
            q = q + dq
            if np.linalg.norm(err) < 1e-3:
                break
        # Enforce joint limits (rad) on limited joints only.
        arm_joint_ids = np.array([model.joint(f"joint{i}").id for i in range(1, 8)])
        limited = model.jnt_limited[arm_joint_ids]
        q = np.where(
            limited,
            np.clip(q, model.jnt_range[arm_joint_ids, 0], model.jnt_range[arm_joint_ids, 1]),
            q,
        )
        return q


def env_arm_qpos_ids(model) -> np.ndarray:
    return np.array([model.joint(f"joint{i}").qposadr for i in range(1, 8)], dtype=int)