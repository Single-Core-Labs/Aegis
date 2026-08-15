from __future__ import annotations

import time

import numpy as np
import pytest

from aegis.config.models import (
    EnvSpec,
    EvalSpec,
    ModelSpec,
    PhysicalAIYaml,
    RobotSpec,
    SafetyLimits,
    SmolVLAPolicySpec,
    TaskSpec,
)
from aegis.envs.mujoco_pick_place import MujocoPickPlaceEnv
from aegis.eval.runner import EvalRunner
from aegis.policies.base import Policy
from aegis.policies.smolvla import PolicyModelError, axis_angle_from_matrix, resolved_rate_velocity
from aegis.safety.fallback import PidToHomeFallback
from aegis.safety.gateway import SafetyGateway
from aegis.telemetry.logger import RunLogger

from test_eval import REPO, _run_pai

SCENE = str(REPO / "assets/scenes/pick_place.xml")


def _base_cfg(**eval_kwargs) -> PhysicalAIYaml:
    kwargs = {"episodes": 1, "seed": 1, "max_steps_per_episode": 100}
    kwargs.update(eval_kwargs)
    return PhysicalAIYaml(
        eval=EvalSpec(**kwargs),
        model=ModelSpec(
            name="stub",
            kind="smolvla",
            endpoint="lerobot/smolvla_libero",
            policy=SmolVLAPolicySpec(instruction="pick up the red cube"),
        ),
        robot=RobotSpec(
            name="franka",
            mjcf_path="assets/menagerie/franka_emika_panda/panda_vel.xml",
            safety=SafetyLimits(max_velocity=0.6, max_force=20.0),
        ),
        environment=EnvSpec(scene_mjcf=SCENE, robot_name="franka"),
        task=TaskSpec(),
    )


class _GarbagePolicy(Policy):
    name = "garbage"

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.array([9.0] * 7 + [5.0])


class _CrashingPolicy(Policy):
    name = "crashing"

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        raise PolicyModelError("cuda OOM (simulated)")


class _SlowPolicy(Policy):
    name = "slow"
    sleep_s = 0.25

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        time.sleep(self.sleep_s)
        return np.zeros(8)


def _run_with(policy: Policy, cfg: PhysicalAIYaml, output_dir) -> list:
    env = MujocoPickPlaceEnv(SCENE, cfg.task, cfg.eval.time_step)
    env.reset(cfg.eval.seed)
    fallback = PidToHomeFallback(home_qpos=env.home_qpos, max_velocity=0.3)
    gateway = SafetyGateway(limits=cfg.robot.safety, joint_names=env.joint_names, fallback=fallback)
    logger = RunLogger(output_dir, "test-run")
    logger.write_run(cfg.model_dump(mode="json"))
    try:
        runner = EvalRunner(cfg, env, policy, gateway, fallback, logger)
        results = runner.run()
    finally:
        env.close()
        logger.close()
    return results


class TestBudgetEnforcement:
    def test_slow_policy_trips_budget_and_fallback(self, tmp_path) -> None:
        cfg = _base_cfg(inference_budget_ms=100.0)
        results = _run_with(_SlowPolicy(), cfg, tmp_path)
        r = results[0]
        assert r.budget_violations > 0, "slow policy must trip the inference budget"
        assert r.violations >= r.budget_violations
        assert r.recoveries >= r.budget_violations, "budget events must engage fallback"

    def test_slow_policy_within_budget_has_no_budget_events(self, tmp_path) -> None:
        cfg = _base_cfg(inference_budget_ms=1000.0)
        results = _run_with(_SlowPolicy(), cfg, tmp_path)
        assert results[0].budget_violations == 0

    def test_garbage_policy_engages_fallback(self, tmp_path) -> None:
        cfg = _base_cfg()
        results = _run_with(_GarbagePolicy(), cfg, tmp_path)
        r = results[0]
        assert r.violations > 0
        assert r.recoveries > 0, "unsafe actions must engage fallback"

    def test_crashing_policy_counts_model_error(self, tmp_path) -> None:
        cfg = _base_cfg()
        results = _run_with(_CrashingPolicy(), cfg, tmp_path)
        r = results[0]
        assert r.model_errors > 0
        assert r.recoveries > 0, "model crash must engage fallback"


class TestAdapterMath:
    def test_axis_angle_roundtrip(self) -> None:
        R = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        aa = axis_angle_from_matrix(R)
        assert np.allclose(np.linalg.norm(aa), np.pi, atol=1e-6)
        assert np.allclose(axis_angle_from_matrix(np.eye(3)), 0.0)

    def test_resolved_rate_shape_and_finiteness(self) -> None:
        J = np.zeros((6, 7))
        np.fill_diagonal(J[:7, :], 1.0)
        vel = resolved_rate_velocity(J, np.ones(6) * 0.01, dt=0.02)
        assert vel.shape == (7,)
        assert np.isfinite(vel).all()
        assert np.abs(vel).max() <= 0.5 + 1e-9
        assert np.allclose(vel[:6], 0.01 / 0.02, atol=1e-3)

    def test_resolved_rate_dls_with_singular_jacobian(self) -> None:
        J = np.zeros((6, 7))
        vel = resolved_rate_velocity(J, np.ones(6) * 0.1, dt=0.02)
        assert np.isfinite(vel).all()
        assert np.abs(vel).max() <= 0.5 + 1e-9


class TestSmolVLAConfig:
    def test_smolvla_model_config_loads(self, tmp_path) -> None:
        result = _run_pai(["validate", "--model", "smolvla_libero"])
        assert result.returncode == 0, result.stderr
        assert "config OK" in result.stdout

    def test_smolvla_requires_endpoint(self) -> None:
        with pytest.raises(ValueError, match="requires model.endpoint"):
            ModelSpec(
                name="x",
                kind="smolvla",
                policy=SmolVLAPolicySpec(instruction="pick up the red cube"),
            )

    def test_endpoint_forbidden_for_scripted(self) -> None:
        from aegis.config.models import ScriptedPolicySpec

        with pytest.raises(ValueError, match="only valid for kind='smolvla'"):
            ModelSpec(
                name="x",
                kind="scripted",
                endpoint="lerobot/smolvla_libero",
                policy=ScriptedPolicySpec(),
            )

    def test_invalid_inference_mode_rejected(self) -> None:
        with pytest.raises(Exception):
            EvalSpec(inference_mode="quantum")