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

    def test_default_velocity_limit_has_headroom_over_adapter_clamp(self) -> None:
        # Regression for the Phase 2 diagnostic: the 0.6 rad/s default limit
        # sat against the adapter's 0.5 clamp, so controller transient
        # overshoot chronically tripped the gateway. The default must keep
        # >=2x headroom over the clamp.
        import yaml

        from aegis.policies.smolvla import MAX_JOINT_VEL

        raw = yaml.safe_load(
            (REPO / "configs" / "robots" / "franka.yaml").read_text(encoding="utf-8")
        )
        limit = raw["safety"]["max_velocity"]
        # per-joint list or uniform float both valid — headroom must hold for every joint
        min_limit = min(limit) if isinstance(limit, list) else limit
        assert min_limit >= 2 * MAX_JOINT_VEL, (
            f"default max_velocity {limit} must keep >=2x headroom over the "
            f"adapter clamp {MAX_JOINT_VEL}"
        )

    def test_adapter_output_stays_under_default_limit_in_env(self) -> None:
        # Drive the real env with resolved-rate-adapted cartesian deltas (the
        # SmolVLA adapter math) toward the cube; assert commanded velocities
        # stay within the adapter clamp and measured velocities stay under the
        # default safety limit with headroom.
        import yaml

        import mujoco

        from aegis.policies.smolvla import MAX_JOINT_VEL, resolved_rate_velocity

        raw = yaml.safe_load(
            (REPO / "configs" / "robots" / "franka.yaml").read_text(encoding="utf-8")
        )
        limit = raw["safety"]["max_velocity"]
        min_limit = min(limit) if isinstance(limit, list) else limit
        max_limit = max(limit) if isinstance(limit, list) else limit
        env = MujocoPickPlaceEnv(
            scene_mjcf=str(REPO / "assets" / "scenes" / "pick_place.xml"),
            task=_base_cfg().task,
            time_step=0.02,
        )
        try:
            env.reset(seed=7)
            hand_id = env.model.body("hand").id
            max_cmd = 0.0
            max_meas = 0.0
            for _ in range(60):
                obs = env.observe()
                jacp = np.zeros((3, env.model.nv))
                jacr = np.zeros((3, env.model.nv))
                mujoco.mj_jacBody(env.model, env.data, jacp, jacr, hand_id)
                J = np.vstack([jacp, jacr])[:, env.arm_qvel_ids].reshape(6, 7)
                delta = obs["object_pos"] - obs["hand_pos"]
                delta = 0.05 * delta / (np.linalg.norm(delta) + 1e-9)
                vel = resolved_rate_velocity(
                    J, np.concatenate([delta, np.zeros(3)]), env.dt
                )
                max_cmd = max(max_cmd, float(np.abs(vel).max()))
                obs, terminated, truncated, info = env.step(
                    np.concatenate([vel, [0.5]])
                )
                max_meas = max(
                    max_meas,
                    float(np.abs(env.state_snapshot()["arm_qvel"]).max()),
                )
        finally:
            env.close()
        assert max_cmd <= MAX_JOINT_VEL + 1e-9
        assert max_cmd <= min_limit / 2 + 1e-9, "adapter clamp must sit at <=limit/2"
        assert max_meas <= max_limit, (
            f"measured velocity {max_meas:.3f} rad/s must stay under the "
            f"default limit {max_limit}"
        )


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


class TestGr00TConfig:
    """Phase A acceptance: config validates, eval fails honestly (exit 2)."""

    def test_groot_model_config_loads(self) -> None:
        result = _run_pai(["validate", "--model", "groot_n17"])
        assert result.returncode == 0, result.stderr
        assert "config OK" in result.stdout

    def test_groot_requires_endpoint(self) -> None:
        from aegis.config.models import GrootPolicySpec

        with pytest.raises(ValueError, match="requires model.endpoint"):
            ModelSpec(
                name="x",
                kind="groot",
                policy=GrootPolicySpec(instruction="pick up the red cube"),
            )

    def test_groot_eval_without_checkpoint_fails_honest_exit_2(self, tmp_path) -> None:
        out = tmp_path / "out"
        result = _run_pai(
            ["eval", "--model", "groot_n17", "--episodes", "1", "--output-dir", str(out)]
        )
        assert result.returncode == 2, result.stdout
        assert "Phase B" in result.stderr, result.stderr

    def test_groot_act_backstop_raises_model_error(self) -> None:
        from aegis.config.loader import ConfigError
        from aegis.config.models import GrootPolicySpec
        from aegis.policies.groot import Gr00tPolicy

        with pytest.raises(ConfigError, match="Phase B"):
            Gr00tPolicy(
                endpoint="nvidia/GR00T-N1.7-LIBERO/libero_object",
                spec=GrootPolicySpec(instruction="pick up the red cube"),
                env=None,  # type: ignore[arg-type]
            )


class _FakeModality:
    def __init__(self, keys, horizon):
        self.modality_keys = list(keys)
        self.delta_indices = list(range(horizon))


def _fake_gr00t_modules(monkeypatch):
    import sys
    import types

    video_keys = ("image", "wrist_image")
    state_keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    calls = {"n": 0}

    class FakeVendorPolicy:
        def __init__(self, embodiment_tag, model_path, device="cpu"):
            self.modality_configs = {
                "video": _FakeModality(video_keys, 1),
                "state": _FakeModality(state_keys, 1),
                "action": _FakeModality(state_keys, 16),
                "language": _FakeModality(
                    ["annotation.human.action.task_description"], 1
                ),
            }

        def get_action(self, obs):
            calls["n"] += 1
            chunk = np.zeros((16, 7), dtype=np.float32)
            chunk[:, 2] = 0.5
            return (
                {k: chunk[:, i].reshape(1, 16, 1) for i, k in enumerate(state_keys)},
                {},
            )

    pkg = types.ModuleType("gr00t")
    data_pkg = types.ModuleType("gr00t.data")
    tags_mod = types.ModuleType("gr00t.data.embodiment_tags")

    class FakeTag:
        value = "libero_sim"

    class FakeEmbodimentTag:
        @staticmethod
        def resolve(tag):
            if str(tag).upper() in ("LIBERO_PANDA", "LIBERO_SIM"):
                return FakeTag()
            raise ValueError(f"Unknown embodiment tag: {tag!r}")

    tags_mod.EmbodimentTag = FakeEmbodimentTag
    policy_pkg = types.ModuleType("gr00t.policy")
    vendor_mod = types.ModuleType("gr00t.policy.gr00t_policy")
    vendor_mod.Gr00tPolicy = FakeVendorPolicy
    monkeypatch.setitem(sys.modules, "gr00t", pkg)
    monkeypatch.setitem(sys.modules, "gr00t.data", data_pkg)
    monkeypatch.setitem(sys.modules, "gr00t.data.embodiment_tags", tags_mod)
    monkeypatch.setitem(sys.modules, "gr00t.policy", policy_pkg)
    monkeypatch.setitem(sys.modules, "gr00t.policy.gr00t_policy", vendor_mod)
    return calls


class _FakeGrootEnv:
    """Real MuJoCo model/data (jacobians work headless), stubbed renderer."""

    def __init__(self, real_env):
        self._real = real_env

    def __getattr__(self, name):
        return getattr(self._real, name)

    def render_images(self, cameras):
        rng = np.random.default_rng(0)
        return {
            c: rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
            for c in cameras
        }


def _groot_repo_stub(tmp_path):
    marker = tmp_path / "gr00t" / "policy" / "gr00t_policy.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("# stub marker for _resolve_gr00t_repo", encoding="utf-8")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "processor_config.json").write_text("{}", encoding="utf-8")
    return tmp_path, ckpt


class TestGr00TAdapter:
    def test_rpy_roundtrip(self) -> None:
        from aegis.policies.groot import mat_to_rpy, rpy_to_mat

        rng = np.random.default_rng(0)
        for _ in range(20):
            rpy = rng.uniform([-np.pi, -np.pi / 2 + 0.1, -np.pi], [np.pi, np.pi / 2 - 0.1, np.pi])
            R = rpy_to_mat(rpy)
            R2 = rpy_to_mat(mat_to_rpy(R))
            assert np.allclose(R, R2, atol=1e-9)

    def test_chunk_buffer_serves_16_steps_per_inference(self, tmp_path, monkeypatch) -> None:
        from aegis.config.models import GrootPolicySpec
        from aegis.policies.groot import Gr00tPolicy

        calls = _fake_gr00t_modules(monkeypatch)
        repo, ckpt = _groot_repo_stub(tmp_path)
        real_env = MujocoPickPlaceEnv(scene_mjcf=SCENE, task=TaskSpec(), time_step=0.02)
        try:
            policy = Gr00tPolicy(
                endpoint=str(ckpt),
                spec=GrootPolicySpec(
                    instruction="pick up the red cube", repo_path=str(repo)
                ),
                env=_FakeGrootEnv(real_env),
            )
            obs = real_env.reset(1)
            assert policy.embodiment_tag == "LIBERO_PANDA"
            first = policy.act(obs)
            assert first.shape == (8,) and np.isfinite(first).all()
            assert calls["n"] == 1
            for _ in range(15):
                a = policy.act(obs)
                assert a.shape == (8,) and np.isfinite(a).all()
            assert calls["n"] == 1, "16-step chunk must serve 16 acts"
            policy.act(obs)
            assert calls["n"] == 2, "chunk exhaustion must re-infer"
        finally:
            real_env.close()

    def test_missing_repo_is_config_error(self, tmp_path) -> None:
        from aegis.config.loader import ConfigError
        from aegis.config.models import GrootPolicySpec
        from aegis.policies.groot import Gr00tPolicy

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "processor_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ConfigError, match="Isaac-GR00T repo not found"):
            Gr00tPolicy(
                endpoint=str(ckpt),
                spec=GrootPolicySpec(
                    instruction="pick up the red cube",
                    repo_path=str(tmp_path / "nope"),
                ),
                env=None,  # type: ignore[arg-type]
            )

    def test_unknown_tag_is_config_error(self, tmp_path, monkeypatch) -> None:
        from aegis.config.loader import ConfigError
        from aegis.config.models import GrootPolicySpec
        from aegis.policies.groot import Gr00tPolicy

        _fake_gr00t_modules(monkeypatch)
        repo, ckpt = _groot_repo_stub(tmp_path)
        with pytest.raises(ConfigError, match="Unknown GR00T embodiment tag"):
            Gr00tPolicy(
                endpoint=str(ckpt),
                spec=GrootPolicySpec(
                    instruction="pick up the red cube",
                    embodiment_tag="NOT_A_ROBOT",
                    repo_path=str(repo),
                ),
                env=None,  # type: ignore[arg-type]
            )