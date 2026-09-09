from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(default=10, ge=1, le=1000)
    seed: int = 42
    max_steps_per_episode: int = Field(default=500, ge=1, le=100_000)
    time_step: float = Field(default=0.02, gt=0)
    episode_timeout_sec: float = Field(default=10.0, gt=0)
    inference_mode: Literal["cpu", "cuda"] = "cpu"
    inference_budget_ms: float = Field(default=200.0, gt=0)


class RandomPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["random"] = "random"
    seed: int = Field(default=0, ge=0)


class ScriptedPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["scripted"] = "scripted"
    home_qpos: Optional[list[float]] = None
    velocity_limit: float = Field(default=0.4, gt=0, le=2.0)


class SmolVLAPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["smolvla"] = "smolvla"
    instruction: str = Field(min_length=1)
    cameras: list[str] = Field(
        default_factory=lambda: ["camera1", "camera2", "camera3"]
    )
    # int8 quantization for VRAM-safe: ~1GB -> ~0.6GB, frees ~0.4GB for Isaac on 6GB
    quantize: Literal["none", "int8"] = Field(default="none")
    headless: bool = False  # Isaac headless: no window, offscreen 256x256 only


ModelPolicySpec = RandomPolicySpec | ScriptedPolicySpec | SmolVLAPolicySpec


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["random", "scripted", "smolvla"]
    policy: ModelPolicySpec
    endpoint: str = Field(default="", min_length=0)
    load_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_endpoint(self) -> "ModelSpec":
        if self.kind == "smolvla" and not self.endpoint:
            raise ValueError("model.kind 'smolvla' requires model.endpoint")
        if self.kind != "smolvla" and self.endpoint:
            raise ValueError(
                f"model.endpoint is only valid for kind='smolvla' (got {self.kind!r})"
            )
        return self


class SafetyLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Per-joint limits: either a single uniform float (backward-compat, e.g. 1.0)
    # or a 7-element list matching ARM_JOINTS order [j1..j7]. Real Franka limits
    # are per-joint (2.175/2.61 rad/s, 87/12 Nm). Uniform mode emits a warning
    # in the report; per-joint mode is the Phase 3 target.
    max_velocity: float | list[float] = Field(default=1.0)
    max_force: float | list[float] = Field(default=40.0)
    max_effort_action: float = Field(default=1.0, gt=0, allow_inf=True)
    reject_nan_actions: bool = True
    action_clamp: Literal["clamp", "reject"] = "reject"
    recovery_steps: int = Field(default=50, ge=1)
    recovery_mode: Literal["resume", "hold"] = "resume"

    @model_validator(mode="after")
    def _check_limits(self) -> "SafetyLimits":
        for name in ("max_velocity", "max_force"):
            v = getattr(self, name)
            if isinstance(v, list):
                if len(v) != 7:
                    raise ValueError(f"{name} per-joint list must have 7 elements (got {len(v)})")
                for i, x in enumerate(v):
                    if not isinstance(x, (int, float)) or not float(x) > 0 or not float(x) != float("inf"):
                        raise ValueError(f"{name}[{i}] must be finite >0 (got {x!r})")
            else:
                if not isinstance(v, (int, float)) or not float(v) > 0 or not float(v) != float("inf"):
                    raise ValueError(f"{name} must be finite >0 (got {v!r})")
        return self

    def is_uniform_velocity(self) -> bool:
        return isinstance(self.max_velocity, (int, float))

    def is_uniform_force(self) -> bool:
        return isinstance(self.max_force, (int, float))


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    mjcf_path: str = Field(min_length=1)
    safety: SafetyLimits


class EnvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sim: Literal["mujoco", "isaaclab", "isaac"] = "mujoco"
    scene_mjcf: str = Field(min_length=1)
    robot_name: str = Field(min_length=1)
    control_mode: Literal["joint_velocity"] = "joint_velocity"
    render_cameras: list[str] = Field(default_factory=list)
    headless: bool = False  # Isaac headless: --headless, no window, RTX Real-Time bounces=1
    usd_scene: str | None = Field(default=None)  # VRAM-safe USD e.g. assets/usd/pick_place_vram_safe.usda


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["pick-place"] = "pick-place"
    success_threshold_m: float = Field(default=0.05, gt=0)
    object_name: str = "object"
    target_name: str = "target"
    max_grasp_attempts: int = Field(default=3, ge=1, le=10)


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = "./outputs"
    run_id: Optional[str] = None
    report_format: Literal["json"] = "json"


class PhysicalAIYaml(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    eval: EvalSpec = EvalSpec()
    model: ModelSpec
    robot: RobotSpec
    environment: EnvSpec
    task: TaskSpec = TaskSpec()
    output: OutputSpec = OutputSpec()

    @model_validator(mode="after")
    def _check_names(self) -> "PhysicalAIYaml":
        if self.environment.robot_name != self.robot.name:
            raise ValueError(
                f"environment.robot_name ({self.environment.robot_name!r}) must match "
                f"robot.name ({self.robot.name!r})"
            )
        return self