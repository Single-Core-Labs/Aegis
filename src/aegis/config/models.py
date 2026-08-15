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

    max_velocity: float = Field(default=1.0, gt=0)
    max_force: float = Field(default=40.0, gt=0)
    max_effort_action: float = Field(default=1.0, gt=0, allow_inf=True)
    reject_nan_actions: bool = True
    action_clamp: Literal["clamp", "reject"] = "reject"
    recovery_steps: int = Field(default=50, ge=1)
    recovery_mode: Literal["resume", "hold"] = "resume"


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    mjcf_path: str = Field(min_length=1)
    safety: SafetyLimits


class EnvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sim: Literal["mujoco"] = "mujoco"
    scene_mjcf: str = Field(min_length=1)
    robot_name: str = Field(min_length=1)
    control_mode: Literal["joint_velocity"] = "joint_velocity"
    render_cameras: list[str] = Field(default_factory=list)


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