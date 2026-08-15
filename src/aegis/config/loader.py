from __future__ import annotations

from pathlib import Path

import yaml

from aegis.config.models import (
    EvalSpec,
    EnvSpec,
    ModelSpec,
    OutputSpec,
    PhysicalAIYaml,
    RobotSpec,
    TaskSpec,
)


class ConfigError(Exception):
    """Raised when a config file is missing, invalid, or fails validation."""


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a mapping at top level")
    return raw


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return p


def load_run_config(
    root_config: Path,
    model_name: str | None = None,
    robot_name: str | None = None,
    task_name: str | None = None,
) -> PhysicalAIYaml:
    """Load and validate the full run config.

    Root YAML holds eval/output plus the default model/robot/task names.
    Component configs live in configs/{models,robots,tasks}/<name>.yaml.
    CLI overrides beat root defaults.
    """
    root = _read_yaml(root_config)
    base = root_config.resolve().parent

    configs_dir = base / "configs"
    model_dir = configs_dir / "models"
    robot_dir = configs_dir / "robots"
    task_dir = configs_dir / "tasks"

    # The --model/--robot/--tasks flags are single names in this POC; the
    # design's root schema keeps them out of the YAML top level for now.
    m_name = model_name or root.get("model_name") or "random"
    r_name = robot_name or root.get("robot_name") or "franka"
    t_name = task_name or root.get("task_name") or "pick-place"

    model_spec = ModelSpec.model_validate(_read_yaml(model_dir / f"{m_name}.yaml"))
    robot_spec = RobotSpec.model_validate(_read_yaml(robot_dir / f"{r_name}.yaml"))
    task_spec = TaskSpec.model_validate(_read_yaml(task_dir / f"{t_name}.yaml"))

    eval_spec = EvalSpec.model_validate(root.get("eval", {}))
    env_spec = EnvSpec.model_validate(root.get("environment", {}))
    output_spec = OutputSpec.model_validate(root.get("output", {}))

    # Resolve asset paths relative to the file that declares them.
    model_spec = model_spec
    robot_spec = robot_spec.model_copy(
        update={"mjcf_path": str(resolve_path(robot_dir, robot_spec.mjcf_path))}
    )
    env_spec = env_spec.model_copy(
        update={"scene_mjcf": str(resolve_path(base, env_spec.scene_mjcf))}
    )

    cfg = PhysicalAIYaml(
        eval=eval_spec,
        model=model_spec,
        robot=robot_spec,
        environment=env_spec,
        task=task_spec,
        output=output_spec,
    )

    # Fail fast on missing asset files (POC honesty contract).
    for label, p in (
        ("robot.mjcf_path", Path(cfg.robot.mjcf_path)),
        ("environment.scene_mjcf", Path(cfg.environment.scene_mjcf)),
    ):
        if not p.is_file():
            raise ConfigError(f"{label} points to a missing file: {p}")
    return cfg