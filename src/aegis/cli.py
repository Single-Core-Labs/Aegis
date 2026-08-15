from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import typer

from aegis.config.loader import ConfigError, load_run_config
from aegis.config.models import (
    PhysicalAIYaml,
    RandomPolicySpec,
    ScriptedPolicySpec,
    SmolVLAPolicySpec,
)
from aegis.envs.mujoco_pick_place import MujocoPickPlaceEnv
from aegis.eval.report import build_report, print_summary, write_report_json
from aegis.eval.runner import EvalRunner
from aegis.policies.random import RandomPolicy
from aegis.policies.scripted import ScriptedPolicy
from aegis.policies.smolvla import SmolVLAPolicy
from aegis.safety.fallback import PidToHomeFallback
from aegis.safety.gateway import SafetyGateway
from aegis.telemetry.logger import RunLogger

app = typer.Typer(
    name="aegis",
    help="Physical AI harness (POC): safety-gated policy evaluation in MuJoCo.",
    no_args_is_help=True,
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_INTERNAL = 3


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def _model_to_cfg_dict(cfg: PhysicalAIYaml) -> dict:
    return cfg.model_dump(mode="json")


@app.command()
def eval(
    model: str = typer.Option("random", "--model", help="model name (configs/models/<name>.yaml)"),
    robot: str = typer.Option("franka", "--robot", help="robot name (configs/robots/<name>.yaml)"),
    sim: str = typer.Option("mujoco", "--sim", help="simulator (POC: mujoco only)"),
    tasks: str = typer.Option("pick-place", "--tasks", help="comma-separated tasks (POC: pick-place only)"),
    episodes: int | None = typer.Option(None, "--episodes", min=1, max=1000),
    seed: int | None = typer.Option(None, "--seed"),
    max_steps: int | None = typer.Option(None, "--max-steps", min=1, max=100_000),
    inference_mode: str | None = typer.Option(
        None, "--inference-mode", help="cpu | cuda (overrides eval.inference_mode)"
    ),
    inference_budget_ms: float | None = typer.Option(
        None, "--inference-budget-ms", min=1.0, help="overrides eval.inference_budget_ms"
    ),
    config: Path = typer.Option(
        Path("physical-ai.yaml"), "--config", help="root config file"
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    run_id: str | None = typer.Option(None, "--run-id"),
    verbose: bool = typer.Option(False, "--verbose", help="debug logging to stderr"),
) -> None:
    """Run a safety-gated evaluation for N episodes and emit a report."""
    try:
        cfg = load_run_config(config, model_name=model, robot_name=robot, task_name=tasks)
        if sim != "mujoco":
            raise ConfigError(f"--sim {sim!r} not supported in POC; use 'mujoco'")
        if tasks != "pick-place":
            raise ConfigError(f"--tasks {tasks!r} not supported in POC; use 'pick-place'")
        overrides: dict = {}
        if episodes is not None:
            overrides["episodes"] = episodes
        if seed is not None:
            overrides["seed"] = seed
        if max_steps is not None:
            overrides["max_steps_per_episode"] = max_steps
        if inference_mode is not None:
            if inference_mode not in ("cpu", "cuda"):
                raise ConfigError(
                    f"--inference-mode {inference_mode!r} not supported; use 'cpu' or 'cuda'"
                )
            overrides["inference_mode"] = inference_mode
        if inference_budget_ms is not None:
            overrides["inference_budget_ms"] = inference_budget_ms
        if overrides:
            cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update=overrides)})
        if output_dir is not None:
            cfg = cfg.model_copy(update={"output": cfg.output.model_copy(update={"dir": str(output_dir)})})
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from exc

    run_id = run_id or cfg.output.run_id or _make_run_id()
    output_dir = Path(cfg.output.dir)
    logger = RunLogger(output_dir, run_id)
    logger.write_run(_model_to_cfg_dict(cfg))

    env = None
    try:
        env = MujocoPickPlaceEnv(
            scene_mjcf=cfg.environment.scene_mjcf,
            task=cfg.task,
            time_step=cfg.eval.time_step,
            render_cameras=cfg.environment.render_cameras,
        )
        env.reset(cfg.eval.seed)

        if cfg.model.kind == "random":
            assert isinstance(cfg.model.policy, RandomPolicySpec)
            policy = RandomPolicy(cfg.model.policy)
        elif cfg.model.kind == "scripted":
            assert isinstance(cfg.model.policy, ScriptedPolicySpec)
            policy = ScriptedPolicy(cfg.model.policy, env)
        elif cfg.model.kind == "smolvla":
            assert isinstance(cfg.model.policy, SmolVLAPolicySpec)
            if cfg.eval.inference_mode == "cuda":
                import torch

                if not torch.cuda.is_available():
                    raise ConfigError(
                        "eval.inference_mode=cuda but torch.cuda is not available"
                    )
            policy = SmolVLAPolicy(
                endpoint=cfg.model.endpoint,
                spec=cfg.model.policy,
                env=env,
                device=cfg.eval.inference_mode,
            )
        else:  # pragma: no cover - guarded by pydantic Literal
            raise ConfigError(f"unknown model kind {cfg.model.kind!r}")

        fallback = PidToHomeFallback(
            home_qpos=env.home_qpos,
            max_velocity=min(0.3, cfg.robot.safety.max_velocity),
        )
        gateway = SafetyGateway(
            limits=cfg.robot.safety,
            joint_names=env.joint_names,
            fallback=fallback,
        )
        runner = EvalRunner(cfg, env, policy, gateway, fallback, logger)
        results = runner.run()
    except Exception as exc:  # internal error: preserve partial logs
        if verbose:
            traceback.print_exc(file=sys.stderr)
        typer.echo(f"internal error: {exc}", err=True)
        logger.close()
        raise typer.Exit(EXIT_INTERNAL) from exc
    finally:
        if env is not None:
            env.close()

    report = build_report(cfg, results, run_id)
    logger.write_report(report)
    logger.close()
    typer.echo(print_summary(report))
    typer.echo(f"report      : {logger.run_dir / 'report.json'}")
    raise typer.Exit(EXIT_OK)


@app.command()
def validate(
    config: Path = typer.Option(Path("physical-ai.yaml"), "--config"),
    model: str = typer.Option("random", "--model"),
    robot: str = typer.Option("franka", "--robot"),
    tasks: str = typer.Option("pick-place", "--tasks"),
) -> None:
    """Validate the config (and asset paths) without running anything."""
    try:
        load_run_config(config, model_name=model, robot_name=robot, task_name=tasks)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from exc
    typer.echo("config OK")
    raise typer.Exit(EXIT_OK)


if __name__ == "__main__":
    app()