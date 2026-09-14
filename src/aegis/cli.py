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
    GrootPolicySpec,
)
from aegis.envs.mujoco_pick_place import MujocoPickPlaceEnv
from aegis.eval.batch import build_batch_report, combo_dir_name, combo_seed
from aegis.eval.report import build_report, print_summary, write_report_json
from aegis.eval.runner import EvalRunner
from aegis.policies.random import RandomPolicy
from aegis.policies.scripted import ScriptedPolicy
from aegis.policies.smolvla import SmolVLAPolicy
from aegis.policies.groot import Gr00tPolicy
from aegis.ros2.bridge import RosBridge, benchmark_latency
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


def _apply_dr_override(cfg: PhysicalAIYaml, dr: bool | None) -> PhysicalAIYaml:
    """Override task.domain_randomization when --dr/--no-dr is passed."""
    if dr is None:
        return cfg
    return cfg.model_copy(update={"task": cfg.task.model_copy(update={"domain_randomization": bool(dr)})})


def _build_env(cfg: PhysicalAIYaml):
    """Select the sim backend. One isolated env per call (batch-safe)."""
    if cfg.environment.sim in ("isaaclab", "isaac"):
        from aegis.envs.isaac_pick_place import IsaacPickPlaceEnv

        return IsaacPickPlaceEnv(
            scene_mjcf=cfg.environment.scene_mjcf,
            task=cfg.task,
            time_step=cfg.eval.time_step,
            render_cameras=cfg.environment.render_cameras,
            usd_scene=cfg.environment.usd_scene or "assets/usd/pick_place_vram_safe.usda",
            headless=cfg.environment.headless,
        )
    return MujocoPickPlaceEnv(
        scene_mjcf=cfg.environment.scene_mjcf,
        task=cfg.task,
        time_step=cfg.eval.time_step,
        render_cameras=cfg.environment.render_cameras,
    )


def _build_policy(cfg: PhysicalAIYaml, env):
    if cfg.model.kind == "random":
        assert isinstance(cfg.model.policy, RandomPolicySpec)
        return RandomPolicy(cfg.model.policy)
    if cfg.model.kind == "scripted":
        assert isinstance(cfg.model.policy, ScriptedPolicySpec)
        return ScriptedPolicy(cfg.model.policy, env)
    if cfg.model.kind == "smolvla":
        assert isinstance(cfg.model.policy, SmolVLAPolicySpec)
        if cfg.eval.inference_mode == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ConfigError(
                    "eval.inference_mode=cuda but torch.cuda is not available"
                )
        return SmolVLAPolicy(
            endpoint=cfg.model.endpoint,
            spec=cfg.model.policy,
            env=env,
            device=cfg.eval.inference_mode,
        )
    if cfg.model.kind == "groot":
        assert isinstance(cfg.model.policy, GrootPolicySpec)
        if cfg.eval.inference_mode == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ConfigError(
                    "eval.inference_mode=cuda but torch.cuda is not available"
                )
        # Phase A: Gr00tPolicy.__init__ raises an honest ConfigError
        # (checkpoint loading lands in Phase B, PolicyServer in Phase D).
        return Gr00tPolicy(
            endpoint=cfg.model.endpoint,
            spec=cfg.model.policy,
            env=env,
            device=cfg.eval.inference_mode,
        )
    raise ConfigError(f"unknown model kind {cfg.model.kind!r}")  # pragma: no cover


def _run_single_cfg(cfg: PhysicalAIYaml, run_id: str, output_dir: Path) -> dict:
    """Run one eval combo: isolated env + policy + gateway, returns the report.

    Raises ConfigError/whatever the backend raises; caller handles exits.
    """
    logger = RunLogger(output_dir, run_id)
    logger.write_run(_model_to_cfg_dict(cfg))
    env = _build_env(cfg)
    try:
        env.reset(cfg.eval.seed)
        policy = _build_policy(cfg, env)
        _limit = cfg.robot.safety.max_velocity
        if isinstance(_limit, list):
            _limit = min(_limit)
        fallback = PidToHomeFallback(
            home_qpos=env.home_qpos,
            max_velocity=min(0.3, _limit),
        )
        gateway = SafetyGateway(
            limits=cfg.robot.safety,
            joint_names=env.joint_names,
            fallback=fallback,
        )
        runner = EvalRunner(cfg, env, policy, gateway, fallback, logger)
        results = runner.run()
    except Exception:
        logger.close()
        raise
    finally:
        env.close()
    report = build_report(cfg, results, run_id)
    logger.write_report(report)
    logger.close()
    return report


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
    headless: bool = typer.Option(False, "--headless", help="Isaac headless: no window, RTX Real-Time bounces=1, 6GB VRAM-safe"),
    quantize: str | None = typer.Option(None, "--quantize", help="SmolVLA quantize: none | int8 (bitsandbytes, ~1GB->0.6GB)"),
    dr: bool | None = typer.Option(
        None, "--dr/--no-dr", help="override task.domain_randomization (seeded MuJoCo DR)"
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
        if sim not in ("mujoco", "isaaclab", "isaac"):
            raise ConfigError(f"--sim {sim!r} not supported; use 'mujoco' or 'isaaclab'")
        if sim in ("isaaclab", "isaac") and cfg.environment.sim != sim:
            cfg = cfg.model_copy(update={"environment": cfg.environment.model_copy(update={"sim": sim})})
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
        # VRAM-safe flags: --headless and --quantize override config
        if headless:
            cfg = cfg.model_copy(update={"environment": cfg.environment.model_copy(update={"headless": True})})
        if quantize is not None:
            if quantize not in ("none", "int8"):
                raise ConfigError(f"--quantize {quantize!r} not supported; use 'none' or 'int8'")
            if cfg.model.kind == "smolvla":
                cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"policy": cfg.model.policy.model_copy(update={"quantize": quantize})})})
        if output_dir is not None:
            cfg = cfg.model_copy(update={"output": cfg.output.model_copy(update={"dir": str(output_dir)})})
        cfg = _apply_dr_override(cfg, dr)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from exc

    run_id = run_id or cfg.output.run_id or _make_run_id()
    output_dir = Path(cfg.output.dir)
    try:
        report = _run_single_cfg(cfg, run_id, output_dir)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from exc
    except Exception as exc:  # internal error: preserve partial logs
        if verbose:
            traceback.print_exc(file=sys.stderr)
        typer.echo(f"internal error: {exc}", err=True)
        raise typer.Exit(EXIT_INTERNAL) from exc

    typer.echo(print_summary(report))
    typer.echo(f"report      : {output_dir / run_id / 'report.json'}")
    raise typer.Exit(EXIT_OK)


@app.command(name="eval-batch")
def eval_batch(
    models: str = typer.Option("random", "--models", help="comma-separated model names (configs/models/<name>.yaml)"),
    robots: str = typer.Option("franka", "--robots", help="comma-separated robot names (configs/robots/<name>.yaml)"),
    tasks: str = typer.Option("pick-place", "--tasks", help="comma-separated tasks (POC: pick-place only)"),
    sim: str = typer.Option("mujoco", "--sim", help="simulator (mujoco | isaaclab)"),
    episodes: int | None = typer.Option(None, "--episodes", min=1, max=1000),
    seed: int | None = typer.Option(None, "--seed"),
    max_steps: int | None = typer.Option(None, "--max-steps", min=1, max=100_000),
    inference_mode: str | None = typer.Option(
        None, "--inference-mode", help="cpu | cuda (overrides eval.inference_mode)"
    ),
    inference_budget_ms: float | None = typer.Option(
        None, "--inference-budget-ms", min=1.0, help="overrides eval.inference_budget_ms"
    ),
    dr: bool | None = typer.Option(
        None, "--dr/--no-dr", help="override task.domain_randomization (seeded MuJoCo DR)"
    ),
    config: Path = typer.Option(
        Path("physical-ai.yaml"), "--config", help="root config file"
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    run_id: str | None = typer.Option(None, "--run-id", help="batch id (default: batch-YYYYMMDDTHHMMSSZ)"),
    verbose: bool = typer.Option(False, "--verbose", help="debug logging to stderr"),
) -> None:
    """Run a sequential batch over robots x models x tasks with one isolated env per combo.

    Combo i uses seed = base_seed + i*1000 (deterministic regardless of order).
    Writes outputs/<batch_id>/runs/<robot>-<model>-<task>-seed<N>/... plus an
    aggregated outputs/<batch_id>/report.json. Parallel episodes are NOT
    implemented (see docs/batching.md).
    """
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    robot_names = [r.strip() for r in robots.split(",") if r.strip()]
    task_names = [t.strip() for t in tasks.split(",") if t.strip()]
    if not model_names or not robot_names or not task_names:
        typer.echo("config error: --models/--robots/--tasks must each list >=1 name", err=True)
        raise typer.Exit(EXIT_CONFIG)
    if sim not in ("mujoco", "isaaclab", "isaac"):
        typer.echo(f"config error: --sim {sim!r} not supported; use 'mujoco' or 'isaaclab'", err=True)
        raise typer.Exit(EXIT_CONFIG)
    for t in task_names:
        if t != "pick-place":
            typer.echo(f"config error: --tasks {t!r} not supported in POC; use 'pick-place'", err=True)
            raise typer.Exit(EXIT_CONFIG)
    if inference_mode is not None and inference_mode not in ("cpu", "cuda"):
        typer.echo(f"config error: --inference-mode {inference_mode!r} not supported", err=True)
        raise typer.Exit(EXIT_CONFIG)

    batch_id = run_id or _make_run_id().replace("run-", "batch-", 1)
    base_seed = seed if seed is not None else 42
    try:
        probe = load_run_config(config)
        batch_root = Path(output_dir) if output_dir is not None else Path(probe.output.dir)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from exc

    combos: list[dict] = []
    combo_reports: list[dict] = []
    idx = 0
    try:
        for robot_name in robot_names:
            for model_name in model_names:
                for task_name in task_names:
                    cfg = load_run_config(
                        config, model_name=model_name, robot_name=robot_name, task_name=task_name
                    )
                    if sim in ("isaaclab", "isaac") and cfg.environment.sim != sim:
                        cfg = cfg.model_copy(
                            update={"environment": cfg.environment.model_copy(update={"sim": sim})}
                        )
                    overrides: dict = {}
                    if episodes is not None:
                        overrides["episodes"] = episodes
                    overrides["seed"] = combo_seed(base_seed, idx)
                    if max_steps is not None:
                        overrides["max_steps_per_episode"] = max_steps
                    if inference_mode is not None:
                        overrides["inference_mode"] = inference_mode
                    if inference_budget_ms is not None:
                        overrides["inference_budget_ms"] = inference_budget_ms
                    if overrides:
                        cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update=overrides)})
                    cfg = _apply_dr_override(cfg, dr)
                    combo_seed_value = combo_seed(base_seed, idx)
                    sub = combo_dir_name(robot_name, model_name, task_name, combo_seed_value)
                    if verbose:
                        typer.echo(
                            f"combo {idx}: robot={robot_name} model={model_name} "
                            f"task={task_name} seed={combo_seed_value}",
                            err=True,
                        )
                    report = _run_single_cfg(cfg, sub, batch_root / batch_id / "runs")
                    combos.append(
                        {"robot": robot_name, "model": model_name, "task": task_name,
                         "seed": combo_seed_value, "run_dir": f"runs/{sub}"}
                    )
                    combo_reports.append(report)
                    idx += 1
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from exc
    except Exception as exc:  # internal error: per-combo partial logs preserved
        if verbose:
            traceback.print_exc(file=sys.stderr)
        typer.echo(f"internal error: {exc}", err=True)
        raise typer.Exit(EXIT_INTERNAL) from exc

    batch_report = build_batch_report(batch_id, combo_reports, combos)
    batch_path = batch_root / batch_id / "report.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    write_report_json(batch_path, batch_report)
    s = batch_report["summary"]
    typer.echo(
        f"batch         : {batch_id}  combos {s['combos']}  "
        f"(success {s['success']} / fail {s['fail']})  "
        f"violations {s['safety_violations']}  recoveries {s['recovery_events']}"
    )
    for combo, rep in zip(combos, combo_reports):
        counts = rep['task_counts']
        typer.echo(f"  - {combo['robot']}/{combo['model']}: {counts}")
    typer.echo(f"report        : {batch_path}")
    raise typer.Exit(EXIT_OK)


@app.command()
def rosbench(
    n: int = typer.Option(100, "--n", help="number of publish samples"),
    mock: bool = typer.Option(True, "--mock/--real", help="use mock in-memory transport (real needs rclpy)"),
) -> None:
    """Benchmark ROS2 bridge publish latency (mock vs real rclpy)."""
    bridge = RosBridge(mock=mock)
    stats = benchmark_latency(bridge, n=n)
    typer.echo(f"ros bridge: {'mock' if bridge._mock else 'real rclpy'}")
    typer.echo(f"  count : {stats.count}")
    typer.echo(f"  p50   : {stats.p50_ms:.3f} ms")
    typer.echo(f"  p95   : {stats.p95_ms:.3f} ms")
    typer.echo(f"  mean  : {stats.mean_ms:.3f} ms  min {stats.min_ms:.3f} max {stats.max_ms:.3f}")
    typer.echo("note: inside-WSL2 vs Windows<->WSL2 bridge latency must be measured separately in WSL2")
    bridge.close()
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