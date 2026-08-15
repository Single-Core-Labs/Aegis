from __future__ import annotations

import time

import numpy as np

from aegis.config.models import PhysicalAIYaml
from aegis.envs.base import Env
from aegis.policies.base import Policy
from aegis.policies.smolvla import PolicyModelError
from aegis.safety.fallback import PidToHomeFallback
from aegis.safety.gateway import SafetyGateway
from aegis.telemetry.logger import RunLogger


class EpisodeResult:
    __slots__ = (
        "episode",
        "seed",
        "outcome",
        "fail_reason",
        "success",
        "steps",
        "duration_s",
        "violations",
        "recoveries",
        "budget_violations",
        "model_errors",
        "inference_latencies_s",
    )

    def __init__(
        self,
        episode: int,
        seed: int,
        outcome: str,
        fail_reason: str | None,
        success: bool,
        steps: int,
        duration_s: float,
        violations: int,
        recoveries: int,
        budget_violations: int,
        model_errors: int,
        inference_latencies_s: list[float],
    ) -> None:
        self.episode = episode
        self.seed = seed
        self.outcome = outcome
        self.fail_reason = fail_reason
        self.success = success
        self.steps = steps
        self.duration_s = duration_s
        self.violations = violations
        self.recoveries = recoveries
        self.budget_violations = budget_violations
        self.model_errors = model_errors
        self.inference_latencies_s = inference_latencies_s


class EvalRunner:
    """Runs episodes: policy -> SafetyGateway -> env, logging everything."""

    def __init__(
        self,
        cfg: PhysicalAIYaml,
        env: Env,
        policy: Policy,
        gateway: SafetyGateway,
        fallback: PidToHomeFallback,
        logger: RunLogger,
    ) -> None:
        self._cfg = cfg
        self._env = env
        self._policy = policy
        self._gateway = gateway
        self._fallback = fallback
        self._logger = logger

    def run(self) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        cfg = self._cfg
        for ep in range(cfg.eval.episodes):
            seed = cfg.eval.seed + ep
            results.append(self._run_episode(ep, seed))
        return results

    def _run_episode(self, ep: int, seed: int) -> EpisodeResult:
        env, gateway = self._env, self._gateway
        cfg = self._cfg.eval
        self._logger.episode_start(ep, seed)
        env.reset(seed)
        gateway.reset_episode()
        self._policy.reset(seed)

        latencies: list[float] = []
        steps = 0
        episode_start = time.perf_counter()
        outcome = "fail"
        fail_reason: str | None = None

        for step in range(1, cfg.max_steps_per_episode + 1):
            obs = env.observe()
            state = env.state_snapshot()

            t0 = time.perf_counter_ns()
            try:
                raw = self._policy.act(obs)
            except PolicyModelError as exc:
                # Model crashed: count like a safety violation, engage fallback.
                inference_s = (time.perf_counter_ns() - t0) / 1e9
                latencies.append(inference_s)
                gated = gateway.model_error(state, str(exc))
                gateway_s = 0.0
            else:
                inference_s = (time.perf_counter_ns() - t0) / 1e9
                latencies.append(inference_s)
                t_gate = time.perf_counter_ns()
                budget_ms = float(cfg.inference_budget_ms)
                if inference_s * 1e3 > budget_ms:
                    # Over budget: never silently drop a slow step. Count it like
                    # a safety violation and run the fallback for this step.
                    gated = gateway.budget_violation(
                        state, measured_ms=inference_s * 1e3, limit_ms=budget_ms
                    )
                else:
                    gated = gateway.filter(raw, state)
                gateway_s = (time.perf_counter_ns() - t_gate) / 1e9

            t0 = time.perf_counter_ns()
            obs, terminated, truncated, info = env.step(gated.command)
            env_s = (time.perf_counter_ns() - t0) / 1e9
            steps = step

            self._logger.step(
                ep,
                step,
                source=gated.source,
                violated=gated.violated,
                violation=gated.violation.as_dict() if gated.violation else None,
                inference_s=inference_s,
                gateway_s=gateway_s,
                env_s=env_s,
                info=info,
            )
            self._logger.trajectory(
                ep,
                step,
                {
                    "qpos": state["qpos"],
                    "action": gated.command,
                    "source": gated.source,
                },
            )

            if terminated:
                outcome = "success"
                fail_reason = None
                break
            if time.perf_counter() - episode_start > cfg.episode_timeout_sec:
                outcome = "fail"
                fail_reason = "timeout"
                break
            if step >= cfg.max_steps_per_episode:
                outcome = "fail"
                fail_reason = "truncated"
                break

        duration_s = time.perf_counter() - episode_start
        budget_violations = sum(
            1 for v in gateway.violations if v.type == "inference_budget"
        )
        model_errors = sum(1 for v in gateway.violations if v.type == "model_error")
        res = EpisodeResult(
            episode=ep,
            seed=seed,
            outcome=outcome,
            fail_reason=fail_reason,
            success=outcome == "success",
            steps=steps,
            duration_s=duration_s,
            violations=gateway.violation_count,
            recoveries=gateway.recovery_count,
            budget_violations=budget_violations,
            model_errors=model_errors,
            inference_latencies_s=latencies,
        )
        self._logger.episode_end(
            ep,
            outcome=res.outcome,
            fail_reason=res.fail_reason,
            steps=res.steps,
            duration_s=res.duration_s,
            violations=res.violations,
            recoveries=res.recoveries,
            success=res.success,
        )
        return res