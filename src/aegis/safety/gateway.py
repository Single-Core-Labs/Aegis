from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aegis.config.models import SafetyLimits
from aegis.safety.checks import (
    Violation,
    check_finite,
    check_forces,
    check_velocities,
)
from aegis.safety.fallback import PidToHomeFallback


@dataclass
class GatedAction:
    command: np.ndarray
    source: str  # "policy" | "fallback"
    violated: bool = False
    violation: Violation | None = None
    clamped: bool = False


@dataclass
class SafetyGateway:
    """Sits between policy output and env.step. No bypass path exists.

    Checks (in order):
      1. NaN/Inf in the action -> fatal, engage fallback
      2. commanded gripper effort |g| > max_effort_action -> clamp or reject
      3. measured joint velocity > max_velocity -> engage fallback
      4. measured joint force > max_force -> engage fallback

    On a reject/fatal violation the gateway switches the action source to the
    fallback for `recovery_steps` (then resumes the policy unless mode="hold").
    During an active fallback window the fallback action is used every step and
    itself re-checked. Violations are counted per step; recoveries count each
    engagement.
    """

    limits: SafetyLimits
    joint_names: list[str]
    fallback: PidToHomeFallback
    gripper_idx: int = 7

    violations: list[Violation] = field(default_factory=list)
    clamp_count: int = 0
    recovery_count: int = 0
    fallback_active: bool = False
    _remaining: int = 0

    def reset_episode(self) -> None:
        self.violations.clear()
        self.clamp_count = 0
        self.recovery_count = 0
        self.fallback_active = False
        self._remaining = 0

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def filter(self, action: np.ndarray, state: dict[str, np.ndarray]) -> GatedAction:
        if self.fallback_active:
            return self._step_fallback(state)

        a = np.asarray(action, dtype=float)
        cmd = a.copy()

        # 1. NaN / Inf -> fatal
        if self.limits.reject_nan_actions:
            v = check_finite(cmd, True)
            if v is not None:
                self._engage(v)
                return self._step_fallback(state, v)

        # 2. commanded gripper effort bound
        g = float(cmd[self.gripper_idx])
        if abs(g) > self.limits.max_effort_action:
            if self.limits.action_clamp == "clamp":
                cmd[self.gripper_idx] = np.clip(g, 0.0, 1.0)
                self.clamp_count += 1
                v = Violation("gripper", "gripper", g, self.limits.max_effort_action)
                self.violations.append(v)
                return GatedAction(cmd, "policy", violated=True, violation=v, clamped=True)
            v = Violation("gripper", "gripper", g, self.limits.max_effort_action)
            self._engage(v)
            return self._step_fallback(state, v)

        # 3. measured velocity
        v = check_velocities(
            state.get("arm_qvel", []), self.joint_names, self.limits.max_velocity
        )
        if v is not None:
            self._engage(v)
            return self._step_fallback(state, v)

        # 4. measured force
        v = check_forces(
            state.get("arm_torque", []), self.joint_names, self.limits.max_force
        )
        if v is not None:
            self._engage(v)
            return self._step_fallback(state, v)

        return GatedAction(cmd, "policy")

    def budget_violation(
        self, state: dict[str, np.ndarray], measured_ms: float, limit_ms: float
    ) -> GatedAction:
        """Inference over the budget counts exactly like a safety violation:
        recorded, fallback engaged for recovery_steps, then policy resumes."""
        v = Violation("inference_budget", "policy", measured_ms, limit_ms)
        self._engage(v)
        return self._step_fallback(state, v)

    def model_error(
        self, state: dict[str, np.ndarray], detail: str
    ) -> GatedAction:
        """The model crashed (or returned a non-finite action): treat it like
        a fatal violation and run the fallback for the recovery window."""
        v = Violation("model_error", "policy", 0.0, 0.0, detail[:120])
        self._engage(v)
        return self._step_fallback(state, v)

    # ------------------------------------------------------------------ helpers

    def _engage(self, v: Violation) -> None:
        self.violations.append(v)
        self.fallback_active = True
        self._remaining = self.limits.recovery_steps
        self.recovery_count += 1

    def _step_fallback(
        self, state: dict[str, np.ndarray], engage_violation: Violation | None = None
    ) -> GatedAction:
        cmd = self.fallback.act(state)
        cmd = np.asarray(cmd, dtype=float)
        # The fallback action itself must pass the checks.
        v = check_finite(cmd, True)
        if v is None:
            v = check_velocities(
                state.get("arm_qvel", []), self.joint_names, self.limits.max_velocity
            )
        if v is None:
            v = check_forces(
                state.get("arm_torque", []), self.joint_names, self.limits.max_force
            )
        if v is not None and engage_violation is None:
            # Fallback also unsafe (should not happen): stay engaged, count it.
            self.violations.append(v)
            return GatedAction(cmd, "fallback", violated=True, violation=v)

        self._remaining -= 1
        if self._remaining <= 0 and self.limits.recovery_mode == "resume":
            self.fallback_active = False
        return GatedAction(
            cmd,
            "fallback",
            violated=engage_violation is not None,
            violation=engage_violation,
        )