from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    type: str  # "nan" | "velocity" | "force" | "gripper" | "inference_budget" | "model_error"
    joint: str
    value: float
    limit: float
    detail: str = ""

    def as_dict(self) -> dict:
        d = {
            "type": self.type,
            "joint": self.joint,
            "value": round(float(self.value), 6),
            "limit": round(float(self.limit), 6),
        }
        if self.detail:
            d["detail"] = self.detail
        return d


def check_finite(action, reject_nan: bool) -> Violation | None:
    import numpy as np

    if not np.isfinite(action).all():
        bad = int(np.flatnonzero(~np.isfinite(action))[0])
        return Violation("nan", f"action[{bad}]", float(action[bad]), 0.0)
    return None


def check_velocities(qvel, joint_names, max_velocity) -> Violation | None:
    import numpy as np

    if max_velocity <= 0:
        return None
    for name, v in zip(joint_names, np.asarray(qvel).ravel()):
        if abs(float(v)) > max_velocity:
            return Violation("velocity", name, float(v), max_velocity)
    return None


def check_forces(torque, joint_names, max_force) -> Violation | None:
    import numpy as np

    if max_force <= 0:
        return None
    for name, f in zip(joint_names, np.asarray(torque).ravel()):
        if abs(float(f)) > max_force:
            return Violation("force", name, float(f), max_force)
    return None