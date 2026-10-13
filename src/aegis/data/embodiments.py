"""Minimal embodiment registry (2 entries; limits-only second robot).

Per ``docs/data-pipeline-decision.md`` revival step 1: constants graduate into
a registry only after validation. Franka values are validated (hardware specs
+ sim benchmarks); UR5e values are conservative starting points marked
``validated: false`` — usable for harness generality tests, not hardware.

Kinematics note: both entries point at the same MuJoCo MJCF today; the UR5e
entry exercises different limits/conventions through the same gateway.
A true second MJCF + adapter is the follow-on.
"""

from __future__ import annotations

EMBODIMENTS: dict[str, dict] = {
    "franka": {
        "tag": "LIBERO_PANDA",
        "rpy_order": "Rz*Ry*Rx",
        "gripper_scale": "finger-mean/0.04 -> [0,1]",
        "robot_config": "configs/robots/franka.yaml",
        "validated": True,
    },
    "ur5e": {
        "tag": "UR5E_PANDA_COMPAT",
        "rpy_order": "Rz*Ry*Rx",
        "gripper_scale": "finger-mean/0.04 -> [0,1]",
        "robot_config": "configs/robots/ur5e.yaml",
        "validated": False,
    },
}


def get_embodiment(name: str) -> dict:
    try:
        return dict(EMBODIMENTS[name])
    except KeyError:
        raise KeyError(f"unknown embodiment {name!r}; known: {sorted(EMBODIMENTS)}") from None


def list_embodiments() -> list[str]:
    return sorted(EMBODIMENTS)
