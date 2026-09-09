from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from aegis.config.models import TaskSpec
from aegis.envs.base import Env
from aegis.envs.mujoco_pick_place import MujocoPickPlaceEnv

try:
    import isaacsim  # noqa: F401

    _HAS_ISAACSIM = True
except Exception:
    _HAS_ISAACSIM = False


class IsaacPickPlaceEnv(Env):
    """Isaac Lab pick-place env — interface scaffold (Phase 3, Step 4).

    API is identical to MujocoPickPlaceEnv so `aegis eval --sim isaaclab`
    produces the same report schema (success, violations, latency).

    Runtime behaviour:
      - If Isaac Sim 6.0.1 + Isaac Lab are available (16GB VRAM, `isaacsim`
        import succeeds), this will delegate to a real Isaac Lab scene.
        TODO: replace the _delegate with a real USD scene + PhysX.
      - Otherwise it **falls back to MujocoPickPlaceEnv** with a warning,
        so the harness stays usable on 6GB laptops and in CI while the
        Isaac Lab install is pending (see CONTEXT.md:16, WSL2_SETUP_STATUS.md).

    This lets Steps 5 (per-joint limits) and 6 (SmolVLA gap measurement)
    be developed and tested now, with no mock numbers — the fallback is
    explicitly recorded in `info["sim"]` and in the report warnings.
    """

    name = "isaac-pick-place"

    ARM_JOINTS = MujocoPickPlaceEnv.ARM_JOINTS

    def __init__(
        self,
        scene_mjcf: str | None = None,
        task: TaskSpec | None = None,
        time_step: float = 0.02,
        render_cameras: list[str] | None = None,
        usd_scene: str | None = None,
        physics: str = "physx",
        headless: bool = False,
    ) -> None:
        self._task = task or TaskSpec()
        self._time_step = time_step
        self._render_cameras = list(render_cameras or [])
        # VRAM-safe defaults per docs/perf-tuning.md
        self._usd_scene = usd_scene or "assets/usd/pick_place_vram_safe.usda"
        self._physics = physics
        self._headless = headless
        self._using_fallback = not _HAS_ISAACSIM

        if self._using_fallback:
            # Reuse MuJoCo scene as fallback; scene_mjcf is required for that path.
            # If caller passed usd_scene only, try the default MuJoCo scene.
            fallback_scene = scene_mjcf or str(Path(__file__).parents[3] / "assets/scenes/pick_place.xml")
            if not Path(fallback_scene).is_file():
                # last resort: use the task's scene_mjcf if loader passed it
                raise FileNotFoundError(
                    f"Isaac Sim not available and fallback MuJoCo scene not found: {fallback_scene}. "
                    "Install Isaac Sim 6.0.1 (16GB VRAM) or provide a valid scene_mjcf."
                )
            warnings.warn(
                "isaacsim not found — IsaacPickPlaceEnv falling back to MujocoPickPlaceEnv. "
                "This is expected on 6GB laptops until NVIDIA provides VRAM-safe scene defaults. "
                f"Fallback scene: {fallback_scene}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._delegate = MujocoPickPlaceEnv(
                scene_mjcf=fallback_scene,
                task=self._task,
                time_step=time_step,
                render_cameras=self._render_cameras,
            )
            self._is_fallback = True
        else:
            # Real Isaac Lab path — scaffold only, not yet implemented.
            # Keeping the delegate pattern so the rest of the harness works
            # while the USD is being authored (see i4h-workflow-create skill).
            raise NotImplementedError(
                "Isaac Sim is installed but IsaacPickPlaceEnv USD scene is not yet authored. "
                "Author it with: workflows/agentic/arena/run.sh --create-env pick_place --from scissor_pick_and_place "
                "then edit the USD in Isaac Sim and set usd_scene=... in configs. "
                "Until then, uninstall isaacsim or pass scene_mjcf to use the MuJoCo fallback."
            )

    # ------------------------------------------------------------------ Env API
    def reset(self, seed: int) -> dict[str, np.ndarray]:
        obs = self._delegate.reset(seed)
        # tag obs so the runner can distinguish fallback vs real Isaac
        if self._using_fallback:
            obs["_sim_fallback"] = np.array([1])
        return obs

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], bool, bool, dict[str, Any]]:
        obs, terminated, truncated, info = self._delegate.step(action)
        # propagate sim tag into info for report honesty
        info = dict(info)
        info["sim"] = "isaaclab-fallback-mujoco" if self._using_fallback else "isaaclab"
        info["isaac_fallback"] = bool(self._using_fallback)
        return obs, terminated, truncated, info

    def observe(self) -> dict[str, np.ndarray]:
        return self._delegate.observe()

    def render_images(self, cameras: list[str] | None = None) -> dict[str, np.ndarray]:
        # Delegate to Mujoco's renderer in fallback mode; real Isaac would use
        # Isaac Sim's camera sensors.
        if hasattr(self._delegate, "render_images"):
            return self._delegate.render_images(cameras)  # type: ignore
        return {}

    def state_snapshot(self) -> dict[str, np.ndarray]:
        return self._delegate.state_snapshot()

    def close(self) -> None:
        self._delegate.close()

    @property
    def action_dim(self) -> int:
        return self._delegate.action_dim

    @property
    def dt(self) -> float:
        return self._delegate.dt

    @property
    def joint_names(self) -> list[str]:
        return self._delegate.joint_names

    @property
    def home_qpos(self) -> np.ndarray:
        return self._delegate.home_qpos

    @property
    def gripper_open_ctrl(self) -> float:
        return self._delegate.gripper_open_ctrl

    @property
    def gripper_closed_ctrl(self) -> float:
        return self._delegate.gripper_closed_ctrl

    @property
    def is_fallback(self) -> bool:
        return self._using_fallback
