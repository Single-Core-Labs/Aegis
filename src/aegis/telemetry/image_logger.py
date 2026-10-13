"""Storage-bounded image logger (Beyond Phase 3 slice, MuJoCo-first).

Captures camera frames at flagged events only (grasp attempts, violations,
recoveries, success, chunk boundaries) — never every step — with a hard
storage budget. PNG via PIL when available, else ``.npy`` fallback so the
module has zero hard deps.

Wiring to ``EvalRunner`` is opt-in (flag ``--image-log`` / ``AEGIS_IMAGE_LOG=1``);
until then this module + ``aegis/data/export.py`` operate offline on
``outputs/<run_id>/`` NDJSON. That keeps the DEFER decision in
``docs/data-pipeline-decision.md`` honest: no videos until frames exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

LOG_EVENTS = frozenset(
    {"grasp_attempt", "violation", "recovery", "success", "chunk_boundary"}
)


class ImageLogger:
    def __init__(self, output_dir: Path, max_storage_mb: float = 100.0) -> None:
        self.output_dir = Path(output_dir) / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_storage_mb = float(max_storage_mb)
        self.current_storage_mb = 0.0
        self._refresh_usage()

    def _refresh_usage(self) -> None:
        total = 0
        for p in self.output_dir.glob("*"):
            if p.is_file():
                total += p.stat().st_size
        self.current_storage_mb = total / (1024 * 1024)

    @staticmethod
    def should_log(event: str) -> bool:
        return event in LOG_EVENTS

    def log_image(
        self,
        image: Any,
        episode_id: int,
        step: int,
        camera_id: Any,
        event: str,
    ) -> Path | None:
        """Save one frame. Returns path or None when over budget."""
        if not self.should_log(event):
            return None
        if self.current_storage_mb >= self.max_storage_mb:
            return None
        filename = f"ep{episode_id}_step{step}_cam{camera_id}_{event}"
        try:
            import numpy as np  # type: ignore

            arr = np.asarray(image)
            try:
                from PIL import Image  # type: ignore

                path = self.output_dir / f"{filename}.png"
                Image.fromarray(arr).save(path)
            except Exception:
                path = self.output_dir / f"{filename}.npy"
                np.save(path, arr)
        except Exception:
            return None
        self.current_storage_mb += path.stat().st_size / (1024 * 1024)
        return path
