"""Offline dataset export: NDJSON run dir -> datasets/ layout.

Reads ``steps.jsonl`` + ``trajectory.jsonl`` (+ ``run.json`` for metadata)
from a finished eval run and writes one file per episode plus
``metadata.json``. Prefers parquet (pandas+pyarrow); falls back to jsonl.

Layout::

    <run_dir>/datasets/
      episode_0.parquet (or .jsonl)
      episode_1.parquet
      metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def export_run(run_dir: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    dest = Path(out_dir) if out_dir is not None else run_dir / "datasets"
    dest.mkdir(parents=True, exist_ok=True)

    steps = _read_jsonl(run_dir / "steps.jsonl")
    trajs = _read_jsonl(run_dir / "trajectory.jsonl")

    # Index trajectory rows by (episode, step) for the state join.
    traj_by_key = {(t.get("episode"), t.get("step")): t for t in trajs}

    episodes: dict[int, list[dict[str, Any]]] = {}
    for s in steps:
        if s.get("event") != "step":
            continue
        ep = int(s.get("episode", -1))
        episodes.setdefault(ep, []).append(s)

    use_parquet = False
    try:
        import pandas  # type: ignore  # noqa: F401
        import pyarrow  # type: ignore  # noqa: F401

        use_parquet = True
    except Exception:
        use_parquet = False

    written: list[str] = []
    for ep, rows in sorted(episodes.items()):
        joined: list[dict[str, Any]] = []
        for s in rows:
            st = s.get("step")
            t = traj_by_key.get((ep, st), {})
            joined.append(
                {
                    "episode_id": ep,
                    "step": st,
                    "action": (t.get("action") if isinstance(t, dict) else None),
                    "qpos": (t.get("qpos") if isinstance(t, dict) else None),
                    "source": s.get("source"),
                    "violated": bool(s.get("violated", False)),
                    "inference_s": s.get("inference_s"),
                    "gateway_s": s.get("gateway_s"),
                    "env_s": s.get("env_s"),
                }
            )
        if use_parquet:
            import pandas as pd  # type: ignore

            fp = dest / f"episode_{ep}.parquet"
            pd.DataFrame(joined).to_parquet(fp, index=False)
        else:
            fp = dest / f"episode_{ep}.jsonl"
            fp.write_text(
                "\n".join(json.dumps(r, default=str) for r in joined),
                encoding="utf-8",
            )
        written.append(fp.name)

    meta = {
        "run_dir": str(run_dir),
        "episodes": sorted(episodes.keys()),
        "files": written,
        "format": "parquet" if use_parquet else "jsonl",
        "note": (
            "state+action export only; no videos/ until image logging lands "
                    "(docs/data-pipeline-decision.md T2)"
        ),
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
