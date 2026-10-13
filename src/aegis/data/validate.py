"""Dataset validation for ``aegis data-check``.

Checks the ``datasets/`` layout produced by ``export.py``: metadata exists,
per-episode files exist, required columns present, no NaN/Inf in numeric
fields, episode continuity. Returns a list of error strings (empty = OK).
Works for both parquet (needs pandas) and jsonl fallback.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REQUIRED_COLS = (
    "episode_id",
    "step",
    "action",
    "qpos",
    "source",
    "violated",
    "inference_s",
)


def _is_bad_number(x: object) -> bool:
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))


def _walk_numbers(v: object) -> bool:
    if _is_bad_number(v):
        return True
    if isinstance(v, (list, tuple)):
        return any(_walk_numbers(i) for i in v)
    return False


def validate_dataset(dataset_dir: str | Path) -> list[str]:
    errors: list[str] = []
    d = Path(dataset_dir)
    meta_p = d / "metadata.json"
    if not meta_p.exists():
        errors.append("metadata.json missing")
        return errors
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"metadata.json unreadable: {exc}"]
    files = meta.get("files", [])
    if not files:
        errors.append("no episode files listed in metadata.json")
    use_parquet = meta.get("format") == "parquet"

    rows_by_ep: dict[int, list[dict]] = {}
    if use_parquet:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            return ["format=parquet but pandas is not installed"]
        for f in files:
            fp = d / f
            if not fp.exists():
                errors.append(f"{f}: file missing")
                continue
            try:
                df = pd.read_parquet(fp)
            except Exception as exc:
                errors.append(f"{f}: parquet unreadable: {exc}")
                continue
            for col in REQUIRED_COLS:
                if col not in df.columns:
                    errors.append(f"{f}: missing column {col}")
            if df.isna().any().any():
                errors.append(f"{f}: contains NaN values")
            for _, row in df.iterrows():
                if _walk_numbers(row.get("action")) or _walk_numbers(row.get("qpos")):
                    errors.append(f"{f}: contains NaN/Inf in action/qpos")
                    break
                rows_by_ep.setdefault(int(row["episode_id"]), []).append(
                    {"step": int(row["step"])}
                )
    else:
        for f in files:
            fp = d / f
            if not fp.exists():
                errors.append(f"{f}: file missing")
                continue
            rows: list[dict] = []
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            if not rows:
                errors.append(f"{f}: empty episode file")
                continue
            for col in REQUIRED_COLS:
                if any(col not in r for r in rows):
                    errors.append(f"{f}: missing column {col}")
            for r in rows:
                if _walk_numbers(r.get("action")) or _walk_numbers(r.get("qpos")):
                    errors.append(f"{f}: contains NaN/Inf in action/qpos")
                    break
                if r.get("action") is None or r.get("qpos") is None:
                    errors.append(f"{f}: null action/qpos (no trajectory join?)")
                    break
            ep = int(rows[0].get("episode_id", -1))
            rows_by_ep[ep] = [{"step": int(r.get("step", -1))} for r in rows]

    for ep, rows in rows_by_ep.items():
        steps = sorted(r["step"] for r in rows)
        if steps != list(range(min(steps), max(steps) + 1)):
            errors.append(f"episode {ep}: non-contiguous steps")
    return errors
