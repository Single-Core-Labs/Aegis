"""Behavior-cloning trainer on exported datasets (numpy-first).

Reads ``datasets/episode_*.{parquet,jsonl}`` (see ``aegis/data/export.py``)
and fits a tiny state(qpos 7)->action(8) model. Uses torch MLP when torch is
installed, else a closed-form ridge-regression fallback so the command works
on a bare checkout. Writes ``bc_policy.npz`` + ``metrics.json``.

Acceptance: train on scripted rollouts, re-eval the checkpoint through the
same gateway, zero unsafe actions during training (gateway is always on in
``AegisRLEnv``; this trainer never commands hardware).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset(dataset_dir: str | Path) -> tuple[Any, Any]:
    """Return (X qpos[N,7], Y action[N,8]) as numpy arrays."""
    import numpy as np  # type: ignore

    d = Path(dataset_dir)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    if meta.get("format") == "parquet":
        import pandas as pd  # type: ignore

        for f in meta["files"]:
            df = pd.read_parquet(d / f)
            for _, row in df.iterrows():
                q, a = row["qpos"], row["action"]
                if q is None or a is None:
                    continue
                xs.append([float(v) for v in list(q)][:7])
                ys.append([float(v) for v in list(a)][:8])
    else:
        for f in meta["files"]:
            for line in (d / f).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                q, a = r.get("qpos"), r.get("action")
                if q is None or a is None:
                    continue
                xs.append([float(v) for v in list(q)][:7])
                ys.append([float(v) for v in list(a)][:8])
    X = np.asarray(xs, dtype=float).reshape(-1, 7)
    Y = np.asarray(ys, dtype=float).reshape(-1, 8)
    return X, Y


def train_ridge(X: Any, Y: Any, l2: float = 1e-3) -> dict[str, Any]:
    import numpy as np  # type: ignore

    n = X.shape[0]
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)  # bias
    A = Xb.T @ Xb + l2 * np.eye(8)
    W = np.linalg.solve(A, Xb.T @ Y)  # [8,8]: 7 state + bias -> 8 action
    pred = Xb @ W
    mse = float(np.mean((pred - Y) ** 2))
    return {"weights": W, "mse": mse, "n": int(n), "backend": "ridge"}


def train_bc(
    dataset_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    l2: float = 1e-3,
) -> dict[str, Any]:
    import numpy as np  # type: ignore

    X, Y = load_dataset(dataset_dir)
    if X.shape[0] == 0:
        raise ValueError("empty dataset: run `aegis export` on a finished run first")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 8))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        for _ in range(max(1, epochs)):
            opt.zero_grad()
            loss = loss_fn(model(Xt), Yt)
            loss.backward()
            opt.step()
        mse = float(loss_fn(model(Xt), Yt).detach().item())
        torch.save(model.state_dict(), out / "bc_policy.pt")
        metrics: dict[str, Any] = {"mse": mse, "n": int(X.shape[0]), "backend": "torch-mlp", "epochs": epochs}
    except Exception:
        res = train_ridge(X, Y, l2=l2)
        np.savez(out / "bc_policy.npz", weights=res["weights"])
        metrics = {"mse": res["mse"], "n": res["n"], "backend": "ridge", "l2": l2}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
