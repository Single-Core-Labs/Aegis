# Multi-Robot / Multi-Task Batching — Spec (P3)

**Status:** Spec only — not implemented until Phase 3 real is green (`DESIGN.md:262` scope discipline).

## Goal

Run `aegis eval` over a Cartesian product of `robots × models × tasks × seeds` in one invocation, with parallel episodes where safe, and a single aggregated `report.json`.

## CLI Sketch (future)

```bash
# Sequential (default, safe):
uv run aegis eval --robots franka,franka_diag --models scripted,random --tasks pick-place --episodes 3 --seed 42

# Parallel episodes (requires --parallel N, isolated envs):
uv run aegis eval --model scripted --episodes 20 --seed 42 --parallel 4 --output-dir outputs/batch-20260909
```

## Design Constraints

- **Safety isolation:** Each episode gets its own `MujocoPickPlaceEnv` instance (no shared `MjData`). Gateway per env.
- **Determinism:** Seed = `base_seed + task_index*1000 + episode` — reproducible regardless of parallel order.
- **No shared GPU:** SmolVLA chunk inference must be serialized or sharded — no concurrent CUDA contexts on 6GB.
- **Report schema:** `report.json:task_counts` becomes `task_counts[task][robot][model]` + `summary` rollup. Backward-compatible: single-task runs keep flat `task_counts: {pick-place: {success, fail}}`.

## Output Layout

```
outputs/batch-20260909/
  report.json          # aggregated across all combos
  runs/
    franka-scripted-pick-place-seed42/
      episodes.jsonl / steps.jsonl / trajectory.jsonl
    franka-random-pick-place-seed42/
      ...
```

## Implementation Notes

- Worker pool: `concurrent.futures.ProcessPoolExecutor` per episode (not per step — too fine-grained).
- Telemetry: each worker writes its own NDJSON, main process merges `report.json`.
- Isaac Lab: parallel envs require `num_envs` in USD — defer until real USD is authored.

## Not Building Now

Tracked here for roadmap visibility; implementation waits for Phase 3 Gates 0-4.
