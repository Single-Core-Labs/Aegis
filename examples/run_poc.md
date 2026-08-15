# Run the Phase 1 POC

End-to-end walkthrough: validate config, evaluate a working scripted
policy, evaluate a failing random policy, and run the test suite.

```bash
# 1. Validate the config (exit 0)
uv run aegis validate

# 2. Scripted policy: 3/3 success, 0 violations, "all episodes succeeded"
uv run aegis eval --model scripted --episodes 3 --seed 42

# 3. Random policy: 0/3 success, ~145 violations / recoveries,
#    "tighten safety limits or fix the policy"
uv run aegis eval --model random --episodes 3 --seed 7

# 4. Test suite (config validation, determinism, acceptance check)
uv run pytest tests/test_eval.py -q
```

## What you should see

Scripted run summary (numbers may drift slightly):

```
task          : pick-place
episodes      : 3   (success 3 / fail 0)
inference     : p50 0.01 ms  p95 0.039 ms   (per-step, cpu)
safety        : violations 0  recoveries 0
gpu hours     : 0.0 (stub — no GPU used in POC)
recommendation: all episodes succeeded — harness ready to evaluate real policies
report      : outputs\run-<timestamp>\report.json
```

Random run summary:

```
episodes      : 3   (success 0 / fail 3)
safety        : violations 145  recoveries 145
recommendation: policy repeatedly violated safety limits (145 violations, 145
                recoveries); tighten safety limits or fix the policy ...
```

## Reading a run directory

Each run creates `outputs/run-<timestamp>/`:

- `report.json` — machine-readable summary (task counts, p50/p95 latency,
  safety stats, recommendation, warnings).
- `episodes.jsonl` — one `episode_start` / `episode_end` event per episode
  (outcome, steps, violations, recoveries).
- `steps.jsonl` — per-step records incl. any `safety_violation` events.
- `trajectory.jsonl` — per-step observations/actions for replay.

Example (from `report.json`):

```json
{
  "run_id": "run-20260815T070329Z",
  "model": "random",
  "robot": "franka",
  "sim": "mujoco",
  "task_counts": { "pick-place": { "success": 0, "fail": 3 } },
  "latency_p50_ms": 0.006,
  "latency_p95_ms": 0.017,
  "safety_violations": 145,
  "recovery_events": 145,
  "recommendation": "policy repeatedly violated safety limits ..."
}
```

## Reproducibility

Episode seeds are `seed + episode` (0, 1, 2 for `--seed 42`), so runs are
deterministic for a given seed, policy, and MuJoCo build. The `scripted`
policy does no random sampling, so its episodes are bit-identical across
re-runs on the same machine.