# GR00T N1.7 Integration — Feature Plan

> **Status:** Phase B IMPLEMENTED, awaiting weights (2026-09-14) — adapter, obs mapping,
> chunk buffer, DLS adapt, determinism seeding, and stubbed-vendor unit tests all green
> (28/28 suite). No real-weights run yet.
> **Needed to run:** one suite subdir of `nvidia/GR00T-N1.7-LIBERO` on local disk
> (`libero_object/` recommended) + `policy.repo_path` pointing at the Isaac-GR00T checkout.
> **Target:** NVIDIA Isaac GR00T N1.7 (`nvidia/GR00T-N1.7-3B` — note `-3B` suffix; `nvidia/GR00T-N1.7` 404s)
> **Ground truths (verified 2026-09-14 against the published checkpoint):** code Apache 2.0,
> **weights NVIDIA Open Model License**; LeRobot I/O; deploy API `Gr00tPolicy.get_action(obs)` /
> PolicyServer; Isaac Lab 3.0 Beta is Lab-2.x-compatible on Isaac Sim 6.0.1.
> ⚠️ The checkpoint's `embodiment_id.json` (52 tags) contains **no `LIBERO_PANDA`**.
> Closest Franka match is `robocasa_panda_omron`, whose modality config is **not** in the
> shipped `processor_config.json` (only 8 configs: oxe/g1/sharpa/xdof variants). Zero-shot
> GR00T on our Franka scene is therefore NOT a tag lookup — the modality config must come
> from the Isaac-GR00T repo code or from post-training (`launch_finetune.py`). Do not cite
> HF-blog LIBERO_PANDA claims; they don't match the checkpoint.

---

## 1. What we are adding (plain language)

A fourth policy adapter — `src/aegis/policies/groot.py` — next to `random`, `scripted`,
and `smolvla`, plus a `configs/models/groot_n17.yaml`. After this feature, this command works:

```bash
aegis eval --model groot_n17 --episodes 3 --seed 42 --max-steps 600
aegis eval-batch --models groot_n17,smolvla_libero,scripted --robots franka --episodes 3 --seed 42
```

The second command is the point: same robot, same limits, same seeds, GR00T vs SmolVLA
vs scripted baseline in one aggregated report — the first head-to-head VLA safety
comparison this harness can produce.

Concretely we add:

1. **`Gr00tPolicy` adapter** — loads the N1.7 checkpoint (or talks to its PolicyServer),
   takes our observation (3×256×256 cameras + 8-D proprioceptive state + language
   instruction), unwraps GR00T's action-chunk output (sequences of relative joint
   motions) into per-step 8-D joint-velocity actions our gateway consumes.
2. **Model config** — `configs/models/groot_n17.yaml` with `kind: groot`, `endpoint`
   (checkpoint path or PolicyServer URL), `embodiment_tag: LIBERO_PANDA`, instruction,
   chunk handling, quantization flag (mirrors `--quantize int8` from SmolVLA).
3. **CLI wiring** — `groot` added wherever `smolvla` is handled today (policy factory,
   `--quantize` override, CUDA-availability guard). No new commands.
4. **Batch + DR + recommendation reuse, free** — `eval-batch`, `--dr`, and `features`
   logging apply to GR00T with zero extra code, because they sit below the policy
   interface.

## 2. What we are improving (not just adding)

| Today | After this feature |
|---|---|
| Only one real VLA (SmolVLA-450M), which scores 0/3 zero-shot with a vision gap | Two real VLAs; SmolVLA's failure becomes a baseline, not a verdict |
| Safety numbers exist only for a third-party model | Safety numbers on NVIDIA's own model, trained on NVIDIA's sim stack — the joint-validation story for the partnership |
| Embodiment support is Franka-by-convention (SmolVLA happens to be LIBERO-tuned) | Explicit embodiment-tag pattern (`LIBERO_PANDA` first) that later extends to humanoids (G1/GR-1) without re-architecting |
| Inference story ends at `cpu/cuda` PyTorch | TensorRT/ONNX export path opened (GR00T ships export support) — first step toward Jetson Thor deployment measurement |

## 3. System architecture

```
                 ┌──────────────────────────────────────────────┐
                 │              AEGIS (unchanged)               │
  language ──▶   │  Gr00tPolicy.act(obs)  (NEW, policies/)      │
  instruction    │    checkpoint or PolicyServer                │
                 │    action-chunk → per-step 8-D joint-vel     │
                 │         │                                   │
                 │         ▼ timed, budget-enforced             │
                 │  SafetyGateway.filter()  (UNCHANGED)         │
                 │    1.NaN → 2.effort → 3.qvel → 4.qfrc        │
                 │         │ violation? → PID-to-home 50 steps  │
                 │         ▼                                   │
                 │  MuJoCo / IsaacLab env  (UNCHANGED)          │
                 │         │                                   │
                 │  RunLogger → report.json + features (UNCHANGED)│
                 └──────────────────────────────────────────────┘
```

The seam is `Policy.act(obs)` (`src/aegis/policies/base.py`). Everything above the
adapter (checkpoint, VLM backbone, diffusion head, chunking) is GR00T's business;
everything below it (gateway, env, logging, batching, DR) is untouched. If the model
crashes or returns non-finite output, the existing `PolicyModelError → model_error`
path engages fallback exactly as it does for SmolVLA today.

Two deployment modes, in order:

- **Phase A — in-process** (`endpoint` = local checkpoint path): simplest, same as
  SmolVLA today. Proves the adapter, the action conversion, and the safety numbers.
- **Phase B — PolicyServer** (`endpoint` = `http://...`): GR00T's reference serving
  pattern (`run_gr00t_server.py`), needed for real-time control and later for
  TensorRT-accelerated serving. Adapter gains an HTTP client; gateway budget
  enforcement covers network latency honestly (over-budget → fallback, logged).

## 4. What it solves — for developers

- **One-line model swap.** A robotics dev evaluates a frontier VLA by writing a YAML
  file, not by building an eval harness. `endpoint` + `embodiment_tag` + instruction
  is the whole integration surface.
- **Attribution, not vibes.** GR00T 0/3 vs SmolVLA 0/3 with identical seeds/limits/DR
  and per-step violation logs tells you *which* model misbehaves *where* (budget?
  velocity? mid-air grips?) instead of "it didn't work."
- **Deterministic re-runs.** Same `seed+episode` rule, same NDJSON schema — a GR00T
  result from today replays bit-identically next month, so regressions in
  post-training are measurable.
- **Failure mining for post-training.** `aegis export` (Slice A, planned) turns
  GR00T's failed episodes into LeRobot-format data — the exact format
  `launch_finetune.py` consumes. Eval failures become training inputs with no
  format wrangling.

## 5. What it solves — for enterprise / partnership

- **Independent safety instrument for NVIDIA's own model.** Today our safety numbers
  cover a third-party checkpoint. GR00T numbers from an independent gateway carry
  weight with safety teams, customers, and regulators that self-reported benchmarks
  don't — "the guardrail vendor measured your model" beats "we measured ourselves."
- **Procurement-grade evidence.** Every run emits `report.json` (violations,
  recoveries, p50/p95, gpu_hours, warnings) + full NDJSON traces. An enterprise
  safety case file practically writes itself from `outputs/<run_id>/`.
- **Embodiment roadmap without re-platforming.** `LIBERO_PANDA` today proves the
  pattern; UNITREE_G1 / GR-1 tags later reuse the same adapter shape, gateway, and
  report contract. The enterprise buys one harness, not one per robot.
- **License hygiene built in.** Reports record `model: groot_n17` + checkpoint
  provenance; docs state the split (code Apache 2.0 / weights Open Model License)
  so legal review starts from facts, not from our marketing.

## 6. Why we are building this (and why now)

1. SmolVLA gave us a measured 0/3 and a diagnosed vision gap — the harness works,
   but a harness with one failing model invites "maybe it's the harness." A second
   VLA, especially one whose LIBERO_PANDA embodiment matches our scene, tests the
   harness instead of the model.
2. NVIDIA is converging its own stack on LeRobot (GR00T 1.7, Teleop, Lab-Arena all
   landed there; Cosmos 3 planned). Our SmolVLA path already speaks LeRobot, so
   GR00T is the cheapest frontier model we will ever integrate — the plumbing is
   mostly built.
3. The partnership conversation needs a joint artifact. "Validate Aegis against
   GR00T N1.7 on the same seed-42 protocol" is concrete, cheap for NVIDIA (their
   model, their embodiment tag, our instrument), and produces numbers both sides
   can cite.
4. It de-risks Phase 3 real: GR00T is trained against Isaac Lab/PhysX, so when our
   Isaac runtime lands, GR00T is the in-distribution eval target — one integration
   serves both the MuJoCo present and the Isaac future.

## 7. Build phases + acceptance

| Phase | Work | Acceptance |
|---|---|---|
| A. Config + stub | `ModelSpec.kind: groot`, `GrootPolicySpec`, `configs/models/groot_n17.yaml`, CLI factory arm that loads config and raises honest `ConfigError` if checkpoint absent | `aegis validate --model groot_n17` → `config OK`; eval without checkpoint → exit 2/3 with clear message, no fabricated numbers |
| B. In-process adapter | `groot.py`: checkpoint load, obs contract, chunk unwrap → 8-D, `PolicyModelError` on failure, CUDA guard | `eval --model groot_n17 --episodes 1` completes with honest report (any score); violations/budget path exercised |
| C. Batch comparison | Docs + verified run: `eval-batch --models groot_n17,smolvla_libero,scripted` | Aggregated report with 3-model nested `task_counts`; committed as first head-to-head |
| D. PolicyServer + TRT (later) | HTTP client mode, TensorRT export measurement | Latency + `gpu_hours` reported; budget violations attributable to network vs inference |

## 8. Risks (honest)

- **Action-space mismatch.** GR00T emits relative-joint action chunks; our DLS adapter
  was built for cartesian deltas. If LIBERO_PANDA's modality config doesn't map
  cleanly, Phase B grows a small resolved-rate rework — contained in the adapter.
- **VRAM.** 3B + Cosmos-Reason2 backbone vs our 6 GB story. Mitigation: big-GPU
  `cuda` first, int8/TRT measurement second, no 6 GB promises until measured.
- **Weight license.** Open Model License, not Apache — docs and reports must say so;
  enterprise legal will check.
- **Upstream churn.** Isaac-GR00T is GA but moving fast (N1.5→N1.6→N1.7 in months);
  pin the checkpoint (`nvidia/GR00T-N1.7`) in config, never `latest`.
