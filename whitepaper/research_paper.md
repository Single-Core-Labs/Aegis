# AEGIS: A Safety-Gated Evaluation Harness for Vision-Language-Action Policies on Physical Robots

> Companion Markdown version of `main.tex` (whitepaper), updated 2026-09-14 with
> batch evaluation, domain randomization, structured recommendation logging, and
> GR00T N1.7 adapter status. All numbers below are measured on the referenced
> commit; nothing is projected or simulated by hand.

**Keywords:** Physical AI, robot safety, VLA, sim-to-real, Isaac Lab, MuJoCo, ROS 2, evaluation

---

## Abstract

Vision-Language-Action (VLA) models — SmolVLA, RT-2, π₀, GR00T — now command robot
joints directly from vision and language. They are powerful but unsafe by default:
a single hallucinated trajectory can exceed joint velocity limits, apply destructive
forces, or drive hardware into collision. Today's robotics stacks have no standard
safety layer between policy and actuator, no unified evaluation protocol, and no
systematic measurement of the sim-to-real gap.

We present **AEGIS**, a minimal, open (Apache-2.0) harness that every action *must*
traverse. AEGIS contributes: (i) a **Safety Gateway** with per-joint
velocity/force/NaN/inference-budget checks and a PID-to-home fallback with no
bypass path; (ii) a **deterministic evaluation harness** (`aegis eval`, plus
`aegis eval-batch` over robots × models × tasks) that emits auditable `report.json`
+ NDJSON telemetry with p50/p95 latency; (iii) **seeded domain randomization** and
**structured recommendation logging** for robustness ablations and future learned
verdicts; and (iv) a **sim-to-real measurement method** across MuJoCo, Isaac Lab,
and real Franka hardware via ROS 2. On a Franka Panda pick-place task: scripted
policy **3/3 success, 0 violations**; random policy **0/3, 4 violations / 2
recoveries** (per-joint limits); a limit-regime A/B showing legacy uniform limits
catch the random policy **9× vs 1×**; and honest SmolVLA-450M zero-shot
characterization (**0/3**) separating limit-induced interruptions from genuine
vision domain gap. The Isaac Lab and ROS 2 scaffold degrades gracefully on 6 GB
laptops with explicit fallback tagging — no fabricated numbers, ever.

---

## 1. Introduction

The convergence of large vision-language models with robot control — VLA — has moved
manipulation from hand-engineered pipelines to end-to-end learned policies. A single
model consumes multi-view images and a language instruction and outputs joint
velocities at 20–50 Hz. This breaks the foundational assumption of industrial
robotics: that every command reaching an actuator is bounded, verified, and safe.

In production settings — healthcare, logistics, manufacturing — an unbounded action
is not a research artifact. It is a stripped gear, a torn tendon, or an unsafe
contact. Yet the ecosystem has not kept pace:

- **No standard safety layer.** Teams re-implement ad-hoc checks, usually only on
  *commanded* actions, not *measured* states. A policy hallucinating 5 rad/s passes
  silently if only the command is clamped.
- **No standard evaluation.** Episode counts, seeds, latency percentiles, and
  violation accounting differ across papers. Cross-paper comparison is impossible;
  failures are not attributable.
- **No honest sim-to-real reporting.** A LIBERO-trained model can fail in a new
  simulator from camera domain shift, but reports conflate this with safety
  interruptions. The gap stays anecdotal.

```
Policy (VLA/Classical) --8-D action--> [ AEGIS Safety Gateway + Eval Harness ] --safe action--> Robot (MuJoCo/Isaac/Real)
                                              ^ measured qvel/qfrc feedback (no bypass by construction)
```

**AEGIS** answers with infrastructure, not another model. Contributions:

1. **Mandatory Safety Gateway** — per-joint limits, NaN/force/budget checks,
   PID-to-home fallback.
2. **Deterministic evaluation harness** — timing from day one, honest reporting
   (stubs flagged, fallbacks tagged), one `report.json` schema across simulators,
   batch mode over robots × models × tasks.
3. **Robustness tooling** — seeded domain randomization and structured
   recommendation features for ablations and future learned verdicts.
4. **Sim-to-real measurement method** — relaxed-limit A/B separating
   limit-induced interruptions from model mis-localization, with visual evidence.
5. **Open NVIDIA-stack scaffold** — Isaac Lab + ROS 2, graceful degradation on
   constrained hardware.

---

## 2. Related Work

**VLA models.** RT-2, OpenVLA, π₀, SmolVLA demonstrate generalist manipulation from
vision+language; NVIDIA Isaac GR00T N1.7 adds an open-weights dual-system VLA
(Cosmos-Reason2 reasoning backbone + diffusion action head) with LeRobot-format I/O
and post-trained embodiments (code Apache-2.0; weights under NVIDIA's Open Model
License). All optimize task loss, not hardware safety. Our work is complementary:
we gate *any* VLA's actions.

**Robot safety.** Control barrier functions, constrained/safe RL, hardware e-stops.
AEGIS operates at the *action-space* layer — a lightweight, policy-agnostic gate
composing with lower-level safety. Unlike learned safety filters, our checks are
interpretable and auditable.

**Evaluation harnesses.** LIBERO, robosuite, and Isaac Lab provide simulation
benchmarks but leave safety to the user. Synthetic-data blueprints (GR00T-Mimic,
Cosmos Transfer) scale training data but evaluate by raw success rate, without
failure attribution. AEGIS unifies evaluation *with* safety accounting — violations,
recoveries, budgets, latencies — under one CLI and report schema.

---

## 3. System Design

### 3.1 Design principles

- **No bypass.** `EvalRunner` only calls `env.step(gateway.filter(...))`.
- **Honest by default.** Mocked/fallback/stubbed values are labeled in `warnings`
  and `info["sim"]`. Fabricating a number is a defect, not a shortcut.
- **Measured, not commanded.** Velocity/force checks read `qvel`/`qfrc`, the
  physical state — never just the policy's command vector.
- **Declarative config.** YAML in, Pydantic-validated (`extra="forbid"`);
  no robot/model constants in application code.
- **Timed from day one.** `perf_counter_ns` on inference/gateway/env; CUDA events
  for per-kernel GPU time when available.

### 3.2 Safety Gateway

Ordered checks on every step: (1) NaN/Inf → fatal; (2) commanded gripper effort
vs `max_effort_action` (clamp or reject); (3) measured `|qvel|` vs per-joint
`max_velocity` (Franka: `[2.175×4, 2.61×3]` rad/s); (4) measured `|torque|` vs
per-joint `max_force` (`[87×4, 12×3]` Nm). Inference over `inference_budget_ms`
or a `PolicyModelError` travels the identical violation path. On violation: count,
emit event, engage PID-to-home fallback (`τ = qfrc_bias + 40·err + 5·vel_err`,
bounded 0.3 rad/s) for `recovery_steps=50`, then `resume` (or `hold`). The fallback
action is itself re-checked.

### 3.3 Evaluation harness

`OctEval loop: env.reset(seed+episode)` → timed `policy.act(obs)` → budget check →
`gateway.filter` → timed `env.step` → NDJSON log. Success = `grasped_ever ∧
dist ≤ 0.05 m`, where grasp = lifted (`z > table+0.025`) ∧ finger–object contacts
with `mj_contactForce > 0.5 N` (sticky). Termination: success | `max_steps`
truncation | wall-clock timeout. `aegis eval-batch` repeats this over
robots × models × tasks with one isolated env per combo and combo seeds
`base + i·1000`, merging nested `task_counts[task][robot][model]` + summary.

### 3.4 Robustness tooling

Seeded MuJoCo domain randomization (`--dr`): light ±0.1 m, object friction
±0.002, camera ±0.02 m, drawn from the episode RNG with nominals restored per
reset — deltas logged to `episodes.jsonl` and step `info`. Every report carries
structured recommendation output (`recommendation_model: heuristic_v1`,
`label/confidence/evidence`, plus a `features` dict) so a future learned
recommender trains on logged runs instead of opinions.

---

## 4. Implementation

Module map (`src/aegis/`): `cli.py` (Typer: `eval`, `eval-batch`, `export`
-deferred, `validate`, `rosbench`) · `config/` (Pydantic schema + loader,
CLI > YAML > defaults) · `envs/` (MuJoCo Franka Menagerie + Isaac Lab scaffold
with honest `isaaclab-fallback-mujoco` tagging) · `policies/` (`random`,
`scripted` 6-phase DLS-IK, `smolvla` 450M chunk-50, `groot` adapter for GR00T
N1.7 LIBERO post-trained checkpoints) · `safety/` (checks/gateway/fallback) ·
`eval/` (runner/metrics/report/recommendation/batch) · `ros2/` (auto-mock bridge
+ latency benchmark) · `telemetry/` (NDJSON logger, timing, OTel + Grafana).

---

## 5. Evaluation

All runs CPU/MuJoCo unless noted; deterministic on stated seeds (byte-identical
repeats verified).

| Run | Policy / sim | Result | Viol / rec | p50/p95 (ms) |
|---|---|---|---|---|
| scripted 3 eps seed 42 | baseline | **3/3** | 0 / 0 | 0.012 / 0.027 |
| random 3 eps seed 7 | negative control | 0/3 | 4 / 2 | 0.008 / 0.020 |
| scripted, `--sim isaaclab` (fallback) | scaffold honesty | **3/3** + `RuntimeWarning` | 0 / 0 | 0.012 / 0.039 |
| scripted `--dr` 3 eps seed 42 | DR ablation | **3/3** + DR warning | 0 / 0 | 0.012 / 0.038 |
| batch: franka/scripted (seed 42) | per-joint | **3/3** | 0 / 0 | 0.012 / 0.026 |
| batch: franka/random (seed 1042) | per-joint | 0/3 | 1 / 1 | 0.009 / 0.023 |
| batch: franka_uniform/scripted (seed 2042) | uniform 1.0 rad/s | **3/3** | 0 / 0 | 0.012 / 0.032 |
| batch: franka_uniform/random (seed 3042) | uniform 1.0 rad/s | 0/3 | **9 / 8** | 0.009 / 0.022 |
| batch summary (4 combos, 12 eps) | — | 6/6 | 10 / 9 | — |
| ROS mock bridge (n=100) | — | — | — | p50 0.000 / p95 0.001 |
| Full test suite | — | **28/28 pass** | — | — |

**Velocity-limit ablation.** Legacy uniform limits catch the random policy 9× vs
per-joint 1× while scripted stays clean under both — the "realistic envelope"
claim reproduced in one batch.

**SmolVLA zero-shot (prior, GPU).** 0/3 at ~33 violations/ep (1 budget + 32
velocity); still 0/3 at relaxed 5.0 rad/s with the hand drifting 0.16 m → 0.60 m
and mid-air grips — i.e. vision domain gap, not limit interruption.

**GR00T N1.7 (adapter implemented, weights pending).** Verified against published
artifacts: base `GR00T-N1.7-3B` refuses `libero_sim` (POSTTRAIN-only tag), so the
target is the post-trained `GR00T-N1.7-LIBERO` suite with the 16-step absolute
chunk contract; RPY order and gripper scale are flagged calibration assumptions.

---

## 6. Limitations (read before citing)

- Sim-only until Phase 3 real: no Isaac PhysX/USD execution, no real `rclpy`, no
  hardware runs. Go/No-Go is explicitly **No-go**.
- Velocity control is torque control (gravity-compensated PD), not a perfect servo.
- Recommendation is 4 hand-written rules (`heuristic_v1`), not learned; unknown
  features are `None`, never zero-filled.
- `gpu_hours` is per-kernel inference time on `cuda`, `0.0` on `cpu`.
- GR00T/SmolVLA numbers need weights + GPU; the 6 GB VRAM story is unmeasured.

---

## 7. Conclusion & Future Work

AEGIS shows that a small, honest, mandatory safety layer plus deterministic
evaluation turns VLA benchmarking from anecdote into evidence: attributable
failures, reproducible numbers, and reports a safety engineer can file. Next, in
order: real ROS 2 + Isaac USD/PhysX runtime (Phase 3), first GR00T-vs-SmolVLA
head-to-head on the same protocol, image-logged runs unlocking dataset export
for the post-training flywheel, and hardware dry-run Gates 1–4.

---

## References

- RT-2, OpenVLA, π₀, SmolVLA (VLA models); LIBERO, robosuite, Isaac Lab
  (benchmarks); MimicGen (synthetic demo amplification).
- NVIDIA Isaac GR00T N1.7 (open VLA, LeRobot I/O, `nvidia/GR00T-N1.7-LIBERO`
  post-trained suites); Isaac Lab 3.0 Beta (Newton/kit-less backends);
  GR00T-Mimic synthetic-motion blueprint (780K trajectories, +40% with
  synthetic+real; Franka stacking 84% via BC).
- AEGIS repo artifacts: `PRD.md`, `README.md`, `CURRENT_STAGE.md`,
  `docs/groot-integration-plan.md`, `docs/data-pipeline-decision.md`.
