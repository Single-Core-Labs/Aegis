# Aegis Platform Vision — From Eval Harness to Robotics Platform

> **Status:** Strategic roadmap — not yet built (except where marked)
> **Context:** Aegis selected as NVIDIA showcase project. This doc captures the path from "safety-gated evaluation tool" to "core platform / service for robotics."
> **Date:** 2026-09-14

---

## 1. Where We Are Today

**Aegis is a safety-gated evaluation harness.** It runs policies through a mandatory safety layer, measures results deterministically, and produces auditable reports. It is excellent at **checking** what a policy does.

| Layer | Status | What It Does |
|---|---|---|
| **Safety Gateway** | ✅ Complete | Per-joint velocity/force/NaN/budget checks, PID-to-home fallback, no bypass path |
| **Eval Harness** | ✅ Complete | `aegis eval`, seeded episodes, p50/p95 latency, `report.json` + NDJSON |
| **Multi-Sim** | ✅ MuJoCo / ⚠️ Isaac scaffold | MuJoCo 3.x full; Isaac Lab falls back with `RuntimeWarning` |
| **Multi-Policy** | ✅ Complete | SmolVLA-450M, GR00T N1.7 (adapter, weights pending), scripted, random |
---

## 3. Layer 1: Evaluate (Today — Polish & Complete)

### 3.1 What Exists
- Full MuJoCo eval loop with scripted/random/SmolVLA policies
- Per-joint Franka limits (`[2.175..2.61] rad/s`, `[87..12] Nm`)
- Deterministic seeded episodes (`seed + episode_id`)
- NDJSON telemetry: `run.json`, `episodes.jsonl`, `steps.jsonl`, `trajectory.jsonl`, `report.json`
- Prometheus/Grafana/OTel exporters
- Sequential `aegis eval-batch` (robots × models × tasks)
- MuJoCo domain randomization (`--dr`: light/friction/camera jitter)
- Structured recommendation (`heuristic_v1`)

### 3.2 Polish for Showcase
- [ ] Get Isaac Sim / Isaac Lab running in **real mode** (Phase 3 real — the #1 priority)
- [ ] Add a **demo walkthrough**: single command that runs a clean eval + shows the report in < 2 minutes
- [ ] Short demo video / GIF of the safety gateway catching a violation (even in MuJoCo — visual is powerful)
- [ ] Polish README lead: one sentence, one diagram, one command, "Built on NVIDIA" badge with versions
- [ ] Add "What's Next" / roadmap section visible in README
- [ ] Draft a one-pager / pitch deck (problem → solution → benchmarks → NVIDIA stack → what's next)

---

## 4. Layer 2: Train — RL + Data Pipeline
### 4.3 Policy Distillation & Compression for Edge

VLA models (SmolVLA-450M, GR00T-3B) are too large for edge robots. The platform needs to **shrink** them.

| Technique | What It Does | Aegis Role |
|---|---|---|
| **Distillation** | Train a small student policy from a large teacher's rollouts | Aegis runs the teacher in sim, logs rollouts, trains the student — then eval **both** through the same safety gate |
| **Quantization** (int8 / FP8) | Shrink model + faster inference | `docs/perf-tuning.md` already mentions `--quantize int8` — extend to measure accuracy vs latency tradeoff per quantization level, report both |
| **Early-exit / dynamic compute** | Skip layers when the action is "easy" | Aegis latency instrumentation (p50/p95 per step) measures whether early-exit helps or hurts — per-step, not just average |
| **LoRA / adapter finetuning** | Cheap finetuning instead of full retrain | Log per-adapter performance through the safety gate — compare which adapters work for which tasks, auto-recommend |

### 4.4 RL for Safety Itself

The most powerful idea — use RL not just for the **task policy**, but for the **safety layer**.

| Idea | What It Means |
|---|---|
| **Learned recovery policy** | Current fallback is PID-to-home (dumb but safe). Train a recovery policy that returns to safe state faster or in a more task-aware way — while staying within safety limits |
| **Configurable safety budget RL** | Instead of fixed `recovery_steps=50`, learn optimal recovery duration per violation type from logged data |
| **Safe RL / Constrained RL** | Train the task policy with a safety constraint (stay under velocity/force limits) — Aegis provides the constraint signal in real-time during training |
| **Learned recommendation (`learned_v1`)** | Replace `heuristic_v1` with a calibrated classifier: `{ready, tighten_limits, fix_policy, raise_steps, investigate_gap}` with confidence + evidence pointers. Features are already logged — needs training data (Isaac + ROS 2 runs) |

---

## 5. Layer 3: Deploy — Edge AI for Robotics

This is where Aegis becomes a **runtime product**, not just a dev tool.

### 5.1 On-Device Safety Gateway (Edge Runtime)
---

## 6. Product / Service Models

### 6.1 Model 1: Aegis Platform (Open-Core)

| Layer | Open Source (Apache 2.0) | Paid / Enterprise |
|---|---|---|
| Safety Gateway + Eval | ✅ | — |
| RL training loop + dataset export | Core or premium | — |
| Edge runtime (C++ gateway, Jetson) | Core | — |
| Fleet dashboard + anomaly detection | — | SaaS / on-prem |
| Continuous validation + compliance reports | — | Enterprise |
| Multi-robot batching + orchestration | — | Enterprise |
| Policy distillation / compression toolkit | — | Premium |
| Support + safety certification path | — | Services |
---

## 7. The Full Flywheel

```
     ┌─────────────────────────────────────────────┐
     │              AEGIS PLATFORM                  │
     │                                             │
     │  EVALUATE ──▶  TRAIN  ──▶  DEPLOY          │
     │  (today)        (RL)       (edge)          │
     │    │               │           │            │
     │    ▼               ▼           ▼            │
     │  report.json   fine-tuned   on-robot       │
     │  + NDJSON       policy      safety +       │
     │                 + safety     telemetry      │
     │                 report                        │
     │                                             │
     │  ◀────────────────────────────────────────  │
     │  Continuous feedback: deploy telemetry      │
     │  feeds back into eval + retraining         │
     └─────────────────────────────────────────────┘
```

1. **Evaluate** a policy safely in sim → get a report
2. **Train** an improved policy using eval data + RL (safe — gateway always on)
3. **Deploy** the policy to the edge with the safety gateway embedded
4. **Monitor** real-world telemetry → feeds back to eval (drift detection) and training (new data)
---

## 8. Prioritized Roadmap

### 🔴 Before NVIDIA Showcase (highest priority)

| # | What | Why |
|---|---|---|
| 1 | **Isaac Sim / Isaac Lab real mode** (Phase 3 real) | Core NVIDIA stack integration. Without it, the showcase is "built for NVIDIA" but doesn't run on it |
| 2 | **Demo walkthrough** (`demo.sh` or notebook) | Showcase visitors need to see something working in < 2 minutes |
| 3 | **README + one-pager polish** | First impression for NVIDIA and the community |

### 🟡 Short-Term (1-3 months)

| # | What | Why |
|---|---|---|
| 4 | **Image logging in RunLogger** (T2) | Unlocks dataset export + VLA retraining story |
| 5 | **Dataset export + `data-check`** (T4) | Turns eval runs into training data — the post-training flywheel |
| 6 | **RL training loop in MuJoCo** (safe, gateway-gated) | Showcases closed-loop capability. Even simple BC on scripted rollouts proves the concept |
| 7 | **Learned recommendation (`learned_v1`)** | Replace heuristic with trained classifier — features are already logged |

### 🟢 Medium-Term (3-6 months)

| # | What | Why |
|---|---|---|
| 8 | **Embodiment registry** (T3 — second robot) | Makes the platform story credible: Aegis works across robots |
| 9 | **Policy distillation + quantization toolkit** | Shrinks VLA models for edge — key for deploy story |
| 10 | **Edge runtime story** (docs + architecture, C++ later) | Shows the deploy path. Safety config as versioned artifact is concrete |
---

## 9. What Makes This a Platform (Not Just a Tool)

| Dimension | Tool (today) | Platform (vision) |
|---|---|---|
| **Scope** | Evaluates one policy on one robot | Trains, deploys, monitors across robots and policies |
| **Output** | A report | A report + a better policy + a deployed runtime + a compliance record |
| **Time horizon** | One run | Continuous: eval → train → deploy → monitor → repeat |
| **User** | Robotics engineer running a test | Robotics team shipping policies, safety engineers tuning limits, fleet operators monitoring robots |
| **Business model** | Open source tool | Open-core platform + SaaS + edge SDK + services |
| **Moat** | Safety gateway design | The flywheel: more runs → more data → better policies → more deployments → more telemetry → better safety |
| 11 | **Shadow mode + canary deployment** | "Harness that never turns off" — powerful differentiator |
| 12 | **Contextual bandit for safety limit tuning** | Automates recommendation beyond 4 rules |

### 🔵 Long-Term (6+ months)

| # | What | Why |
|---|---|---|
| 13 | **C++ edge gateway** (TensorRT, Jetson-native) | Real on-robot safety runtime |
| 14 | **Fleet dashboard + anomaly detection** | SaaS / enterprise product story |
| 15 | **Continuous validation + compliance reporting** | Product for regulated industries |
| 16 | **RL for safety** (learned recovery, constrained RL) | Safety layer becomes smarter, not just a fixed gate |
| 17 | **Multi-robot batching + orchestration** (parallel, Isaac `num_envs`) | Scales testing throughput |
| 18 | **Safety-as-a-Service (SaaS)** | Cloud certification service for robot AI policies |
5. **Certify** the whole chain for compliance (signed, auditable reports)

The safety gateway is the **thread that runs through everything** — sim, training, edge, compliance.

**Why it works:** The safety gateway and eval are the open hook — everyone needs them. The value is in what you build on top: training, deployment, fleet management, compliance.

### 6.2 Model 2: Safety-as-a-Service (SaaS)

A cloud service where robotics teams:

1. **Upload their policy** (or point to a HuggingFace checkpoint)
2. **Select their robot** from an embodiment registry
3. **Run safety evaluation** in the cloud (Aegis on NVIDIA GPUs, MuJoCo + Isaac)
4. **Get a safety report** with Go/No-Go recommendation
5. **Certify** — the report is signed, auditable, exportable for compliance

**"Safety certification as a service"** for robot AI policies. Every time a team updates their policy, they re-run the cert. If the safety record degrades, they know before deployment.

### 6.3 Model 3: Edge SDK + Runtime

A stripped-down Aegis runtime that ships **on the robot**:

- Embedded safety gateway (C++/TensorRT)
- Policy inference (TensorRT, ONNX, or native)
- Telemetry logger (local + optional cloud sync)
- OTA config updates (safety limits, policy version)
- Shadow mode + canary deployment

This is what a robot OEM would embed in their product — **"powered by Aegis safety"** becomes a selling point.

### 6.4 Model 4: RL Training Platform

A service that takes your policy + robot + task, runs closed-loop RL in simulation (safety-gated, so training never damages hardware), and returns a fine-tuned policy + safety report:

```
Upload policy → Aegis runs RL in sim (safety-gated) → Returns improved policy + eval report
```

Leverages everything: the env, the safety gate (makes RL safe to run unattended), the telemetry (measures improvement), the report (proves the result).

The safety gateway today runs in Python alongside the policy. For real robots, it runs **on the edge device** — Jetson Orin, or even a microcontroller for the lowest-latency checks.

| Enhancement | Why It Matters |
|---|---|
| **C++/TensorRT safety checks** | Gateway checks (NaN, velocity clamp, force clamp) are simple math — run in C++ at microsecond latency on the robot's realtime controller |
| **Jetson-native runtime** | Isaac + ROS 2 + Aegis gateway all on a Jetson Orin — deploy the same safety config from sim to real robot |
| **Safety config as versioned artifact** | `franka.yaml` safety limits are versioned alongside the policy. Deploy policy v3 + safety config v3 together. Roll back both. |
| **OTA safety updates** | Push new safety limits without touching the policy. A safety engineer can tighten limits on 100 robots overnight. |

### 5.2 Edge Inference Pipeline

| Component | What It Does |
|---|---|
| **Policy server / sidecar** | Policy runs as a separate process (or separate accelerator), sends actions to the safety gateway via localhost/ROS 2 — decouples policy lifecycle from safety |
| **Inference budget enforcement at edge** | If the policy takes too long, the gateway cuts it off and falls back. Prevents a stuck model from hanging the robot. Already spec'd as `inference_budget_ms` — make it real at the edge |
| **Deterministic replay / shadow mode** | Deploy a new policy in "shadow mode" — it sees real sensor data, produces actions, but the safety gateway only logs them, never executes. Compare shadow vs active policy before switching over |
| **Continuous validation in production** | Every real-robot action passes through the safety gateway and gets logged. If violation rate spikes, the system flags it before it becomes an accident. **The harness never turns off.** |

### 5.3 Fleet-Level Observability

| Idea | What It Means |
|---|---|
| **Per-robot report streams** | Each robot streams its telemetry to a central collector. See every robot's safety record in one dashboard |
| **Anomaly detection across fleet** | Robot #47 shows velocity violations that robot #46 never does → something is wrong with #47 (calibration? wear? obstacle?). Pattern detection across the fleet |
| **Sim-to-real drift monitoring** | Compare each robot's real-world violation/latency profile to its sim profile. Drift = something changed (hardware wear, environment, policy degradation) |
| **Compliance reporting** | For regulated industries (healthcare, manufacturing), generate an auditable "safety record" for every robot, every day — automatically from telemetry. **This is a product for hospitals, factories.** |

This is where Aegis becomes a **closed-loop learning system**, not just a checker.

### 4.1 Closed-Loop RL Training Environment

**Key insight:** You already have the env (`MujocoPickPlaceEnv`), the safety gate, the telemetry, and the report schema. RL inserts a training loop between eval episodes.

| Enhancement | What It Is | Why It Matters | How It Fits Aegis |
|---|---|---|---|
| **Reward shaping** | Add dense rewards on top of existing sparse success metric (`grasped_ever && dist≤0.05m`) | RL needs rewards — approach bonus, alignment bonus, grasp force bonus | Reuse existing success criteria as sparse reward; add shaped components |
| **Episode rollouts for BC/RL** | Store full trajectories for behavior cloning or RL training | Every eval episode becomes training data | Extend `trajectory.jsonl` + `steps.jsonl` to store complete rollouts with states, actions, rewards |
| **Contextual bandit for safety limits** | Learn which safety limits are too tight vs too loose per task/policy | Automates the recommendation beyond 4 hand-written rules | Use existing recommendation features (`violations_per_ep`, `recoveries_per_ep`, `success_rate`) as bandit reward signal |
| **Multi-task env** | Extend beyond pick-place: reach, push, stack, etc. | Real robots do more than one task | Extend `configs/tasks/` — same env, different goals |
| **Domain randomization for training** | Wider DR for robustness during training (not just eval ablation) | Policies trained with DR generalize better | Already have MuJoCo DR — expand parameter ranges, add to Isaac when real |

**Safety guarantee:** The Safety Gateway is **always on during training**. No broken gears. No unsafe contacts. RL training is as safe as eval — because it's the same gateway.

### 4.2 Dataset Export → Post-Training Flywheel

You already spec'd and deferred this (`docs/data-pipeline-decision.md`). To go from eval to platform, **this is required.**

| Component | What It Does | Trigger |
|---|---|---|
| **Embodiment registry** | Maps robot-specific conventions (RPY order, gripper scale, joint limits, action space) to canonical form | T3: second embodiment (humanoid, WidowX, etc.) |
| **LeRobot-compatible export** | Writes rollout data (state, action, image, reward, done) into parquet + metadata that post-training pipelines consume | T4: post-training loop starts |
| **Image logging** (flagged, storage-bounded) | Captures camera frames at key moments (grasp attempts, violations, chunk boundaries) — needed for VLA retraining | T2: image logging lands in RunLogger |
| **`data-check` tool** | Validates exported datasets before training — catches bad rollouts, missing fields, format drift | T4: post-training loop starts |
| **First real GR00T run** | Validates RPY order + gripper scale as facts, not guesses | T1: first real-weights eval |

**The flywheel:**
```
Eval run → logged rollout → dataset export → finetune policy → re-eval through safety gate → better policy → repeat
```

Every eval produces a dataset. Every dataset can improve a policy. Every improved policy goes back through the same safety gate. **Aegis becomes the loop, not just the check.**
| **Telemetry** | ✅ Complete | NDJSON → Prometheus/Grafana/OTel exporters |
| **Recommendation** | ✅ `heuristic_v1` | 4 hand-written rules with structured `{label, confidence, evidence}` + `features` |

**What's missing for the platform story:** Phase 3 real runtime (Isaac Sim/Lab PhysX+USD, real ROS 2, hardware dry-run), dataset export, RL training loop, edge deployment runtime.

---

## 2. The Platform Vision — Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│                   AEGIS PLATFORM                            │
│                                                             │
│  EVALUATE (today)  ──▶  TRAIN (RL + data)  ──▶  DEPLOY    │
│  safety-gated        closed-loop policy     on-robot       │
│  benchmarking         learning              safety +       │
│                       + dataset export       telemetry      │
│                                                             │
│  ◀──────────────────────────────────────────────────────   │
│  Continuous feedback: deploy telemetry → eval + retraining │
└─────────────────────────────────────────────────────────────┘
```

The **Safety Gateway is the thread that runs through everything**: sim, training, edge deployment, compliance reporting. It's always on. It always logs. It never lets an unsafe action reach hardware.