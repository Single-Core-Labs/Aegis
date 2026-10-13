## What You've Built

**Aegis** is a safety gate + test harness for AI robot controllers. It sits between the AI brain and the robot body, checking every action for safety before it reaches the motors, and producing standardized test reports. Right now it works with simulation (MuJoCo + Isaac Lab fallback) and integrates real VLA models like SmolVLA and GR00T.

---

## What More You Can Add

Here's everything organized by what's **ready now** vs. what needs **hardware/sim setup first** vs. **future ideas**.

---

### 🟢 Ready to Build Now (MuJoCo-only, specs exist)

These are already spec'd out and just need implementation on the MuJoCo side:

| Feature | What It Is | Why It Matters |
|---|---|---|
| **Parallel episode batching** (`--parallel N`) | Run multiple episodes at the same time instead of one-after-another | Faster evaluations — currently `eval-batch` is sequential only |
| **Learned recommendation classifier** (`learned_v1`) | Currently you have 4 hand-written rules that say "ready" / "tighten limits" / "fix policy". Train a real ML model on your logged run data to do this automatically | Smarter, more nuanced recommendations as you collect more data |
| **Dataset export** | Export the runs you've logged into a format you can use for training or analysis later | Turns your eval runs into training data for post-training loops |
| **More policies** | Add more VLA/controller policies beyond SmolVLA + GR00T + scripted + random | Compare different AI models on the same safety benchmark |
| **More robots** | Add a second robot embodiment (different arm, different joint limits) | Proves the harness works across hardware, not just Franka |

---

### 🟡 Needs Phase 3 Real Setup First (WSL2 + Isaac Sim + ROS 2)

These need the real NVIDIA stack running before they can work:

| Feature | What It Is | Blocker |
|---|---|---|
| **Real Isaac Sim / Isaac Lab** | Replace the MuJoCo fallback with actual PhysX + USD simulation | Needs WSL2, Isaac Sim 6.0.1 installed, USD scene authored, 6GB VRAM-safe tuning applied |
| **Real ROS 2 Humble** | Connect to a real robot via ROS 2 topics instead of the mock bridge | Needs `rclpy`, WSL2 Ubuntu 22.04, `ros2 topic list` working |
| **Hardware dry-run + Go/No-Go** | Test on the actual Franka Panda robot with E-stop, safety barriers, NaN injection test | Needs Gates 1-4 from PRD §12 passed — physical safety setup |
| **Prometheus / Grafana live dashboard** | Real-time metrics dashboard instead of just offline JSON exports | Needs real ROS 2 latency data to be meaningful |
| **Isaac 6GB perf numbers** | Measure how fast SmolVLA runs on RTX 4050 (6GB) with VRAM-safe settings | Needs real Isaac + CUDA on the target hardware |

---

### 🔵 Spec'd / Backlog Ideas (Beyond Phase 3)

These are written up in docs but not yet prioritized for build:

| Feature | What It Is | Value |
|---|---|---|
| **Camera calibration (P3a)** | Match simulator camera positions to LIBERO training camera positions | Could close the vision gap that makes SmolVLA fail (0/3 → hopefully much better) |
| **Domain randomization expansion** | Already have light/friction/camera jitter in MuJoCo. Expand to more parameters, or add to Isaac | Makes policies more robust, better sim-to-real transfer |
| **Multi-robot fleet** | Run evaluations across multiple robots at once | Scales up testing throughput |
| **RL training loop** | Use the harness + logged data to train policies with reinforcement learning | Turns Aegis into a training platform, not just eval |
| **Contact-force grasp detection** | Already started — refine to detect and log grasp quality from force data | Better success measurement than just "close to target" |
| **Learned safety filter** | Currently safety checks are hand-written rules (interpretable). A learned filter could catch more subtle issues | More comprehensive safety, but less auditable — that's why it's not the default |

---

### 📋 Quick Priority Order

If I had to rank by **impact per effort** for where you are today:

1. **Parallel batching** (`--parallel N`) — big time savings on eval, MuJoCo-only, well-spec'd
2. **Learned recommendation** (`learned_v1`) — features are already logged, just needs training data + a classifier
3. **Dataset export** — turns your eval outputs into training data, unlocks post-training flywheel
4. **Second robot embodiment** — proves generalizability, makes the harness more credible
5. **Phase 3 real** (Isaac + ROS 2 + hardware) — biggest leap, needs environment setup first
6. **Camera calibration + expanded DR** — could meaningfully improve SmolVLA success rate
7. **RL training / dashboard / fleet** — bigger projects, feed off the above

---

### The Big Picture

Aegis is currently a **sim-only evaluation safety harness**. The natural progression is:

```
Sim safety gate (done) → Real sim (Isaac) → Real robot (ROS 2 + Franka) → Fleet scale
                        → Learn from runs (recommendation, dataset, RL)
```

Everything beyond Phase 3 is about **making the harness more useful**: faster, smarter recommendations, more robots/models, and eventually closing the loop to actually train better policies using the data you collect.