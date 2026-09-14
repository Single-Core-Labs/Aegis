# Data Pipeline (Embodiment Registry + LeRobot Export): Do We Need It?

> **Status:** DECIDED 2026-09-14 — **DEFER**. A full `src/aegis/data/` package
> (registry + schema + mapping + export + `data-check`) was implemented and then
> **reverted** before merge. This doc records why, and the exact triggers that
> would reverse the decision. Nothing in this doc is built; everything cited as
> "existing" can be verified in the tree.

---

## 1. What was built and reverted

One session produced, then deleted: `src/aegis/data/{__init__,embodiments,schema,
mapping,export,check}.py`, `tests/test_data.py` (11 tests, all passing at deletion),
`aegis export` + `aegis data-check` CLI commands, extra trajectory keys in
`EvalRunner`, and a registry-driven rewrite of `policies/groot.py`. The suite was
28/28 green both with and without it. The revert was clean: no references remain,
`groot.py` is back to explicit constants, trajectory logs keep their 3-key shape.

## 2. What is REAL about the problem (evidence, not vibes)

- **Ambiguity is real out there.** Isaac-GR00T's data layer exists for exactly this:
  `ModalityConfig` (key lists + horizons + normalization scheme per embodiment),
  `EmbodimentTag` registry with PRETRAIN/POSTTRAIN/FINETUNE_ONLY categories,
  `VLAStepData` canonical step, `pose.py` rotation-order handling, `stats.py`
  per-key statistics, plus `repair_lerobot_metadata` / `validate_hf_config_alignment`
  tools. A frontier lab hit this wall and built machinery. We will too — eventually.
- **We already have the first symptoms.** `groot.py` carries two hardcoded,
  unvalidated conventions: RPY order (`Rz*Ry*Rx`) and gripper scale
  (finger-mean/0.04 → [0,1]). Today that's 2 constants with comments. At 3
  embodiments it becomes a bug farm — the registry idea is the right *shape*.

## 3. What is NOT yet true (why building now is premature)

1. **One consumer, zero weights.** The only reader of any registry would be the
   GR00T adapter, which has never run against real weights. Codifying the RPY
   order and gripper scale into an authoritative-looking "registry" before a
   single real inference would emboss guesses as architecture. The comments in
   `groot.py` are more honest than a registry row right now.
2. **The export's main payoff is unreachable.** The LeRobot export wrote valid
   parquet + stats — but RunLogger records no camera frames, so no `videos/`
   dir. State+action export serves failure mining, which today is a 5-line
   `steps.jsonl` grep. VLA retraining (the reason to export) needs image logging
   first, which is a separate, unbuilt feature.
3. **No second embodiment.** `franka_pick_place` + `libero_sim` are the same robot
   seen twice. A mapping layer earns its keep at 2+ genuinely different
   conventions (e.g. a humanoid tag, or WidowX-style keys). Until then it is
   abstraction without a second use case — exactly what `agent.md` scope
   discipline forbids.
4. **Maintenance without users.** Stats formats, LeRobot v2-vs-v3 layout drift,
   MP4 encoding (no encoder in this env: no imageio/av) — all real costs, paid
   immediately, for zero current readers.

## 4. Decision: DEFER, with triggers

Do not build `src/aegis/data/` until **any** trigger fires:

- **T1 — First real GR00T run.** A real-weights eval validates (or corrects) the
  RPY order and gripper scale. *Then* those constants graduate from comments
  into a registry, because they'd be facts.
- **T2 — Image logging lands in RunLogger.** Export gains its `videos/` dir and
  the retraining payoff unlocks. (Requires an encoder dep decision: imageio/av
  vs torchcodec, plus storage budget — frames dwarf NDJSON.)
- **T3 — Second embodiment.** A humanoid tag, WidowX keys, or a second policy
  family needing the same mapping. Two consumers justify the abstraction.
- **T4 — Post-training loop starts.** `launch_finetune.py`-style flow needs
  dataset stats + format guarantees — i.e. `data-check` earns its existence.

## 5. What we keep instead (cheap, honest)

- The two assumptions stay as **comments + named constants** in `groot.py`
  (`FINGER_QPOS_RANGE_M`, RPY order note) — visible, grepable, zero machinery.
- The embodiment findings stay recorded: `docs/groot-integration-plan.md`
  (POSTTRAIN-only `libero_sim`, no `LIBERO_PANDA` in the base checkpoint) and
  `configs/models/groot_n17.yaml` header.
- The reverse-engineering notes in §2 above, so the rebuild starts from
  conclusions, not re-discovery.

## 6. Revival plan (when a trigger fires, in order)

1. Registry + mapping only (`embodiments.py`, `mapping.py`), seeded from
   validated constants — no export yet.
2. Rewire `groot.py` onto it (the deleted diff is the template, not the law —
   re-derive from whatever T1 taught us).
3. RunLogger image capture behind a flag (storage-bounded, e.g. chunk-boundary
   frames only, mirroring the SmolVLA pattern).
4. Export + `data-check` (+ `datasets/` layout, encoder dep, `.gitignore` entry).
5. Second-embodiment onboarding as the acceptance test for the whole layer.
