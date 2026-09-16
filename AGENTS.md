# AGENTS.md — Quantization Deployment Boundary

This project studies how far an LLM should be quantized before the additional serving benefit no longer justifies the additional degradation in model behavior. The experiment compares complete deployment configurations of the same base model across a candidate precision ladder of BF16 -> FP8 -> FP4, measuring both model-quality degradation and serving improvement under a fixed deployment environment. The project is measurement-driven: no precision level, quantization scheme, or model is required to win, and the measured tradeoff is itself the result.

## Git

- **Never add Claude as an author or co-author on commits or pushes.** The user is always the sole author — no `Co-Authored-By: Claude` trailer and no `Generated with Claude Code` line.
- Inspect the repository before assuming the default branch, remote, existing layout, or build/test commands.
- Do not commit benchmark claims that are not backed by tracked result artifacts.

## Doc map

This file is the always-loaded hub. Keep detailed methodology and measured state in `docs/`.

- **`docs/PROJECT_SPEC.md`** — authoritative six-part research design and scope. *Read before changing the research question, model strategy, study dimensions, success criteria, or final deliverable.*
- **`docs/HARDWARE_PROFILE.md`** — measured machine profile and the constraints it places on the study. *Read before choosing models, precision formats, inference backends, or interpreting serving results.*
- **`docs/DECISIONS.md`** — current project decisions, open design gates, and the reasoning behind them. *Read before treating any design choice as settled or reopening a previously resolved question.*
- **`docs/QUANTIZATION_CONFIGS.md`** — reproducible definition of every deployment configuration. *Read before creating, loading, or benchmarking a BF16/FP8/FP4 checkpoint.*
- **`docs/EXPERIMENTAL_CONTRACT.md`** — controls, workload definitions, run validity, warmup, repetition, telemetry, and comparability rules. *Read before collecting timing or memory results.*
- **`docs/EVALUATION_RIG.md`** — quality metrics, serving metrics, uncertainty treatment, and marginal tradeoff calculations. *Read before implementing or changing evaluation code.*
- **`docs/LIMITATIONS.md`** — claims the study cannot make and known threats to validity. *Read before interpreting final results or writing conclusions.*

**Code layout.** `scripts/harness/` is the shared measurement harness — `common.py` (identity,
telemetry, statistics), `driver.py` (one timed cell), `server.py` (vLLM lifecycle),
`orchestration.py` (cell identity, resume, launch preflight). `run_pilot.py` and `run_sweep.py` are
runners over it; `analyze.py` turns pilot cells into the pilot verdict; `analyze_ceiling.py`
adjudicates the SLO-ceiling replication against its pre-registered criterion; `selftest.py` exercises
the cell state machine and the ceiling phase against a stub engine with no GPU. Renamed from `scripts/pilot/` on 2026-08-24 —
the harness outlives the pilot.

`scripts/harness/quality/` is the quality arm, KL first — `positions.py` (retained-position
contract), `kl_math.py` (float64 KL, trajectory bootstrap), `qcommon.py` (identity, provenance,
`KL_SPEC`), `qengine.py` (engine lifecycle, observed identity), `trajectories.py` (the frozen BF16
continuations), `collect_kl.py`, `analyze_kl.py`, `gates.py`, `preflight.py` (P12), `floor_study.py`
(the production-scale replication floor), `launch_variance.py` (BF16 launch identity as a nuisance
variance component — **proposed, not adopted**; supplements `analyze_kl.py` and changes nothing it
owns), `qselftest.py`. It imports `common.py` and
`server.py` and edits neither: the serving contract is frozen. `scripts/logits_probe.py`, `scripts/compute_kl.py` and
`scripts/harness/correctness_gate.py` are **historical qualification/prototype paths**, kept
byte-unchanged so `results/qualification/` and `results/pilot/` stay reproducible. They are not the
quality runner and their KL numbers are not quality results.

## Research contract

The unit under study is a **complete deployment configuration**, not a precision label in isolation. A configuration includes, where applicable:

- base checkpoint and tokenizer
- weight precision
- activation precision
- KV-cache precision
- quantization algorithm / format
- calibration procedure
- group or block size
- inference backend
- kernels actually exercised on the target GPU

For the same base model and equivalent workloads, compare successive configurations and measure:

```text
quality change + serving change
        -> marginal tradeoff
        -> deployment boundary / knee
```

The core question is not “is FP4 faster than BF16?” It is:

> What do we gain by taking the next quantization step, what do we lose, and is the additional serving gain still large enough to justify the additional quality degradation?

Detailed definitions live in `docs/PROJECT_SPEC.md` and `docs/EVALUATION_RIG.md`.

## Target regime / scope

The primary study is deliberately narrow:

```text
Hardware:       one fixed, profiled GPU machine
Precision:      candidate BF16 -> FP8 -> FP4
Reference:      BF16 deployment of the same base model
Serving:        single-GPU inference
Workloads:      prefill-heavy, balanced, decode-heavy
Concurrency:    low load through saturation
Multi-GPU:      out of scope for the primary study
```

A model ladder is **not required** to answer the core within-model tradeoff question. Additional models are only justified if the project explicitly expands to test whether the measured boundary generalizes across model sizes or families. See `docs/DECISIONS.md`.

## Core hypothesis

> Under a fixed model, hardware environment, serving stack, and workload contract, the quality loss and serving benefit of successive quantization steps can be measured precisely enough to determine whether further quantization remains worthwhile.

A valid result may favor BF16, FP8, FP4, different choices for different workloads, or conclude that the differences are not resolvable with sufficient confidence.

**What it cannot claim.** A boundary measured on this machine, model, backend, and workload is not automatically a universal boundary for other hardware, models, quantization algorithms, or serving stacks.

**The comparison bar is the same base model under the same serving stack and equivalent workload**, not theoretical compression ratios, vendor peak throughput, or a weaker implementation path.

## Workflow rules

### Plan before acting
- Enter plan mode for non-trivial work or architectural / experimental decisions.
- Prefer the cheapest measurement that can invalidate an assumption before building a large harness around it.
- If evidence contradicts a current design claim, update the tracked doc that owns that claim instead of preserving both versions.

### Measurement discipline
- Never quote a benchmark number that is not in a tracked result artifact; name the artifact when using the number in documentation.
- Distinguish measured facts from assumptions, estimates, and vendor/backend capability claims.
- Do not infer usable acceleration from a datatype name alone. A quantization format is part of the study only after the selected backend is verified to execute an appropriate path on the target GPU.
- Before timed runs, verify the GPU is otherwise idle and record enough telemetry to detect power, thermal, clock, or memory-state contamination.
- Do not compare runs produced under different workload definitions, tokenizer/input data, generation lengths, scheduler settings, cache policy, or software stack unless the difference itself is the intended experimental variable.
- Warmup, repetition count, saturation criterion, and run-validity rules belong in `docs/EXPERIMENTAL_CONTRACT.md`, not ad hoc benchmark scripts.

### Verification before done
- A new quantization or serving path is not “supported” until it is exercised successfully on the target machine.
- A benchmark path is not “correct” until outputs are validated against the relevant reference before timing.
- A result is not final until the raw artifact, configuration, and reproduction command are tracked.
- Before quoting a committed number, verify the current tree still reproduces the configuration that generated it.

### Scope discipline
- Do not turn the project into a broad model benchmark unless a tracked decision explicitly changes the research question.
- Do not add quantization methods merely because they are popular. They must answer a research need and have a fair serving implementation on the target hardware.
- Do not optimize kernels or inference code as the primary project goal. Optimization is only relevant when needed to make deployment configurations fairly comparable.

## Code style: comments

- No paragraph-style or multi-line block comments explaining what code does.
- Comments only where intent is not obvious from the code itself: non-obvious tradeoffs, gotchas, or why a seemingly natural alternative is wrong.
- Keep comments short and explain **why**, not **what**.
- Do not duplicate configuration values in comments when they already live in a config or tracked specification.

## Current focus

**The D11 serving sweep is COMPLETE** (124 cells, 14.75 h, `results/sweep/`, zero defect cells).
Headline: maximum concurrency within the 50 ms TPOT P95 SLO is **21 / 57 / 70** for BF16 / FP8 / FP4
(refined bisection, **n=1**), i.e. FP8 at 2.71x and FP4 at 3.33x BF16, with KV-pressure walls at
[17,18], [38,39], [47,48] predicted exactly by peak-footprint arithmetic. The SLO ceiling is not the
wall: every configuration serves past its wall before breaching latency. Below-wall throughput
differences are reproducible, concurrency-dependent and **unattributed** — quote the concurrency with
the number, never as a bandwidth or weight-residency benefit.

**The quality arm is built and smoke-validated. The 64-trajectory production run has NOT been run.**

```text
0-4. serving sweep + refinement                                DONE 2026-08-25
4b. ceiling replication                                        PARTIAL 2026-09-14, 1/18 cells
5. quality run
   P0-P6  contract, numerics, engine lifecycle, G7, G9         DONE 2026-08-25
   P7     64 BF16 trajectories frozen                          DONE 2026-08-26
   P8-P9  collect_kl.py / analyze_kl.py                        DONE 2026-08-26
   P10    real smoke, 4 traj x 10 pos x 3 configs              DONE 2026-08-26
   P11    replication floor + fp16/fp32 storage gate           DONE 2026-08-26
   P12    full preflight, 35/35                                DONE 2026-08-26
   G2'    production BF16 floor, 3 launches x 640 cells        DONE 2026-08-26
   review harness audit, 7 defects closed                     DONE 2026-09-01
   D13-4  BF16 launch variance, estimator + tests + results    PROPOSED 2026-09-12, NOT adopted
   P13    64-trajectory production KL run                      <- next, NOT authorised
          BLOCKED on the replication-floor disposition, not on the rig
   PPL and downstream tasks still need D14/D15
6. marginal tradeoff: quality loss vs sustainable concurrency
```

### Locked engine profile — `graph_2048` (G9)

`enforce_eager=False`, `max_num_batched_tokens=2048`, prefix caching on, `detokenize=False`. Chosen
because it is the only profile reproducing the serving sweep's KV capacities for all three
configurations (44,688 / 97,888 / 120,944). **Neither control is numerically inert**: flipping
`enforce_eager` moves FP4 by 3.74e-02 nats and its argmax on 4 of 40 cells; 2048-vs-8192 moves FP4
by 2.15e-02 and 6 of 40. BF16 sits at its own floor under both. See `LIMITATIONS.md`.

### Smoke KL, 4 trajectories x 10 positions (NOT a result — n=4)

```text
              headline nats     95% CI                    floor      signal/floor
BF16||FP8        3.690e-03      [1.748e-03, 6.016e-03]    2.08e-04        17.7x
BF16||FP4        2.945e-02      [2.315e-02, 3.576e-02]    2.08e-04       141.3x
FP8||FP4         3.566e-02      [2.627e-02, 4.506e-02]    3.91e-11       9.1e+08x
```

Floors are the **production-scale** BF16 figure (3 launches, 640 cells); the KL values are still n=4.

The barred subtraction proxy would have put FP8->FP4 at 2.576e-02, low by a factor of 1.38.

### What the gates returned

```text
G7  numerics vs the historical EPS formula      PASS
G9  engine profile                              MEASURED, profile locked
    replayability                               NOT REPLAYABLE (51/64) -- informational
G2  replication floor (n=4)                     FAIL on BF16 (8.1% of FP8 signal vs 1% bound)
G2' replication floor, 3 launches x 640 cells   FAIL on BF16 (5.6%); floor 2.084e-04
G3  cache equivalence                           FAIL on BF16 (4.74%), below BF16's own floor
G4  fp16 vs fp32 storage                        FAIL -- fp32 stays
```

Two pre-registered bounds fail, both on BF16 and both traceable to one cause: **BF16 is the
non-reproducible configuration.** Thresholds were not relaxed after seeing the results.

The earlier "BF16 reproduces 0 of 40 cells" figure was pair-specific small-sample noise and is
**superseded**: at production scale BF16 reproduces 28-39% of 640 cells across launches, and one of
the six pairs reproduces 20 of the same 40 cells the smoke scored 0 on. What is robust is that the
floor is worst at short contexts (3.78e-04 at p=1, 3.01e-05 at p=2048) and is carried by a small
number of unstable cells rather than uniform jitter.

## Last session

**Session 10 — started the ceiling replication (it was killed at 1 of 18 cells), and synthesised
what the BF16 tests have actually established.** No new BF16 serving cells exist.

- **Ceiling replication is PARTIAL** (`175ae12`). `run_sweep.py --job ceiling` ran, completed
  `FP8 C=56 rep2` (48.774 ms, SLO True, 1402.4 tok/s), and took SIGTERM partway through C=57;
  engines tore down cleanly through the finally blocks and the GPU is idle. The landed cell
  replicates its rep-1 counterpart to **0.18 ms** (48.774 vs 48.958) and 0.4% on throughput.
  **The two cells that actually fix K=57 — C=57 and C=58 — did not run.**
- **Resume is clean; the dry-run's count is not.** `replicate_ceiling_group` consults
  `orch.done_keys` before each group and again per cell, so C=56 is skipped rather than
  duplicated. But `ceiling_dry_run` reports the static plan and still prints "18 cells, 6
  launches" — cosmetic, and misleading in exactly this situation. **17 cells remain.**
- **The client-headroom question is answered on the cells that carry it.** Driving FP8 at C=56 and
  1402 tok/s the client used `client_cpu_cores = 0.021` — ~2% of one core on a 32-core host —
  with `host_cpu_busy_frac = 0.0348`. Not close to limiting. The completed sweep's 124 cells still
  carry none of this.
- **The BF16 synthesis, from tracked artifacts.** BF16 is simultaneously the **least** reproducible
  configuration on the quality axis and the **most** reproducible on the serving axis, and the two
  are not in tension — they measure bitwise determinism of a 128k logit vector versus an aggregate
  P95 latency. Quality: BF16 self-KL 2.084e-04 against FP8 1.51e-11 and FP4 3.91e-11, a factor of
  **1.4e7**; FP8/FP4 are exactly 0.0 at every retained position but p=1 and p=2048, BF16 nowhere.
  Serving: three reps at C=16 give 32.599 / 32.578 / 32.607 ms, a **0.029 ms** spread, and K=21
  clears the SLO by 2.41 ms — about 83 noise widths. **BF16's ceiling is the one that never needed
  replicating; FP8 (0.34 ms) and FP4 (0.43 ms) are why the pass exists.**
- **G3 fails on a number below its own floor.** Cache-on-vs-off moves BF16 by 1.749e-04 = 4.74% of
  the FP8 signal against a 1% bound, but BF16's replication floor is 2.084e-04. The cache effect is
  not resolvable from relaunch noise, so G3 is a measurement about the rig, not about caching.
- **Docs correction** (`569946c`). My earlier "committed numbers are not bit-reproducible" was an
  environment mistake, not a property of the artifacts: under `envs/qnt` they reproduce exactly.

## Known issues / unresolved premises

- **The BF16 replication floor fails its pre-registered bound at production scale**: 2.084e-04 nats
  from 3 launches over 640 cells, i.e. **5.6%** of the n=4 BF16→FP8 KL against a 1% bound, leaving
  that comparison 17.7x above noise. BF16→FP4 passes at 0.7%. Worse per position: on the provisional
  FP8 curve the floor is 95% and 148% of the signal at p=2048 and p=512. This remains the binding
  constraint on the quality axis and is a property of the *reference*. **Four dispositions are now
  on the table and none has been taken**: accept it as a stated resolution limit; restrict FP8
  claims to the positions that clear it; re-register an averaged reference (barred as written);
  or D13's fourth — carry BF16 launch identity as a nuisance variance component, which is built,
  reviewed and measured but **not adopted**.
- **The floor is a range, not a point** (measured 2026-09-12). Six ordered pairs from three launches
  are a U-statistic carrying **2 df**, not five, so the naive over-pairs SE understates the floor's
  own uncertainty **3.2x**. A delete-one-launch jackknife gives **2.084e-04, 95% CI
  [1.16e-04, 3.01e-04]** — "5.6% of the FP8 signal" is really **3–8%**, and the n=4 figure of
  2.984e-04 that the hub calls superseded sits **inside** that interval. Anywhere the floor is
  quoted as a point, it is being quoted more precisely than it was measured.
- **The floor is not the only reference-side nuisance, and not always the larger one.** The spread
  of per-launch BF16→FP4 headlines (5.18e-04) **exceeds the floor (2.08e-04) by 2.5x**, while for
  BF16→FP8 it sits below it — the two diagnostics do not track each other and their ordering flips
  across the ladder, because the floor is second order in the launch perturbation and the signal's
  launch sensitivity is first order. "BF16→FP4 sits 141x above the floor" understates the
  reference-side nuisance on that pair by ~2.5x.
- **P13 currently plans one BF16 launch, and that carries unmodelled variance.** Projected to 64
  trajectories, BF16 launch identity would contribute ~10% (FP8) and ~15% (FP4) of the headline
  variance at R=1, falling to ~3–6% at R=3. Whether to add BF16 launches at P13 is open and is
  cheap to decide now: the prediction `SD_launch = sigma_proj * sqrt(2 * signal)` with
  `sigma_proj ≈ 2.1e-03` is registerable before the run.
- **Seeded generation is not replayable** — 51 of 64, earliest divergence at token 3. Reproduction
  goes through the tracked `trajectories.json` and its hash, never by rerunning generation.
- **Every quality figure so far is n=4 and is not a result.** The smoke exists to validate the rig.
  That includes every BF16→FP8/FP4 number in `launch_variance.json`: FP8 and FP4 full-vocabulary
  distributions exist only on the smoke's four trajectories. Only the BF16↔BF16 arm there is
  production scale.
- **Quality work runs under `envs/qnt`, not the login shell's `python3`.** The pinned environment
  reproduces every committed number bit-identically; the default one drifts 3.3e-14 and imports
  cleanly enough to do it silently on analysis-only paths. The stored distributions, not the
  derived summaries, are the durable artifact — `dist/*.npy` is gitignored while a BF16 launch
  provably does not regenerate itself, so `launch_variance.json` records the SHA-256 of every
  matrix it consumed.
- **The refined serving ceilings are still n=1** (21 / 57 / 70) and the FP8/FP4 margins are thin:
  C=K clears the 50 ms bound by 0.34 ms (FP8) and 0.43 ms (FP4) against a matched-cell spread that
  BF16 measures at **0.029 ms** (three reps at C=16: 32.599 / 32.578 / 32.607) — so those two are
  ~12 and ~15 noise widths, not three to four, but still the thin ones. **BF16's 2.41 ms is ~83
  noise widths and is not in question.**
- **The ceiling replication is PARTIAL — 1 of 18 cells, and the landed one is FP8 C=56, not a
  cell that fixes any K.** `FP8 C=56 rep2` measured 48.774 ms against rep 1's 48.958, a 0.18 ms
  replication. **17 cells remain, ~3.5 h.** Resume and finish with:

  ```text
  /home/sathvik/miniconda3/envs/qnt/bin/python scripts/harness/run_sweep.py --job ceiling
  /home/sathvik/miniconda3/envs/qnt/bin/python scripts/harness/analyze_ceiling.py
  ```

  Resume is safe — `replicate_ceiling_group` checks `orch.done_keys` per group and per cell, so the
  landed cell is skipped rather than duplicated. **`--dry-run` still prints "18 cells, 6 launches"**
  because `ceiling_dry_run` reports the static plan without consulting `done_keys`; cosmetic, but it
  will misstate the remaining work to whoever checks next. The pass is pre-registered in
  `EXPERIMENTAL_CONTRACT.md`; `analyze_ceiling.py` adjudicates against that criterion and has never
  been run on real cells.
- **The run must be launched under `envs/qnt`.** The login shell's `python3` (3.13) cannot import
  vLLM at all. Verify the GPU is idle and cool first — it was 15 MiB / 0% / 34 °C at handoff.
- **`PREFILL_PROBE` inherited the decode SLO** and lost BF16's C=8 point; it needs a TTFT-based
  criterion before it is re-run.
- **The below-wall throughput gap is unattributed**, and that is a settled position.
- **Latin-square carryover is unbalanced**; queue *time* is still not recorded, only depth;
  `meets_slo` is survivor-biased.
- **No perplexity or downstream-task axis yet.** D14 and D15 are open; chat-formatted tasks stay
  blocked by the `chat_template` deviation.
- **Checkpoint provenance has one open deviation** (shorter `chat_template`), and **FP4 calibration
  sensitivity is untested** — one draw, 128 ultrachat samples, seed 0.
- **Three dispatch-verification gaps, found by the harness audit and deliberately not fixed.**
  `server.py:137,142` carries the same truncate-before-normalise pattern that was fixed in
  `qengine.observed_identity` — it sits on the frozen serving path and was left alone. FP4's
  forbidden list contains `"emulation"`, which is not in `server.KERNEL_PATTERNS`, so that pattern is
  unenforceable. And BF16 logs contain no matching kernel lines at all, so the reference
  configuration's `dispatch_verdict.ok` is satisfied by silence. None blocks P13; all three should be
  settled before the serving numbers are written up.
- **`results/quality/preflight.json` is one key and one check stale** — `every_cell_rederive_checked`
  is superseded by `contract_enforced_by_build_all`, and the count moves 35 -> 36 with engine.
  Regenerating it costs three GPU launches. **`results/quality/gates/engine_profile/manifest.json`
  predates the current spec** (`4ef13273db16d285`) and aborts on the KL_SPEC guard before any other
  check.
- **Any re-run of `collect()` dirties the tree**: `collection_<short>.json` is tracked and carries a
  fresh `timestamp`, so `floor_study.py` without `--analyze-only` aborts at launch 2's
  `require_clean_tree`. Pre-existing and separate from the resume-provenance fix.
- **`results/quality/smoke/kl_summary.json` still carries the n=4-floor ratios** (12.37x / 98.72x)
  while the hub quotes the production-floor ones (17.7x / 141.3x). Regenerating it is now possible
  and would change a committed number, so it needs its own decision.
- **The completed sweep carries no host/client CPU telemetry**, so for the 124 cells behind the
  21 / 57 / 70 headline the claim that the GPU rather than the benchmark client was limiting is
  still an argument. **The one replication cell that carries it puts the client at 2% of one core**
  (`client_cpu_cores` 0.021, `host_cpu_busy_frac` 0.0348, at FP8 C=56 / 1402 tok/s) — strong
  evidence the argument was right, but it is one cell at one concurrency and the sweep's own cells
  remain uninstrumented.
- **BF16 splits across the two axes, and that split is the main BF16 result so far.** It is the
  least reproducible configuration on quality — self-KL 2.084e-04 against FP8 1.51e-11 and FP4
  3.91e-11, a factor of 1.4e7, and bit-identical in only 28-39% of 640 cells where the quantized
  paths are exactly 0.0 almost everywhere — and the most reproducible on serving (0.029 ms across
  three reps). **The mechanism is unconfirmed**, and the harness currently cannot confirm it:
  BF16's `dispatch_verdict` is satisfied by silence (empty `kernel_lines`) while FP8 and FP4 each
  log one line naming a fixed Cutlass/FlashInfer kernel, so there is no positive evidence of which
  GEMM BF16 dispatched. A split-k or atomic reduction is the plausible story and is **not
  measured**. This is the same gap listed under dispatch verification below, but it now has a
  result resting on it.
- **The GPU is power-limited at 145 W**, so every number is measured under a power ceiling.

At the end of a session, overwrite `Current focus`, `Last session`, and `Known issues / unresolved premises` in place. Git history is the changelog; this file should remain a current-state hub.
