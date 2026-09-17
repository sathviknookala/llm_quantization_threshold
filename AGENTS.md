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
variance component — **adopted for P13**; supplements `analyze_kl.py` and changes nothing it owns),
`dispatch_verify.py` (additive positive dispatch evidence — the inherited verdict is satisfied by
silence for BF16 and cannot enforce FP4's `"emulation"`), `p13.py` (the registered production run:
`REGISTERED`, collection, and the resolution rule), `qselftest.py`. It imports `common.py` and
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

**The quality arm has a production result. P13 is DONE** (3,200 cells, `results/quality/kl/`,
registration `b42d228a8afdd8df`). **The D11 serving sweep is COMPLETE** (124 cells,
`results/sweep/`). **The serving ceiling replication is still PARTIAL at 1 of 18 cells** and was
deliberately not advanced this session.

```text
                     headline nats     95% CI                       floor fraction (pre-reg / own)
BF16->FP8             5.557797e-03     [3.610931e-03, 8.751935e-03]      3.75%  /  3.36%
BF16->FP4             5.298014e-02     [3.649696e-02, 7.717467e-02]      0.39%  /  0.35%
FP8->FP4  (direct)    5.679499e-02     [3.848252e-02, 8.535338e-02]      n/a -- FP8 is the reference
BF16<->BF16 floor     1.865e-04        [1.002e-04, 2.729e-04]            this run, 3 launches
```

**The ladder is not additive and the direct measurement is what shows it.** `FP8->FP4` exceeds
`BF16->FP4`: paired on the shared 64 trajectories the difference is **3.815e-03 nats, 95%
[4.99e-04, 8.50e-03], P(<=0) = 0.0083**. The marginal intervals overlap; the paired one does not.
The barred subtraction proxy would have said 4.742e-02, low by **1.198x**.

Serving headline unchanged: maximum concurrency within the 50 ms TPOT P95 SLO is **21 / 57 / 70**
for BF16 / FP8 / FP4 (refined bisection, **n=1**), i.e. FP8 at 2.71x and FP4 at 3.33x BF16, with
KV-pressure walls at [17,18], [38,39], [47,48] predicted exactly by peak-footprint arithmetic. The
below-wall throughput gap is reproducible, concurrency-dependent and **unattributed**.

```text
0-4. serving sweep + refinement                                DONE 2026-08-25
4b. ceiling replication                                        PARTIAL 2026-09-14, 1/18 cells
5. quality run
   P0-P12  contract, numerics, engine, gates, preflight        DONE 2026-08-26
   G2'     production BF16 floor, 3 launches x 640 cells       DONE 2026-08-26
   review  harness audit, 7 defects closed                     DONE 2026-09-01
   D13-4   BF16 launch variance as a nuisance component        ADOPTED 2026-09-16
   P13     64-trajectory production KL run                     DONE 2026-09-16
   review  3 reviewers; formulation, implementation, artifacts DONE 2026-09-16
   PPL and downstream tasks still need D14/D15
6. marginal tradeoff: quality loss vs sustainable concurrency  <- next
```

### The floor disposition, taken 2026-09-16

**1 + 4, with 2 as a pre-registered rule; 3 stays barred.** The floor is accepted as a stated
resolution limit, quantified by carrying BF16 launch identity as a nuisance variance component, and
where it bites at a position that position is *named* rather than dropped. **G2 and G2' remain
recorded failures** — the 1% bound was not relaxed, no floor is subtracted from any reported KL, and
no pooled or averaged BF16 distribution exists anywhere. The locked single-reference headline is
still the headline; the launch-averaged value supplements it. See `DECISIONS.md` D13.

### What P13 settled, and what it did not

- **The registered nuisance model is not supported.** Observed pooled `sigma_proj` **6.18e-04**
  against **2.1e-03** registered; launch CV 1.577% / 0.173% against a law-at-observed prediction of
  4.00% / 1.29%; the two configurations' `sigma_proj` differ by **2.93x** where n=4 had them
  agreeing to 3%. The registered verdict is `MIXED`, and that label flatters a conjunction over two
  arms from the same three launches against 12x-wide intervals. Post-hoc and labelled as such: the
  law's sharp SD-ratio prediction is 3.103, observed **1.058**, and the two arms' launch-effect
  vectors are anti-correlated at **r = -0.880**.
- **BF16 launch identity is a small nuisance**: 0.1% (FP8) and 0.0% (FP4) of headline variance,
  against ~10-15% at R=1 from the n=4 projection. Like for like the miss is larger: 0.088% vs 3.70%
  at R=3, 0.26% vs 10.2% at R=1. **R=3 was cheap insurance the data says was barely needed — and is
  the only reason we can say so.**
- **Resolution: FP4 10/10 positions, FP8 9/10**, p=2048 noise-limited (ratio 0.65). The smoke's
  worry that p=512 would be unresolvable does not survive — it resolves at 3.95x. Two caveats:
  **none of the twenty classifications depends on the launch term** (inflation 1.0001-1.0081), and
  at p=2048 the binding uncertainty is **trajectory sampling**, not relaunch noise.
- **"Resolved" licenses less than it sounds like.** It means the divergence exceeds the floor's
  upper launch bound — *not* "separable from relaunch noise". The floor is second order in the
  launch perturbation while the signal is first order.

### Locked engine profile — `graph_2048` (G9)

`enforce_eager=False`, `max_num_batched_tokens=2048`, prefix caching on, `detokenize=False`. The only
profile reproducing the serving sweep's KV capacities for all three configurations (44,688 / 97,888 /
120,944), which P13 reproduced again on all five collections. **Neither control is numerically
inert**: flipping `enforce_eager` moves FP4 by 3.74e-02 nats and its argmax on 4 of 40 cells;
2048-vs-8192 moves FP4 by 2.15e-02 and 6 of 40. See `LIMITATIONS.md`.

### What the gates returned

```text
G7  numerics vs the historical EPS formula      PASS
G9  engine profile                              MEASURED, profile locked
    replayability                               NOT REPLAYABLE (51/64) -- informational
G2  replication floor (n=4)                     FAIL on BF16
G2' replication floor, 3 launches x 640 cells   FAIL on BF16 -- 3.75% of the PRODUCTION FP8
                                                signal against a 1% bound
G3  cache equivalence                           FAIL on BF16 (4.74%), below BF16's own floor
G4  fp16 vs fp32 storage                        FAIL -- fp32 stays
```

Two pre-registered bounds fail, both on BF16 and both traceable to one cause: **BF16 is the
non-reproducible configuration.** Thresholds were not relaxed after seeing the results, and P13 did
not relax them either.

## Last session

**Session 11 — deferred the serving ceiling, took the BF16-floor disposition, and ran P13 to a
result.** No GPU time was spent on serving.

- **The disposition is taken and was registered before any FP8/FP4 cell existed** (`1746d28`,
  `4d813a0`). `scripts/harness/quality/p13.py` carries the machine-readable registration and
  refuses to collect a cell unless that registration is itself committed.
- **Two harness defects had to be fixed first.** The repeated-launch clean-tree defect — a run now
  declares its own output roots, only those are excused, the source tree stays at full strength and
  HEAD is pinned; `--allow-dirty` was not used anywhere. And dispatch verification that BF16
  satisfied *by silence*: `dispatch_verify.py` is additive (the kernel patterns are hashed into
  `KL_SPEC`, so editing them would strand every committed artifact) and requires named positive
  evidence, which for BF16 is the resolved quantization method, weight dtype, attention backend and
  compile step rather than a kernel class that does not exist.
- **P12 refreshed, 39/39, three genuinely fresh engines.** The first refresh attempt returned 36/36
  while replaying August's probes and launching nothing; the probe artifacts were removed and the
  freshness flag now keys off an explicit reuse flag. Count is 39 not 36 because the three positive
  dispatch-evidence checks are new.
- **P13 ran clean**: 3,200 cells, five collections in the registered interleaved order
  (BF16 L1 → FP8 → BF16 L2 → FP4 → BF16 L3), 7 minutes of scoring, tree clean at `3aea394`.
- **Three reviewers went over the formulation, the implementation and the artifacts.** Every
  headline, CI, per-launch value and floor pair was independently re-derived **bit-identically from
  the raw matrices**. Every finding was about labelling, durability, or two wrong claims — not about
  a measured number.
- **One of those wrong claims was mine.** I attributed the 5.65% → 3.36% move in the floor fraction
  entirely to the larger production signal. It is mostly that, but 3.36% also swapped in this run's
  own floor for the pre-registered one. Both are now reported and the move is decomposed.
- **Defects only production scale could surface**, found and fixed before the final analysis: 4 of
  64 trajectories reproduce bit-identically across all three BF16 launches, so their floor is
  exactly 0.0 and the per-trajectory ratio was `inf`/`nan`; and the BF16 floor was being quoted
  against `FP8||FP4`, which has no BF16 reference.
- **A guard caught its own data and was right to.** Re-assembling the finished root marked three
  real launches `launched=false`; independence is now a property of the cells
  (`cells_scored_by_launch`, `cells_first_collected_timestamp`), with byte-distinctness of the
  consumed matrices still the real bar.

## Known issues / unresolved premises

- **G2' fails against the production signal: 3.75% (pre-registered floor) or 3.36% (this run's) of
  BF16→FP8, against a 1% bound.** This is the binding resolution limit on the quality axis and it is
  a property of the *reference*. The disposition quantifies it; it cannot fix it, and that is
  structural — the floor is second order in the launch perturbation while the signal is first order,
  so no value of R moves the ratio toward a pass.
- **The registered nuisance model is contradicted by its own sharpest consequences** (SD ratio
  3.103 predicted vs 1.058 observed; `sigma_proj` 2.93x apart across configurations; launch-effect
  vectors at r = -0.880). The registered verdict is `MIXED` only because the criterion is a weak
  conjunction. **Anything that relies on pooling one coupling constant across configurations should
  be treated as unsupported** until a design with more launches says otherwise.
- **The launch apparatus is operationally inert in the resolution rule at n=64** — 0 of 20
  classifications depend on it. It is worth keeping because it is what *established* that, but it
  should not be presented as load-bearing.
- **Two disclosed conservatisms in the registered rule, deliberately not corrected.** The launch
  inflation uses `sigma2_A + sigma2_E/T` where a CI on the mean wants `sigma2_A/R` (~4.4x larger,
  conservative); `floor_hi` omits the floor's own trajectory uncertainty (anti-conservative). Both
  are reported in `p13_summary.json`; correcting the first after seeing the data would run in the
  resolution-friendly direction.
- **`F = 3.123` for the FP8 launch effect is knife-edge and uncorroborated** — p = 0.047, and the
  FP4 arm on the same three launches gives `F = 0.857`. Do not build on it.
- **BF16 still has no positive GEMM-kernel evidence.** The additive verifier rules out a quantized
  path and proves the engine resolved no quantization method, but *which* BF16 GEMM dispatched is
  still unknown, and the split-k/atomic-reduction story for BF16's non-reproducibility remains
  **unmeasured**.
- **`server.py`'s three dispatch gaps are untouched** — the truncate-before-normalise pattern, the
  unenforceable `"emulation"` pattern, and BF16's silence. They sit on the frozen serving path and
  should be settled before the serving numbers are written up. The quality arm routes around them
  additively.
- **The ceiling replication is PARTIAL — 1 of 18 cells, ~3.3 h remaining**, deferred this session.
  Resume is safe (`replicate_ceiling_group` checks `done_keys` per group and per cell); `--dry-run`
  still prints the static "18 cells, 6 launches" and will misstate the remaining work.

  ```text
  /home/sathvik/miniconda3/envs/qnt/bin/python scripts/harness/run_sweep.py --job ceiling
  /home/sathvik/miniconda3/envs/qnt/bin/python scripts/harness/analyze_ceiling.py --write
  ```

- **The refined serving ceilings are still n=1** (21 / 57 / 70). FP8 and FP4 clear the 50 ms bound
  by 0.34 ms and 0.43 ms against a matched-cell spread BF16 measures at 0.029 ms — ~12 and ~15 noise
  widths. BF16's 2.41 ms is ~83 and is not in question.
- **Seeded generation is not replayable** — 51 of 64. Reproduction goes through the tracked
  `trajectories.json` and its hash, never by rerunning generation.
- **Quality work runs under `envs/qnt`, not the login shell's `python3`.** Committed artifacts
  reproduce to a median 8.6e-14 relative — *not* bit-identically, as `569946c` implies; the large
  relative gaps sit on cells whose absolute value is ~1e-08 nats. Two fresh runs are bit-identical
  to each other.
- **`results/quality/smoke/kl_summary.json` still carries n=4-floor ratios.** P13 supersedes the
  smoke entirely; regenerating it would change a committed number to restate a superseded artifact.
- **The completed sweep carries no host/client CPU telemetry.** The one replication cell that does
  puts the client at 2% of one core (FP8 C=56 / 1402 tok/s).
- **`PREFILL_PROBE` inherited the decode SLO** and needs a TTFT-based criterion before it is re-run.
- **Latin-square carryover is unbalanced**; queue *time* is still not recorded, only depth;
  `meets_slo` is survivor-biased.
- **No perplexity or downstream-task axis yet.** D14 and D15 are open; chat-formatted tasks stay
  blocked by the `chat_template` deviation.
- **Checkpoint provenance has one open deviation** (shorter `chat_template`), FP4's `tokenizer.json`
  carries a 2048 truncation the base and FP8 do not — inert here because the rig feeds frozen token
  IDs with `detokenize=False`, and all 640 context hashes match across all five collections — and
  **FP4 calibration sensitivity is untested** (one draw, 128 ultrachat samples, seed 0).
- **`scripts/harness/selftest.py` is flaky on its timing-dependent stub cases** — three runs this
  session gave three different failure sets (t3/t4, t10, t8). The quality suite `qselftest.py` is
  355/355 and stable.
- **The GPU is power-limited at 145 W**, so every number is measured under a power ceiling.
