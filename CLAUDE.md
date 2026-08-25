# CLAUDE.md — Quantization Deployment Boundary

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
runners over it; `analyze.py` turns pilot cells into the pilot verdict; `selftest.py` exercises the
cell state machine against a stub engine with no GPU. Renamed from `scripts/pilot/` on 2026-08-24 —
the harness outlives the pilot.

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

**The pilot's science gate is CLEARED. The sweep is blocked on engineering, not on physics.**
P2, P3, P5 pass, P4 is valid, the correctness gate is clean — and **P1 FAILED and stays FAIL**.

**Amended 2026-08-24: P1 does not gate the sweep and is not rerun.** Its pre-registered tolerances
were not relaxed and it is not rewritten as a PASS. P1 tested a *predictive shortcut* — that the
below-wall throughput ratio could be read off the weight-size ratio — and reproducibly falsified it.
Losing a shortcut is a reason to run the measurement, not to postpone it. No replacement predictive
model is required, and the post-hoc three-term traffic model in D10 must **not** be promoted into a
pre-sweep gate. Verdict in `results/pilot/PILOT_DECISION.md`, criterion in `EXPERIMENTAL_CONTRACT.md`
"Exit criteria".

```text
P1  weight-size-ratio prediction       FAIL    informative falsification, recorded, does not gate
P2  BF16 KV wall                       PASS    bracket [17, 18], inside the 15-25 band
P3  repetition count                   PASS    rho 0.0020 / 0.0064  -> 3 repetitions locked
P4  achievable HBM bandwidth           VALID   620.1 GB/s read, 92.3% of 672.0 GB/s spec
P5  harness validation                 PASS    23 invariants, 46 cells
    correctness gate                   CLEAN   32 real C4 contexts, dispatch reconfirmed
```

**The interpretation the project now holds.** Quantization has two distinct resource effects, and
only the first follows from checkpoint size:

```text
smaller weight residency  ->  more VRAM free for KV  ->  greater KV capacity   (P2 confirmed: wall [17,18])
smaller weight traffic    ->  *potential* throughput benefit, magnitude unknown
```

Decode also moves KV and other traffic and is subject to kernel efficiency, compute and scheduling
effects, so **compression ratio does not imply proportional throughput gain**. The magnitude of the
actual serving benefit is an **experimental output of the sweep**, not an inference from checkpoint
size.

**What may not be claimed.** The below-wall gap has *not* been uniquely attributed to HBM bandwidth,
to KV dilution, or to kernel efficiency — this measurement cannot separate them. Supported: the
sub-wall differences are reproducible; the weight-ratio prediction does not explain them
quantitatively; the sweep will measure the realized differences; P2 independently confirmed the
capacity-side motivation. Report the below-wall gap as the raw measured quantity, always with its
concurrency attached, and never as a bandwidth or weight-residency benefit.

```text
0. D10/D11 corrected; P1 exit criterion amended               DONE 2026-08-24
1. build the D11 sweep orchestrator                            <- current step
2. implement SKIPPED_PAST_SLO
3. reconcile the cell wall cap with the 4-period window, and make
   CELL_TIMEOUT classification agree with the contract
4. serving sweep                       overnight, GPU-exclusive, 3 reps locked
5. quality run                         KL first; PPL/tasks need D14/D15
```

Steps 1-3 are the whole remaining blocker list. After them the sweep is cleared with no further
P1 experiment. The sweep must retain: the H9 stationarity gate and whole-period windows, H10
per-launch KV-capacity recording with idle-GPU preflight/teardown, counterbalanced order, the frozen
corpus, 3 repetitions with per-repetition engine restart.

## Last session

**Session 4 — built and ran the serving pilot; P1 failed and the sweep is not cleared.**

- Resolved **D16** (was blocking): C4 `en` validation shard 0, 512 prompts of exactly 512 tokens plus
  64 of exactly 8192, each from a distinct document, prefix-disjointness enforced by hash over the
  first 64 content tokens, ultrachat excluded. Emitted as token IDs so tokenization cannot drift.
- Built `scripts/pilot/`: corpus prep, HBM microbenchmark, server control, a contract-enforcing
  request driver, orchestrator with resume, correctness gate, analyzer with pre-registered criteria.
- **P4**: 620.1 / 564.7 / 545.8 GB/s (read / triad / copy), CV ~0.1%, 21.3x L2 working set. Recorded
  in `HARDWARE_PROFILE.md`, which no longer says "NOT YET RECORDED".
- **Correctness gate CLEAN** on 32 real held-out contexts; BF16 self-KL exactly 0 under batch-order
  reversal; dispatch reconfirmed on all three rungs.
- **P2 PASS**: wall at [17, 18], reproducible on both sides. Peak-footprint basis predicted 17 and is
  the better predictor; the mean-occupancy basis (29) overstates it.
- **P3 PASS**: rho = 0.0020 / 0.0064 against a 0.25 criterion. 3 repetitions locked.
- **P1 FAIL**: see Current focus. Reported against the pre-registered criteria rather than adjusted.
- Found **H9**: the workload is *periodic*, not flat — phase-aligned slots make throughput oscillate
  +/-11% with period `out_tokens * C / throughput` (predicted 39.8 s vs 40 s observed). A flatness
  gate cannot fire; steady state must be defined as stationarity over >= 1 period, and windows must
  span whole periods. Six cells were discarded and re-run under the corrected gate.
- Found **H10**: the serving path allocates more KV than the offline path (BF16 44,688 vs 39,664),
  and a launch begun before a previous engine released VRAM silently sized a 10%-smaller cache.
- Four harness bugs caught before they could corrupt results: Prometheus `_total` suffix nulling all
  four D11 counters; a window-only pressure test that would have inverted P2 on an aborted cell;
  empty-text tokens dropping from the ITL series; and a token-count invariant that compared
  incommensurable time spans and, in one revision, passed vacuously on a missing field.

## Known issues / unresolved premises

- **The below-wall gap is unattributed, and that is now a settled position rather than an open gate.**
  It is reproducible and concurrency-dependent. What produces it — weight traffic, common KV traffic,
  kernel efficiency, scheduling — is not separated by this measurement and the study does not claim
  to separate it. Report it as the realized difference at a stated concurrency.
- **The three-term KV-traffic model is post-hoc and stays that way.** It reproduces R_FP4(12) to 0.3%
  because it was fitted on that point. It is documented in D10 as a candidate explanation, is **not**
  a sweep gate, and validating it is not required. If anyone wants to test it later, the pilot
  already holds unfitted points (BF16 at C=15/16, FP8 and FP4 at C=24) and the test costs no GPU
  time — a follow-up, not a dependency.
- **FP4's low-batch bandwidth shortfall is unexplained.** 86% of measured read bandwidth at C=1
  against FP8's 97% is consistent with per-GEMM overhead but has not been attributed.
- **Throughput inversion near the wall is confounded with clock throttling.** SM clock fell
  1872 -> 1728 MHz at C=17 under a pinned 145 W cap, so ~7.7 of the 13% throughput drop is clock, not
  capacity. Identify the wall by preemption and KV saturation, never by throughput shape.
- **The measured wall is 4 percentage points of KV from the clean side.** C=17 sits at 96.2% KV max
  with zero preemption; C=18 at 98.5% with preemption. The bracket is tight and depends on the KV
  capacity of the specific engine launch (H10).
- **Pilot variance is a floor, not an estimate.** P3's rho comes from minutes-long windows; the sweep
  is 6-8 hours with thermal drift.
- **Checkpoint provenance still has one open deviation** (`chat_template`, 348 chars vs official).
  Weights are SHA256-identical. Re-pin to `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f2` when the
  license lands.
- **Qualification serving and KL numbers remain non-citable** — single-request, no warmup, and 16
  repetitive synthetic contexts respectively.
- **Pilot numbers are diagnostic and may not be cited as results.**
- **Three engineering blockers stand between here and the sweep** — orchestrator, `SKIPPED_PAST_SLO`,
  and the wall-cap/window reconciliation. None is a physics question; see `Current focus`.
- **The quality arm has two open gates**, D14 (perplexity corpus) and D15 (task set), and
  chat-formatted tasks stay blocked by the `chat_template` deviation.
- **Calibration sensitivity is untested and FP4-only.** One draw, 128 ultrachat samples, seed 0.
- **`PREFILL_PROBE` is plumbing-checked only.** One C=1 cell per configuration; the 1/2/4/8 sweep has
  not run.
- **No quality or serving outcome is predetermined.**

At the end of a session, overwrite `Current focus`, `Last session`, and `Known issues / unresolved premises` in place. Git history is the changelog; this file should remain a current-state hub.
