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
runners over it; `analyze.py` turns pilot cells into the pilot verdict; `selftest.py` exercises the
cell state machine against a stub engine with no GPU. Renamed from `scripts/pilot/` on 2026-08-24 —
the harness outlives the pilot.

`scripts/harness/quality/` is the quality arm, KL first — `positions.py` (retained-position
contract), `kl_math.py` (float64 KL, trajectory bootstrap), `qcommon.py` (identity, provenance,
`KL_SPEC`), `qengine.py` (engine lifecycle, observed identity), `trajectories.py` (the frozen BF16
continuations), `collect_kl.py`, `analyze_kl.py`, `gates.py`, `qselftest.py`. It imports `common.py`
and `server.py` and edits neither: the serving contract is frozen. `scripts/logits_probe.py`, `scripts/compute_kl.py` and
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
5. quality run
   P0-P6  contract, numerics, engine lifecycle, G7, G9         DONE 2026-08-25
   P7     64 BF16 trajectories frozen                          DONE 2026-08-26
   P8-P9  collect_kl.py / analyze_kl.py                        DONE 2026-08-26
   P10    real smoke, 4 traj x 10 pos x 3 configs              DONE 2026-08-26
   P11    replication floor + fp16/fp32 storage gate           DONE 2026-08-26
   P12    full preflight, 35/35                               DONE 2026-08-26
   P13    64-trajectory production KL run                      <- next, NOT authorised
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
BF16||FP8        3.690e-03      [1.748e-03, 6.016e-03]    2.98e-04        12.4x
BF16||FP4        2.945e-02      [2.315e-02, 3.576e-02]    2.98e-04        98.7x
FP8||FP4         3.566e-02      [2.627e-02, 4.506e-02]    3.91e-11       9.1e+08x
```

The barred subtraction proxy would have put FP8->FP4 at 2.576e-02, low by a factor of 1.38.

### What the gates returned

```text
G7  numerics vs the historical EPS formula      PASS
G9  engine profile                              MEASURED, profile locked
    replayability                               NOT REPLAYABLE (51/64) -- informational
G2  replication floor                           FAIL on BF16 (8.1% of FP8 signal vs 1% bound)
G3  cache equivalence                           FAIL on BF16 (4.74%), below BF16's own floor
G4  fp16 vs fp32 storage                        FAIL -- fp32 stays
```

Two pre-registered bounds fail, both on BF16 and both traceable to one cause: **BF16 is the
non-reproducible configuration.** It reproduces 0 of 40 cells across launches while FP8 and FP4
reproduce 36 and 34. Thresholds were not relaxed after seeing the results.

## Last session

**Session 6 — locked the engine profile, froze the trajectories, built and smoke-validated the whole
KL path.** Five persistent reviewers inspected the implementation before any GPU time.

- **`graph_2048` locked** on G9's evidence, and the eager/chunking sensitivity recorded as a result
  rather than a configuration footnote.
- **Four blockers closed before running.** `run_job` released VRAM outside its `try`, so a killed
  parent orphaned an engine — the incident already in this file, reintroduced; the helper feeding the
  floor and cache gates lacked its siblings' completeness checks, so a short batch would have left
  `-inf` rows in the very floor everything is read against; two pre-registered thresholds were hashed
  into `KL_SPEC` and never evaluated, leaving both gates unable to fail on substance; and the
  checkpoint hash cache keyed on size+mtime, which a timestamp-preserving deploy leaves untouched
  (reproduced returning a stale digest).
- **A worst-cell floor comparison spanning different grid sizes was refused.** A maximum is an
  extreme-value statistic: under a null with no signal, max-of-640 beats max-of-40 in 95% of draws.
- **Two defects found by running, not reading.** The replayability gate defaulted to 4 trajectories,
  which would have submitted groups of 4 against the freeze's 16 and confounded batching with launch
  noise. And SIGTERM to a parent blocked in `wait()` orphaned an EngineCore holding 22 GiB — the
  earlier `finally` only rescued SIGINT.
- **P7 frozen**: 64 trajectories, 2048 tokens each, zero preemptions, `trajectory_set_hash`
  `1dd71faeb242c7c1`, engine identity matching G9's BF16 `graph_2048` launch exactly.
- Prefix-cache reuse **measured**: 32,192 hits of 42,876 queries, against a 76.1% ceiling.

## Known issues / unresolved premises

- **The BF16 replication floor fails its pre-registered bound**, at 8.1% of the BF16→FP8 KL against
  1%. The FP8 comparison sits only 12.4x above its noise floor. This is the binding constraint on the
  whole quality axis and it is a property of the *reference*, not of the quantized paths: BF16
  reproduces 0 of 40 cells across launches, FP8 and FP4 reproduce 36 and 34. Decide before P13
  whether to accept it as a stated resolution limit or to average repeated BF16 launches.
- **Seeded generation is not replayable** — 51 of 64, earliest divergence at token 3. Reproduction
  goes through the tracked `trajectories.json` and its hash, never by rerunning generation.
- **Every quality figure so far is n=4 and is not a result.** The smoke exists to validate the rig.
- **The refined serving ceilings are still n=1** (21 / 57 / 70). Replicating the three bisections is
  ~9 cells and is the first thing to do before quoting them.
- **`PREFILL_PROBE` inherited the decode SLO** and lost BF16's C=8 point; it needs a TTFT-based
  criterion before it is re-run.
- **The below-wall throughput gap is unattributed**, and that is a settled position.
- **Latin-square carryover is unbalanced**; queue *time* is still not recorded, only depth;
  `meets_slo` is survivor-biased.
- **No perplexity or downstream-task axis yet.** D14 and D15 are open; chat-formatted tasks stay
  blocked by the `chat_template` deviation.
- **Checkpoint provenance has one open deviation** (shorter `chat_template`), and **FP4 calibration
  sensitivity is untested** — one draw, 128 ultrachat samples, seed 0.
- **The GPU is power-limited at 145 W**, so every number is measured under a power ceiling.

At the end of a session, overwrite `Current focus`, `Last session`, and `Known issues / unresolved premises` in place. Git history is the changelog; this file should remain a current-state hub.
