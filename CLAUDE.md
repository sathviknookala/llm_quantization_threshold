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

**The workload and concurrency contract is locked. Phase 1 is a memory-focused study.** Model
qualification is complete, the precision ladder is locked, and D10/D11 are now resolved: one primary
decode-dominated workload swept across concurrency, with an explicit latency SLO as the saturation
criterion. **No final benchmark collection has started** — the next step is a pilot, then the harness.

Current conclusions:

1. The core experimental dimension is the quantization/deployment configuration.
2. BF16 is the common high-precision reference; it is not the maximum precision supported by the GPU.
3. BF16 fit is the binding constraint, and at 8B it is genuinely tight: 14.99 GiB of weights leaves
   4.84 GiB of KV cache. This is the regime where quantization buys capacity, not just speed.
4. The ladder is locked: BF16 -> FP8 -> NVFP4, all three natively executed on SM120.
5. KV-cache precision is held at BF16 across the whole ladder, so measured capacity gains are
   attributable to weight residency alone. FP8 KV cache is the natural phase 2, not a phase-1 gap.
6. Hardware is fixed experimental context/control, not an independent variable.
7. The driver (575.64.03 / CUDA 12.9) caps the serving stack at vLLM 0.19.1. This costs nothing the
   ladder needs — SM120 CUTLASS FP8 and FP4 are both present in that build.
8. **Both halves of the benefit come from weight residency shrinking.** Under the locked workload the
   sweep is memory-bandwidth-bound throughout, so the split is bandwidth vs capacity, not compute vs
   capacity. This supersedes the earlier compute-vs-capacity framing in D11.

Locked workload and sweep (D10, D11):

```text
DECODE_PRIMARY    512 in / 2048 out    concurrency 1,4,8,12,16,24,32,48,64,96   3 repetitions
PREFILL_PROBE    8192 in /   32 out    concurrency 1,2,4,8                       no repetitions
SLO               TPOT P95 <= 50 ms    defines saturation and the headline capacity metric
```

KV wall per configuration under `DECODE_PRIMARY` (2,560-token peak footprint, 128 KiB KV/token):

```text
              weights     KV cache    KV tokens   KV wall (peak / mean)   kernel
BF16          14.99 GiB    4.84 GiB      39,664         15  /  25         default BF16 GEMM
FP8            8.49 GiB   11.31 GiB      92,608         36  /  60         CutlassFP8ScaledMMLinearKernel
FP4 (NVFP4)    5.65 GiB   14.57 GiB     119,360         46  /  77         SM120 CUTLASS FP4 (flashinfer)
```

**Next — planned order.** The pilot is defined in `EXPERIMENTAL_CONTRACT.md` with five jobs (P1-P5)
and exit criteria; it validates the contract's free parameters and produces no results.

```text
1. serving harness + pilot (P1-P5) + KL correctness gate   <- current step
2. serving sweep                                            overnight, GPU-exclusive
3. quality run                                              KL first; PPL/tasks need D14/D15
```

The correctness gate sits *before* the sweep, not after: timing a corrupted checkpoint wastes the
most expensive resource in the project. The quality harness can be **written** while the sweep runs;
only its execution serializes on the GPU. Inside the quality arm the BF16 continuations and reference
distributions must be produced first, then each quantized configuration streams against them.

Phase 1 deliverable is the quality-loss vs sustainable-concurrency curve for this one model.

## Last session

**Session 3 — resolved D10 and D11; narrowed phase 1 to a memory-focused lens.**

- Adopted a memory-focused lens and locked one primary workload, 512 in / 2048 out, replacing the
  prefill-heavy / balanced / decode-heavy trio. `BALANCED` dropped; a cheap `PREFILL_PROBE` kept so
  the arithmetic benefit is observed in bounded form rather than not at all.
- Established that decode arithmetic intensity is approximately batch size, so the whole sweep stays
  bandwidth-bound and the KV walls (15 / 36 / 46) sit clear of compute saturation by construction.
- Reframed the D11 decomposition from compute-vs-capacity to **bandwidth-vs-capacity**, which also
  strengthens the portability argument: every GPU reads weights, not every GPU has FP4 tensor cores.
- Derived from the coarse smoke artifacts that all three configurations decode at ~620-660 GB/s, so
  the batch-1 speedup may be fully explained by weight bytes. Recorded as a falsifiable pilot
  prediction, explicitly not as a result.
- Locked the concurrency points, an explicit TPOT P95 <= 50 ms SLO as the saturation criterion, a
  cell-abort rule, steady-state warmup rules, and run-order counterbalancing.
- Added hazards H7 (prefix caching can silently delete the workload) and H8 (occupancy grows 5x, so
  the KV wall is a band and preemption-by-recompute makes degradation superlinear).
- Extended the quality rig to teacher-forced KL at ten strided positions across the 2,048-token
  generation, since measuring quality at position 1 of a 2,048-position workload was the central gap.
- Then measured the first formulation of that rig infeasible and replaced it. Full-vocab
  `prompt_logprobs` over 2,560 positions returns 328M `Logprob` objects (~16 GB host RAM per request)
  and disables prefix-cache reads. Ten truncated-prefix passes over the already-proven D13 path cost
  ~64 MB and keep caching. Recorded in D13 so it is not reintroduced.
- Defined the pilot formally (P1-P5 plus a KL correctness gate) and added the sub-wall bandwidth-bound
  check: P1 now runs at concurrency 1, 8 and 12, because D11's whole decomposition rests on the sweep
  being bandwidth-bound and nothing had measured that.
- Opened D14 (perplexity corpus and token budget) and D15 (downstream task set), converting the last
  unowned `TBD`s in the quality rig into tracked gates.

## Known issues / unresolved premises

- **Checkpoint provenance has one open deviation.** Weights are SHA256-identical to official, but
  `tokenizer_config.json` carries a shorter `chat_template` (348 chars vs the official multi-KB
  template). Within-model comparability is unaffected; absolute instruct-task scores would not be
  comparable to published Llama numbers. Re-pin to `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f2`
  when the license is approved.
- **Qualification serving numbers are not benchmark numbers.** They are single-request, single-run,
  no warmup, no repetition. The BF16 short-workload figure (2.7 tok/s) is warmup contamination.
- **The ~620-660 GB/s bandwidth finding is derived, not measured.** It comes from a two-point prefill
  subtraction over coarse smoke artifacts and assumes decode cost is equal at 2k and 16k context. It
  is a pilot prediction and must not be quoted as a result.
- **Memory bandwidth is not recorded in `HARDWARE_PROFILE.md`.** Under a memory-focused lens it is the
  governing hardware constant, and neither the spec figure nor an achieved measurement exists yet.
- **Qualification KL values are a feasibility demo, not a quality result.** 16 highly repetitive
  synthetic contexts. They must not be cited as measured degradation.
- **Repetition count is provisional at 3.** Pilot variance has not been measured; raise it if
  run-to-run spread is large relative to the FP8-to-FP4 gap.
- **The teacher-forced KL rig measures divergence given BF16's trajectory**, not free-running drift of
  each configuration's own generation. Whether divergence compounds through the sampling loop is
  untested.
- **The sweep is assumed memory-bandwidth-bound throughout, and that is not yet measured.** D11's
  bandwidth-vs-capacity decomposition depends on it. Pilot job P1 tests it at concurrency 1, 8 and 12;
  if the ratio decays before the BF16 wall at 15, the decomposition must be reopened.
- **The quality arm has two open gates.** D14 (perplexity corpus and token budget) and D15 (downstream
  task set) are unresolved, and chat-formatted tasks stay blocked by the `chat_template` deviation.
- **Calibration sensitivity is untested, and applies only to FP4.** FP8 needs no calibration, so the
  calibration-robustness study is FP4-only. The single FP4 draw used 128 ultrachat samples, seed 0.
- **FP4's first load pays a ~61 s flashinfer JIT build.** Cached now, but it must never leak into a
  serving metric.
- **Idle PCIe state is not benchmark state.** Negotiated link generation still unverified under load.
- **The GPU is power-limited at 145 W** and clocks fall under load, so throughput is measured under a
  power ceiling rather than at fixed clocks.
- **Phase 1 measures one workload shape.** The arithmetic benefit of low precision is not measured by
  the sweep, and prefill-dominated deployments are out of scope. See `LIMITATIONS.md`.
- **No quality or serving outcome is predetermined.** The ladder being natively executable says
  nothing about whether FP4's degradation is worth its gain.

At the end of a session, overwrite `Current focus`, `Last session`, and `Known issues / unresolved premises` in place. Git history is the changelog; this file should remain a current-state hub.
