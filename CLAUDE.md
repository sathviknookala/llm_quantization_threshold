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

**Model qualification is complete and the precision ladder is locked.** Llama 3.1 8B Instruct is
QUALIFIED as the phase-1 primary model. BF16, FP8 (W8A8 E4M3) and FP4 (NVFP4 W4A4) all build from
one verified checkpoint, serve on the target GPU, and each dispatches to its intended native kernel.
The quality rig's full-logits/KL path is measured feasible. **No final benchmark collection has
started** — workload token counts and concurrency points are still open.

Current conclusions:

1. The core experimental dimension is the quantization/deployment configuration.
2. BF16 is the common high-precision reference; it is not the maximum precision supported by the GPU.
3. BF16 fit is the binding constraint, and at 8B it is genuinely tight: 14.99 GiB of weights leaves
   4.84 GiB of KV cache (1.21x concurrency at 32k). This is the regime where quantization buys
   capacity, not just speed.
4. The ladder is locked: BF16 -> FP8 -> NVFP4, all three natively executed on SM120.
5. KV-cache precision is held at BF16 across the whole ladder, so measured capacity gains are
   attributable to weight residency alone.
6. Hardware is fixed experimental context/control, not an independent variable.
7. The driver (575.64.03 / CUDA 12.9) caps the serving stack at vLLM 0.19.1. This costs nothing the
   ladder needs — SM120 CUTLASS FP8 and FP4 are both present in that build.

Measured at qualification (coarse — see hazards H1/H2 in `EXPERIMENTAL_CONTRACT.md`):

```text
              weights     KV cache    KV tokens   conc@32k   kernel
BF16          14.99 GiB    4.84 GiB      39,664     1.21x    default BF16 GEMM
FP8            8.49 GiB   11.31 GiB      92,608     2.83x    CutlassFP8ScaledMMLinearKernel
FP4 (NVFP4)    5.65 GiB   14.57 GiB     119,360     3.64x    SM120 CUTLASS FP4 (flashinfer)
```

**Next:** freeze workload token counts (D10) and concurrency points (D11) in
`EXPERIMENTAL_CONTRACT.md`, then build the real quality + serving harness under the warmup rules.
Phase 1 deliverable is the quality-loss vs serving-throughput curve for this one model.

## Last session

**Session 2 — environment build and model qualification.**

- Built two conda envs: `qnt` (vLLM 0.19.1 serving) and `qnt-quant` (llmcompressor 0.10.0.3),
  split because their `compressed-tensors` pins are irreconcilable.
- Discovered the driver ceiling: CUDA 13 wheels do not run on driver 575, capping vLLM at 0.19.1.
  Established this costs nothing, since SM120 CUTLASS FP8/FP4 exist in that build.
- Qualified Llama 3.1 8B Instruct: locked exact identity (8,030,261,248 params, all BF16).
- Worked around the gated repo by verifying an ungated mirror byte-identical by SHA256 against the
  official checksums, recomputed locally. License request still pending.
- Built and served FP8 and NVFP4 from the same checkpoint; captured explicit kernel-dispatch
  evidence for both and confirmed the NVFP4 emulation path was unused.
- Proved the KL rig: full 128,256-entry logits from the serving engine, token-identical contexts,
  250 KiB/context storage.
- Recorded six measurement hazards found in the process, including a first-call warmup artifact that
  would have made BF16 look 24x slower than FP8.

## Known issues / unresolved premises

- **Checkpoint provenance has one open deviation.** Weights are SHA256-identical to official, but
  `tokenizer_config.json` carries a shorter `chat_template` (348 chars vs the official multi-KB
  template). Within-model comparability is unaffected; absolute instruct-task scores would not be
  comparable to published Llama numbers. Re-pin to `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f2`
  when the license is approved.
- **Qualification serving numbers are not benchmark numbers.** They are single-request, single-run,
  no warmup, no repetition. The BF16 short-workload figure (2.7 tok/s) is warmup contamination.
- **Qualification KL values are a feasibility demo, not a quality result.** 16 highly repetitive
  synthetic contexts. They must not be cited as measured degradation.
- **Workload and concurrency definitions are still open (D10, D11).** Nothing may be collected as
  final data until they are frozen.
- **Calibration sensitivity is untested, and applies only to FP4.** FP8 needs no calibration, so the
  calibration-robustness study is FP4-only. The single FP4 draw used 128 ultrachat samples, seed 0.
- **FP4's first load pays a ~61 s flashinfer JIT build.** Cached now, but it must never leak into a
  serving metric.
- **Idle PCIe state is not benchmark state.** Negotiated link generation still unverified under load.
- **The GPU is power-limited at 145 W** and clocks fall under load, so throughput is measured under a
  power ceiling rather than at fixed clocks.
- **No quality or serving outcome is predetermined.** The ladder being natively executable says
  nothing about whether FP4's degradation is worth its gain.

At the end of a session, overwrite `Current focus`, `Last session`, and `Known issues / unresolved premises` in place. Git history is the changelog; this file should remain a current-state hub.
