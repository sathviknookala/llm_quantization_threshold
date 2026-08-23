# Project Decisions

This document records current design decisions and open gates. It is not a narrative changelog; update decisions in place when new evidence resolves them, and rely on git history for prior states.

## D1 — Research question

**Status:** LOCKED

**Decision:** The project measures the marginal quality-versus-serving tradeoff of successive LLM quantization steps and identifies the deployment boundary where further quantization stops being worthwhile under the tested conditions.

**Why:** This preserves a useful result regardless of which precision wins.

---

## D2 — Comparison unit

**Status:** LOCKED

**Decision:** The unit of comparison is a complete deployment configuration, not a datatype label.

A configuration includes the base checkpoint plus the relevant weight, activation, KV-cache, calibration, quantization, backend, and kernel choices.

**Why:** “FP8” or “FP4” alone does not uniquely determine either quality behavior or serving performance.

---

## D3 — High-precision reference

**Status:** LOCKED

**Decision:** BF16 is the common high-precision deployment reference.

**Correction:** BF16 is **not** the GPU's maximum supported precision. It is the chosen reference precision for this experiment.

**Implication:** The largest primary model must fit and serve properly in BF16, because every lower-precision comparison needs the same-model reference.

---

## D4 — Hardware scope

**Status:** LOCKED

**Decision:** The primary study uses one fixed RTX PRO 4000 Blackwell machine.

**Why:** The goal is not to compare hardware. Fixing the machine makes serving differences easier to attribute to deployment configuration and workload.

**Implication:** Hardware is a control/context variable, not a primary independent variable.

---

## D5 — Candidate precision ladder

**Status:** LOCKED for the primary study (2026-08-22)

```text
BF16 -> FP8 (W8A8 E4M3) -> FP4 (NVFP4 W4A4, group 16)
```

**Unlock condition met.** All three configurations were built from one checkpoint, served on the
target GPU, and each was shown to dispatch to its intended kernel — including an SM120-specific
CUTLASS FP4 GEMM for the FP4 rung, with the NVFP4 emulation path confirmed unused. Full definitions
and evidence: `QUANTIZATION_CONFIGS.md`. Artifacts: `results/qualification/`.

Measured at qualification (coarse, see hazards H1/H2 in `EXPERIMENTAL_CONTRACT.md`):

```text
                weights    KV cache      KV tokens   concurrency@32k
BF16            14.99 GiB   4.84 GiB        39,664       1.21x
FP8              8.49 GiB  11.31 GiB        92,608       2.83x
FP4 (NVFP4)      5.65 GiB  14.57 GiB       119,360       3.64x
```

**Why this is a real ladder and not three labels.** Each step both shrinks weight residency and
enlarges KV capacity while KV *precision* is held constant at BF16, so the capacity gain is
attributable to weights alone.

---

## D6 — Model strategy

**Status:** RESOLVED for phase 1 (2026-08-22) — Option A, one focused primary model

**Decision:** Llama 3.1 8B Instruct is the primary model. 8,030,261,248 parameters, native BF16.

**Qualification result:** QUALIFIED. BF16 is fully GPU-resident with no offload, all three
representative workloads (128 / 2048 / 16384 prompt tokens) ran without OOM, and peak process VRAM
was 22,845 MiB of 24,467 MiB.

**Why 8B is the right size here rather than something smaller.** BF16 leaves only 4.84 GiB for KV
(1.21x concurrency at 32k context). That is deliberately tight: it is precisely the regime where
quantization buys *capacity*, which is one of the two outcomes this project separates. A smaller
model would have made the memory axis uninteresting.

**Provenance caveat:** served from a SHA256-verified ungated mirror while the Llama license request
is pending. See `QUANTIZATION_CONFIGS.md` for the exact deviation and the re-pin action.

**Phase 2 (not started):** additional models only to test generalization, per the original
criterion. Do not expand until phase 1 produces the quality-vs-serving curve.

---

## D7 — FP4 representation

**Status:** LOCKED (2026-08-22) — NVFP4 W4A4, group size 16

**Decision:** `nvfp4-pack-quantized`, 4-bit float weights and activations, `tensor_group` strategy,
group size 16, symmetric; `lm_head` excluded. Produced by llmcompressor 0.10.0.3 with 128
calibration samples from `HuggingFaceH4/ultrachat_200k`.

**Why NVFP4 and not MXFP4.** NVFP4 is the format for which this stack exposes a native SM120 CUTLASS
GEMM. Dispatch resolves to `flashinfer.gemm.get_gemm_sm120_module_cutlass_fp4()`.

**Why W4A4 and not the weight-only NVFP4A16.** W4A16 would store 4-bit weights but dequantize and
compute in BF16 — quantized storage with no native FP4 math. That is the exact failure mode this
project must avoid. W4A4 puts activations in FP4 too, so the GEMM runs on FP4 inputs.

Both `MXFP4`/`MXFP4A16` and `NVFP4A16` remain available in the tool if a secondary comparison is
ever justified, but they are not part of the primary ladder.

---

## D8 — FP8 representation

**Status:** LOCKED (2026-08-22) — FP8 E4M3 W8A8, dynamic per-token activations

**Decision:** compressed-tensors `float-quantized`. Weights 8-bit float, static, per-channel
symmetric (`memoryless_minmax` observer). Activations 8-bit float, dynamic, per-token symmetric.
`lm_head` excluded. KV cache left at BF16.

**Calibration: none required.** Weight scales are computed from the weights and activation scales
are dynamic at runtime, so this rung has no calibration-draw sensitivity. Quantization took 22.8 s.

**Consequence for the evaluation rig.** The calibration-robustness study in `EVALUATION_RIG.md`
therefore applies to the **FP4 rung only**. FP8 has no calibration draw to vary. This asymmetry is a
property of the recipe, not an oversight.

**Kernel path:** `CutlassFP8ScaledMMLinearKernel` for `CompressedTensorsW8A8Fp8`.

---

## D9 — Inference backend

**Status:** LOCKED (2026-08-22) — vLLM 0.19.1

**Decision:** vLLM `0.19.1` in conda env `qnt` (Python 3.12.13, torch 2.10.0+cu128,
compressed-tensors 0.15.0.1). Offline checkpoint production runs separately in conda env
`qnt-quant` (llmcompressor 0.10.0.3). Artifacts: `results/system/env_qnt_2026-08-22.json`,
`results/system/env_qnt-quant_2026-08-22.json`.

**Why this version and not the newest.** Driver 575.64.03 exposes CUDA 12.9 and cannot run CUDA 13
wheels. vLLM >= 0.20.0 pins torch >= 2.11.0, whose default PyPI wheel is cu13x. vLLM 0.19.1 is the
newest release that installs on a cu128 torch. Verified failure: vLLM 0.27.1 pulled torch
2.13.0+cu130 and `torch.cuda.is_available()` was `False`. See `HARDWARE_PROFILE.md`.

**Why the ceiling is acceptable.** The reason to want a newer vLLM would be SM120 low-precision
kernel coverage, and that is already present: this build reports
`cutlass_scaled_mm_supports_fp8(120) = True` and `cutlass_scaled_mm_supports_fp4(120) = True`.
The ladder therefore does not depend on a driver upgrade.

**Unlock condition met.** All three configurations loaded and dispatched to their intended kernels
(BF16 default GEMM, `CutlassFP8ScaledMMLinearKernel`, SM120 CUTLASS FP4 via flashinfer), with the
NVFP4 emulation path confirmed unused.

**Operational requirement.** The FP4 kernel is JIT-compiled, so the serving process needs `ninja`
and `nvcc` on `PATH`:

```bash
export PATH=/home/sathvik/miniconda3/envs/qnt/bin:/home/sathvik/cuda-12.9/bin:$PATH
export CUDA_HOME=/home/sathvik/cuda-12.9
```

Launching by absolute interpreter path without this fails at engine init with
`FileNotFoundError: 'ninja'`.

**Revisit if:** a required FP4 or KV-cache feature turns out to be missing from 0.19.1. The escape
hatch is a driver upgrade to r580+, which is a tracked decision because it changes the recorded
machine identity — not an incidental fix.

---

## D10 — Workload token counts

**Status:** OPEN

Three workload classes are locked conceptually:

```text
prefill-heavy
decode-heavy
balanced
```

Exact input/output token counts remain open until pilot measurements establish useful, realistic regimes.

Final counts must be frozen in `EXPERIMENTAL_CONTRACT.md` before final data collection.

---

## D11 — Concurrency strategy

**Status:** PROVISIONAL — strategy settled, exact points still to freeze

Sweep from low concurrency toward saturation rather than benchmark one arbitrary concurrency level.

**Why:** Quantization changes both compute cost and available KV-cache capacity, so its value depends
strongly on load. At a single concurrency point these two effects are indistinguishable.

### The sweep must decompose the compute benefit from the capacity benefit

A quantization step delivers two different things at once:

```text
compute benefit    lower-precision GEMMs execute faster
capacity benefit   smaller weights free VRAM -> more KV -> larger batches
```

Both are real and a deployment gets both, so the **headline result is the combined benefit**. But the
two do not generalize equally: the compute benefit travels to other hardware, while the capacity
benefit is a function of this machine's 24 GiB ceiling (see `LIMITATIONS.md`). A result that cannot
separate them cannot say which part a reader should expect to reproduce.

**The separation is already present in the shape of a single sweep** and does not require a second
experimental arm:

- **Below BF16's KV limit**, every configuration holds every sequence. KV binds for no one, so the
  throughput gap is the **compute benefit alone**.
- **Above BF16's KV limit**, BF16 begins queuing and preempting while FP8/FP4 keep batching. The gap
  *widens*, and that widening is the **capacity contribution**.

Concurrency at which each configuration becomes KV-limited, from measured KV capacity
(39,664 / 92,608 / 119,360 tokens — `results/qualification/qualification_summary.json`):

```text
 ctx tokens |   BF16    FP8  NVFP4 | BF16-starved but others not
        512 |     77    180    233 | 78..233
       1024 |     38     90    116 | 39..116
       2048 |     19     45     58 | 20..58
       4096 |      9     22     29 | 10..29
       8192 |      4     11     14 | 5..14
      16384 |      2      5      7 | 3..7
      32768 |      1      2      3 | 2..3
```

### Consequences for freezing the points

1. **The sweep must bracket the BF16 threshold.** At 2048-token contexts all the structure lives
   between concurrency 19 and 58. A sweep of 1/2/4/8 sits wholly in the compute-only region and would
   report the benefit as ~1.7x while missing the capacity effect entirely. A sweep that starts at 64
   sits wholly past it and would attribute the whole gain to precision.

2. **Concurrency points must be chosen per workload, not shared across workloads.** The thresholds
   above move with context length, so one shared list cannot bracket all three workload classes. This
   cuts against the natural instinct to sweep identical values everywhere.

3. **The long-context workload cannot decompose the two effects.** At 32k contexts BF16 holds one
   sequence, so capacity binds immediately at concurrency 2 and there is no compute-only region to
   measure. The decomposition is available at short and medium contexts and structurally unavailable
   at long ones. Report the long-context result as combined-only rather than implying a split that the
   data cannot support.

4. **Log per point whether each configuration was KV-limited, plus preemption counts.** vLLM exposes
   both. Without them the two regions cannot be told apart after the fact, and the decomposition
   becomes an argument instead of a measurement.

### Rejected alternative — do not re-litigate

A two-arm design was considered and rejected: one arm with KV pinned to the BF16-feasible budget for
all configurations (isolating compute), one arm with KV floating (deployment-realistic).

Rejected because it doubles the sweep to buy a portability claim that D4 already places out of
scope, and because the pinned arm measures a configuration nobody would deploy — it deliberately
handicaps FP8/FP4 and would understate the real benefit, risking a knee in the wrong place. The
single floating-KV sweep is both the deployment-realistic measurement and, read by region, the
decomposition. `num_gpu_blocks_override` is the mechanism that would pin KV if this is ever revisited
(see hazard H5 in `EXPERIMENTAL_CONTRACT.md`).

**Unlock condition:** freeze exact per-workload concurrency points here and in
`EXPERIMENTAL_CONTRACT.md` together with D10's token counts. The two decisions are not separable —
the thresholds above are a function of context length, so token counts must be fixed first.

---

## D12 — Optional GPTQ / AWQ / INT4 comparisons

**Status:** DEFERRED

These are not part of the core study unless they answer a specific secondary question and can be served fairly on the target stack.

**Why:** Adding method families too early would shift the project from precision-boundary measurement toward a broad PTQ-method benchmark.

---

## D13 — Quality-evaluation path (full logits for KL)

**Status:** LOCKED (2026-08-22) — vLLM full-vocab logprobs, measured feasible

**Decision:** obtain next-token distributions from the **same vLLM engine that serves the
configuration**, via `SamplingParams(logprobs=128256)` with the engine started at
`max_logprobs=128256`, driving the model with explicit `prompt_token_ids`.

**Why not HF transformers for the quality side.** Running the compressed-tensors checkpoints under
transformers would not exercise the CUTLASS FP8/FP4 kernels, so measured quality would not belong to
the deployed configuration. D2 requires the unit of comparison to be the whole deployment.

**Measured feasibility** (artifact: `results/qualification/kl_feasibility.json`):

```text
full vocab returned:        128,256 entries per context (not top-k truncated)
probability mass:           sums to 1.0
context token identity:     identical token IDs across BF16 / FP8 / FP4
storage per context (fp16): 250 KiB
   1,000 contexts:            0.24 GiB
  10,000 contexts:            2.39 GiB
 100,000 contexts:           23.89 GiB
```

Storage is tractable: the BF16 reference distributions are written once and each quantized
configuration is then streamed against them, so only one reference set is ever held.

**Feed token IDs, not text.** Contexts are supplied as `prompt_token_ids` so tokenization cannot
drift between configurations. `compute_kl.py` aborts if the stored contexts are not token-identical.

**Note on scope:** this is a next-token-distribution rig (one distribution per context). If dense
per-position KL is wanted later, `prompt_logprobs` returns every prompt position in one pass and
would raise storage by roughly the context length — re-evaluate before adopting.
