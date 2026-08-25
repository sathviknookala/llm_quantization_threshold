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

## D10 — Workload definition

**Status:** LOCKED (2026-08-23) — one primary decode-heavy workload plus a supporting prefill probe

**Decision.** Phase 1 measures one primary workload and one cheap supporting probe. The three-shape
trio previously described in `PROJECT_SPEC.md` is narrowed and `BALANCED` is dropped.

```text
DECODE_PRIMARY    512 input / 2048 output tokens    full concurrency sweep, 3 repetitions
PREFILL_PROBE    8192 input /   32 output tokens    concurrency 1-8, no repetition structure
```

### The lens: this is a memory-focused study

Both halves of the measured benefit are consequences of one thing — weight residency shrinking — and
the workload is chosen so that a single sweep separates them:

```text
below the KV wall   fewer weight BYTES READ per decode step    -> bandwidth benefit
above the KV wall   fewer weight BYTES RESIDENT -> more KV      -> capacity benefit
```

This supersedes the earlier compute-vs-capacity framing. See D11.

### Why 512 / 2048

**KV arithmetic.** This model stores 128 KiB of KV per token
(32 layers x 8 KV heads x 128 head_dim x 2 for K/V x 2 bytes). That constant reproduces the measured
capacity to within 0.04% — 4.84 GiB / 128 KiB = 39,649 tokens against 39,664 recorded in
`results/qualification/qualification_summary.json` — so the wall arithmetic below is trustworthy.

Peak footprint per sequence is 2,560 tokens (0.3125 GiB). Concurrency at which each configuration
becomes KV-limited:

```text
              KV wall (peak)   KV wall (mean occupancy, 1,536 tok)
BF16                      15                                   25
FP8                       36                                   60
FP4                       46                                   77
```

**The walls sit below compute saturation, by construction.** Decode arithmetic intensity is
approximately the batch size, so at batch 15-46 an 8B model is deeply memory-bandwidth-bound and
nowhere near the roofline crossover. The entire sweep stays bandwidth-bound. This is the property a
shorter decode shape lacks: at 128/1024 the BF16 wall lands at 34 (62 by mean occupancy), where it
risks confounding with compute saturation and the capacity effect may be unobservable.

**Occupancy grows 5x over a sequence's life** (512 -> 2,560 tokens). The KV wall is therefore a band
between the peak and mean figures above rather than a single concurrency, and preemption arrives
progressively rather than at a threshold. That is a property to measure, not a defect — see hazard H8
in `EXPERIMENTAL_CONTRACT.md`.

**Prefill is negligible in this shape.** Using the coarse prefill rate derived below, 512 input
tokens cost roughly 0.11 s against roughly 51 s of decode at BF16 batch 1 — under 0.3% of the
request. The workload is effectively pure decode, which is what makes the bandwidth reading clean.

### Falsifiable prediction for the pilot

Subtracting prefill from the qualification smoke runs via the two-point difference between the
medium (2048/128) and long (16384/128) workloads, then dividing decode rate into weight bytes:

```text
          weight bytes   decode tok/s (derived)   implied bandwidth
BF16          16.10 GB                    40.3             649 GB/s
FP8            9.12 GB                    72.3             659 GB/s
FP4            6.07 GB                   101.8             618 GB/s
```

All three land within ~6% of each other, and the predicted speedup ratios from weight size alone
(1.77x, 2.65x) track the derived ratios (1.79x, 2.53x).

**This is an estimate, not a result.** It comes from single-run, no-warmup artifacts
(`results/qualification/*_smoke.json`, labelled coarse under hazards H1/H2) and assumes decode cost
is identical at 2k and 16k context, which it is not exactly. It must not be quoted as a measurement.

It is recorded here because it gives the pilot a sharp test: **below the wall, the throughput ratio
should equal the weight-size ratio.** If it holds, the entire below-wall region is explained by one
number. If it does not, that is worth discovering before committing a full sweep.

### Why the study narrows to one workload

The memory lens needs *resolution near the walls* more than it needs shape variety. Three walls
(15 / 36 / 46) all live inside a narrow concurrency band, and distinguishing the bandwidth region
from the capacity region requires dense sampling through it. Dropping workload breadth is what pays
for that density within the same GPU-time budget.

`BALANCED` is dropped specifically because 512/2048 has absorbed its role. Balanced existed to be the
clean decomposition workload; this shape decomposes better, with the wall at 15 rather than 17 and no
prefill contribution to confound the reading.

### Why the prefill probe is kept

At these batch sizes decode is pure bandwidth, so the primary workload never observes low precision's
**arithmetic** benefit at all. Without the probe, `LIMITATIONS.md` would have to say the study does
not measure the compute benefit of quantization in any form. The probe is ~20 minutes of GPU time and
lets the project state a bounded observation instead of nothing.

It is explicitly **not** a sweep arm: no repetition structure, no saturation search, and its numbers
are reported as a supporting observation rather than as part of the tradeoff curve.

### Companion definitions frozen with the token counts

Token counts alone do not define a workload. Each of the following can invalidate the sweep silently:

1. **Prefix caching OFF for serving runs.** vLLM V1 enables automatic prefix caching by default. With
   shared prompt prefixes it makes prefill nearly free after the first request, producing a fast,
   clean, successful-looking run that measures cache hits. Launch with prefix caching disabled and log
   the hit rate regardless. See hazard H7. The quality rig deliberately enables it (D13,
   `EVALUATION_RIG.md`); that exception does not extend to anything timed.
2. **Real prompt corpus, prefix-disjoint.** Prompts are drawn from a fixed held-out corpus,
   tokenized and chunked to exactly 512 tokens, distinct per request, seeded, and reused
   byte-identically across all three configurations. The repeating filler sentence used during
   qualification is not acceptable for final runs.
3. **`ignore_eos=True`, exact output length.** Otherwise output length becomes a dependent variable
   of precision: a configuration that emits EOS earlier does less work and looks faster for the wrong
   reason. This makes the workload synthetic but comparable, which is the correct trade for the
   serving arm.
4. **Fixed-exact lengths, not sampled.** Homogeneous batches maximise comparability. The cost is that
   real traces are heterogeneous and homogeneous batches make the scheduler's job easier, likely
   overstating throughput for all three configurations near-equally. Recorded in `LIMITATIONS.md`.

### Rejected alternatives — do not re-litigate

- **128 in / 1024 out.** BF16 wall at 34 (62 by mean occupancy) risks landing at or past compute
  saturation, which would leave the capacity benefit unobservable and confounded.
- **Keeping `BALANCED` (2048/256).** Redundant once 512/2048 provides a cleaner decomposition, and its
  cost is paid in concurrency resolution, which is the scarcer resource under this lens.
- **4096 output tokens.** Roughly quadruples the low-concurrency cells; at BF16 batch 1 a single
  request would take over 100 s. The budget is better spent on concurrency points and repetitions.
- **A full prefill-heavy sweep arm.** Prefill dominance requires long prompts, which collapse the
  BF16 wall to concurrency 2-4 and remove any bandwidth-only region. Prefill purity and
  decomposability trade directly against each other; the probe keeps the observation without paying
  for the arm.

### Out of scope, and why the objection is anticipated

The first question a memory-focused study invites is "why not quantize the KV cache too — that is the
direct memory lever?" KV precision is deliberately held at BF16 across the whole ladder (D5) so that
measured capacity gains are attributable to **weight residency alone**. Quantizing KV would confound
weight-residency capacity with KV-density capacity and make the decomposition in D11 unreadable.

FP8 KV cache is the natural phase 2 and is not a gap in phase 1.

---

## D11 — Concurrency sweep

**Status:** LOCKED (2026-08-23) — points frozen for `DECODE_PRIMARY`

```text
1, 4, 8, 12, 16, 24, 32, 48, 64, 96
```

Ten points, deliberately dense through 12-48 where all three KV walls sit (15 / 36 / 46 by peak
footprint). 3 configurations x 10 points x 3 repetitions = 90 timed cells.

`PREFILL_PROBE` uses `1, 2, 4, 8` with no repetition structure.

**Why a sweep rather than one concurrency level:** quantization changes both the bytes read per
decode step and the KV capacity available. At a single point those two effects are indistinguishable.

### The sweep decomposes bandwidth benefit from capacity benefit

A quantization step delivers two things at once, and under this workload both are consequences of
weight residency shrinking:

```text
bandwidth benefit   fewer weight bytes read per decode step -> faster per-token
capacity benefit    fewer weight bytes resident -> more KV  -> more concurrent sequences
```

Both are real and a deployment gets both, so the **headline result is the combined benefit**. But
they do not generalize equally, and the separation is already present in the shape of a single sweep:

- **Below the BF16 wall** every configuration holds every sequence. KV binds for no one, so the
  throughput gap is the **bandwidth benefit alone**. The prediction in D10 says this region should be
  explained entirely by the weight-size ratio.
- **Above the BF16 wall** BF16 begins queuing and preempting while FP8/FP4 keep batching. The gap
  *widens*, and that widening is the **capacity contribution**.

**Correction to the earlier framing.** This decision previously described the split as *compute*
versus capacity, and argued that the compute half travels to other hardware while the capacity half
is a function of this machine's 24 GiB ceiling. Under a bandwidth-bound decode workload the
below-wall benefit is bandwidth, not FLOPs — and that **strengthens** the portability argument rather
than weakening it. Every GPU must read weights; not every GPU has FP4 tensor cores. The below-wall
result generalizes more strongly than the compute framing claimed. The capacity half remains
machine-specific (`LIMITATIONS.md`).

The corollary is a real narrowing, recorded honestly: because the sweep is bandwidth-bound
throughout, it does **not** measure low precision's arithmetic/tensor-core benefit. That is what
`PREFILL_PROBE` exists to observe, in bounded form.

### The saturation criterion is an SLO, and it is mandatory

"Maximum sustainable concurrency" is meaningless without a latency bound — concurrency can always be
pushed higher by accepting worse latency. Under a memory lens this stops being one option among
several and becomes the denominator of the headline metric.

```text
SLO:  TPOT P95 <= 50 ms   (20 output tok/s per user, an interactive floor)
```

Feasible for all three at low concurrency — BF16 at batch 1 is roughly 25 ms — and it binds at
different concurrencies per configuration, which is precisely the deliverable:

> BF16 serves N concurrent users at 20 tok/s each; FP8 serves N'; FP4 serves N''.

### Required per-point logging

Without these the two regions cannot be told apart after the fact and the decomposition becomes an
argument rather than a measurement:

- whether each configuration was KV-limited at that point;
- preemption count;
- recomputed-token count (vLLM preempts by recomputation, so preempted sequences redo prefill);
- queue depth and time-in-queue;
- KV-block utilisation.

### Cell-abort rule

At concurrency 64 the BF16 configuration needs 163,840 KV tokens against 39,664 available — a 4x
oversubscription that will thrash. Predeclared so the decision is not made ad hoc mid-run:

> A cell is aborted and recorded as `SLO_VIOLATED` if TPOT P95 exceeds 10x the SLO or the cell
> exceeds its wall-clock cap. Once a configuration violates the SLO at concurrency C, higher
> concurrency points for that configuration are skipped and recorded as `SKIPPED_PAST_SLO`.

Aborted and skipped cells are results, not missing data. Do not stop a sweep merely because one
configuration "looks fast enough."

### Rejected alternative — do not re-litigate

A two-arm design was considered and rejected: one arm with KV pinned to the BF16-feasible budget for
all configurations (isolating bandwidth), one arm with KV floating (deployment-realistic).

Rejected because it doubles the sweep to buy a portability claim that D4 already places out of
scope, and because the pinned arm measures a configuration nobody would deploy — it deliberately
handicaps FP8/FP4 and would understate the real benefit, risking a knee in the wrong place. The
single floating-KV sweep is both the deployment-realistic measurement and, read by region, the
decomposition. `num_gpu_blocks_override` is the mechanism that would pin KV if this is ever revisited
(see hazard H5 in `EXPERIMENTAL_CONTRACT.md`).

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

**Extended 2026-08-23 for position-resolved KL.** D10's decode-dominated workload made
single-position KL insufficient — it would evaluate quality at position 1 of a 2,048-position
generation. The rig now measures teacher-forced KL at ten strided positions along a BF16-generated
continuation, using **truncated prefixes**: for each position `p`, feed `prompt_token_ids[:p]` and
read the next-token distribution through this same proven path. Full policy in `EVALUATION_RIG.md`.

**`prompt_logprobs` was specified first and then measured infeasible (vLLM 0.19.1, 2026-08-23).** The
earlier note here suggested it as the natural route to per-position KL. It is not:

- full-vocab prompt logprobs over a 2,560-position context returns 328,335,360 `Logprob` objects per
  context, on the order of 16 GB of host RAM in flight per request;
- `sampling_params.py:425` sets `skip_reading_prefix_cache = self.prompt_logprobs is not None`, so it
  also forfeits prefix-cache reuse of the repeated prefill.

Ten truncated-prefix passes cost 1.28 million objects (~64 MB) and keep prefix caching available.
Do not reintroduce the one-pass formulation.

---

## D14 — Perplexity corpus and token budget

**Status:** OPEN (raised 2026-08-23)

`EVALUATION_RIG.md` requires perplexity against fixed held-out corpora but has never named them. This
gate is now explicit rather than an unowned `TBD`, because the quality arm cannot be built without it.

To decide:

- corpus or corpora, and the exact revision/slice;
- token budget per configuration;
- chunking and stride policy, which must match the KL rig's 512-token chunking unless there is a
  stated reason to diverge;
- whether perplexity is computed from the serving engine (consistent with D13's reasoning that
  quality must belong to the deployed configuration) or from a separate path.

**Constraint carried from D13.** The same argument that forced KL through the serving engine applies
here: perplexity computed under a path that does not exercise the CUTLASS FP8/FP4 kernels would not
be the perplexity of the deployed configuration. Do not resolve this by reaching for a transformers
scoring script.

**Constraint carried from the checkpoint deviation.** The corpus should be scored from raw token
continuations, not chat-formatted prompts, while the `chat_template` provenance issue is open.

---

## D15 — Downstream task set

**Status:** OPEN (raised 2026-08-23)

`EVALUATION_RIG.md` calls for a deliberately limited suite spanning distinct capabilities and records
that model selection no longer blocks the choice. The set itself is still unchosen.

To decide:

- which tasks, and how many;
- harness and scoring implementation;
- whether any are deferred until the license lands.

**Blocked in part.** The served checkpoint carries a non-official `chat_template` (348 chars against
the official multi-KB template). Chat-formatted task scores would be internally comparable across
BF16/FP8/FP4 but not comparable to published Llama 3.1 numbers. Either prefer tasks scored from raw
token continuations, or defer chat-formatted tasks until `meta-llama/Llama-3.1-8B-Instruct`
@ `0e9e39f2` can be re-pinned.

**Scope discipline.** The suite exists to detect capability-level degradation the KL and perplexity
views might miss, not to produce a leaderboard. Resist expansion; see D12's reasoning about method
families, which applies equally to task families.

---

---

## D16 — Prompt corpus for serving workloads and KL contexts

**Status:** RESOLVED (2026-08-23) — option (a), C4 `en` validation

**Decision.** Serving prompts and KL contexts are drawn from `allenai/c4`, file
`en/c4-validation.00000-of-00008.json.gz`, tokenized with the served tokenizer and chunked to
exactly the workload's input length.

```text
Corpus version:      c4-en-validation-shard0-v1
Selection seed:      20260823          (deterministic shuffle over the first 60,000 documents)
DECODE_PRIMARY:      512 prompts x exactly 512 tokens   = [BOS] + 511 content tokens
PREFILL_PROBE:        64 prompts x exactly 8192 tokens  = [BOS] + 8191 content tokens
Prompt-set hash:     2681c604332813f2e893b252bc512efaac15316874d22352917492e5a40b130f  (512 tok)
                     ead1b35968a2bd3a267996ba250603c45fe47e5ff97505637f774f1493813f07  (8192 tok)
Prefix-disjointness: SHA256 over the first 64 content token IDs, uniqueness enforced
Source disjointness: no C4 document is reused across prompts; each 512-token prompt is one document
```

Artifacts: `results/pilot/corpus/*_manifest.json`. Body files hold the token IDs and are gitignored
as regenerable at the fixed seed; the manifests carry the hashes that make a run reproducible.
Reproduce with `python scripts/pilot/corpus.py --workload DECODE_PRIMARY`.

**Why (a) and not (b).** For pure serving timing the token content is irrelevant, so random token
IDs would have been defensible on measurement grounds. Real text is chosen because it costs one prep
script and preserves D10's link between the quality and serving distributions — the KL contexts are
literally the first 32 serving prompts, same corpus, same chunking.

**Why C4 and not something else.** General web text rather than chat-formatted, so it sidesteps the
`chat_template` provenance deviation; cleanly disjoint from `HuggingFaceH4/ultrachat_200k`, which
calibrated the FP4 rung (D7) and is therefore excluded by the calibration-leakage rule in
`EXPERIMENTAL_CONTRACT.md`.

**Exact-length construction.** Prompts are emitted as **token IDs**, not text, so tokenization
cannot drift between configurations. One slot is reserved for BOS because the completions endpoint
passes `prompt_token_ids` through without adding it — so the count is exact rather than
exact-plus-or-minus-one. The 8192-token probe prompts concatenate consecutive distinct documents,
since C4 documents are rarely that long; document identity is still disjoint across prompts.

**Relationship to D14.** This gate covers the serving prompts and the KL contexts only. D14's
perplexity corpus remains a separate open choice.

---

