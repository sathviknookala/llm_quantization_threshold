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

The workload is chosen so that a single sweep separates two effects by *where KV binds* — an
observable, not an attribution:

```text
below the KV wall   nobody is capacity-limited  -> realized throughput difference, measured
above the KV wall   fewer weight BYTES RESIDENT -> more KV -> capacity benefit
```

**Corrected 2026-08-24.** This previously read "fewer weight bytes read per decode step -> bandwidth
benefit", asserting that both halves are consequences of weight residency shrinking. Pilot P1
falsified the predictive form of that claim and the study does not isolate what produces the
sub-wall difference, so the below-wall row states what is measured and attributes nothing. The
capacity row stands and was confirmed by P2. See D11.

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

### Falsifiable prediction for the pilot — TESTED 2026-08-23, FALSIFIED

**The assumption was:** below the KV wall the throughput gain from a quantization step is roughly the
weight-size ratio between the two configurations. BF16:FP8:FP4 weights are 16.10 : 9.12 : 6.07 GB, so
the expected speedups were **1.77x and 2.65x**, and the below-wall region would be "explained by one
number".

**That is false.** Pilot P1 measured it at concurrency 1, 8 and 12, three repetitions each, per-cell
CV 0.01-0.15% (`results/pilot/p1_bandwidth_assumption.csv`):

```text
  C     BF16      FP8      FP4    R_FP8    dev    R_FP4     dev
  1    36.06    66.01    87.88    1.831  +3.7%    2.437   -8.1%
  8   268.31   449.99   566.99    1.677  -5.0%    2.113  -20.3%
 12   381.30   618.08   763.77    1.621  -8.2%    2.003  -24.5%
```

FP4 lands outside a +/-20% tolerance at C=8 and C=12, and **both** ratios decline with concurrency —
FP8 by 11.5% and FP4 by 17.8% between C=1 and C=12. A weight-size ratio is a single constant and
cannot reproduce a concurrency-dependent gap.

### What replaces it: measure the benefit, do not predict it

**The primary consequence of P1 is subtractive.** The project no longer assumes that compression
ratio predicts serving speedup. Quantization has two distinct resource effects, and only the first
is a capacity statement that follows directly from checkpoint size:

```text
reduced weight residency  ->  more VRAM free for KV  ->  higher feasible concurrency, outward wall
reduced weight traffic    ->  less weight-attributable traffic per decode step
                          ->  *potential* sub-wall throughput benefit, magnitude unknown
```

Total decode behaviour also contains KV traffic, other memory traffic, kernel efficiency, compute
effects and scheduling effects. So `compression ratio` does **not** imply proportional throughput
gain, and the magnitude of the realized serving benefit is an **experimental output of the sweep**,
not something inferred beforehand. The sweep reports the observed below-wall throughput empirically
and separately measures the movement and usefulness of the KV wall.

### Post-hoc candidate explanation — documented, not adopted

**This model was fitted after seeing P1's data and is not a project premise.** It is recorded because
it is the most plausible account on hand and a reasonable follow-up if the question is ever pursued
directly. Nothing downstream depends on it, and it is explicitly **not** a pre-sweep gate — see
"Status" at the end of this section.

The bytes a decode step must move are not only the weights:

```text
bytes per decode step  =  weights  +  KV read  +  other
                          ~~~~~~~     ~~~~~~~     ~~~~~
                          shrinks     common      roughly constant
                          with        across      per configuration
                          precision   the ladder
```

and therefore

```text
                      W_BF16 + KV(C) + O_BF16
throughput ratio  =  -------------------------
                      W_quant + KV(C) + O_quant
```

Only the **first** term shrinks when weight precision drops. The weight-size ratio is the limiting
case of this expression when the other two terms go to zero, which at 512/2048 they do not.

**Why KV is common to all three rungs.** KV precision is deliberately held at BF16 across the whole
ladder (D5), so per-token KV cost is the same 128 KiB for every configuration — confirmed by
measurement: 44,688 / 97,888 / 120,944 tokens at 131,072 bytes each. KV traffic therefore appears
*identically* in numerator and denominator and pulls the ratio toward 1. It compresses the larger
ratio harder, which is why FP4 degrades faster than FP8, and it grows with concurrency, which is why
both ratios decline.

**The arithmetic.** Dividing per-step bytes by the independently measured 620.1 GB/s read ceiling
(P4) and subtracting the weight and KV terms leaves a residual that is roughly constant per
configuration and does not scale with concurrency. Only the 620.1 GB/s figure is a measurement here;
the decomposition around it is not:

```text
cfg    C    tok/s   step bytes        W        KV     other   other%
BF16   1     36.1       17.20G   16.10G     0.22G    0.88G     5.1%
BF16   8    268.3       18.49G   16.10G     1.83G    0.56G     3.0%
BF16  12    381.3       19.51G   16.10G     2.62G    0.80G     4.1%
FP8    1     66.0        9.39G    9.12G     0.19G    0.08G     0.9%
FP8    8    450.0       11.02G    9.12G     1.81G    0.09G     0.8%
FP8   12    618.1       12.04G    9.12G     2.80G    0.12G     1.0%
FP4    1     87.9        7.06G    6.07G     0.20G    0.78G    11.1%
FP4    8    567.0        8.75G    6.07G     1.73G    0.95G    10.9%
FP4   12    763.8        9.74G    6.07G     2.80G    0.88G     9.0%
```

FP8 is almost entirely accounted for by weights plus KV — a 1% residual. BF16 and FP4 each carry
roughly 0.7-0.9 GB of additional per-step cost, which for FP4 is 9-11% of its whole step and is the
reason its ratio undershoots even at concurrency 1.

**Read `other` as a residual, not a measured traffic term.** It absorbs genuine extra traffic
(activations, quantization scales) *and* any failure to reach the read ceiling. FP4 achieves only 86%
of measured read bandwidth at batch 1 against FP8's 97%, so a large part of FP4's residual is
execution efficiency in the SM120 CUTLASS FP4 path rather than bytes moved. The two are not separated
by this measurement.

**What is *not* overturned.** The sub-wall region showed zero preemption, zero recompute, clean
scaling and stable TPOT at every sub-wall point — it is not capacity-limited below the wall, which is
the property D11's ladder actually depends on. What failed is the claim that *weight* bytes alone set
the ratio.

**Status — post-hoc, not a gate. Settled 2026-08-24.** The three-term model was constructed after
seeing the data and its `other` term is fitted per configuration. It reproduces R_FP4(12) to 0.3%,
which is unsurprising for a model fitted on that point. It is therefore **not** a validated
prediction, it is **not** promoted into a pre-sweep gate, and validating it is **not** required to
run the primary experiment. Should anyone wish to test it later, the pilot already holds points it
was not fitted on (BF16 at C=15/16, FP8 and FP4 at C=24) and the test costs no GPU time — but that is
a follow-up, not a dependency.

**What may and may not be claimed about the below-wall gap.** The gap has *not* been uniquely
attributed to HBM bandwidth, nor to KV dilution, nor to kernel efficiency; this measurement cannot
separate them. The supported claims are narrower:

- quantized configurations show reproducible sub-wall throughput differences;
- the weight-ratio prediction does not explain them quantitatively;
- the full sweep will measure those realized differences;
- P2 independently confirmed the BF16 KV-capacity wall at [17, 18], which preserves the capacity-side
  motivation of the study on its own evidence.

### Why the study narrows to one workload

The memory lens needs *resolution near the walls* more than it needs shape variety. Three walls
(15 / 36 / 46) all live inside a narrow concurrency band, and distinguishing the bandwidth region
from the capacity region requires dense sampling through it. Dropping workload breadth is what pays
for that density within the same GPU-time budget.

`BALANCED` is dropped specifically because 512/2048 has absorbed its role. Balanced existed to be the
clean decomposition workload; this shape decomposes better, with the wall at 15 rather than 17 and no
prefill contribution to confound the reading.

### Why the prefill probe is kept

At these batch sizes the primary workload does not isolate low precision's **arithmetic** benefit at
all. Without the probe, `LIMITATIONS.md` would have to say the study does not measure the compute
benefit of quantization in any form. The probe is ~20 minutes of GPU time and
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

**Ladder resolution — recorded 2026-08-24, points unchanged.** These ten points do not resolve the
FP8 wall (~38) from the FP4 wall (~47): both sit in the single 32-48 gap, so the sweep alone returns
`[32, 48]` for each. The grid spacing there is 1.5x against a 1.24x KV separation. The locked points
are **not** re-spaced — that would rewrite a pre-registered set after seeing pilot data. Instead a
separate `SWEEP_REFINE` phase bisects each bracket afterwards (`EXPERIMENTAL_CONTRACT.md`,
"Wall-refinement pass"). Reported walls must state which phase produced them.

**Why a sweep rather than one concurrency level:** quantization changes both the bytes read per
decode step and the KV capacity available. At a single point those two effects are indistinguishable.

### Pilot outcome — the ladder holds, the predictive shortcut does not (2026-08-23)

P1 and P2 tested this decision. **P1 FAILED, P2 PASSED.** Diagnostic pilot numbers, not results;
artifacts under `results/pilot/`.

**The below-wall region is not capacity-limited, as assumed.** Throughput scales with concurrency,
TPOT P95 stays within 27.7-31.3 ms for BF16 across C=1-12, and no configuration preempts below the
wall. That is what the two-region reading of the sweep needs, and it survives.

**But the below-wall gap is not the weight-size ratio.** FP4's ratio falls from 2.437 at C=1 to 2.003
at C=12 — a 17.8% drift that breaches P1's stability criterion, and deviations of -20.3% and -24.5%
from the predicted 2.652 that breach its tolerance. FP8 stays inside both. Full evidence in D10.

**Consequence for reading the sweep — amended 2026-08-24.** The below-wall gap is measured where no
configuration is KV-limited, so it is not a capacity reading. It is **not**, on this evidence,
uniquely a bandwidth reading either: weight traffic, common KV traffic, kernel efficiency and
scheduling all sit inside it and the pilot cannot separate them. It is **concurrency-dependent** and
it is reported as what it is — the realized throughput difference at a stated concurrency. Two rules
follow, and they are the same ones stated under "Resolved 2026-08-24" below:

- the sub-wall region cannot be summarised by one ratio per configuration; report it as a curve, and
  attach the concurrency to any single-number claim;
- do not call it "the bandwidth benefit" or "the weight-residency benefit". Reduced weight traffic is
  one contributor among several, and the measurement does not isolate it. Report the realized
  difference; attribute nothing.

**The wall is set by peak occupancy, not by a peak-to-mean band.** P2 measured the BF16 transition
directly:

```text
measured KV capacity, this engine      44,688 tokens
median occupancy                       ~1,600 tokens/sequence   (D10 assumed 1,536 - close)
max occupancy                          ~2,530 tokens/sequence   (D10 assumed 2,560 - close)

peak-basis prediction  44,688 / 2,560 = 17.5
measured bracket       last clean C=17, first pressured C=18     reproduced in 2 repetitions
```

D10's occupancy arithmetic was right. What was wrong is the claim that the wall is "a band between
the peak and mean figures". Preemption begins when the *instantaneous* aggregate footprint exceeds
capacity, which happens at the crest of the occupancy oscillation (H9), so the **peak basis is the
wall predictor and the mean-occupancy basis is not an upper edge of anything** — it would have
predicted headroom to C=29, wrong by eleven concurrency points. The band is one concurrency wide.

Restated wall estimates on the peak basis, using each configuration's measured serving-path capacity
rather than the qualification figures (H10 — capacity is not reproducible across engine launches, so
these must be recomputed per engine instance):

```text
BF16   44,688 KV tokens / 2,560 = 17   measured bracket [17, 18]
FP8    97,888 KV tokens / 2,560 = 38   not yet measured
FP4    not yet measured under the serving path
```

**Degradation past the wall is milder than H8 predicted, at least at C=18.** BF16 at C=18 recorded 7
preemptions per cell and throughput of 437 tok/s against 490 at C=16 — a regression, not a collapse.
The superlinear preemption-recompute feedback loop H8 describes was not observed one point past the
wall. Note also that the wall's first symptom is **throughput regression at C=17** (490 -> 425) with
*zero* preemptions, so a pressure test keyed only on preemption counters detects the wall one
concurrency point late.

### Pilot outcome 2026-08-23 — the ladder is confirmed, the below-wall prediction is withdrawn

**Status of this decision: the concurrency ladder is confirmed; the below-wall *prediction* is
withdrawn and replaced by measurement.** Pilot P1 FAILED and P2 PASSED
(`results/pilot/PILOT_DECISION.md`). **Amended 2026-08-24: P1's failure does not gate the sweep and
is not rerun** — see `EXPERIMENTAL_CONTRACT.md` "Exit criteria".

- **P2 PASS.** The BF16 KV wall was located at **C in [17, 18]**, reproducibly: C=17 clean twice
  (0 preemptions across 32 telemetry samples), C=18 pressured twice (4 preemptions in 4 separate
  samples, KV P95 0.974). That is inside the pre-registered 15-25 neighbourhood, so the locked
  concurrency points do bracket the wall and the ladder stands. Predicted from the *measured* KV
  capacity of this engine (44,688 tokens) the peak-footprint basis gives 17 and the mean-occupancy
  basis 29; the measured wall sits at the peak-footprint edge, so **peak footprint is the better
  predictor for this workload** and the mean-occupancy column overstates the wall.
- **P1 FAIL.** The below-wall throughput ratio is not the weight-size ratio, and it shrinks as
  concurrency rises (see D10). The consequence is that the sweep **measures** the below-wall benefit
  rather than predicting it from checkpoint size. No replacement predictor is required first.

**Resolved 2026-08-24 — how the below-wall gap is reported.** The sentence "below the BF16 wall the
throughput gap is the bandwidth benefit alone" was wrong as written and has been corrected above. The
decision is to **report the below-wall gap as the raw, deployment-relevant measured quantity**, at
each concurrency, without attributing it to any single mechanism. Nothing is divided out of the
headline number, because the number a deployment actually experiences includes all of it.

Three reporting obligations follow:

- **Always name the concurrency.** The below-wall gap is not a constant. FP4 measured 2.44x at
  concurrency 1 and 2.00x at concurrency 12. "FP4 is 2.4x faster" is only true at batch 1.
- **Never label it a weight-residency or weight-bandwidth benefit,** and do not label it a bandwidth
  benefit either. It is the realized throughput difference; the study does not resolve what produces
  it. Any mechanistic attribution must be computed separately and labelled a modelled quantity, not a
  measurement.
- **Never derive it from the compression ratio.** The magnitude of the serving benefit is an output
  of the sweep, not an inference from checkpoint size. That is the whole content of P1.

**Unaffected.** The sub-wall region is free of capacity limitation — no preemption, no recompute,
clean scaling — which is the premise the two-region reading needs. Hardware bandwidth is now measured
independently at 620 GB/s read (P4), so sub-wall throughput can be stated against a real ceiling
rather than a self-referential one.

### The sweep separates a capacity effect from a throughput effect — amended 2026-08-24

A quantization step has two distinct resource consequences. Only the first follows directly from
checkpoint size:

```text
quantization -> smaller weight residency -> more VRAM free for KV -> greater KV capacity
             -> smaller weight traffic   -> potential throughput benefit, magnitude unknown
```

The capacity leg is an arithmetic consequence of a smaller checkpoint on a fixed 24 GiB card, and P2
confirmed the mechanism directly by locating the BF16 wall at [17, 18]. The throughput leg is only a
*potential*: total decode behaviour also contains KV traffic, other memory traffic, kernel efficiency,
compute effects and scheduling effects, so `compression ratio` does **not** imply proportional
throughput gain. **The magnitude of the actual serving benefit is an experimental output of the
sweep, not something inferred from the compression ratio.** That is what P1 settled.

Both are real and a deployment gets both, so the **headline result is the combined benefit**. The two
regions of a single sweep separate them by *where KV binds*, which is an observable, not an
attribution:

- **Below the BF16 wall** every configuration holds every sequence and KV binds for no one, so the
  throughput gap is not a capacity effect. It is the **realized throughput difference at that
  concurrency** — reproducible, concurrency-dependent, and smaller than the weight-size ratio. It is
  reported as measured and is not attributed to a mechanism. See D10.
- **Above the BF16 wall** BF16 begins queuing and preempting while FP8/FP4 keep batching. The gap
  *widens*, and that widening is the **capacity contribution**, whose mechanism P2 did confirm.

**Consequence for how the sweep is reported.** The below-wall gap is the right deployment-relevant
quantity and is reported as measured, always with its concurrency attached. What it may **not** be
called is "the weight-residency benefit", "the bandwidth benefit", or anything else that names a
cause the measurement did not isolate.

**Correction to the earlier framing.** This decision previously described the split as *compute*
versus capacity, then as *bandwidth* versus capacity. Neither is supported. The sub-wall gap has not
been attributed to bandwidth, to arithmetic, or to anything else, so the portability argument that
rested on it — "every GPU must read weights, so the sub-wall result travels" — is **withdrawn as a
claim** and left as an open question. The capacity half remains machine-specific
(`LIMITATIONS.md`), and how far the throughput half generalizes is simply not something this study
establishes.

The corollary is a real narrowing, recorded honestly: at these batch sizes the primary workload does
not isolate low precision's arithmetic/tensor-core benefit. That is what `PREFILL_PROBE` exists to
observe, in bounded form.

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

