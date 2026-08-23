# Limitations

This document should evolve with the project. It records the boundaries of what the experiment can support even if the measurements themselves are precise.

## Hardware specificity

The primary study is performed on one RTX PRO 4000 Blackwell workstation. Serving speedups, memory capacity, kernel behavior, and the location of the quality-serving boundary are therefore hardware-specific.

The study does not by itself establish the same numeric boundary on A100, H100, B200, other Blackwell SKUs, consumer GPUs, or cloud inference accelerators.

### The tight BF16 baseline amplifies the measured capacity benefit

Measured at qualification (`results/qualification/qualification_summary.json`), BF16 leaves only
4.84 GiB for KV cache on this 24 GiB card — 39,664 tokens, 1.21x concurrency at 32k context. FP8 and
FP4 then show 2.3x and 3.0x the KV tokens.

Those ratios depend on the small denominator. The KV *gain* is created by weights vacating a scarce
24 GiB budget, so the scarcer that budget, the larger the gain looks:

```text
this machine (24 GiB):  BF16 KV headroom is the binding constraint -> capacity gain is large
a larger card (80 GiB): an 8B BF16 model already has ample KV room  -> capacity gain shrinks
                        toward irrelevance, and the case for quantization rests on bandwidth alone
```

This cuts in a specific direction: **the study is biased toward making quantization look
worthwhile**, on the memory/concurrency axis, relative to a VRAM-rich deployment of the same model.

Consequences for the final claim:

- Do not present the concurrency/capacity multiplier as a property of FP8 or FP4. It is a property of
  this model at this precision *on a 24 GiB GPU*.
- Report the bandwidth benefit and the capacity benefit separately, as `HARDWARE_PROFILE.md`
  implication 3 requires. Under the locked decode workload the below-wall half is bandwidth, not
  compute, and it travels better than the capacity half (see below).
- A knee located mainly by capacity gains is the most hardware-specific result this project can
  produce, and should be labelled as such.

The inverse bias is also worth stating: a model large enough that BF16 barely fits at all would
exaggerate the benefit further, and one small enough to leave BF16 with abundant headroom would hide
it. 8B on 24 GiB was chosen knowing it sits in the regime where the memory axis is measurable — that
is a deliberate experimental choice, not a neutral observation.

## Backend specificity

The serving result depends on the selected inference backend, scheduler, kernels, and software versions. A precision format that performs poorly in one stack may perform differently in another.

The final claim should describe the measured deployment configuration, not generalize from “FP8” or “FP4” as abstract categories.

## Model specificity

If the primary study uses one model, the result establishes a within-model deployment boundary, not a universal LLM quantization boundary.

If a small model ladder is added, it provides evidence about generalization across those tested models only. It still does not establish a universal law across architectures or training recipes.

## Workload specificity

Quantization value may differ between:

- prefill-heavy workloads;
- decode-heavy workloads;
- balanced workloads;
- low concurrency;
- saturated serving.

A single quoted speedup should never be treated as the whole serving result unless the workload is named.

### Phase 1 measures one workload, and it is decode-dominated

D10 narrows phase 1 to `DECODE_PRIMARY` (512 in / 2048 out) plus a supporting prefill probe. This
buys concurrency resolution near the KV wall at the cost of workload breadth, and the cost must be
stated plainly:

- **The study does not measure the arithmetic benefit of low precision.** Decode arithmetic intensity
  is approximately the batch size, so across the whole locked sweep (concurrency 1-96) an 8B model is
  memory-bandwidth-bound and nowhere near the roofline crossover. What is measured below the KV wall
  is fewer weight *bytes read*, not fewer FLOPs. `PREFILL_PROBE` gives a bounded observation of the
  arithmetic path at concurrency 1-8; it is not a sweep and must not be extrapolated into the
  tradeoff curve.
- **Prefill-dominated deployments are out of scope.** RAG, long-document summarization, and
  large-context agentic prefill are exactly the regimes the primary workload does not cover.
- **No balanced/mixed shape is measured**, so nothing is known about how the boundary moves between
  the two regimes.
- **The result is a concurrency-dependent boundary for one shape**, not a workload-dependent map.

### The bandwidth half travels better than the compute framing suggested

An earlier version of D11 argued that the compute half of the benefit generalizes to other hardware
while the capacity half does not. Under a bandwidth-bound decode workload the below-wall benefit is
bandwidth, not FLOPs, and that generalizes *more* strongly: every GPU must read weights, whereas FP4
tensor-core support is specific to recent architectures.

This is a limitation stated in the favourable direction, and it should not be overread. It says the
below-wall *mechanism* is portable, not that the numbers are. The realized ratio still depends on the
target card's achievable bandwidth and on whether that card's decode regime is also bandwidth-bound
at the concurrency of interest.

### The workload is synthetic in two controlled ways

- **`ignore_eos=True` with exact output lengths.** Required so that output length does not become a
  dependent variable of precision, but real deployments have variable-length generations and a
  distribution of early stops.
- **Fixed-exact, homogeneous prompt and output lengths.** Real traces are heterogeneous. Homogeneous
  batches make the scheduler's job easier and likely overstate throughput — hopefully near-equally
  for all three configurations, but that equality is an assumption, not a measurement.

## Precision labels are underspecified

`BF16`, `FP8`, and `FP4` do not uniquely identify a deployment. Weight precision, activation precision, KV-cache precision, scale format, group/block size, calibration method, backend, and kernel path can all change the result.

Final conclusions must refer to configuration IDs from `QUANTIZATION_CONFIGS.md`.

## Quality is sampled along the generation, not measured densely

The KL rig retains ten strided positions out of a 2,048-token generation (`EVALUATION_RIG.md`).
Degradation between retained positions is interpolated, not observed, and any behavior that appears
and resolves inside a stride interval is invisible.

More fundamentally, teacher-forced KL against a BF16-generated continuation measures divergence
*given BF16's trajectory*. It does not measure how far a quantized configuration's own free-running
generation drifts from BF16's, which is the quantity a user of a decode-heavy deployment actually
experiences. The two coincide only if divergence does not compound through the sampling loop, which
this study does not test.

## Quality metrics are incomplete views of behavior

KL divergence is sensitive but does not directly state whether a behavioral change matters to users. Perplexity is corpus-dependent. Downstream benchmarks are sparse and task-dependent.

The project therefore uses several quality views rather than claiming that one metric fully captures “model quality.”

## Downstream benchmark coverage

The task suite will intentionally be limited to keep the project focused. Stable performance on the selected tasks does not prove absence of degradation on untested capabilities, languages, domains, safety behavior, long-context behavior, or rare-token distributions.

## Calibration dependence

Calibration-dependent quantization may vary with calibration dataset, draw, size, and preprocessing. A single quantized checkpoint may not represent the expected performance of the method.

The project should characterize this where relevant but may not exhaust every calibration choice.

## Benchmark-client and host effects

The host is strong enough that it is not expected to dominate normal single-GPU serving, but request generation, tokenization, synchronization, or client-side bottlenecks can contaminate high-concurrency results.

CPU and client behavior should be monitored during saturation testing before claiming the GPU is the limiting resource.

## Power / thermal / clock behavior

The GPU uses dynamic clocks and power management. Final results represent performance under the locked stock power/environment policy, not an architecture-level theoretical maximum.

Run-to-run telemetry is needed to detect thermal or power-state anomalies.

## Statistical resolution

A measured difference smaller than benchmark or evaluation uncertainty may be practically real but unresolved by this experiment. In that case the correct conclusion is that the study cannot distinguish the configurations at the required precision, not that they are identical.

## No predetermined knee

The project may find:

- FP8 is the preferred boundary;
- FP4 remains worthwhile;
- BF16 remains preferable for some workload;
- different workloads have different boundaries;
- no single sharp knee exists.

The analysis should not force a unique boundary when the measured frontier does not support one.
