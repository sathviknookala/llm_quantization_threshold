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
- Report the below-wall throughput benefit and the capacity benefit separately, as
  `HARDWARE_PROFILE.md` implication 3 requires. The study does not establish which of the two travels
  better (see below).
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
- **The headline ceilings are unreplicated.** The 21 / 57 / 70 max-in-SLO figures come from a
  repetition-1 bisection (n=1). The n=3 ladder points bracketing them agree to <=0.22%, so the risk
  is low, but these numbers have not been repeated and must be labelled n=1 wherever quoted.
  Replicating the three bisections costs about nine cells.
- **`PREFILL_PROBE`'s BF16 arm stops at C=4.** The 50 ms TPOT SLO is a decode criterion and should
  not have governed a 32-output-token shape; it fired at C=2 for BF16, so C=8 was skipped. The point
  may be KV-infeasible anyway (65,792 tokens needed against 44,688), but it is absent for a
  misapplied rule rather than a measurement, and the probe needs a TTFT-based criterion before re-run.
- **The KV walls are resolved to different precisions.** BF16's was bracketed to [17, 18] by a
  dedicated pilot search. The sweep's locked ladder bracketed FP8 and FP4 only to [32, 48] — the same
  interval for both — and the `SWEEP_REFINE` bisection resolved them to [38, 39] and [47, 48]
  (2026-08-25). Any reported wall must still name the phase that produced it: a ladder bracket may not
  be compared numerically against a bisected one.
- **Counterbalancing balances position, not carryover.** The three-repetition Latin square is cyclic,
  so FP8 follows BF16 in two of three repetitions and FP4 in none. Balancing carryover across three
  treatments needs six sequences. A thermal preflight gate and a pre-registered drift test stand in
  for what the design cannot balance; the residual is reported, not assumed away.
- **"Concurrent users" is not what is measured.** The driver is closed-loop with a fixed number of
  in-flight requests, `ignore_eos`, and homogeneous 512/2048 prompts — no arrival process, no
  variability. The headline quantity is **maximum in-flight requests within the SLO**. Phrasing it as
  concurrent users overstates it.
- **`meets_slo` is survivor-biased.** It is computed only over requests that both start and finish
  inside the timed window, so starved and still-in-flight requests contribute nothing. At high
  concurrency it can read true on a cell where much of the offered load is not being served. Queue
  depth and `window_completed_requests` are recorded alongside it and must be read with it.

### The below-wall benefit is not attributed, and its portability is unknown

**Withdrawn 2026-08-24.** This section previously argued that the below-wall benefit is a *bandwidth*
effect rather than an arithmetic one, and therefore travels well because every GPU must read weights
while not every GPU has FP4 tensor cores. **That argument is withdrawn.** Pilot P1 showed the gap is
not predictable from weight size, and nothing in the study isolates which of weight traffic, common
KV traffic, kernel efficiency, or scheduling produces it. An unattributed effect cannot be argued to
port on the strength of one of its candidate causes.

What can be said: the capacity half remains a function of this machine's 24 GiB ceiling, and how far
the below-wall throughput half generalizes is an open question this study does not answer.

**Measured qualification (pilot P1, 2026-08-23).** The below-wall benefit is *not* a single number
that travels — it is concurrency-dependent, and it is not predictable from weight size alone.

- FP4's measured ratio falls from 2.44 at concurrency 1 to 2.00 at concurrency 12 against a
  weight-size prediction of 2.65. Any claim of the form "FP4 is N times faster below the wall" is
  incomplete unless it names the concurrency. D10 records a post-hoc traffic model as a candidate
  explanation; it is fitted, not validated, and no conclusion rests on it.
- **FP4's advantage erodes faster than FP8's** as load rises: FP8 drifted 11.5% between concurrency 1
  and 12, FP4 17.8%. A deployment sizing on a batch-1 measurement will over-estimate FP4's benefit
  more than it over-estimates FP8's.
- FP4 additionally sustains a lower fraction of achievable memory bandwidth than FP8 (86% versus 97%
  of a measured 620 GB/s read ceiling at batch 1). The shortfall is unexplained and unattributed; it
  is consistent with per-GEMM overhead in the NVFP4 path of this backend, but that has not been
  demonstrated and is not implied by the format.

The portability claim therefore holds for the *mechanism* (fewer weight bytes read per step) but not
for the *magnitude*. Transporting a specific speedup to another GPU requires knowing that machine's
KV-to-weight traffic ratio at the intended concurrency, and its FP4 kernel efficiency.

### The KV wall is set by peak occupancy, and capacity depends on the serving path and launch hygiene

Two measured constraints on any capacity claim (pilot P2, hazards H9/H10):

- The wall sits where *instantaneous* aggregate KV footprint exceeds capacity, i.e. at the crest of
  the occupancy oscillation, not at mean occupancy. Predicting from mean occupancy overestimated
  sustainable concurrency by eleven points.
- KV capacity depends on the serving path and on the GPU being genuinely free at launch. The offline
  `LLM` path and `vllm serve` report different capacities for the same configuration and flags (BF16
  39,664 vs 44,688 tokens), and a launch begun before a previous engine released VRAM silently sized
  a 10% smaller cache. Across nine clean serving launches BF16 and FP8 were exactly reproducible, so
  this is a launch-hygiene hazard rather than inherent nondeterminism (H10). A wall concurrency must
  still be reported with the capacity of the engine instance that produced it.

### The study measures weight quantization under a fixed KV-cache policy

KV precision is held at BF16 on every rung (D5), so nothing here measures the benefit of quantizing
the KV cache. This is deliberate — it is what makes the measured capacity gain attributable to weight
residency alone — but it bounds the claim in a way worth stating plainly.

At this workload the KV term is not a small correction. At concurrency 12 a decode step reads roughly
2.8 GB of KV against 6.07 GB of FP4 weights, so for the FP4 rung KV is already a third of per-step
traffic, and by concurrency 24 it is comparable to the weights. A deployment that quantized KV as
well would see a materially larger benefit than any number this study reports.

**Therefore:** these results are the benefit of weight quantization *under a BF16 KV policy*, not the
benefit of low-precision serving in general, and they understate what a fully quantized deployment
achieves. `FP8_KV_VARIANT` is reserved in `QUANTIZATION_CONFIGS.md` and is the natural phase 2.

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

## What the KL sample can and cannot support

The KL result rests on **64 sampled contexts**. That is the statistical sample, and the uncertainty
reported with it is context-sampling uncertainty at n = 64 — not uncertainty over models, corpora,
hardware, or quantization draws. The ten retained positions inside a trajectory are correlated
repeated measurements on one context and are never counted as independent observations.

The interval is a plain percentile bootstrap. At n = 64 on a right-skewed, non-negative statistic
like KL, percentile intervals are known to under-cover, more so in the upper tail, so "95%" is
nominal rather than guaranteed. The bootstrap bias and standard error are reported with every
interval so a reader can see the skew rather than infer it.

## Quality numbers carry a measurement floor

KL between two deployments is only interpretable against the noise of the measurement itself. Two
independent BF16 launches scored on the same trajectories produce a non-negative self-KL — the
replication floor — and any BF16 -> FP8 or FP8 -> FP4 movement of comparable magnitude is not
resolvable by this rig. The floor is reported alongside every KL result for that reason.

Two related quantities are measured rather than assumed, and each bounds interpretation: the
difference between scoring with prefix caching on versus off, and the difference between eager and
CUDA-graph execution. Where a measured effect is not large relative to these, the study says so
instead of reporting the effect.

**A floor ratio is not a magnitude.** Under CUDA graphs FP8 and FP4 replicate to ~1e-11 nats, so a
difference of a few nanonats reads as many multiples of the floor while being numerically nothing.
Every comparison here reports the absolute nats first and the ratio as a reproducibility diagnostic;
"above the replication floor" means reproducible, not important.

## The execution profile changes quantized model outputs, and that is a result, not a footnote

The quality axis runs `graph_2048` — CUDA graphs, `max_num_batched_tokens = 2048` — because that is
the execution profile the measured serving axis ran under, confirmed by its exactly reproducing the
sweep's KV capacities for all three configurations. Choosing it was necessary for comparability. But
G9 also measured what the alternatives do, and the finding stands on its own
(`results/quality/gates/engine_profile.json`, 4 trajectories x 10 positions = 40 cells per
comparison):

```text
                   eager vs graph              chunked 2048 vs unchunked 8192
BF16    5.81e-04 nats, top-1 40/40       4.28e-04 nats, top-1 40/40
FP8     4.03e-03 nats, top-1 39/40       1.63e-10 nats, top-1 40/40
FP4     3.74e-02 nats, top-1 36/40       2.15e-02 nats, top-1 34/40
```

FP4's sensitivity to `enforce_eager` — 3.74e-02 nats at the headline, 2.95e-01 at the worst cell — is
the same order as the historical BF16-to-FP4 position-1 median. In other words, for the FP4
deployment an execution-profile change can move the next-token distribution about as far as the
quantization step itself does at short contexts, and it flips the argmax on a tenth of the sampled
cells. BF16 is indistinguishable from its own replication floor under both flips, so the sensitivity
is a property of the quantized paths, not of the rig in general.

Two consequences travel with every quality number:

- KL values here describe these checkpoints **under this execution profile**. A deployment serving
  the same FP4 checkpoint eagerly, or with unchunked prefill, is not guaranteed the same
  distribution, and this study has not measured how far that generalizes.
- The study has not attributed the effect to a mechanism. Graph capture, kernel autotuning
  selection, and prefill chunk boundaries all differ between the profiles; separating them was not
  attempted.

## The quality engine is the deployed configuration, but not the serving process

Quality is measured through the same checkpoints, backend and kernels as the serving axis, verified
by dispatch. It is not measured through the same *process*: the serving sweep ran an HTTP server
under its own scheduler settings, while the quality rig drives an in-process engine with prefix
caching deliberately enabled. Engine controls that can affect execution are pinned and recorded from
observed state, and the ones that differ from serving are enumerated rather than glossed. A claim
that quality and serving were measured under one identical runtime would be stronger than the
evidence.

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
