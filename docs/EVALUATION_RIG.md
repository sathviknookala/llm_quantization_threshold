# Evaluation Rig

## Purpose

The project needs to measure two things for each deployment configuration:

1. **what changed in model behavior?**
2. **what improved in serving?**

The final analysis joins those measurements into marginal quantization tradeoffs.

---

# A. Model-quality evaluation

## 1. Token-level KL divergence

Primary sensitive metric:

```text
D_KL(P_BF16 || P_quantized)
```

For the same token context and aligned vocabulary distribution, compare the BF16 next-token probability distribution with the quantized distribution.

### Why this metric exists

Task accuracy is coarse and may remain unchanged even when the model's probability distribution moves. Token-level KL provides a dense behavioral-distance signal across many contexts.

### Required outputs

At minimum, save enough data to report:

- number of token contexts;
- mean KL;
- median KL;
- selected quantiles;
- tail statistics if informative;
- uncertainty around the aggregate;
- comparison identity: BF16 vs which configuration.

Do not retain only a single mean if doing so prevents tail or uncertainty analysis later.

### Context and position policy — LOCKED 2026-08-23

D10 makes the primary workload 512 input / 2048 output tokens. A next-token KL measured at a single
position from a 512-token context would evaluate quality at **position 1 of a 2,048-position
generation**. That gap was peripheral under a mixed-workload design; under a decode-dominated one it
is central, and the rig is extended accordingly.

**Teacher-forced continuation KL at strided positions, via truncated prefixes.**

1. Generate a continuation from the BF16 reference for each context (512 prompt + 2048 generated).
   This token sequence is the fixed evaluation context; it is stored and reused byte-identically.
2. For each strided position `p`, feed `prompt_token_ids[:p]` to every configuration and take the
   next-token distribution with the ordinary `SamplingParams(logprobs=128256)` call proven in D13.
3. Strided positions:

```text
1, 8, 32, 64, 128, 256, 512, 1024, 1536, 2048
```

Ten passes per context per configuration. Feeding the prefix that ends at `p` and reading the
next-token distribution is exactly teacher-forced KL at position `p`, so this is equivalent to the
one-pass formulation and not an approximation of it.

**Why not `prompt_logprobs` in a single pass — measured against vLLM 0.19.1, 2026-08-23.** The
one-pass formulation was specified first and does not work at this scale:

- Full-vocab prompt logprobs over a 2,560-position context returns **328,335,360 `Logprob` objects
  per context** — on the order of 16 GB of host RAM in flight per request even at an optimistic
  50 bytes per entry. There is no cap preventing the request; `max_logprobs` is already 128256. It
  simply is not feasible.
- `sampling_params.py:425` sets `skip_reading_prefix_cache = self.prompt_logprobs is not None`, so
  requesting prompt logprobs also disables prefix-cache reads and the repeated prefill cannot be
  amortized.

The truncated-prefix formulation costs 10 x 128,256 = 1.28 million objects per context (~64 MB) and,
because it uses `logprobs` rather than `prompt_logprobs`, keeps prefix caching available — so the ten
passes over a shared context reuse the prefill. Storage is unchanged at 250 KiB per retained
position, 2.5 MiB per context, 250 MiB at 100 contexts.

**Prefix caching is ENABLED for the quality rig.** This is the opposite of the serving contract, and
deliberately so: here the shared prefix is the same token sequence by construction, caching changes
no returned distribution, and it is what makes ten passes affordable. Prefix caching remains disabled
for all serving runs (H7).

**Contexts are drawn from the same corpus and the same 512-token chunking as the serving workload**
(corpus itself is open gate **D16**),
so quality and serving are measured on the same distribution. Narrowing to one workload removes the
need to stratify KL across context-length buckets: one stratum, sampled along the generation axis
instead of across prompt lengths.

**Report KL as a function of position**, not only pooled. Whether degradation is flat or accumulates
over a long generation is a first-order question for a decode-heavy deployment, and pooling hides it.

### Comparisons

Primary:

```text
BF16 vs FP8
BF16 vs FP4
```

Marginal analysis must also quantify the additional behavioral movement associated with:

```text
FP8 -> FP4
```

This may be computed directly where the metric definition is appropriate, or interpreted through the two BF16-anchored distributions. The final method should be declared before analysis.

## 2. Perplexity

Evaluate fixed held-out LM corpora with identical preprocessing and tokenization.

Report:

```text
PPL_BF16
PPL_configuration
absolute Delta PPL
relative Delta PPL where useful
```

The corpus/corpora and token budget are an open tracked gate: **D14**. Two constraints already bind
the answer — perplexity must be computed through the serving engine for the same reason D13 requires
it of KL, and the corpus should be scored from raw token continuations while the `chat_template`
provenance deviation is open.

## 3. Downstream tasks

Use a deliberately limited suite rather than a broad leaderboard.

Desired capability coverage may include:

- factual knowledge;
- general understanding;
- reasoning;
- mathematical reasoning;
- instruction following if appropriate.

Task set: open tracked gate **D15** — no longer blocked by model selection (Llama 3.1 8B Instruct,
D6), but partly blocked by checkpoint provenance: the current tokenizer carries a non-official
`chat_template`, so any chat-formatted task yields scores that are internally comparable across
BF16/FP8/FP4 but **not** comparable to published Llama 3.1 numbers until the official
`tokenizer_config.json` is re-pinned. Prefer tasks scored from raw token continuations, or defer
chat-formatted tasks until the license lands.

Rules:

- use identical examples across configurations;
- prefer paired analysis of the same items;
- preserve per-example outputs/scores when feasible;
- do not repeatedly resample benchmark subsets merely to create run-to-run variance.

## 4. Calibration robustness

Only required for configurations whose quality depends on calibration.

Possible factors:

```text
calibration draw
calibration sample count
```

Candidate sample counts may include values such as 16, 32, 64, 128, but these are not locked.

Measure how calibration changes:

- KL;
- perplexity;
- downstream scores.

The goal is to determine whether the measured precision penalty is stable or whether it depends materially on one calibration sample.

---

# B. Serving evaluation

## 1. Memory

Measure separately where the backend permits:

- model-resident GPU memory after load;
- peak GPU memory during serving;
- memory available for KV cache / scheduler use;
- maximum sustainable concurrency before memory or SLO failure.

Do not reduce memory evaluation to weights-only arithmetic; the runtime footprint is the serving quantity of interest.

## 2. Throughput

Measure as appropriate:

- requests/sec;
- input tokens/sec;
- output tokens/sec;
- total tokens/sec.

Output-token throughput is especially important for decode-heavy workloads; input-token throughput is especially important for prefill-heavy workloads.

## 3. Latency

Measure:

- TTFT;
- TPOT;
- inter-token latency if separately meaningful;
- end-to-end request latency;
- P50;
- P95;
- P99 only when sample count supports a stable estimate.

Do not summarize an entire serving regime with one latency number if quantization changes prefill and decode differently.

## 4. Concurrency / saturation

Sweep the locked D11 concurrency values through the SLO boundary.

### Headline serving metric — LOCKED 2026-08-23

```text
maximum concurrency sustained within  TPOT P95 <= 50 ms
```

Under a memory-focused lens this, not raw tok/s, is the primary serving quantity: "maximum
sustainable concurrency" has no meaning without a latency bound, because concurrency can always be
raised by accepting worse latency. The deliverable sentence is *BF16 serves N concurrent users at
20 tok/s each; FP8 serves N'; FP4 serves N''.*

Aggregate throughput at the SLO boundary is reported alongside it.

### Read the sweep by region

```text
below the BF16 KV wall   throughput gap = realized token-rate difference at that concurrency
above the BF16 KV wall   widening gap   = capacity benefit  (fewer weight bytes resident -> more KV)
```

**Corrected 2026-08-23 after pilot P1, amended 2026-08-24.** D10 predicted the below-wall ratio would
equal the weight-size ratio (1.77x / 2.65x). It does not: measured 1.83x / 2.44x at concurrency 1,
falling to 1.62x / 2.00x at concurrency 12. The prediction is falsified and the compression ratio is
no longer treated as a predictor of speedup. No replacement predictor is adopted; the sweep measures
the realized difference instead.

Consequences for reporting:

- the below-wall gap is **concurrency-dependent**; quote it with the concurrency attached, never as a
  single speedup number;
- it may not be described as a weight-residency, weight-bandwidth, or bandwidth benefit. The study
  does not isolate what produces it. Any mechanistic split must be labelled modelled, not measured;
- it may not be derived from or sanity-checked against the compression ratio;
- the capacity half above the wall is unaffected by this correction, and P2 confirmed its mechanism
  directly.

At these batch sizes the primary workload does **not** isolate the arithmetic benefit of
low precision. `PREFILL_PROBE` supplies a bounded observation of that; do not extrapolate it into the
tradeoff curve.

---

# C. Marginal tradeoff analysis

The project is not primarily interested in isolated absolute measurements. It asks what changes when taking the **next** quantization step.

For a generic serving metric `S` where larger is better:

```text
Delta S(BF16 -> FP8) = S_FP8 - S_BF16
Delta S(FP8  -> FP4) = S_FP4 - S_FP8
```

For a quality-degradation metric `Q` where larger is worse:

```text
Delta Q(BF16 -> FP8)
Delta Q(FP8  -> FP4)
```

The final analysis should present the serving gain next to the corresponding additional quality cost rather than collapsing them prematurely into one arbitrary scalar score.

Possible final visualization:

```text
x-axis: serving improvement
        primary:   max concurrency sustained at TPOT P95 <= 50 ms
        secondary: aggregate throughput at the SLO boundary; KV tokens resident

y-axis: model-quality degradation
        KL (pooled and by generation position) / PPL / task view
points: BF16, FP8, FP4 configurations
facets: generation position, and, if adopted, model
```

The workload facet is gone in phase 1 — D10 narrows the study to one primary workload, so the curve
is traced along concurrency rather than across shapes. The facet that replaces it is **generation
position**, which asks whether the quality cost of a step is constant over a long generation or grows
with it.

A single universal “knee” may not exist. The result may instead be a concurrency-dependent boundary:
a step that is not worth taking at low load may be clearly worth taking past the BF16 KV wall,
because the capacity benefit only exists there.

---

# D. Uncertainty and repeatability

## Quality

Use paired examples whenever possible. Quantify uncertainty on aggregate KL, perplexity changes, and task-score differences using a method appropriate to the metric and dependence structure.

Calibration-dependent methods should separate calibration-draw variation from evaluation-sample uncertainty where the experiment supports doing so.

## Serving

Estimate run-to-run variation from repeated benchmark runs under the locked contract. Do not treat thousands of requests inside one benchmark process as thousands of independent hardware experiments if the dominant noise occurs at the run level.

Pilot measurements should determine how many repetitions are needed to resolve practically meaningful differences.

---

# E. Validation before timing

Every serving implementation must pass a correctness gate before its timing is treated as meaningful.

At minimum:

- confirm the intended checkpoint/configuration loads;
- verify generated/logit behavior is not obviously corrupted;
- confirm the workload harness sends the intended token lengths and request counts;
- inspect the backend path sufficiently to detect unintended fallback when possible.

The exact correctness tolerances for BF16 vs lower-precision outputs are metric-specific and should not be confused with the downstream quality evaluation itself.

---

# F. Result artifacts

Final raw outputs should be machine-readable and preserve the dimensions needed for later re-analysis.

Suggested logical schema:

```text
model_id
configuration_id
workload_id
concurrency
run_id
metric_name
metric_value
units
sample_count
software/env identity
artifact timestamp
```

Quality-specific outputs should preserve example/token identity where feasible. Calibration experiments should preserve calibration-draw identity.
