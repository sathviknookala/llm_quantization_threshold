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

**Teacher-forced continuation KL at strided positions.**

1. Generate a continuation from the BF16 reference for each context (512 prompt + 2048 generated).
2. Feed the *same* token sequence to all three configurations as `prompt_token_ids`.
3. Recover per-position distributions in one pass with `prompt_logprobs` (D13 already notes this
   returns every prompt position per pass).
4. Retain only the strided positions:

```text
1, 8, 32, 64, 128, 256, 512, 1024, 1536, 2048
```

**Why strided and not dense.** Storing every position is out of reach — 250 KiB x 2,560 positions is
640 MiB per context, 64 GiB at 100 contexts. Ten retained positions is 2.5 MiB per context and
250 MiB at 100 contexts, which is tractable on the same terms D13 established.

**Contexts are drawn from the same corpus and the same 512-token chunking as the serving workload**,
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

The corpus/corpora and token budget remain **TBD**.

## 3. Downstream tasks

Use a deliberately limited suite rather than a broad leaderboard.

Desired capability coverage may include:

- factual knowledge;
- general understanding;
- reasoning;
- mathematical reasoning;
- instruction following if appropriate.

Task set: **still TBD, but no longer blocked** — the model is selected (Llama 3.1 8B Instruct,
D6). Note the constraint from checkpoint provenance: the current tokenizer carries a non-official
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
below the BF16 KV wall   throughput gap = bandwidth benefit  (fewer weight bytes read per step)
above the BF16 KV wall   widening gap   = capacity benefit   (fewer weight bytes resident -> more KV)
```

Both halves are consequences of weight residency shrinking. The below-wall region carries a
falsifiable prediction from D10: the throughput ratio there should equal the weight-size ratio. Test
it explicitly rather than assuming it.

Because the sweep is bandwidth-bound throughout, it does **not** measure the arithmetic benefit of
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
