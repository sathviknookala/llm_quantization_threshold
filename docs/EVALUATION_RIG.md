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

1. Generate one continuation from the BF16 reference for each context: 512 prompt tokens, then
   exactly 2,048 generated tokens. Only BF16 generates. The token sequence is frozen, hashed, and
   reused byte-identically by BF16, FP8 and FP4 scoring alike.
2. Retained positions are positions **in the generated continuation**, not offsets into the
   concatenated sequence:

```text
1, 8, 32, 64, 128, 256, 512, 1024, 1536, 2048
```

3. For retained position `p`, every configuration is fed the context

```text
context(p) = prompt[0:512] + continuation[0:p-1]      length = 511 + p
target(p)  = continuation[p-1]                        never appended before scoring
```

   and the returned next-token distribution is the one that predicts `target(p)`. Distributions come
   from the ordinary `SamplingParams(logprobs=128256)` call proven in D13, driven by explicit
   `prompt_token_ids`.

Worked, so the convention cannot be re-derived incorrectly from prose:

```text
p =    1   context = prompt[0:512]                     len  512   predicts continuation[0]
p =    8   context = prompt[0:512] + cont[0:7]         len  519   predicts continuation[7]
p =  512   context = prompt[0:512] + cont[0:511]       len 1023   predicts continuation[511]
p = 2048   context = prompt[0:512] + cont[0:2047]      len 2559   predicts continuation[2047]
```

**Corrected 2026-08-25.** This section previously said "feed `prompt_token_ids[:p]`", which is wrong
under either reading and silently so. Read against the concatenated 2,560-token sequence, `p = 1`
feeds a single BOS token and `p = 1..512` all target positions inside the *original prompt*, never
touching generated content — the exact defect this section exists to remove; `p = 2048` would score
generated token 1537. Read against the 512-token prompt instead, `p = 1024`, `1536` and `2048`
collapse to three byte-identical contexts, because a Python slice beyond a list's length returns the
whole list without raising. The formula above replaces that wording; D13 carried an independent copy
of the same error and is corrected with it.

Ten passes per context per configuration. Feeding the prefix that ends before `target(p)` and
reading the next-token distribution is exactly teacher-forced KL at generation position `p`, so this
is equivalent to the one-pass formulation and not an approximation of it. Contexts within a
trajectory are strictly nested — `context(p_i)` is a byte-identical prefix of `context(p_j)` for
`p_i < p_j` — and that nesting is asserted at runtime before any scoring call is issued.

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
passes over a shared context reuse the prefill.

**Storage — fp32, revised 2026-08-25.** Earlier revisions of this section costed storage at fp16
(250 KiB per retained position, 2.5 MiB per context). Persisted-logprob precision is now a
pre-registered numerical gate rather than an assumption, and fp32 is the default (see *Persisted
distribution precision* below), so the real figures are **501 KiB per retained position, 4.89 MiB per
trajectory, 939 MiB for the locked 3 x 64 x 10 grid**. That is negligible against the machine's free
disk and buys the study out of a numerical argument it would otherwise have to keep making.

**Prefix caching is ENABLED for the quality rig.** This is the opposite of the serving contract, and
deliberately so: here the shared prefix is the same token sequence by construction and caching is
what makes ten passes affordable. Prefix caching remains disabled for all serving runs (H7).

That caching "changes no returned distribution" was asserted here from 2026-08-23 and never
measured. It is now an empirical gate: cache-on versus cache-off distributions are compared on a
fixed subset for BF16 **and** FP4, and the observed difference is reported against the replication
floor below. Failure aborts and escalates rather than silently falling back, because scoring with
caching off reintroduces exactly the cost the truncated-prefix design exists to avoid.

The ten nested prefixes are submitted in **ascending length order** so the shorter prefix's blocks
are committed before the longer prefix asks for them, and observed cache-reuse counters are recorded
per trajectory rather than assumed.

**Contexts are drawn from the same corpus and the same 512-token chunking as the serving workload**
(corpus resolved in **D16**), so quality and serving are measured on the same distribution. The
locked evaluation sample is the **first 64 prompts of the frozen `DECODE_PRIMARY` set**, taken in
their stored order. They are not redrawn, reshuffled or resampled, and they are supplied as token
IDs rather than reconstructed text. Narrowing to one workload removes the
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

**Method declared 2026-08-25, as this section required: the marginal step is computed directly as
`D_KL(P_FP8 || P_FP4)`, with the FP8 distribution as the reference.** All three configurations are
scored on the same contexts, so the FP8-referenced pair is measured, not inferred.

Deriving it as `D_KL(P_BF16 || P_FP4) - D_KL(P_BF16 || P_FP8)` is **barred**. KL is not additive, the
difference of two BF16-anchored divergences is not a divergence between FP8 and FP4, and it can take
either sign for reasons that have nothing to do with the FP8 -> FP4 step.

### BF16 trajectory generation — LOCKED 2026-08-25

Each of the 64 prompts receives **exactly one** BF16 reference continuation. Only `BF16_REFERENCE`
generates. FP8 and FP4 never generate their own evaluation histories for this experiment: their
distributions are always measured on BF16's trajectory, which is what makes the three configurations
comparable at all.

```text
max_tokens           2048          temperature          0.7
min_tokens           2048          top_p                0.9
ignore_eos           true          top_k                -1   (disabled; this build canonicalises to 0)
seed                 20260823      min_p                0
detokenize           false         presence_penalty     0
                                   frequency_penalty    0
                                   repetition_penalty   1.0
```

The continuation must contain **exactly 2,048 generated token IDs**; a short or long trajectory
aborts the run. Token IDs are taken directly from the engine's output token-ID list and are never
obtained by decoding to text and re-encoding — BPE merges across the prompt/continuation boundary
would silently produce a different sequence from the one actually sampled.

Trajectories are frozen with the source prompt token IDs **inlined**, so the artifact is
self-contained rather than depending on the corpus body, which is gitignored as regenerable. The
artifact is hashed over canonical parsed content, and every later scoring pass records the hash it
consumed.

Generation is submitted in groups of **16**, below BF16's measured KV wall of `[17, 18]` (D11), so no
trajectory is produced under preemption. What recompute-preemption does to a seeded per-request
generator is not something this project has measured, so it is avoided rather than assumed benign.

**Freezing was gated on the engine profile and is now released.** Generation and scoring share one
profile — `graph_2048` — because the alignment check compares a sampled token against the *scoring*
engine's distribution; freezing under a different profile would couple the two through an unmeasured
choice.

Once written, `results/quality/trajectories.json` is **immutable experimental input**. Every KL
context in the study is a prefix of one of these continuations, so regenerating them is a new
experiment identity, not a rerun — the freezing path refuses to overwrite, and no later phase
regenerates during resume.

**Replayability is measured, not assumed — and the answer is no.** Whether the locked seed and
profile reproduce the same token IDs across independent BF16 launches was tested directly by
regenerating all 64 trajectories in a second launch under an identical engine identity, seed,
profile and group size (`results/quality/gates/replayability.json`):

```text
identical trajectories   51 / 64
diverging                13
earliest divergence      token 3 of 2048
median divergence        token 30
```

The two launches resolved to the same `engine_identity_hash` (`ca16377ea4206028`) and the same
44,688-token KV cache, so this is not a configuration difference. Sampling at `temperature = 0.7`
draws from a distribution whose logits differ in the last bits between launches — non-deterministic
reduction order in the kernels — and a seeded RNG stream lands on a different token whenever those
last bits straddle a sampling boundary. Once it does, the trajectories diverge for good.

**This does not invalidate the design; it is the reason the artifact is frozen.** Once generated and
validated, the token IDs in `trajectories.json` and their hash *are* the evaluation contexts. Nothing
downstream regenerates them, resume never re-derives them, and every scoring pass records the
`trajectory_set_hash` it consumed. What would be invalid is treating the generation *procedure* as
the contract; the generated *tokens* are the contract.

### Grid completeness — LOCKED 2026-08-25

Every trajectory must carry **all ten** retained positions in every configuration. Missing,
duplicated, mislabeled and out-of-vector positions are hard failures at collection and again at
analysis. A partial trajectory is never averaged: the headline is a mean of per-trajectory means, so
nine positions would silently be weighted as if they were ten. The statistical sample stays exactly
64 trajectory means.

Stored cells are not trusted. Analysis independently reconstructs each context and target from the
frozen trajectory, its trajectory index and its recorded position, and requires a byte-for-byte match
with what was stored — the check that catches a position-label scramble, which no same-cell
self-consistency check can see.

Seeded generation is not assumed to be replayable: vLLM's per-request seed fixes sampling given the
logits, but the logits depend on batch composition. Replayability is measured and recorded; the
guarantee the experiment actually relies on is the frozen artifact and its hash, not re-derivation.

### Aggregation — LOCKED 2026-08-25

```text
K_c          = mean KL over trajectory c's 10 retained positions      -> 64 values
headline_KL  = mean(K_1 .. K_64)
```

computed separately for `BF16||FP8`, `BF16||FP4` and `FP8||FP4`. The position-resolved curve across
all ten retained positions is preserved and reported alongside the headline; it is not a diagnostic
but a first-class result, since whether degradation is flat or accumulates over a generation is the
question a decode-heavy deployment actually asks.

Under this balanced design — every trajectory contributing exactly ten positions — the mean of
trajectory means is *algebraically identical* to pooling all 640 cells. The aggregation order is
specified anyway because the two diverge as soon as group sizes differ, and because the order is what
makes the resampling unit unambiguous. **A trajectory with fewer than ten valid positions aborts the
run**; the grid is never silently aggregated unbalanced.

### Uncertainty — LOCKED 2026-08-25

The independent statistical unit is the **trajectory**, not the retained position. There are 64
independent units, not 640: the ten positions within a trajectory are correlated repeated measurements
on one sampled context.

```text
draws                10,000        resampling unit    trajectory (whole position structure travels with it)
sample               64 trajectories with replacement from the original 64
statistic            that resample's headline mean
interval             percentile 95%: lower = 2.5th, upper = 97.5th
interpolation        linear (matching common.quantiles)
seed                 pre-registered constant, recorded in the run manifest
```

The original 64-trajectory headline remains the point estimate; the bootstrap distribution estimates
its context-sampling uncertainty. Position-resolved CIs use the **same** draw indices, so the ten
position curves are jointly consistent. The same indices are shared across the three pairs, which
means a paired quantity such as "is the FP8 -> FP4 step costlier than BF16 -> FP8" must be computed
from the shared draws directly and **never** inferred from whether two marginal CIs overlap.

Report the bootstrap bias and standard error beside every interval. Percentile intervals at n = 64 on
a right-skewed non-negative statistic are not guaranteed nominal coverage; that caveat travels with
the number (see `LIMITATIONS.md`). A bias-corrected variant is deferred, not adopted.

### Engine profile — LOCKED 2026-08-25 (G9)

The quality rig drives an in-process engine while the serving axis ran an HTTP server, so the engine
controls that can change execution are pinned, recorded from **observed** state rather than from
requested flags, and — where they can move a number — measured rather than assumed. vLLM is
documented in this project to deviate silently from what was requested (H10), which is why a
requested-flag hash is not an engine identity.

```text
detokenize                false   pinned; the logprob path otherwise detokenises every returned
                                  vocab entry individually, which nothing here reads
max_num_batched_tokens     2048   the value the measured serving axis actually ran under; the
                                  offline engine would otherwise default to 8192
enable_prefix_caching      true   the deliberate H7 exception
max_logprobs             128256   differs from serving's default of 20; a request-validation bound
                                  with no effect on kernels, memory or dispatch
enforce_eager             false   CUDA graphs; decided by G9, see below
```

The profile is named **`graph_2048`**. Both equivalences were measured before it was fixed
(`results/quality/gates/engine_profile.json`, 16 launches, 40 cells each, 4 provisional
trajectories):

- **eager versus CUDA-graph execution** — identical contexts and checkpoint, with only
  `enforce_eager` flipped;
- **chunked versus unchunked prefill** — `max_num_batched_tokens = 2048`, which splits the longest
  retained context (2,559 tokens) across scheduler steps, against the offline default of 8192.

```text
                floor(graph)   floor(eager)   eager_vs_graph   chunk_vs_unchunk
BF16               3.984e-04      3.834e-04        5.812e-04          4.280e-04
FP8                1.140e-11      2.765e-06        4.032e-03          1.629e-10
FP4                7.798e-11      4.632e-06        3.739e-02          2.153e-02
```

Headline nats, mean-within-trajectory then across trajectories. **Neither control is numerically
inert.** For BF16 both differences sit at its own replication floor, but FP4 moves 3.74e-02 nats
when `enforce_eager` flips (top-1 changes in 4 of 40 cells) and 2.15e-02 nats between 2048 and 8192
(top-1 changes in 6 of 40). FP8 moves 4.03e-03 nats under the eager flip. These are quantified
results about the engine, not merely a configuration choice; `LIMITATIONS.md` carries them.

`graph_2048` is selected because it is the profile the quality axis must share with the measured
serving axis:

- it is the only profile that reproduces the serving sweep's KV capacities for **all three**
  configurations — 44,688 / 97,888 / 120,944, matching `results/sweep/`. Eager lands
  44,864 / 98,064 / 121,120 and 8192 lands 41,728 / 95,888 / 119,360;
- 2048 is the batching profile the completed sweep actually ran;
- it gives the lowest headline replication floor for every configuration, and 5 orders lower than
  eager for FP8 and FP4.

The decision releases trajectory freezing. Locking `enforce_eager` changes `KL_SPEC`, so the G9
artifact carries the earlier spec hash `4ef13273db16d285` — the hash under which it was measured.

### Reading a difference against the replication floor

Every engine-level difference is reported as an **absolute magnitude in nats first**, with the ratio
to the measured floor as a secondary reproducibility diagnostic. The two must not be conflated: FP8's
chunking difference is 15.7x its replication floor and simultaneously 6.15e-09 nats, which is
nothing. `above_replication_floor` says a difference is reproducible, not that it matters. No
post-hoc materiality threshold is applied to KL values; the raw numbers are preserved and the
thresholds that exist are correctness trip-wires with pre-registered bounds.

### Persisted distribution precision — GATED 2026-08-25

Full-vocabulary distributions are persisted as **fp32 by default**, and KL is computed in **float64**
from log-normalised values (`logp = lp - logsumexp(lp)`), with no epsilon floor inside the logarithm.
A zero comparison probability where the reference has mass is a validity failure, not a value to be
floored.

Storage precision is a pre-registered gate, not an assumption. From one set of collected fp32
distributions, KL is recomputed with both operands rounded to the candidate representation, and the
candidate is adopted only if per-cell relative error stays within its pre-registered bound above an
absolute-nats floor.

**The gate has not been run.** No fp32 distribution array exists anywhere in this repository yet —
every persisted logprob array to date is fp16 — so there is currently nothing to compare against and
no measured fp16-versus-fp32 figure may be quoted. fp32 is the default on the grounds that it costs
under a gigabyte and removes the question, not on the grounds of a measurement. Whether fp16 would
pass is an open expectation until the gate runs on collected fp32 data.

### Replication floor — LOCKED 2026-08-25

BF16 is scored against BF16 across two **independent engine launches** on a fixed trajectory subset,
under the real ten-prefix workload. The resulting self-KL is the measurement's noise floor and is
reported alongside every KL result, so a `BF16 -> FP8` or `BF16 -> FP4` effect can be read against
engine and run noise rather than against zero.

This is a distinct quantity from the pre-sweep correctness gate's self-check, which compared batch
orderings within a single launch and returned exactly 0.0 — a bit-identity result under one batch
shape that says nothing about the batching this rig actually uses.

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

**Measured 2026-08-25.** The sweep produced the headline metric for all three configurations
(`results/sweep/`, 124 cells, zero defects):

```text
                max concurrency within TPOT P95 <= 50 ms      tok/s at ceiling
                ladder point   refined (n=1)   first breach
BF16                      16              21             22              488.2
FP8                       48              57             58             1401.2
FP4                       64              70             71             1745.5
```

FP8 sustains 2.71x BF16's concurrency, FP4 3.33x BF16 and 1.23x FP8. The refined figures come from
the `SWEEP_REFINE_SLO` bisection and are **repetition 1 only**; label them n=1. The ladder points
bracketing them are n=3 with <=0.22% spread. KV-pressure walls, separately refined, are [17,18],
[38,39] and [47,48] — each exactly what the peak-footprint arithmetic predicted.

These are the serving axis only. **The marginal tradeoff and any deployment boundary require the
quality axis, which is not measured**, so nothing here locates a knee.

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
