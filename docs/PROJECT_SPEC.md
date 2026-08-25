# Quantization Project Specification

## Research question

The project asks:

> **At what point does further quantization hurt model quality enough that the additional serving benefit is no longer justified?**

More technically:

> **Where is the knee / deployment boundary of the quality–serving tradeoff curve, where the marginal serving benefit from the next quantization step stops compensating for the marginal model-quality loss?**

The project is measurement-driven rather than optimization-driven. It does not depend on FP4 being faster enough, FP8 being quality-neutral, or any particular configuration winning. The measured tradeoff is the result.

The current candidate precision progression is:

```text
BF16 -> FP8 -> FP4
```

The exact deployment configurations represented by those labels remain to be locked after backend validation.

---

# 1. Model strategy

## Core requirement

The core research question can be answered with **one carefully chosen base model**. For a fixed model, successive quantization configurations can be compared directly on quality degradation and serving improvement.

A model ladder is therefore **not a required experimental dimension**.

## Why additional models may still be useful

Additional models answer a different question:

> Does the measured quantization boundary generalize across model sizes or model families?

Multiple models may reveal that:

- quantization sensitivity differs by architecture or training recipe;
- a small model is overhead-bound while a larger model is more memory- or compute-bound;
- the serving benefit of lower precision changes with model scale;
- the quality-serving knee occurs at different precision levels for different models.

## Current open decision

Two valid designs remain:

### Option A — focused primary study

```text
one approximately 7–8B model
    x BF16 / FP8 / FP4
    x workload
    x concurrency
```

Advantages:

- maximizes depth and statistical care;
- keeps the independent-variable structure clean;
- concentrates effort on the actual research question;
- avoids turning the project into a benchmark survey.

### Option B — small replication ladder

```text
~1B -> ~3B -> ~7–8B
```

Each model is tested across the same quantization and workload contract.

Advantages:

- tests whether the deployment boundary changes with model scale;
- separates small-model overhead effects from larger-model memory/compute effects;
- supports a stronger generalization claim.

The model ladder should only be adopted if that extra generalization question is worth the additional experiment count.

## Model-selection constraints

For every chosen model:

- the BF16 reference must fit and serve properly on the target GPU;
- all primary quantization configurations must be available through a fair serving path;
- the same tokenizer/checkpoint lineage must be used across precision comparisons;
- enough VRAM must remain in BF16 for KV cache, runtime workspaces, scheduler state, and realistic serving measurements;
- the model should have suitable evaluation coverage and an unambiguous license / checkpoint source.

Because BF16 is the common reference, the practical maximum model size is constrained by the BF16 deployment rather than by what can fit only after FP8/FP4 compression.

---

# 2. Quantization ladder

## Candidate ladder

```text
BF16 reference -> FP8 -> FP4
```

The experiment is about **deployment configurations**, not datatype names alone.

For each configuration, record:

- base checkpoint;
- weight precision;
- activation precision;
- KV-cache precision;
- quantization format / algorithm;
- calibration requirements;
- calibration corpus and preprocessing;
- group / block size and scale format;
- quantization tool/version;
- inference backend/version;
- kernels/path exercised on the target GPU;
- whether the path is native/efficient on the target GPU;
- resulting checkpoint/artifact identifier.

## BF16

BF16 is the high-precision experimental reference for the same base model. It is not being treated as “unquantized truth” in an abstract mathematical sense; it is the deployment baseline against which the lower-precision configurations are measured.

## FP8

The exact FP8 recipe is open. Do not lock “FP8” until the project specifies weight precision, activation precision, KV-cache behavior, scaling/calibration method, backend, and runtime kernels.

## FP4

The exact FP4 recipe is open. Candidate Blackwell-oriented formats may include NVFP4 or MXFP4, but the primary study should choose a specific reproducible deployment recipe rather than treating all FP4 schemes as interchangeable.

## Optional comparison baselines

INT4 / GPTQ / AWQ or other weight-only PTQ methods may be added only if they answer a specific secondary question and have a fair implementation on the target GPU/backend. They are not required for the core BF16 -> FP8 -> FP4 study.

---

# 3. Evaluation + serving measurement rig

The rig has two complementary sides: model-quality degradation and serving performance.

## A. Model-quality degradation

### Token-level KL divergence

Primary sensitive measurement:

```text
D_KL(P_BF16 || P_quantized)
```

For the exact same token contexts, compare the BF16 next-token distribution with each quantized deployment.

Report at minimum:

- mean token-level KL;
- distribution of per-token KL;
- useful quantiles / tail behavior;
- uncertainty around aggregate estimates.

Primary comparisons:

```text
BF16 -> FP8
BF16 -> FP4
```

The incremental comparison is also central:

```text
FP8 -> FP4
```

because the project asks what the **next** quantization step buys and costs.

### Perplexity

Evaluate fixed held-out language-model corpora using identical examples and preprocessing.

Report:

```text
PPL_BF16
PPL_quantized
Delta PPL
relative Delta PPL where useful
```

### Downstream task degradation

Use a deliberately limited suite that spans distinct capabilities rather than a broad leaderboard benchmark.

Possible categories:

- factual knowledge;
- general language understanding;
- reasoning;
- mathematical reasoning;
- instruction-following if the chosen model and harness make this meaningful.

Use the same evaluation examples for every deployment configuration and analyze paired differences.

### Calibration robustness

For calibration-dependent methods, quantify whether quality results depend materially on the calibration draw or calibration size.

Possible design:

```text
independent calibration draws
x selected calibration sizes
```

Measure effects on KL, perplexity, and task performance. The purpose is not to manufacture run-to-run variance; it is to determine whether the apparent quality loss is stable or calibration-sensitive.

## B. Serving performance

Serve each deployment configuration through the same inference stack and measure at minimum:

- model-resident GPU memory;
- peak GPU memory;
- request throughput;
- input tokens/sec;
- output tokens/sec;
- total tokens/sec where meaningful;
- time to first token (TTFT);
- time per output token (TPOT);
- inter-token latency if separately useful;
- end-to-end request latency;
- P50 latency;
- P95 latency;
- P99 latency where sample count supports it;
- maximum sustainable concurrency.

### Workload regimes — narrowed 2026-08-23 (D10)

Phase 1 adopts a **memory-focused lens** and one primary workload rather than a trio of shapes:

```text
DECODE_PRIMARY    512 in / 2048 out    primary, full concurrency sweep
PREFILL_PROBE    8192 in /   32 out    supporting observation only
```

This is a deliberate scope change from the earlier prefill-heavy / balanced / decode-heavy design,
recorded as a tracked decision. The reasoning: the memory lens needs **resolution near the KV wall**
more than it needs shape variety, and dropping workload breadth is what pays for that resolution
within the same GPU-time budget. Full justification, wall arithmetic, and rejected alternatives are
in D10.

Workload shape is therefore not a primary experimental variable in phase 1. The tradeoff curve is
traced along concurrency alone.

### Concurrency sweep

Locked in D11 and `EXPERIMENTAL_CONTRACT.md`:

```text
DECODE_PRIMARY:  1, 4, 8, 12, 16, 24, 32, 48, 64, 96
```

Dense through 12-48, where all three configurations' KV walls sit. Saturation is defined by an
explicit SLO (`TPOT P95 <= 50 ms`), not by inspecting a throughput curve.

The concurrency sweep carries the whole comparison. **Corrected 2026-08-24 after pilot P1** — the
sweep separates two effects by *where KV binds*, which is observable, and does not attribute either
to a mechanism the measurement cannot isolate:

1. **throughput** — the realized token-rate difference below the KV wall, where no configuration is
   capacity-limited. It is measured, concurrency-dependent, and **not** predictable from the weight
   compression ratio (P1 falsified that prediction; see D10);
2. **capacity** — fewer weight bytes resident leaves more VRAM for KV, raising feasible concurrency;
   visible as the widening gap above the KV wall, and confirmed directly by P2's location of the
   BF16 wall at [17, 18].

The magnitude of the serving benefit is an experimental output of this sweep, not an inference from
checkpoint size. At these batch sizes the primary workload does not isolate the arithmetic/tensor-core
benefit of low precision; `PREFILL_PROBE` supplies a bounded observation of that instead.

---

# 4. Experimental controls and workload definitions

Final serving comparisons are valid only when the runs represent equivalent workloads and differ only in intended experimental variables.

Hold constant where appropriate:

- base checkpoint;
- tokenizer;
- evaluation examples;
- prompt text;
- prompt lengths;
- requested output lengths;
- generation / sampling parameters;
- inference engine and version;
- batching policy;
- scheduler configuration;
- GPU and host machine;
- GPU power configuration;
- driver / CUDA / framework / backend versions;
- KV-cache policy unless it is intentionally part of the configuration;
- warmup procedure;
- cache state policy;
- benchmark duration;
- number of repeated measurements;
- telemetry collection.

The hardware is a **fixed control/context**, not a primary independent variable.

The final run-validity rules, workload token counts, saturation criterion, telemetry cadence, repetition count, and software pins live in `EXPERIMENTAL_CONTRACT.md`.

---

# 5. Success criteria

The project must preserve the “cannot fail” property: a scientifically useful negative or mixed result is still success.

Success does **not** require:

- FP4 to beat FP8;
- FP8 to preserve BF16 quality;
- any fixed throughput improvement;
- any specific knee location;
- any quantization method to win.

## Measurement success

For each tested:

```text
model x deployment configuration x workload
```

the rig should reliably produce quality measurements such as:

```text
Delta KL
Delta PPL
Delta task performance
```

and serving measurements such as:

```text
Delta throughput
Delta latency
Delta VRAM
Delta maximum sustainable concurrency
```

with enough repeatability to distinguish configuration effects from benchmark noise.

## Research success

For every successive quantization step, quantify the marginal tradeoff:

```text
BF16 -> FP8
FP8  -> FP4
```

The project succeeds if it can defensibly answer:

> What do I gain by taking the next quantization step, what do I lose, and how confidently can I measure both?

Example outcomes that would all count as valid results:

- FP8 is nearly quality-neutral and provides a large serving improvement, while FP4 adds little serving value but measurable quality loss;
- FP4 provides meaningful additional value for decode-heavy workloads but not prefill-heavy workloads;
- the boundary changes materially with concurrency;
- the boundary differs across model sizes/families if multiple models are tested;
- quality degradation is too calibration-sensitive to treat one quantized checkpoint as representative;
- task accuracy appears stable while KL detects significant behavioral change.

---

# 6. Final research product

The final product should be more than a benchmark table.

The central artifact is a **quantization deployment frontier / boundary** showing model-quality degradation against serving improvement for each tested deployment configuration and workload.

The final write-up should answer:

> **How far should this LLM be quantized on this deployment stack before the marginal serving benefit stops justifying the marginal degradation in model behavior?**

If multiple models are tested, the stronger second-order question becomes:

> **Does that boundary generalize across model sizes or families?**

The final repository should contain:

- reproducible quantization configurations;
- quality evaluation harness;
- serving benchmark harness;
- locked experimental contract;
- raw results;
- analysis scripts;
- final figures;
- reproduction instructions;
- `docs/LIMITATIONS.md`;
- polished technical report / write-up.

The centerpiece should be one or more figures that make the marginal tradeoff visible, with uncertainty shown where it materially affects interpretation.

---

# Immediate next sequence

1. Resolve the primary model strategy: one focused model vs a small replication ladder.
2. Validate candidate inference backends and exact FP8/FP4 paths on the target GPU.
3. Lock reproducible deployment configurations in `QUANTIZATION_CONFIGS.md`.
4. Pilot workloads to choose meaningful token-count profiles and concurrency levels.
5. Freeze `EXPERIMENTAL_CONTRACT.md` before final data collection.
6. Implement / validate the quality and serving harnesses against that contract.
