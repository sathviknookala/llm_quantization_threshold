# LLM Quantization Serving–Quality Benchmark

A controlled study of the serving benefit and distributional shift from quantizing **Llama 3.1 8B Instruct** across **BF16, FP8, and NVFP4**. The benchmark pairs a vLLM concurrency sweep with teacher-forced next-token KL divergence on a single **RTX PRO 4000 Blackwell 24 GB GPU**.

**Headline: moving from FP8 to NVFP4 provides only 1.23× the SLO-compliant concurrency, while the direct FP8→FP4 step produces 10.22× the KL divergence of the BF16→FP8 step.**

The serving ceilings are measured once at the refined boundary (**n=1**); confirmatory replication remains incomplete. KL measures distributional change, not a proportional loss in task accuracy.

## Project Objective

At what point does the additional serving benefit from further quantization stop justifying the additional change in model behavior?

The project measures two axes:

- **Serving capacity:** maximum in-flight requests while p95 time per output token (TPOT) stays at or below **50 ms**.
- **Distributional shift:** KL divergence between next-token distributions evaluated on identical token histories.

BF16 is the deployment reference. FP8 and NVFP4 are complete deployment configurations, including their activation precision, calibration, kernels, and runtime settings. The result describes a tradeoff for this setup; selecting a deployment still requires an application-specific tolerance for behavioral change.

## Experimental Setup

| Component | Measured configuration |
| --- | --- |
| Model | Llama 3.1 8B Instruct |
| GPU | NVIDIA RTX PRO 4000 Blackwell, SM120, 24 GB class |
| Power limit | 145 W |
| Serving stack | vLLM 0.19.1; PyTorch 2.10.0+cu128 |
| Runtime | Python 3.12.13; CUDA toolkit 12.9; driver 575.64.03 |
| Primary workload | 512 input tokens → 2,048 output tokens |
| Evaluation text | Frozen prompts from C4 English validation |
| KV-cache precision | BF16 for every configuration |

The checkpoint came from the NousResearch mirror, with weight shards verified byte-identical to the official checkpoint. A chat-template difference remains documented; these measurements use raw token IDs. See [checkpoint provenance](docs/QUANTIZATION_CONFIGS.md) and the [hardware profile](docs/HARDWARE_PROFILE.md).

### Quantization configurations

| Configuration ID | Weights | Activations | Execution path |
| --- | --- | --- | --- |
| `BF16_REFERENCE` | BF16 | BF16 | Default BF16 GEMM |
| `FP8_PRIMARY` | FP8 E4M3, static per-channel scales | Dynamic per-token FP8 | CUTLASS FP8 scaled GEMM |
| `FP4_PRIMARY` | NVFP4, group size 16 | Dynamic NVFP4, group size 16 | FlashInfer/CUTLASS SM120 FP4 GEMM |

FP8 uses a calibration-free dynamic-activation recipe. NVFP4 uses 128 UltraChat calibration samples, truncated to 2,048 tokens, with seed 0. Both recipes exclude `lm_head` from quantization. Checkpoints are produced offline with llmcompressor 0.10.0.3 in a separate environment.

**KV precision is fixed; KV capacity is allowed to grow.** Smaller weight allocations leave more VRAM available for the same BF16 KV cache. The measured capacity gain therefore includes the benefit of fitting more cached tokens, without changing their precision.

Exact recipes and creation commands are in [Quantization Configurations](docs/QUANTIZATION_CONFIGS.md).

## Harness Design

The implementation separates serving measurements from quality measurements:

| Component | Responsibility |
| --- | --- |
| [Serving harness](scripts/harness/) | Engine lifecycle, closed-loop requests, warmup, timing, telemetry, resumable concurrency sweeps |
| [Quality harness](scripts/harness/quality/) | Frozen trajectories, full-vocabulary scoring, KL computation, bootstrap intervals, reference-launch analysis |
| [Tracked results](results/) | Cell records, configuration identities, manifests, quality summaries, and gate outcomes |

The harness records checkpoint and context hashes, runtime controls, and observed engine identity. Quality collection checks token alignment, complete position coverage, numerical validity, storage precision, and dispatch evidence before interpreting the distributions.

### Serving methodology

A closed-loop client maintains a fixed number of in-flight requests. Every request uses exactly 512 prompt tokens and 2,048 generated tokens, with greedy decoding, early stopping disabled, and serving prefix caching disabled.

The original sweep contains **124 recorded cells** across the coarse ladder, supporting prefill probe, and boundary refinements. Coarse decode measurements use three repetitions; the refined SLO boundaries use one. Warmup and measurement windows account for request lifetimes and KV-occupancy cycles.

TPOT is computed per completed request, then summarized at the 95th percentile. The reported SLO applies to requests that start and finish within the measurement window. Queue depth and completion counts accompany the latency result.

### Quality methodology

The quality run freezes **64 BF16-generated continuations**, each containing 2,048 tokens. Every configuration scores the same next-token prediction at ten retained generation positions:

```text
1, 8, 32, 64, 128, 256, 512, 1024, 1536, 2048
```

At position `p`, the input is the 512-token prompt followed by the first `p − 1` frozen continuation tokens. The target token is excluded from the input. This gives **640 scored contexts per collection** over the full 128,256-token vocabulary.

For reference distribution $P$ and comparison distribution $Q$:

$$
D_{\mathrm{KL}}(P\|Q)
= \sum_{v \in V} P(v)\log\frac{P(v)}{Q(v)}.
$$

Log-probabilities are stored in float32 and normalized and compared in float64. The headline averages the ten positions within each trajectory, then averages across trajectories. **10,000 bootstrap resamples of whole trajectories** produce nominal 95% confidence intervals; the ten positions are not treated as independent samples.

The production run collected **3,200 cells** in the order BF16 launch 1 → FP8 → BF16 launch 2 → FP4 → BF16 launch 3. The first BF16 launch remains the headline reference; the additional launches quantify reference-side variation.

Quality uses the locked `graph_2048` profile: CUDA graphs enabled and `max_num_batched_tokens=2048`. It shares the checkpoints and selected execution paths with serving, but uses an in-process engine and enables prefix caching for nested prefixes. Those differences are recorded explicitly.

See the [Evaluation Rig](docs/EVALUATION_RIG.md) and [Experimental Contract](docs/EXPERIMENTAL_CONTRACT.md).

## Measured Results

### Serving capacity

Maximum concurrency under **TPOT p95 ≤ 50 ms**, from the refined **n=1** measurements:

| Configuration | Maximum in-flight requests | First SLO breach | TPOT p95 at ceiling | Output throughput at ceiling |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 21 | 22 | 47.59 ms | 488.18 tokens/s |
| FP8 | 57 | 58 | 49.66 ms | 1,401.23 tokens/s |
| NVFP4 | 70 | 71 | 49.57 ms | 1,745.46 tokens/s |

FP8 supports **2.71×** BF16's concurrency. NVFP4 supports **3.33×** BF16's concurrency, or **1.23×** FP8's.

Source: [serving cell records](results/sweep/cells.jsonl), `SWEEP_REFINE_SLO` entries. The file also contains one subsequent ceiling-replication cell; the original sweep accounts for 124 records.

### Distributional shift

| Direct comparison | Mean KL, nats | Nominal 95% bootstrap CI | Worst retained cell, nats |
| --- | ---: | ---: | ---: |
| BF16 → FP8 | 0.005558 | [0.003611, 0.008752] | 0.3383 |
| BF16 → NVFP4 | 0.052980 | [0.036497, 0.077175] | 2.4477 |
| FP8 → NVFP4 | 0.056795 | [0.038483, 0.085353] | 2.6036 |

Source: [production KL summary](results/quality/kl/kl_summary.json). The arrow denotes the KL direction: the left-hand configuration supplies the reference distribution.

### The marginal tradeoff

Using the measured point estimates:

$$
\text{Concurrency gain from FP8 to FP4}
= \frac{70}{57}
\approx 1.23\times
$$

$$
\text{Ratio of successive-step KL divergences}
= \frac{D_{\mathrm{KL}}(P_{\mathrm{FP8}}\|P_{\mathrm{FP4}})}
{D_{\mathrm{KL}}(P_{\mathrm{BF16}}\|P_{\mathrm{FP8}})}
= \frac{0.056795}{0.005558}
\approx 10.22\times.
$$

FP8 captures most of the measured capacity improvement, while the next quantization step introduces substantially more distributional movement for a smaller capacity gain. This supports FP8 as a **candidate knee in the tradeoff for the tested configuration**.

The FP8→FP4 divergence is measured directly. KL is not additive, so subtracting the two BF16-referenced values would not measure that step. The 10.22× value compares adjacent-step divergences; it is not a task-error multiplier.

## Reproducing and Inspecting Results

Run commands from the repository root.

### Inspect committed measurements

The serving inventory reads the recorded cells without launching a GPU job:

```bash
python scripts/harness/summarize_sweep.py
```

Read the quality summaries and rederive the headline ratio:

```bash
python - <<'PY'
import json
from pathlib import Path

pairs = json.loads(
    Path("results/quality/kl/kl_summary.json").read_text()
)["pairs"]

for name, result in pairs.items():
    ci = result["headline_bootstrap"]
    print(f"{name}: {result['headline_nats']:.6f} nats "
          f"[{ci['ci_low']:.6f}, {ci['ci_high']:.6f}]")

ratio = pairs["FP8||FP4"]["headline_nats"] / pairs["BF16||FP8"]["headline_nats"]
print(f"Successive-step KL ratio: {ratio:.2f}x")
print(f"FP8-to-FP4 concurrency ratio: {70 / 57:.2f}x")
PY
```

### Recompute or collect measurements

Use the pinned `qnt` environment for serving and quality analysis, and `qnt-quant` for checkpoint production. Model paths are defined in [common.py](scripts/harness/common.py). NVFP4 initialization requires `nvcc` and `ninja` on the serving process's `PATH`.

With the original distribution shards and required artifacts present:

```bash
conda run -n qnt python scripts/harness/quality/p13.py --analyze
```

For collection on the configured measurement machine, follow the checkpoint setup, corpus preparation, and preflight requirements in the [Experimental Contract](docs/EXPERIMENTAL_CONTRACT.md) and [Evaluation Rig](docs/EVALUATION_RIG.md). The production entry point is:

```bash
conda run -n qnt python scripts/harness/quality/p13.py --collect --analyze
```

**A clone contains summaries, frozen trajectories, and provenance, but not checkpoints or the full distribution `.npy` files.** Exact distribution-level recomputation requires the original shards. Fresh collection is a new measurement: BF16 relaunches are not bit-reproducible, and regenerating continuations from the same seed does not recreate the frozen trajectory set.

## Scope and Remaining Work

- **Serving replication is incomplete:** 1 of 18 confirmatory cells is recorded. The refined 21/57/70 ceilings remain n=1; FP8 and FP4 clear the SLO by only about 0.34 and 0.43 ms.
- **KL is a conditional distribution metric:** it covers 64 contexts at ten positions on BF16-generated histories. Perplexity, downstream-task accuracy, and quantized models' own free-running trajectories have not been evaluated.
- **The BF16 reference has a measurable floor:** production self-KL is approximately `1.865e-4` nats. The pre-registered 1% floor criterion still fails for BF16→FP8: the floor is 3.75% of the signal using the pre-run estimate, or 3.36% using this run's estimate. No floor is subtracted and no threshold was relaxed. See the [P13 analysis](results/quality/kl/p13_summary.json).
- **The result is deployment-specific:** one model, GPU, software stack, workload shape, and NVFP4 calibration draw. Runtime settings affect the distributions; the observed throughput gains have not been attributed to a single kernel or bandwidth mechanism.
- **The client is a controlled workload:** fixed-length closed-loop requests do not represent a real user arrival process. The SLO excludes unfinished requests, and the original sweep lacks host/client CPU telemetry.

The serving sweep and production KL run are complete. Confirmatory serving replication and task-level quality evaluation remain open. Detailed qualifications and unresolved measurement issues are tracked in [Limitations](docs/LIMITATIONS.md) and [Decisions](docs/DECISIONS.md).
