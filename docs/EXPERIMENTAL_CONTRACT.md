# Experimental Contract

## Purpose

This file defines when two benchmark runs are comparable and when a run is valid. Freeze this contract before final benchmark collection.

Anything still marked `TBD` is a design item, not permission for a benchmark script to choose an arbitrary value.

## Fixed environment

Primary study:

```text
Machine:           profiled single-GPU workstation in HARDWARE_PROFILE.md
GPU count:         1
Power policy:      stock/default 145 W limit unless a later decision changes it
Inference backend: TBD
Software pins:     TBD
```

Final runs must record:

- NVIDIA driver;
- CUDA/runtime/container identity;
- PyTorch/framework version if used by the serving path;
- inference backend version;
- quantization tool version;
- model/checkpoint revision;
- relevant backend flags.

## Variables intentionally changed

Primary experimental variables:

```text
Deployment configuration: BF16 / FP8 / FP4 recipe
Workload profile:         prefill-heavy / balanced / decode-heavy
Concurrency:              selected sweep through saturation
```

Conditional variable:

```text
Model: only if the project adopts a multi-model generalization design
```

Calibration draw / calibration size may be varied inside the calibration-robustness sub-experiment for calibration-dependent formats.

## Controls

Hold constant within a serving comparison:

- base checkpoint;
- tokenizer and revision;
- prompt content;
- prompt lengths;
- requested output lengths;
- generation parameters;
- random/sampling policy where applicable;
- backend;
- scheduler settings;
- batching policy;
- host machine;
- GPU power policy;
- software versions;
- benchmark client/request-generation logic;
- cache policy unless intentionally varied;
- warmup policy;
- timed-run duration / request count;
- telemetry collection.

Hold constant within quality comparisons:

- evaluation examples;
- tokenization;
- context construction;
- metric implementation;
- prompt/template formatting;
- decoding policy for generated-answer tasks;
- scoring implementation.

## Workload profiles

Exact token counts are not yet locked.

### PREFILL_HEAVY

```text
Input tokens:   TBD — long
Output tokens:  TBD — short
Purpose:        emphasize prompt processing / prefill
```

### BALANCED

```text
Input tokens:   TBD — moderate
Output tokens:  TBD — moderate
Purpose:        representative mixed serving regime
```

### DECODE_HEAVY

```text
Input tokens:   TBD — short
Output tokens:  TBD — long
Purpose:        emphasize autoregressive decode
```

Token counts should be selected after pilot runs but **before** final comparisons. Do not tune them separately for each precision configuration.

## Concurrency sweep

Current candidate structure:

```text
1 -> 4 -> 8 -> 16 -> ... -> saturation
```

Exact points: **TBD after pilot**.

Define “saturation” before final runs. A usable definition should be based on an observable serving condition such as throughput flattening while latency grows rapidly, backend queue saturation, memory capacity, or an explicit SLO boundary.

Do not stop a concurrency sweep merely because one configuration “looks fast enough.”

## Warmup

**TBD after backend selection.**

The final contract must specify:

- model-load exclusion;
- number or duration of warmup requests;
- whether CUDA graph / compilation / kernel autotuning warmup is required;
- whether warmup is repeated after changing workload shape or concurrency;
- what cache state is allowed at the start of timed runs.

Warmup must be sufficient to exclude one-time initialization from steady-state serving results.

## Repetitions and duration

**TBD after pilot variance measurement.**

The final rule must specify:

- number of independent benchmark repetitions;
- timed duration or completed-request count per repetition;
- how median / mean and confidence intervals are computed;
- whether outlier runs are ever discarded and, if so, under what predeclared rule.

Do not choose repetition count solely from convenience. Use pilot variance to determine what is needed to resolve effects of practical interest.

## GPU state and telemetry

Before every final timed run:

- verify no unintended process is materially using GPU compute or memory;
- verify the expected GPU and software stack;
- record initial GPU memory state;
- allow the benchmark warmup policy to reach steady state.

During final serving runs, collect enough telemetry to observe at least:

- GPU utilization;
- memory utilization / used VRAM;
- power draw;
- temperature;
- SM / graphics clock;
- memory clock;
- PCIe link generation/width at least during validation or representative load.

The exact sampling tool/cadence is **TBD**.

## Run validity

A final serving run is invalid if any of the following occurs:

- the intended model/configuration fails correctness validation;
- another process materially contaminates GPU memory or utilization;
- the backend crashes or silently falls back to an unintended execution path;
- the workload differs from the locked profile;
- generation stops early in a way that changes the intended output-token count distribution unless that behavior is explicitly part of the workload;
- the benchmark client cannot supply requests fast enough to sustain the target load;
- thermal/power behavior is clearly abnormal relative to the locked environment;
- result rows are missing or only partially written;
- the run uses different software/configuration without being labeled as a separate treatment.

Invalid runs should be preserved or logged as invalid rather than silently deleted when they reveal a systematic failure mode.

## Quality-run validity

A quality comparison is invalid if:

- examples differ between precision configurations;
- prompt/tokenization preprocessing differs unintentionally;
- the BF16 and quantized logits are not aligned to the same token positions for KL;
- generated-answer scoring changes between runs;
- a quantized checkpoint is produced from a different base checkpoint/revision;
- calibration data leaks into an evaluation set where that would bias the result.

## Measurement hazards found during model qualification (2026-08-22)

These were observed on this exact stack while qualifying Llama 3.1 8B Instruct. Each is a way to
produce a wrong number that still *looks* like a successful run. Artifacts:
`results/qualification/`.

### H1 — First-call warmup contamination is large enough to invert a comparison

The BF16 qualification run reported **2.7 tok/s** for its first (short) workload and 35.4 tok/s for
the next one on the same engine. The 2.7 figure is cold inductor compile plus CUDA-graph capture
attributed to the first request, not a decode rate.

Taken naively this would have made BF16 look ~24x slower than FP8 on short prompts. The real
medium-workload ratio is about 1.7x.

**Rule:** discard the first request(s) against every engine instance. No timed measurement may come
from the first generate call after engine init. The warmup policy in this document governs; the
qualification numbers in `results/qualification/*_smoke.json` deliberately do **not** satisfy it and
are labelled coarse.

### H2 — Cold-start cost is per-configuration and unequal

First-load costs differ sharply by configuration and are not inference:

```text
BF16   ~39 s  CUDA-graph memory profiling (tightest KV budget) + cold inductor compile
FP8     ~2 s  graph memory profiling
FP4    ~61 s  flashinfer JIT build of the SM120 CUTLASS FP4 GEMM on first forward pass
```

Once warm, engine init drops to ~2-5 s (measured 2.08 / 1.68 / 4.90 s vs 58.48 / 20.03 / 79.49 s
cold). The flashinfer artifact is cached at
`~/.cache/flashinfer/0.6.6/120a/cached_ops/fp4_gemm_cutlass_sm120/`.

**Rule:** never include engine startup in a serving metric, and prefer reusing one engine process
across measurements for a configuration. Report cold-start separately if it is of interest; do not
let an unequal one-time JIT cost leak into FP4's throughput.

### H3 — `torch.cuda.max_memory_allocated()` reads 0.0 in the launching process

vLLM v1 runs the model in a separate `EngineCore` process, so parent-process torch memory counters
see nothing. The qualification artifacts record `torch_max_mem_alloc_GiB: 0.0` for this reason — it
is an artifact of where the counter was read, not a real measurement.

**Rule:** GPU memory must come from `nvidia-smi` (or from inside the worker process). Do not report
parent-process torch allocator counters.

### H4 — Do not misread the shutdown race as a failed run

vLLM prints:

```text
ERROR ... Engine core proc EngineCore died unexpectedly, shutting down client.
```

during normal interpreter exit, *after* results are written. All three qualification runs printed it
and all three completed and wrote valid artifacts.

**Rule:** run success is determined by the result artifact, not by absence of that line.

### H5 — `num_gpu_blocks_override=512` in the log does NOT pin KV capacity — investigated, benign

Every run logs, during CUDA-graph memory profiling:

```text
Overriding num_gpu_blocks=0 with num_gpu_blocks_override=512
```

This was flagged as possibly pinning KV capacity and defeating the concurrency sweep. **It does
not.** It belongs to the transient profiling dry-run. The allocation that actually serves is larger
and differs per configuration:

```text
BF16    39,664 tokens = 2,479 blocks
FP8     92,608 tokens = 5,788 blocks
FP4    119,360 tokens = 7,460 blocks
```

512 blocks at block_size 16 would be 8,192 tokens, which matches none of them. The measured KV
capacity differences between configurations are real. No fix is needed; do not "correct" this.

### H6 — The GPU is power-limited, so clocks are not constant

SM clock fell from 2272 MHz idle-boost to 1657-1980 MHz under load with power pinned at the 145 W
limit. Throughput differences between configurations are therefore measured under a power ceiling,
not at fixed clocks.

**Rule:** record power, SM clock, and temperature per run (the qualification harness already does)
so a clock-limited run can be distinguished from a genuine configuration difference.

## Result identity

Every final result artifact should be attributable to a unique combination of:

```text
model_id
configuration_id
workload_id
concurrency
run_id
software/environment identity
```

Calibration experiments additionally need calibration draw / sample-count identity.
