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
Inference backend: vLLM 0.19.1  (LOCKED, D9)
Software pins:     Python 3.12.13, torch 2.10.0+cu128, compressed-tensors 0.15.0.1,
                   driver 575.64.03, CUDA toolkit 12.9; conda env `qnt`
                   checkpoint production: llmcompressor 0.10.0.3; conda env `qnt-quant`
Model:             Llama 3.1 8B Instruct, 8,030,261,248 params  (LOCKED, D6)
Configurations:    BF16_REFERENCE, FP8_PRIMARY, FP4_PRIMARY  (LOCKED, D5)
KV-cache precision: BF16 for all configurations (held constant, not a variable)
```

Full environment artifacts: `results/system/env_qnt_2026-08-22.json`,
`results/system/env_qnt-quant_2026-08-22.json`. Machine profile:
`results/system/profile_2026-08-22.txt`.

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
Concurrency:              locked sweep through the SLO boundary (D11)
```

Workload shape is **not** a primary variable in phase 1. D10 narrows the study to one primary
workload plus a supporting probe, so the tradeoff curve is traced along concurrency alone.

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

**LOCKED 2026-08-23 (D10).** Phase 1 is a memory-focused study with one primary workload and one
supporting probe. The earlier prefill-heavy / balanced / decode-heavy trio is superseded.

### DECODE_PRIMARY

```text
Input tokens:    512  exact
Output tokens:  2048  exact, ignore_eos=True
Peak KV/seq:    2,560 tokens = 0.3125 GiB   (128 KiB per token)
Purpose:        primary tradeoff curve; bandwidth-bound decode across the KV wall
```

Concurrency at which each configuration becomes KV-limited:

```text
              KV wall (peak)   KV wall (mean occupancy, 1,536 tok)
BF16                      15                                   25
FP8                       36                                   60
FP4                       46                                   77
```

Occupancy grows 5x over a sequence's life, so the wall is a band between these two columns rather
than a single concurrency. See hazard H8.

### PREFILL_PROBE

```text
Input tokens:   8192  exact
Output tokens:    32  exact, ignore_eos=True
Purpose:        bounded observation of the arithmetic benefit, which DECODE_PRIMARY cannot see
Scope:          concurrency 1, 2, 4, 8. No repetition structure. Not part of the tradeoff curve.
```

### Workload definitions frozen alongside the token counts

These are part of the workload, not harness details. Any one of them can invalidate a sweep silently.

- **Prefix caching disabled.** Launch with automatic prefix caching off and log the hit rate anyway.
  See hazard H7.
- **Prompt corpus.** Fixed held-out corpus, tokenized and chunked to exactly 512 (or 8192) tokens,
  prefix-disjoint, distinct per request, seeded, and reused byte-identically across all
  configurations. The repeating filler sentence used at qualification is not acceptable for final
  runs.
- **`ignore_eos=True` with exact output length.** Otherwise output length becomes a dependent
  variable of precision.
- **Fixed-exact lengths, not sampled.** Homogeneous batches; the realism cost is recorded in
  `LIMITATIONS.md`.
- **Greedy decoding** (`temperature=0`) for serving runs, so sampling variance does not enter timing.

Token counts are not tuned per precision configuration.

## Concurrency sweep

**LOCKED 2026-08-23 (D11).**

```text
DECODE_PRIMARY:  1, 4, 8, 12, 16, 24, 32, 48, 64, 96
PREFILL_PROBE:   1, 2, 4, 8
```

Ten points for the primary workload, dense through 12-48 where all three KV walls sit.
3 configurations x 10 points x 3 repetitions = 90 timed cells.

### Saturation criterion

Saturation is defined by an explicit SLO, not by inspection of a throughput curve:

```text
SLO:  TPOT P95 <= 50 ms   (20 output tok/s per user)
```

The headline capacity metric per configuration is **maximum concurrency sustained within the SLO**.

### Cell-abort rule

Predeclared so the decision is never made ad hoc mid-run:

- A cell is aborted and recorded as `SLO_VIOLATED` if TPOT P95 exceeds **10x the SLO** (500 ms) or if
  the cell exceeds its wall-clock cap of **15 minutes**.
- Once a configuration violates the SLO at concurrency C, higher concurrency points for that
  configuration are skipped and recorded as `SKIPPED_PAST_SLO`.

Aborted and skipped cells are results and must be written to the artifact, not omitted. Do not stop a
sweep merely because one configuration "looks fast enough."

### Per-point logging requirements

Without these the bandwidth region and the capacity region cannot be distinguished after the fact:

- whether the configuration was KV-limited at that point;
- preemption count;
- recomputed-token count;
- queue depth and time-in-queue;
- KV-block utilisation;
- prefix-cache hit rate (expected zero; a non-zero value invalidates the cell).

### Run ordering

Configurations must be **counterbalanced or randomized across repetitions**, not run in a fixed
BF16 -> FP8 -> FP4 order. Over a multi-hour sweep the card heats and clocks fall (H6), so a fixed
order systematically favours whichever configuration always runs first on a cool card.

## Warmup

**Backend is now selected (vLLM 0.19.1), so this gate is unblocked. Minimum rules already
established by measurement — see hazards H1 and H2:**

- **Discard the first request against every engine instance.** No timed measurement may come from
  the first generate call after engine init. Measured basis: the BF16 qualification run reported
  2.7 tok/s on its first workload and 35.4 tok/s on the next against the same engine.
- **Exclude engine startup entirely**, and prefer reusing one engine process across measurements for
  a configuration. Cold start is unequal across the ladder (BF16 ~39 s graph-memory profiling,
  FP8 ~2 s, FP4 ~61 s flashinfer JIT), so a shared warmup budget would silently favour FP8.
- **Never let the FP4 flashinfer JIT build enter a serving metric.** It is cached at
  `~/.cache/flashinfer/0.6.6/120a/cached_ops/fp4_gemm_cutlass_sm120/` and must be warm before timing.

**Locked 2026-08-23 — steady-state entry.** `DECODE_PRIMARY` needs more than first-call warmup,
because per-sequence KV occupancy grows 5x over a sequence's life (H8). A run begins un-limited and
only becomes KV-limited partway through, so a window opened too early measures the transient rather
than the regime.

- **Model load and engine init are excluded** from every metric. Reuse one engine process across all
  concurrency points for a configuration.
- **Warmup requests:** discard all completions until the client has held the target concurrency for
  at least **one full sequence lifetime** (2,048 output tokens at the observed rate), then discard
  one further generation per slot. Only after that does the timed window open.
- **Steady-state gate — revised 2026-08-23 after H9.** The timed window may not open until *both*:
  (a) KV-block utilisation has been non-decreasing across two consecutive 10 s telemetry samples, or
  has reached its ceiling; and (b) output throughput is **stationary** — the mean over the last k
  telemetry intervals is within 6% of the mean over the k intervals before, with k sized to span at
  least one oscillation period. Do **not** require throughput to be *flat*: this workload is periodic
  (H9) and a flatness test over part of a period can never fire. Measured example: drift 0.11% while
  instantaneous 10 s windows swung 588 / 575 / 685 tok/s.
- **Timed windows must span whole periods.** At least 4 sequence lifetimes, with
  `periods_in_window` recorded per cell, so the sawtooth averages out instead of biasing the mean.
- **Warmup is repeated after every change of concurrency**, not only after engine start. Changing
  concurrency changes the steady-state occupancy, so the previous steady state does not carry over.
- **Cache state:** prefix caching disabled; flashinfer JIT artifacts warm before timing.

Warmup must be sufficient to exclude both one-time initialization and the occupancy transient from
steady-state serving results.

## Repetitions and duration

**LOCKED 2026-08-23 — 3 repetitions, confirmed by pilot P3.**

Pilot P3 measured run-to-run spread against the FP8-to-FP4 gap it must resolve
(`results/pilot/p3_repeatability.csv`):

```text
point            C    FP8        FP4        delta     95% half-width   rho = H/delta
sub-wall         8    449.99     566.99     117.00    0.238            0.0020
high-concurrency 24  1000.30    1187.81     187.51    1.201            0.0064
```

Criterion was rho <= 0.25; measured rho clears it by 39-122x. Per-cell CV is 0.012-0.152% at n=3.
Three repetitions therefore resolve the FP8-vs-FP4 serving difference with very large margin.

**Caveat that the number does not capture.** P3 estimates variance from ~2-6 minute windows. The
sweep runs 6-8 hours, over which the card heats and clocks fall (H6) — the pilot already saw SM clock
move 1885 -> 1728 MHz between concurrency points under a pinned 145 W cap. Short-run spread is a
floor on long-run spread, not an estimate of it, which is why counterbalanced run ordering remains a
rule rather than something the variance figure lets us skip.

- **Repetitions:** 3 independent runs per cell, each with its own warmup and its own engine-process
  restart between repetitions of the same configuration. Repetitions are counterbalanced against
  configuration order (see "Run ordering").
- **Timed window:** `max(120 s, 4 x concurrency completed requests)`, capped by the 15-minute
  cell-abort limit. The request-count floor matters at low concurrency, where a single BF16 request
  takes roughly 51 s; the duration floor matters at high concurrency, where requests complete fast
  enough that a count-based window would be too short to be a regime measurement.
- **Aggregation:** report the median across repetitions with the full spread; the run-level sample
  size is 3, not the number of requests inside a run. Do not treat thousands of requests inside one
  process as thousands of independent hardware experiments.
- **Latency percentiles** are computed within a run across requests, then summarised across runs.
  Note that `DECODE_PRIMARY` yields 2,048 inter-token samples per request, so TPOT percentiles are
  well resolved even at concurrency 1; TTFT and end-to-end percentiles are the ones limited by
  request count.
- **Outliers:** no run is discarded for being slow. A run is discarded only if it meets a
  `Run validity` failure condition, and it is retained in the artifact marked invalid.

The repetition count is provisional precisely because pilot variance has not been measured. If pilot
run-to-run spread is large relative to the FP8-to-FP4 gap, raise it before final collection rather
than reporting an unresolved difference.

## Pilot

**Defined 2026-08-23.** Several rules in this document are marked provisional "after pilot". This
section says what the pilot is, so that phrase has a referent.

The pilot exists to **validate the free parameters of the contract**, not to produce a small version
of the experiment. Nothing it measures is a result, and no pilot number may be quoted as one.

### Jobs

**P1 — Test the weight-size-ratio prediction below the wall. RUN 2026-08-23: FAILED.** D10 predicted
that in the bandwidth-bound region the throughput ratio between configurations equals the weight-size
ratio (BF16:FP8:FP4 = 16.10 : 9.12 : 6.07 GB, so 1.77x and 2.65x over BF16). Measured at concurrency
1, 8 and 12, three repetitions: 1.83x / 2.44x falling to 1.62x / 2.00x. The prediction is falsified —
per-step traffic is `weights + KV + other` and the KV term is common across the ladder, so the ratio
is concurrency-dependent and below the weight-size ratio. The region is still memory-bound. See D10
for the corrected interpretation and `results/pilot/PILOT_DECISION.md` for the evidence.

Running it at several sub-wall points is the substance of the check, not a refinement of it. The
entire bandwidth-vs-capacity decomposition in D11 assumes the sweep is memory-bandwidth-bound
throughout, and that assumption is currently supported by an arithmetic-intensity argument with no
measurement behind it. If the ratio holds at concurrency 1 but decays by 12, the card is entering
compute saturation before the BF16 KV wall at 15, the below-wall region stops being a clean bandwidth
reading, and D11's decomposition needs revisiting before the sweep runs.

**P2 — Confirm the BF16 KV wall.** It should appear between concurrency 15 (peak footprint basis) and
25 (mean occupancy basis). Confirm by observing preemption onset and KV-block saturation, not by
throughput shape alone. If the wall is materially outside that band the D11 concurrency points do not
bracket it and must be re-chosen.

**P3 — Measure run-to-run variance.** The 3-repetition rule is provisional. Estimate the run-level
spread at two or three concurrency points and compare it to the FP8-to-FP4 gap. If spread is large
relative to that gap, raise the repetition count before final collection rather than reporting an
unresolved difference.

**P4 — Record achievable memory bandwidth.** The governing hardware constant for this study is absent
from `HARDWARE_PROFILE.md`. Capture the vendor spec figure and an achieved measurement on this card.

**P5 — Verify the workload plumbing.** Prefix-cache hit rate is zero, output length is exactly 2048
for every request, prompts are prefix-disjoint, the steady-state gate fires as specified, and the
per-point counters required by D11 (preemption, recompute, queue depth, KV-block utilisation) are
actually emitted by this vLLM build.

### Correctness gate — runs with the pilot, before the sweep

Section E of `EVALUATION_RIG.md` requires a correctness gate before any timing is treated as
meaningful. It belongs here rather than after the sweep: timing a corrupted checkpoint wastes the
most expensive resource in the project.

Run a small KL smoke — order 16-32 contexts, position 1 only — across BF16/FP8/FP4. This is a
**sanity gate, not a quality result**, and reuses the D13 path. An implausibly large divergence means
a broken checkpoint or an unintended execution path, and the sweep must not start.

Confirm alongside it that each configuration still dispatches to its intended kernel, since the
qualification evidence predates any harness code.

### Exit criteria

The sweep may start when P1-P5 pass and the correctness gate is clean. P1 or P2 failing is not a
reason to proceed with an adjusted interpretation — both feed decisions (D10's prediction, D11's
bracketing) that would have to be reopened first.

### Readiness — checked 2026-08-23

Present and verified on this machine:

- both quantized checkpoints on disk (`checkpoints/`, FP8 8.5 GB, NVFP4 5.7 GB);
- GPU idle (15 MiB used, 0% utilisation, 5 W);
- `vllm/benchmarks/serve.py` ships in this build with `--max-concurrency`, `--random-input-len`,
  `--random-output-len`, `--ignore-eos`, `--request-rate`;
- the counter *names* D11 requires exist in `vllm/v1/metrics/`. **Superseded by measurement — see
  hazard H11.** Three of the four emit usable values (`num_preemptions`, `kv_cache_usage`,
  `num_waiting_reqs`); `recomputed_tokens` measures a one-token prefix-cache artifact, not
  preemption recompute, and is structurally always zero once prefix caching is disabled. The
  earlier claim here that it makes H8's loop "measurable rather than inferred" was wrong.
  Note also that Prometheus exposes counters with a `_total` suffix, so scraping the source-level
  name returns nothing.

Not present:

- **No harness code.** `scripts/` holds a 60-line single-request qualification smoke with no
  concurrency, warmup, or counter scraping. Four pieces are needed: sweep driver, corpus prep,
  bandwidth measurement, and adaptation of the existing KL scripts to real contexts.
- **The stock client is partial.** `vllm bench serve`'s `--max-concurrency` is a semaphore over a
  fixed `--num-prompts`. It has no warmup-discard window, no steady-state entry, and no cell abort,
  so the contract's rules must wrap it rather than configure it.
- **The prompt corpus is unnamed** — open gate D16.

### Open items — all resolved by the pilot run of 2026-08-23

1. **P4 measurement method.** Resolved: a standalone CUDA `float4` streaming microbenchmark
   (copy / triad / read), CUDA-event timed, 1 GiB arrays against a 48 MiB L2. Independent of decode
   throughput, so it is not circular with P1. Result 620.1 GB/s read, 92.3% of the 672.0 GB/s figure
   derived from driver-reported bus width and memory clock. `results/pilot/p4_hbm_bandwidth.json`.
2. **P3 threshold.** Resolved: rho = H/delta <= 0.25, escalating 3 -> 5 -> 7 repetitions. Measured
   rho = 0.0020 / 0.0064. Repetition count locked at 3.
3. **P2 tolerance.** Resolved: the transition must fall inside the pre-registered 15-25 band.
   Measured bracket [17, 18], reproducible on both sides. PASS.
4. **Correctness-gate trigger values.** Resolved and pre-registered before running: BF16 self-KL
   median <= 1e-6 and max <= 1e-4 nats; FP8 median KL <= 0.1; FP4 median KL <= 0.5; top-1 agreement
   >= 0.90 (FP8) and >= 0.75 (FP4); probability mass normalized to 1e-3. All passed.
   `results/pilot/correctness_gate.json`.
5. **`PREFILL_PROBE` plumbing.** Resolved: one C=1 cell per configuration at 8192 in / 32 out, all
   `OK` with zero prefix-cache hits — the H7 hazard is the one that bites hardest on this shape.
6. **SLO feasibility above batch 1.** Resolved favourably: BF16 TPOT P95 is 29.6 ms at C=8 and
   31.3 ms at C=12, inside the 50 ms bound. BF16 first breaches the SLO between C=17 (39.8 ms) and
   C=18 (41.3 ms), i.e. essentially at its own KV wall, so its max-concurrency-at-SLO does not
   collapse below the wall and the headline comparison is not distorted.
7. **Pilot artifact path and schema.** Resolved: `results/pilot/` with `manifest.json`, one CSV per
   job, `p4_hbm_bandwidth.json`, `p5_harness_validation.json`, `correctness_gate.json`,
   `PILOT_DECISION.md`, and `cells.jsonl` carrying full per-cell identity.
8. **Cost estimate.** Actual: 43 cells, about 5 hours GPU-exclusive including engine restarts, plus
   roughly 1 hour discarded to a gate correction (H9) and a teardown stall.

### What the pilot will not tell you even if it passes

P3 estimates variance from short runs. The sweep is 6-8 hours, over which the card heats and clocks
fall (H6). Short-run spread is a floor on long-run spread, not an estimate of it — which is why run
ordering is counterbalanced by rule rather than trusted to the variance figure.

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

**Cadence locked 2026-08-23: 10 s.** The steady-state gate in "Warmup" is defined in terms of two
consecutive telemetry samples, so the cadence is part of the contract rather than a harness choice.
Engine-side counters (KV-block utilisation, preemption, recompute, queue depth) are sampled on the
same 10 s tick so GPU telemetry and scheduler state can be aligned per point.

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
- the run uses different software/configuration without being labeled as a separate treatment;
- the prefix-cache hit rate is non-zero (H7);
- the timed window opened before the occupancy transient settled (H8);
- the requested output length was not enforced exactly, so the output-token count varies with
  configuration.

A cell recorded as `SLO_VIOLATED` or `SKIPPED_PAST_SLO` under the cell-abort rule is a **valid
result**, not an invalid run.

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

## Measurement hazards identified during workload design (2026-08-23)

These were not observed at qualification. They are consequences of the locked `DECODE_PRIMARY` shape
and were identified while resolving D10/D11. Both produce runs that look successful.

### H7 — Automatic prefix caching can silently delete the workload

vLLM V1 enables automatic prefix caching by default. The qualification harness drove every request
with the same repeating filler sentence, so requests shared long prefixes.

With caching on and shared prefixes, prefill after the first request is served from cache and costs
almost nothing. A `PREFILL_PROBE` run under those conditions measures cache hits, not prefill, and
reports an 8,192-token prompt as nearly free. `DECODE_PRIMARY` is less exposed because its prompt is
short, but a cached prefix also perturbs KV occupancy and therefore the wall position.

**Rule:** disable prefix caching for all **serving** runs, use a prefix-disjoint prompt corpus, and
log the hit rate regardless. A non-zero hit rate invalidates the cell.

**The quality rig is the deliberate exception.** `EVALUATION_RIG.md` enables prefix caching for
teacher-forced KL, where the shared prefix is the same token sequence by construction, caching changes
no returned distribution, and it is what makes ten truncated-prefix passes affordable. That exception
does not extend to anything timed.

### H8 — KV occupancy grows during the run, so the KV wall is a band and preemption is superlinear

Under `DECODE_PRIMARY` a sequence's footprint grows 5x over its life, from 512 to 2,560 tokens. Two
consequences:

**The wall is not a threshold.** At fixed concurrency the system starts un-limited and becomes
KV-limited partway through. The BF16 wall is at concurrency 15 by peak footprint but 25 by mean
occupancy, and the true onset lies between. Reporting a single "wall concurrency" without saying
which basis was used is a category error.

**Degradation past the wall is superlinear, not a plateau.** vLLM's default preemption mode is
recomputation, so a preempted sequence discards its KV and redoes its prefill. Past the wall this
creates a feedback loop: preemption causes recompute, recompute consumes throughput, which lengthens
residency, which causes more preemption.

This is the effect the memory lens exists to measure — but only if it is logged. Preemption count and
recomputed-token count must be recorded per point (D11). Without them the collapse is indistinguishable
from a generic slowdown and the capacity contribution becomes an argument instead of a measurement.

**Rule:** open the timed window only after the occupancy transient has settled (see "Warmup"), record
preemption and recompute counters per point, and state which basis (peak or mean) any quoted wall
concurrency uses.

### H9 — Phase-aligned slots make this workload periodic, not flat — measured 2026-08-23

Observed during the pilot on `DECODE_PRIMARY` at FP8, concurrency 12. Engine-reported output
throughput does not settle to a constant. It oscillates with a stable period and a flat mean:

```text
669.6  620.4  578.4 | 628.5  667.1  616.8  576.0 | 635.9  664.8  614.4  574.8 | 641.9 ...
peak ~665 tok/s   trough ~570 tok/s   amplitude +/-7.7%   period ~40 s
```

The pattern is unchanged 300 s apart, so this is neither a warmup transient nor thermal drift.

**Cause.** In a closed-loop driver every slot starts a new request the instant the previous one
finishes, and every request is exactly 2,048 output tokens. The slots therefore stay phase-aligned:
all sequences grow 512 -> 2,560 tokens together, decode-attention traffic grows with them, throughput
sags; then all sequences retire together and it snaps back. The period is one sequence lifetime,

```text
period = output_tokens x concurrency / output_throughput
```

which predicted 39.8 s against the ~40 s observed.

**Consequence 1 — "steady state" must be defined per period.** A gate that requires throughput to be
flat within a few percent over a fraction of a period can never fire, and the cell fails for a
definitional reason rather than a physical one. This happened once in the pilot before the rule was
corrected. Steady state here means **stationary**, not flat: compare the mean over the last k
telemetry intervals against the mean over the k before, with k spanning at least one period.

**Consequence 2 — timed windows must span whole periods.** A window ending mid-period is biased by
up to the oscillation amplitude. The pilot requires at least 4 periods per window and records
`periods_in_window` per cell.

**Consequence 3 — this is part of why the KV wall is a band.** H8 derives the band from occupancy
growth within a sequence. H9 adds that at fixed concurrency the *aggregate* footprint sweeps the
same range every period, so preemption onset is periodic near the wall rather than sustained.
Preemption must therefore be judged over whole periods too, which is why the pressure test counts
the number of telemetry samples showing preemption rather than a single non-zero reading.

**Rule:** gate on stationarity over at least one period, size timed windows to an integer number of
periods, and record `period_estimate_s`, `periods_in_window` and `oscillation_amplitude_rel` per
cell. Staggering slot start times would remove the oscillation and is worth considering for the
sweep, but it changes the locked workload definition and is therefore a tracked decision, not a
harness choice.

### H10 — The serving path allocates more KV than the offline path, and a contaminated launch allocates less — measured 2026-08-23

Reported KV capacity for the same configuration and flags is **not** the qualification figure:

```text
                offline LLM path      serving path (vllm serve)
BF16            39,664 tokens         44,688 tokens   (4.84 -> 5.46 GiB)
FP8             92,608 tokens         97,888 tokens
FP4            119,360 tokens        116,960 / 120,944 tokens
```

Across the nine engine launches of pilot P1, BF16 reported 44,688 tokens on **every** launch and FP8
reported 97,888 on every launch. Only FP4 varied, between 116,960 and 120,944 (3.4%), which is
consistent with its JIT-compiled GEMM affecting the memory-profiling peak.

**One outlier is explained, and is a warning.** An early BF16 serving launch reported 40,432 tokens,
10% below the reproducible 44,688. That launch began while a previous engine process was still
releasing VRAM. vLLM v1 runs `EngineCore` in a separate process, so the parent can exit while the GPU
is still occupied, and the memory-profiling step then sizes the KV cache against whatever is left.
This is silent: the engine starts, serves correctly, and reports a smaller cache.

**Rules.**

- Wait for the GPU to be genuinely released between configurations, not merely for the parent process
  to exit. The pilot harness waits for used memory to fall below a threshold, escalates to a forced
  kill, and re-checks; the idle preflight then polls rather than aborting on a slow teardown.
- Record `kv_cache_tokens` **per cell**, from the engine instance that actually served it, and compute
  any predicted KV wall from that figure rather than from a committed constant.
- Treat a KV capacity materially below the configuration's reproducible value as evidence of a
  contaminated launch and discard the cell.

### H11 — `recomputed_tokens` cannot measure preemption recompute in this build — measured 2026-08-23

D11 lists "recomputed-token count" as a required per-point observable, and the pilot readiness note
in this document claimed `recomputed_tokens` "is what makes H8's preemption-by-recompute loop
measurable rather than inferred". **Both are wrong.** The counter exists, emits, and is structurally
incapable of reporting what D11 wants.

`vllm/v1/metrics/stats.py`, `PromptTokenStats.update_from_output`:

```python
recomputed = 1 if (num_cached_tokens + 1 == prompt_len) else 0
self.recomputed_tokens += recomputed
```

It increments by **at most one token per prefill**, and only in the single edge case where the whole
prompt was a prefix-cache hit except the last token, which the scheduler forces the model to
recompute so the forward pass has an input. It has nothing to do with preemption.

Under the locked serving configuration prefix caching is disabled (H7), so `num_cached_tokens` is
always 0, the condition is never satisfied, and `vllm:prompt_tokens_recomputed` is **always exactly
zero**. Measured: BF16 at concurrency 18 recorded 7 preemptions per cell across two repetitions with
`recomputed_tokens = 0` in both.

**Rule.** Do not use `recomputed_tokens` as a pressure indicator. The KV-capacity pressure signal for
this study is:

```text
vllm:num_preemptions_total     sharp and reproducible: 0 at C<=17, 7 at C=18, both reps
vllm:kv_cache_usage_perc       max 0.962 at C=17 -> 0.985 at C=18
throughput regression          peaks at C=16, falls at C=17 and beyond
TPOT P95 inflation             32.6 ms at C=16 -> 39.8 at C=17 -> 41.3 at C=18
```

The *effect* of preemption-by-recompute remains measurable through throughput and latency; the
recompute **volume** is not directly counted in this build. D11's per-point logging requirement must
be amended accordingly rather than left naming a counter that cannot fire.

This is the exact failure mode P5 exists to catch: a metric name present in the source is not
evidence that a usable value arrives at runtime.

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
