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

Added 2026-08-25, once the KL rig made them load-bearing:

- the frozen trajectory artifact and its hash — every configuration is scored on the same token
  sequence, never on its own generated history;
- the retained-position contract (context construction at each position, and the position vector);
- the engine profile, item by item: `detokenize`, `max_num_batched_tokens`, eager/graph mode,
  prefix-cache policy, `max_model_len`, `gpu_memory_utilization`, `max_num_seqs`, `kv_cache_dtype`,
  `max_logprobs`, seed;
- request submission order within a trajectory;
- persisted storage precision and the KL working precision;
- the checkpoint content actually loaded, identified by hash rather than by path.

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

**Prompt reuse across concurrency waves — amended 2026-08-24.** The corpus is 512 prompts and the
driver draws round-robin, so a cell that issues more than 512 requests wraps. At C=96 a cell issues
roughly 900. The rule is therefore: **distinct within a concurrency wave; reuse across waves is
permitted and must be recorded** (`prompt_wraps`, `prompt_indices_unique`). No two simultaneous
requests share a prompt while C <= 512, which is the property that matters for KV, and prefix caching
is off. The corpus is deliberately **not** regenerated — that would change `prompt_set_hash` and
break the provenance chain back to D16.

**Pre-registered cell parameters for the sweep.** Fixed before the run, hashed into
`sweep_config_hash`, and serialized into every cell record:

```text
margin                    1.5      budget headroom over the close conjunction
min_periods               4        whole periods per timed window
min_requests_factor       4        completed requests per unit concurrency
window_floor_s            120      (DECODE_PRIMARY) / 60 (PREFILL_PROBE)
warmup_lifetimes          2        sequence lifetimes discarded before the gate
gate_timeout_s            420      floor; raised to 3x period at high concurrency
hard_cap_s                1800     (DECODE_PRIMARY) / 900 (PREFILL_PROBE)
abort_tpot_ms             500      10x the SLO. P2 disabled this to observe the pressured
                                   regime; the sweep keeps it, because the skip rule already
                                   stops the ladder and this is only a pathological-cell guard
nonstationary_ratio       1.25     period growth that marks a cell as having left steady state
```

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

### Wall-refinement pass — additive, D11's locked points unchanged

**Added 2026-08-24.** The locked ladder cannot resolve the FP8 and FP4 walls from each other: both
fall inside the single 32-48 gap.

```text
configuration   wall        ladder bracket   width
BF16            17          [16, 24]         8
FP8             ~38         [32, 48]         16
FP4             ~47         [32, 48]         16
```

The grid spacing where all three ceilings live is 1.5x; the FP8-to-FP4 KV separation is 1.24x. A 1.5x
grid cannot resolve a 1.24x difference, so the sweep alone can return the same bracket for both — a
null on the marginal capacity benefit of the FP8-to-FP4 step that would be indistinguishable from a
real null, on the exact question the study exists to answer. It also contradicts D10's own rationale
for the narrow workload, which traded shape variety to buy resolution near the walls.

**The locked ten points are run unchanged.** After them, a separate phase bisects each
configuration's `[last_clean, first_pressured]` bracket using the pilot's preemption-based
`pressured()` predicate, repetition 1 only, roughly four cells per configuration. It carries its own
job label (`SWEEP_REFINE`) so the pre-registered D11 set stays clean and the refinement is visibly
additive rather than a retroactive re-spacing. Refinement points lie below the SLO-violation point,
so they do not interact with the skip rule.

### Ceiling replication pass — pre-registered 2026-08-31, additive

**Added 2026-08-31, before any cell was run.** The headline metric — maximum concurrency sustained
within the SLO — came out of the `SWEEP_REFINE_SLO` bisection at repetition 1 only, while every
ladder point around it is n=3. The margins are thin enough that the label matters:

```text
config   K    tpot P95 at K   margin    tpot P95 at K+1   margin
BF16     21       47.59 ms   +2.41 ms          50.74 ms   -0.74 ms
FP8      57       49.66 ms   +0.34 ms          50.78 ms   -0.78 ms
FP4      70       49.57 ms   +0.43 ms          50.64 ms   -0.64 ms
```

Matched-cell TPOT P95 spread across the existing repetitions is about 0.1 ms (`results/sweep/cells.jsonl`;
FP8 C=48 reads 40.01 / 39.95 / 39.91, FP4 C=64 reads 44.44 / 44.42 / 44.38), so FP8 and FP4 sit
three to four noise widths from flipping and BF16 about twenty-four.

**A confirmatory triplet, not a re-run of the search.** The bisection's answer is determined
entirely by the SLO verdicts at K and K+1; re-running it would re-probe interior points
(BF16 20; FP8 56, 60; FP4 68, 72, 80) that say nothing about the boundary. The phase measures

```text
BF16_REFERENCE   20, 21, 22
FP8_PRIMARY      56, 57, 58
FP4_PRIMARY      69, 70, 71        C=69 is new; repetition 1 bisected 68 -> 70
```

at **repetitions 2 and 3**, bringing the headline to the ladder's n=3. Nine cells per repetition,
eighteen in total. One engine launch per (configuration, repetition) — H10 holds that KV capacity is
not reproducible across launches, so launch is a variance source the replication must include rather
than amortise across configurations. Configuration order follows the existing Latin square; cells
ascend within a launch. Job label `SWEEP_CEILING_REP`.

**`SWEEP_SPEC` is unchanged and the cells carry `sweep_config_hash = df0f0f124d987a5c`.** The
measurement contract — workload, cell budgets, server controls, SLO — is identical to the cells
these are compared against; only the job label and the concurrency points are new. The skip rule is
deliberately not applied: C=K+1 is *expected* to breach the SLO and is half the evidence that places
the ceiling, so `SKIPPED_PAST_SLO` would delete the result.

**Criterion.** Per repetition, the observed ceiling `K_r` is the largest C in the triplet meeting
the SLO, admissible only from cells with status `OK` and a non-null `meets_slo`, and only if the
triplet's verdicts are monotone. No cell is excluded after the fact; non-OK cells are reported as
defects or infeasibilities.

```text
per repetition
  CONFIRMED          K meets the SLO, K+1 does not
  MOVED_DOWN         K fails, K-1 meets the SLO -- the ceiling is K-1 in this repetition
  UNRESOLVED_ABOVE   K+1 meets the SLO; the ceiling is above the triplet
  UNRESOLVED_BELOW   K and K-1 both fail; the ceiling is below the triplet
  NON_MONOTONE       more than one pass->fail transition
  INCOMPLETE         K, or the point needed to place the ceiling, was not measured

overall
  CONFIRMED          every repetition returns K            report K at n=3
  MOVED              every repetition agrees on K' != K    K' supersedes K
  UNSTABLE           repetitions disagree                  report the observed range and drop
                                                           the point estimate
  NOT_YET_REPLICATED fewer than three usable triplets
```

`UNRESOLVED_ABOVE` records and stops rather than extending the probe set: widening the search after
seeing the data is the retroactive re-spacing the additive-phase discipline exists to prevent. A
wider bracket would have to be pre-registered as its own phase.

Recorded but not gating: TPOT P95 and its margin to 50 ms at every point, matched-cell spread
against repetition 1, per-launch `kv_cache_tokens` (H10), and `num_preemptions_delta`.

**Reproduction.** `python3 scripts/harness/run_sweep.py --job ceiling` (add `--dry-run` to print the
eighteen cells without launching an engine); `python3 scripts/harness/analyze_ceiling.py --write`
applies the criterion above and writes `results/sweep/ceiling_replication.json`. The phase is
deliberately excluded from `--job all` so an additive pass over a completed artifact is never a side
effect of re-invoking the sweep.

### Saturation criterion

Saturation is defined by an explicit SLO, not by inspection of a throughput curve:

```text
SLO:  TPOT P95 <= 50 ms   (20 output tok/s per user)
```

The headline capacity metric per configuration is **maximum concurrency sustained within the SLO**.

### Cell-abort rule

**Revised 2026-08-24.** The previous rule read: "A cell is aborted and recorded as `SLO_VIOLATED` if
TPOT P95 exceeds 10x the SLO (500 ms) or if the cell exceeds its wall-clock cap of 15 minutes." That
conflated two different events under one status, and the harness disagreed with it anyway — it
emitted `CELL_TIMEOUT`, which is marked invalid, i.e. missing data rather than a result.

**Budgets are derived from the measured period, not flat.** A flat cap is anti-correlated with what
it should protect: past a KV wall throughput falls, the period lengthens, and the cell needs *more*
wall clock precisely when it is producing the capacity result. Per phase:

```text
warmup budget    max(warmup_wall_cap_s, warmup_lifetimes * period * margin)
gate budget      max(gate_timeout_s, 3 * period)
measure budget   margin * max(window_floor_s, min_periods * period,
                              (min_requests_factor + 1) * period)
hard ceiling     hard_cap_s   absolute anti-hang backstop, 30 min for DECODE_PRIMARY
```

The measure budget must dominate the **whole** close conjunction, not one term of it. Sizing it on
`min_periods` alone starves any short-period workload: `PREFILL_PROBE` has a 1-2 s period against a
60 s window floor, so a `min_periods`-only budget would expire every probe cell. The
`min_requests_factor + 1` term is measured, not decorative — `in_window_records` excludes the first
period, so the request-count floor needs about one extra period, and the pilot's cells recorded 4.24
to 5.16 periods in window against `min_periods = 4`.

**Classification at expiry, in order.** The discriminator is *not* SLO state alone: a configuration
that is genuinely too slow and a budget that is too small both present as missing the SLO, so an
SLO-only test fires the defect branch only in the benign case.

| condition | status | `outcome_class` |
|---|---|---|
| TPOT P95 > 10x SLO during measure | `SLO_VIOLATED` | measured |
| at expiry, TPOT P95 above the SLO | `SLO_VIOLATED` | measured |
| at expiry, within SLO but period grew > 1.25x | `NONSTATIONARY` | infeasible |
| at expiry, within SLO at a stable period | `CELL_TIMEOUT` | defect |
| gate never fired | `STEADY_STATE_NOT_REACHED` | infeasible |
| hard ceiling with no tokens streaming | `CELL_HUNG` | defect |
| engine died / harness exception | `SERVER_DIED` / `HARNESS_ERROR` | defect |

`CELL_TIMEOUT` is therefore a genuine defect signal rather than a catch-all, and should be
unreachable in a correctly budgeted run.

**`outcome_class` exists because `valid_result` was overloaded.** `valid_result` answers "is there a
timing measurement that passed the invariants". `outcome_class` answers "what does this cell tell the
study": `measured`, `infeasible`, or `defect`. The headline max-concurrency-at-SLO metric reads
`measured` union `infeasible`; a `defect` is never evidence about the hardware. Any status not
explicitly mapped defaults to `defect`, never `infeasible` — a harness failure must not be able to
masquerade as a capacity finding.

**The skip predicate is an SLO-only allowlist.** Once a configuration violates the SLO at concurrency
C, higher points for that configuration **in that repetition** are skipped and recorded as
`SKIPPED_PAST_SLO`:

```text
violated  <=>  status == SLO_VIOLATED  or  (status == OK and meets_slo is False)
```

Nothing else truncates a ladder. `STEADY_STATE_NOT_REACHED`, `INVALID`, `CELL_TIMEOUT`, `CELL_HUNG`,
`SERVER_DIED` and `HARNESS_ERROR` abort the **cell**, never the **ladder** — otherwise a harness
event silently deletes the remaining points and the deletion is then reported as the configuration's
serving ceiling.

Two guards on the boundary, because per-repetition skipping makes the boundary point ragged and the
surviving repetitions are the ones that did *not* violate:

- repetition 1 runs exactly one point **past** the first violation, so monotonicity is tested rather
  than assumed. BF16's pilot throughput is already non-monotonic (490 -> 425 -> 437 at C=16/17/18,
  with C=17 confounded by a 1872 -> 1728 MHz clock drop), so one unlucky cell must not be able to
  truncate a ladder permanently;
- a ladder point counts toward the headline metric only if it violated in **at least 2 of 3**
  repetitions, and the analyzer must print n per point and refuse to emit a median where n < 3.

Do not conflate pressure with SLO violation. Pilot BF16 at C=18 shows 4 preemptions at 98.5% KV and
still meets the SLO at 41.3 ms; keying the skip on preemption would truncate that ladder eight
concurrency points early.

Aborted and skipped cells are results and must be written to the artifact, not omitted. Do not stop a
sweep merely because one configuration "looks fast enough."

### Per-point logging requirements

Without these the two regions cannot be distinguished after the fact:

- whether the configuration was KV-limited at that point;
- preemption count;
- queue **depth** (`num_waiting_reqs`). **Time-in-queue is not recorded** — the driver has no
  per-request queue-time observable in this build. Recorded here as a known gap rather than left
  as an unmet requirement;
- KV-block utilisation;
- prefix-cache hit rate (expected zero; a non-zero value invalidates the cell);
- per-launch KV capacity in tokens, checked against the configuration's reproducible value before
  any cell runs (H10);
- SM-clock throttle reasons (`sw_power_cap`, `hw_slowdown`, `sw_thermal_slowdown`), without which a
  power-cap event cannot be separated from a thermal one after the fact;
- the full `CellConfig` used, serialized into the record. The pilot ran with `wall_cap_s` of 1200
  and 1500 against a documented 15 minutes and nobody noticed, because not one config field was
  ever written to the artifact;
- prompt-corpus wrap count. The corpus holds 512 prompts; above roughly C=48 a cell issues more
  requests than that and wraps.

**`recomputed_tokens` is deliberately absent.** D11 originally required it. H11 established that the
counter is structurally incapable of reporting preemption recompute in this build and is always zero
once prefix caching is disabled. It is still recorded, but it is not a requirement and no analysis
may rest on it.

### Run ordering

Configurations must be **counterbalanced or randomized across repetitions**, not run in a fixed
BF16 -> FP8 -> FP4 order. Over a multi-hour sweep the card heats and clocks fall (H6), so a fixed
order systematically favours whichever configuration always runs first on a cool card.

**What the Latin square does and does not balance — recorded 2026-08-24.** `LATIN` is a cyclic
square: each configuration occupies each ordinal position exactly once, so *position* is balanced.
*Carryover* is not. FP8 follows BF16 in 2 of 3 repetitions and FP4 in 0 of 3. Balancing carryover for
three treatments requires six sequences (a Williams design); with three repetitions it is not
achievable, and the skip rule makes it worse because BF16 skips the most cells and therefore forms
the shortest block. This is stated rather than implied to be handled.

**Two mitigations, and a test rather than an assumption.**

- Preflight waits for an idle **and thermally comparable** GPU before every launch, not just idle
  memory. That removes most of the ordering confound directly, which counterbalancing alone cannot.
- H6 is **tested, not assumed**. Across the pilot's 43 decode cells the card sat at 83-86 C and
  exactly 145.0 W in every single cell, and SM clock tracked the operating point rather than elapsed
  time: matched `(configuration, C)` cells run hours apart agree to within 1.07% on clock and 0.33%
  on throughput. That is the baseline a real drift signal must exceed. The sweep pre-registers the
  comparison of matched cells between repetition 1 and repetition 3, and reports the residual of
  within-cell throughput regressed on wall-clock offset and block position. Counterbalancing without
  a residual diagnostic is a ritual.

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
  This guarantee is **not** traded away for a predictable budget: the period count stays exact and
  token-derived (see "Repetitions and duration"), and the *budget* is sized from the period frozen at
  gate-fire. Pilot oscillation amplitude reaches 0.357 relative, so a window covering 2.8 periods
  instead of 4 would carry roughly 10% residual bias — the same order as the FP8-to-FP4 gap the sweep
  exists to measure, and biased in the flattering direction on exactly the degraded cells.
- **Warmup is repeated after every change of concurrency**, not only after engine start. Changing
  concurrency changes the steady-state occupancy, so the previous steady state does not carry over.
- **Cache state:** prefix caching disabled; flashinfer JIT artifacts warm before timing.
- **Warmup truncation is recorded, not invalidating.** Warmup truncates where throughput has
  collapsed, i.e. on the past-the-wall capacity cells. Promoting it to an invalidity reason would
  destroy exactly the measurements the budget revision exists to save, so it is a first-class
  recorded field feeding `outcome_class`. The period-derived warmup budget stops it firing
  spuriously at high concurrency in the first place.

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
sweep ran 14.75 hours (2026-08-25), over which the card heats and clocks fall (H6) — the pilot saw SM clock
move 1885 -> 1728 MHz between concurrency points under a pinned 145 W cap. Short-run spread is a
floor on long-run spread, not an estimate of it, which is why counterbalanced run ordering remains a
rule rather than something the variance figure lets us skip.

- **Repetitions:** 3 independent runs per cell, each with its own warmup and its own engine-process
  restart between repetitions of the same configuration. Repetitions are counterbalanced against
  configuration order (see "Run ordering").
- **Timed window — single definition, revised 2026-08-24.** This document previously carried two
  incompatible rules: `max(120 s, 4 x concurrency completed requests)` here, and "at least 4 sequence
  lifetimes" under Warmup. The window closes when **all three** hold:

  ```text
  elapsed  >= window_floor_s
  periods  >= min_periods          periods = tokens_in_window / (out_tokens * concurrency)
  done     >= min_requests_factor * concurrency
  ```

  The period count is **token-derived and exact**, not a rate estimate. Using a live throughput
  estimate made the requirement a moving target that receded as throughput fell, so a degrading cell
  could never satisfy it. Verified against all 46 pilot cells: the token formula reproduces the
  recorded `periods_in_window` to within 2.2%, with no cell disagreeing by more than 3%. The
  request-count floor matters at low concurrency, where a single BF16 request takes roughly 51 s; the
  duration floor matters at high concurrency, where requests complete fast enough that a count-based
  window would be too short to be a regime measurement.
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
the measured ratio is concurrency-dependent and, for FP4, well below the weight-size ratio. The
sub-wall region showed no preemption, no recompute and stable TPOT throughout. D10 records a
post-hoc `weights + KV + other` traffic model as a candidate explanation; it is not established, and
nothing downstream depends on it. See D10 and `results/pilot/PILOT_DECISION.md` for the evidence.

Running it at several sub-wall points was the substance of the check, and the several points are
what produced the finding: the ratio held to +3.7% / -8.1% at concurrency 1 and decayed to -8.2% /
-24.5% by concurrency 12. A single-point P1 would have read as a near-miss instead of a falsification.

**What P1 established, and what it did not.** It established that the ratio is concurrency-dependent
and that weight compression does not quantitatively predict throughput speedup. It did *not*
establish a replacement attribution: the below-wall gap has not been uniquely attributed to HBM
bandwidth, to KV dilution, or to kernel efficiency, and the pilot cannot separate them. The sub-wall
region does remain free of the failure mode the check was watching for — zero preemption, zero
recompute, clean scaling, stable TPOT at every sub-wall point — so the region is not entering
capacity limitation before the BF16 wall, which is the property D11's ladder depends on.

**Amended 2026-08-24: this verdict does not gate the sweep.** See "Exit criteria" below.

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

**Amended 2026-08-24. P1 is an informative falsification, not a required PASS.**

The original clause read: "The sweep may start when P1-P5 pass and the correctness gate is clean.
P1 or P2 failing is not a reason to proceed with an adjusted interpretation — both feed decisions
(D10's prediction, D11's bracketing) that would have to be reopened first." That clause was
discharged, not waived: P1 failed, D10 and D11 were reopened, and the interpretation was corrected
before anything was cleared. What follows replaces it.

> P1 falsified D10's original proportional-scaling prediction. This does not invalidate the serving
> sweep; it removes the assumption that weight compression ratio quantitatively predicts throughput
> speedup. The full sweep reports observed below-wall throughput empirically and separately
> measure the movement and usefulness of the KV-capacity wall.

The current criteria:

```text
P1   informative     verdict recorded as measured; does not gate
P2   must PASS       the ladder must bracket the measured wall, or the points are wrong
P3   must PASS       repetition count must resolve the FP8-FP4 gap
P4   must be VALID   the bandwidth ceiling must be measured independently
P5   must PASS       the harness must be measuring what it claims to measure
gate must be CLEAN   no timing a corrupted checkpoint or an unintended execution path
```

**P1 is not rerun and no replacement gate takes its place.** Specifically, the post-hoc three-term
traffic model in D10 must **not** be promoted into a pre-sweep predictive gate. It was fitted after
seeing P1's data and is documented as a candidate explanation and a possible follow-up, nothing more.
Requiring it to be validated first would reinstate exactly the error P1 exposed — predicting the
serving benefit instead of measuring it.

**Why a falsification clears rather than blocks.** P1 tested a *predictive shortcut*: that the
below-wall throughput ratio could be read off the weight-size ratio. Losing the shortcut is a reason
to run the sweep, not a reason to postpone it, because the sweep is the measurement the shortcut was
standing in for. What would have blocked the sweep is P2 failing — that would mean the concurrency
points do not bracket the phenomenon and the sweep would measure the wrong region. P2 passed.

**P1's result is preserved exactly.** Its pre-registered tolerances are unchanged, its verdict stays
FAIL, and it is not rewritten as a PASS. See `results/pilot/PILOT_DECISION.md` and D10.

### Execution readiness — separate from the exit criteria above

The pilot's science gate is not the whole clearance. These are engineering and contract defects
found after the pilot, and the sweep does not start until they are resolved:

1. the D11 sweep orchestrator does not exist — `scripts/pilot/run_pilot.py` covers P1/P2/P3/P5 only;
2. `SKIPPED_PAST_SLO` is unimplemented; `driver.py` carries only the 10x-SLO abort;
3. the cell wall cap and the four-period window are unreconciled at the top of the ladder, and
   `CELL_TIMEOUT` is classified invalid where this document calls a wall-cap overrun a result
   (see "Cell-abort rule" and the note under it);
4. documentation that still makes unsupported causal claims about the below-wall gap must be
   corrected.

The sweep must retain, unchanged: the H9 periodic-stationarity gate and whole-period windows; H10
per-launch KV-capacity recording with idle-GPU preflight and teardown; counterbalanced configuration
order; the frozen corpus and prompt-set hash; 3 repetitions with a per-repetition engine restart.

Once those are resolved the serving sweep is cleared, with no further P1 experiment.

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

P3 estimates variance from short runs. The sweep ran 14.75 hours, over which the card heats and clocks
fall (H6). Short-run spread is a floor on long-run spread, not an estimate of it — which is why run
ordering is counterbalanced by rule rather than trusted to the variance figure.

## Sweep execution record — 2026-08-25

The sweep ran to completion. 124 cells, 14.75 h GPU-exclusive, `results/sweep/`,
`sweep_config_hash = df0f0f124d987a5c`. **Zero defect cells, zero invalid reasons, zero cells with an
incomplete window.** Twelve engine launches for the locked ladder plus prefill, three for
`SWEEP_REFINE` and three for `SWEEP_REFINE_SLO`.

The execution-readiness blockers listed under "Exit criteria" were all discharged before the run, and
the contract's own rules held up in practice:

- the derived budget mattered — BF16 at C=32 returned a valid 5.04-period window at **908 s**, past
  the flat 900 s cap this document used to specify;
- the SLO-only skip predicate mattered — BF16 at C=18 and FP8 at C=48 both preempt heavily while
  meeting the SLO, so a pressure-keyed skip would have truncated FP8 a full ladder step early;
- H6 was tested rather than assumed and produced no drift signal (0.20% throughput between
  repetitions 1 and 3);
- H10 did not occur: per-launch `kv_cache_tokens` was identical for every launch of a configuration.

**One rule was misapplied and cost a cell.** The 50 ms TPOT SLO is a decode criterion, but it also
governed `PREFILL_PROBE`, whose shape emits 32 output tokens. It fired at C=2 for BF16, so C=8 was
recorded `SKIPPED_PAST_SLO` rather than measured. BF16 at C=8 would need 65,792 KV tokens against
44,688 and may be infeasible regardless, but the recorded reason is a rule applied outside its
domain. **`PREFILL_PROBE` needs its own saturation criterion — TTFT-based — before it is re-run**, and
until then its BF16 arm is bounded at C=4.

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

Extended 2026-08-25 for the KL rig. A run is also invalid if:

- the observation grid is incomplete, duplicated, or unbalanced — any trajectory contributing fewer
  than its full set of retained positions invalidates the run rather than being aggregated as-is;
- a consumed trajectory, prompt-set, or checkpoint-content hash differs from the one the run
  contracted for;
- any KL value is non-finite, or negative beyond floating-point tolerance;
- the reference distribution assigns mass where the comparison distribution has none;
- the pre-run correctness/self-KL gate fails;
- dispatch verification does not reproduce the configuration's expected kernel, or a forbidden
  fallback pattern appears;
- distributions collected under different observed engine identities are pooled within one
  configuration;
- the persisted storage dtype differs from the one the run contracted for;
- the working tree was dirty and the override was not explicitly requested and recorded;
- the returned distribution is not the full vocabulary, is not finite, or does not normalise within
  the recorded tolerance.

A failed empirical gate — replication floor, cache equivalence, precision, engine-profile
equivalence — is a **measurement about the rig**, and is reported as such. It invalidates a quality
result only where the contract above says it does; it is not silently absorbed into the number.

**Three of them have failed, and the results stand as measured** (2026-08-26). The BF16 replication
floor is 5.6% of the provisional BF16→FP8 KL against a 1% bound; cache-on-versus-off is 4.74% on
BF16, itself below that floor; and fp16 storage misses its per-cell relative bounds by an order of
magnitude, so fp32 is retained. No threshold was relaxed after the fact, and no result was averaged
or re-referenced to convert a failure into a pass. Each failure travels with the number it bounds.

Two rules follow from what those gates measured, and both are binding on how results are read:

- **A worst-cell value may only be compared against a floor measured over the same grid size.** A
  maximum is an extreme-value statistic: the BF16 worst-cell floor grew 2.13x, from 3.01e-03 over 40
  cells to 6.43e-03 over 640, under identical underlying noise. Comparing maxima across different
  cell counts measures sample size. The analysis path refuses this comparison rather than reporting
  it with a caveat.
- **A ratio to the floor is a reproducibility diagnostic, never a magnitude.** Under CUDA graphs FP8
  and FP4 replicate to ~1e-11 nats, so a few nanonats reads as many multiples of the floor while
  being numerically nothing. Absolute nats are reported first.

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
