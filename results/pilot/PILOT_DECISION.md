# PILOT_DECISION

Pilot measurements are **diagnostic**. Nothing in this file is an experiment result and no
number here may be cited as measured degradation or measured serving benefit.

Generated 2026-08-23T22:53:36-0400 - 43 cells.
Corpus `c4-en-validation-shard0-v1`, prompt-set hash `2681c604332813f2...`.

## P1 - memory-bandwidth-bound assumption: **FAIL**

| C | BF16 tok/s | FP8 tok/s | FP4 tok/s | R_FP8 | R_FP4 | expected | BF16 KV pressure |
|---|---|---|---|---|---|---|---|
| 1 | 36.06 | 66.01 | 87.88 | 1.831 | 2.437 | 1.765 / 2.652 | none |
| 8 | 268.31 | 449.99 | 566.99 | 1.677 | 2.113 | 1.765 / 2.652 | none |
| 12 | 381.3 | 618.08 | 763.77 | 1.621 | 2.003 | 1.765 / 2.652 | none |

- PASS - ordering_bf16_lt_fp8_lt_fp4
- FAIL - ratios_within_20pct_of_d11
- FAIL - ratio_stable_across_concurrency
- PASS - subwall_free_of_bf16_kv_pressure
- PASS - all_cells_present
- finding: C=8 FP4: ratio 2.113 deviates -20.3% from expected 2.652
- finding: C=12 FP4: ratio 2.003 deviates -24.5% from expected 2.652
- finding: C=12 FP4: ratio drifted -17.8% vs C=1 (collapse threshold 15%)

## P2 - BF16 KV wall: **PASS**

- measured KV capacity this configuration: **44688 tokens**
- predicted wall, peak-footprint basis (2,560 tok/seq): **17**
- predicted wall, mean-occupancy basis (1,536 tok/seq): **29**
- **measured wall bracket: C_KV^BF16 in [17, 18]**
- pre-registered D11 neighbourhood: [15, 25]

| C | state | tok/s | TPOT P95 ms | KV P95 | preempt | recomputed | waiting max | status |
|---|---|---|---|---|---|---|---|---|
| 12 | clean | 382.21 | 31.3 | 0.645 | 0.0 | 0.0 | 0.0 | OK |
| 15 | clean | 464.16 | 32.25 | 0.834 | 0.0 | 0.0 | 0.0 | OK |
| 16 | clean | 489.61 | 32.59 | 0.874 | 0.0 | 0.0 | 0.0 | OK |
| 17 | clean | 425.35 | 39.84 | 0.907 | 0.0 | 0.0 | 0.0 | OK |
| 17 | clean | 425.96 | 39.79 | 0.91 | 0.0 | 0.0 | 0.0 | OK |
| 18 | pressured | 437.3 | 41.33 | 0.974 | 4.0 | 0.0 | 1.0 | OK |
| 18 | pressured | 437.27 | 41.33 | 0.959 | 4.0 | 0.0 | 0.0 | OK |

- PASS - transition_observed
- PASS - transition_in_d11_band_15_25
- PASS - not_limited_materially_below_15
- PASS - not_unconstrained_above_25
- PASS - transition_reproducible

## P3 - repetition count: **PASS**

| point | C | mean FP8 | mean FP4 | delta | SE | 95% half-width | rho | resolves |
|---|---|---|---|---|---|---|---|---|
| sub_wall | 8 | 449.99 | 566.99 | 117.0 | 0.075 | 0.238 | 0.002 | True |
| high_concurrency | 24 | 1000.3 | 1187.81 | 187.51 | 0.433 | 1.201 | 0.0064 | True |

- criterion rho <= 0.25
- **locked repetition count for the full serving sweep: 3**

Coefficient of variation, every configuration tested:

| cell | n | mean tok/s | sd | CV % | TPOT P95 CV % |
|---|---|---|---|---|---|
| BF16@C1 | 3 | 36.06 | 0.036 | 0.1 | 0.12 |
| BF16@C12 | 3 | 381.3 | 0.09 | 0.024 | 0.019 |
| BF16@C8 | 3 | 268.31 | 0.07 | 0.026 | 0.033 |
| FP4@C1 | 3 | 87.88 | 0.133 | 0.152 | 0.105 |
| FP4@C12 | 3 | 763.77 | 0.175 | 0.023 | 0.046 |
| FP4@C24 | 3 | 1187.81 | 0.441 | 0.037 | 0.015 |
| FP4@C8 | 3 | 566.99 | 0.118 | 0.021 | 0.033 |
| FP8@C1 | 3 | 66.01 | 0.015 | 0.023 | 0.024 |
| FP8@C12 | 3 | 618.08 | 0.167 | 0.027 | 0.026 |
| FP8@C24 | 3 | 1000.3 | 0.606 | 0.061 | 0.083 |
| FP8@C8 | 3 | 449.99 | 0.053 | 0.012 | 0.017 |

## P4 - achievable HBM bandwidth: **VALID**

Independent CUDA streaming kernels, CUDA-event timed, working set 1.0 GiB per array (21.3x L2). Not derived from decode throughput.

| kernel | median GB/s | P95 | max | CV % | vs spec |
|---|---|---|---|---|---|
| copy_read_plus_write | 545.64 | 546.86 | 547.66 | 0.145 | 0.812 |
| triad_2read_1write | 564.7 | 565.69 | 566.11 | 0.107 | 0.84 |
| read_only | 620.08 | 620.62 | 620.82 | 0.107 | 0.923 |

- spec bandwidth 672.0 GB/s (192-bit x 14.001 GHz x 2); derived from CUDA device attributes: bus width x memory clock x 2 (DDR). Not a vendor datasheet quote.
- kernel correctness validated: {'copy': True, 'triad': True}
- device-memory only; no host-device transfer is involved

## P5 - harness validation: **PASS**

- PASS - exact_input_512_tokens
- PASS - exact_output_2048_requested
- PASS - no_output_length_violation
- PASS - no_input_length_violation
- PASS - finish_reason_length_only
- PASS - frozen_corpus_identical
- PASS - prefix_caching_off_in_config
- PASS - prefix_cache_hits_zero
- PASS - no_request_failures
- PASS - kernel_dispatch_as_intended
- PASS - kv_cache_usage_recorded
- PASS - preemption_counter_recorded
- PASS - recomputed_tokens_counter_recorded
- PASS - waiting_requests_counter_recorded
- PASS - telemetry_recorded
- PASS - window_spans_whole_periods
- PASS - steady_state_gate_fired
- PASS - chunk_count_matches_usage_tokens
- PASS - engine_token_rate_agrees
- PASS - counters_exercised_nonzero
- PASS - cell_status_classification
- PASS - prefill_probe_plumbing
- PASS - slo_visibility

SLO visibility (TPOT P95 <= 50.0 ms): BF16 within SLO at C=8 and C=12 = **True**
- BF16@C12: 31.3 ms (True), 31.3 ms (True), 31.3 ms (True)
- BF16@C8: 29.6 ms (True), 29.6 ms (True), 29.6 ms (True)
- FP4@C12: 15.5 ms (True), 15.5 ms (True), 15.5 ms (True)
- FP4@C8: 14.0 ms (True), 14.0 ms (True), 14.0 ms (True)
- FP8@C12: 19.2 ms (True), 19.2 ms (True), 19.2 ms (True)
- FP8@C8: 17.6 ms (True), 17.6 ms (True), 17.6 ms (True)

## Correctness gate: **CLEAN**

32 real held-out C4 contexts, position 1, D13 path. Sanity trip-wire, not a quality result.

| comparison | median nats | p90 | max | finite | non-negative | top-1 agreement |
|---|---|---|---|---|---|---|
| BF16 vs BF16_selfcheck | 0.000e+00 | 0.000e+00 | 0.000e+00 | True | True | 1.0000 |
| BF16 vs FP8 | 5.080e-03 | 1.741e-02 | 9.304e-02 | True | True | 0.9375 |
| BF16 vs FP4 | 4.552e-02 | 1.720e-01 | 1.392e+00 | True | True | 0.8125 |

- PASS - all_configs_loaded
- PASS - dispatch_reconfirmed
- PASS - full_vocab_all_passes
- PASS - distributions_finite
- PASS - distributions_normalized
- PASS - bf16_self_kl_near_zero
- PASS - fp8_kl_finite_nonnegative
- PASS - fp4_kl_finite_nonnegative
- PASS - fp8_not_pathological
- PASS - fp4_not_pathological
- PASS - fp8_top1_agreement_ok
- PASS - fp4_top1_agreement_ok

## Clearance

**NOT CLEARED for the full serving sweep.**

- P1 = FAIL

P1 or P2 failing means the physical assumptions behind D11 are not supported by the
pilot. The decision to reopen is D11's, not the harness's - the evidence is reported
as measured and the interpretation is not adjusted to fit it.

## Decisions that must be reopened

- D11 bandwidth-vs-capacity decomposition, and D10's weight-size-ratio prediction

