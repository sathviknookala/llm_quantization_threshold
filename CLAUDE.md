# CLAUDE.md — Quantization Deployment Boundary

This project studies how far an LLM should be quantized before the additional serving benefit no longer justifies the additional degradation in model behavior. The experiment compares complete deployment configurations of the same base model across a candidate precision ladder of BF16 -> FP8 -> FP4, measuring both model-quality degradation and serving improvement under a fixed deployment environment. The project is measurement-driven: no precision level, quantization scheme, or model is required to win, and the measured tradeoff is itself the result.

## Git

- **Never add Claude as an author or co-author on commits or pushes.** The user is always the sole author — no `Co-Authored-By: Claude` trailer and no `Generated with Claude Code` line.
- Inspect the repository before assuming the default branch, remote, existing layout, or build/test commands.
- Do not commit benchmark claims that are not backed by tracked result artifacts.

## Doc map

This file is the always-loaded hub. Keep detailed methodology and measured state in `docs/`.

- **`docs/PROJECT_SPEC.md`** — authoritative six-part research design and scope. *Read before changing the research question, model strategy, study dimensions, success criteria, or final deliverable.*
- **`docs/HARDWARE_PROFILE.md`** — measured machine profile and the constraints it places on the study. *Read before choosing models, precision formats, inference backends, or interpreting serving results.*
- **`docs/DECISIONS.md`** — current project decisions, open design gates, and the reasoning behind them. *Read before treating any design choice as settled or reopening a previously resolved question.*
- **`docs/QUANTIZATION_CONFIGS.md`** — reproducible definition of every deployment configuration. *Read before creating, loading, or benchmarking a BF16/FP8/FP4 checkpoint.*
- **`docs/EXPERIMENTAL_CONTRACT.md`** — controls, workload definitions, run validity, warmup, repetition, telemetry, and comparability rules. *Read before collecting timing or memory results.*
- **`docs/EVALUATION_RIG.md`** — quality metrics, serving metrics, uncertainty treatment, and marginal tradeoff calculations. *Read before implementing or changing evaluation code.*
- **`docs/LIMITATIONS.md`** — claims the study cannot make and known threats to validity. *Read before interpreting final results or writing conclusions.*

**Code layout.** `scripts/harness/` is the shared measurement harness — `common.py` (identity,
telemetry, statistics), `driver.py` (one timed cell), `server.py` (vLLM lifecycle),
`orchestration.py` (cell identity, resume, launch preflight). `run_pilot.py` and `run_sweep.py` are
runners over it; `analyze.py` turns pilot cells into the pilot verdict; `selftest.py` exercises the
cell state machine against a stub engine with no GPU. Renamed from `scripts/pilot/` on 2026-08-24 —
the harness outlives the pilot.

## Research contract

The unit under study is a **complete deployment configuration**, not a precision label in isolation. A configuration includes, where applicable:

- base checkpoint and tokenizer
- weight precision
- activation precision
- KV-cache precision
- quantization algorithm / format
- calibration procedure
- group or block size
- inference backend
- kernels actually exercised on the target GPU

For the same base model and equivalent workloads, compare successive configurations and measure:

```text
quality change + serving change
        -> marginal tradeoff
        -> deployment boundary / knee
```

The core question is not “is FP4 faster than BF16?” It is:

> What do we gain by taking the next quantization step, what do we lose, and is the additional serving gain still large enough to justify the additional quality degradation?

Detailed definitions live in `docs/PROJECT_SPEC.md` and `docs/EVALUATION_RIG.md`.

## Target regime / scope

The primary study is deliberately narrow:

```text
Hardware:       one fixed, profiled GPU machine
Precision:      candidate BF16 -> FP8 -> FP4
Reference:      BF16 deployment of the same base model
Serving:        single-GPU inference
Workloads:      prefill-heavy, balanced, decode-heavy
Concurrency:    low load through saturation
Multi-GPU:      out of scope for the primary study
```

A model ladder is **not required** to answer the core within-model tradeoff question. Additional models are only justified if the project explicitly expands to test whether the measured boundary generalizes across model sizes or families. See `docs/DECISIONS.md`.

## Core hypothesis

> Under a fixed model, hardware environment, serving stack, and workload contract, the quality loss and serving benefit of successive quantization steps can be measured precisely enough to determine whether further quantization remains worthwhile.

A valid result may favor BF16, FP8, FP4, different choices for different workloads, or conclude that the differences are not resolvable with sufficient confidence.

**What it cannot claim.** A boundary measured on this machine, model, backend, and workload is not automatically a universal boundary for other hardware, models, quantization algorithms, or serving stacks.

**The comparison bar is the same base model under the same serving stack and equivalent workload**, not theoretical compression ratios, vendor peak throughput, or a weaker implementation path.

## Workflow rules

### Plan before acting
- Enter plan mode for non-trivial work or architectural / experimental decisions.
- Prefer the cheapest measurement that can invalidate an assumption before building a large harness around it.
- If evidence contradicts a current design claim, update the tracked doc that owns that claim instead of preserving both versions.

### Measurement discipline
- Never quote a benchmark number that is not in a tracked result artifact; name the artifact when using the number in documentation.
- Distinguish measured facts from assumptions, estimates, and vendor/backend capability claims.
- Do not infer usable acceleration from a datatype name alone. A quantization format is part of the study only after the selected backend is verified to execute an appropriate path on the target GPU.
- Before timed runs, verify the GPU is otherwise idle and record enough telemetry to detect power, thermal, clock, or memory-state contamination.
- Do not compare runs produced under different workload definitions, tokenizer/input data, generation lengths, scheduler settings, cache policy, or software stack unless the difference itself is the intended experimental variable.
- Warmup, repetition count, saturation criterion, and run-validity rules belong in `docs/EXPERIMENTAL_CONTRACT.md`, not ad hoc benchmark scripts.

### Verification before done
- A new quantization or serving path is not “supported” until it is exercised successfully on the target machine.
- A benchmark path is not “correct” until outputs are validated against the relevant reference before timing.
- A result is not final until the raw artifact, configuration, and reproduction command are tracked.
- Before quoting a committed number, verify the current tree still reproduces the configuration that generated it.

### Scope discipline
- Do not turn the project into a broad model benchmark unless a tracked decision explicitly changes the research question.
- Do not add quantization methods merely because they are popular. They must answer a research need and have a fair serving implementation on the target hardware.
- Do not optimize kernels or inference code as the primary project goal. Optimization is only relevant when needed to make deployment configurations fairly comparable.

## Code style: comments

- No paragraph-style or multi-line block comments explaining what code does.
- Comments only where intent is not obvious from the code itself: non-obvious tradeoffs, gotchas, or why a seemingly natural alternative is wrong.
- Keep comments short and explain **why**, not **what**.
- Do not duplicate configuration values in comments when they already live in a config or tracked specification.

## Current focus

**The D11 serving sweep is COMPLETE.** 124 cells, 14.75 h GPU-exclusive, finished 2026-08-25
14:00:56. Artifacts in `results/sweep/` (`cells.jsonl`, `run.log`, `manifest.json`,
`sweep_config_hash = df0f0f124d987a5c`). **Zero defect cells, zero invalid reasons, zero incomplete
windows.**

Phase-1 deliverable's serving axis is measured. The quality axis is not, so the deployment boundary
is **not** located yet.

### Headline — maximum concurrency within the SLO (TPOT P95 <= 50 ms)

Ladder points are n=3 (median, full spread). The **refined** ceilings come from the
`SWEEP_REFINE_SLO` bisection and are **n=1, repetition 1 only**.

```text
              ladder point   refined ceiling   first breach   tok/s at ceiling
BF16                    16                21             22              488.2
FP8                     48                57             58             1401.2
FP4                     64                70             71             1745.5
```

The ladder understated every configuration by 24-31%; the bisection is what makes these numbers
usable. Ratios at the refined ceilings: FP8 **2.71x** BF16 concurrency, FP4 **3.33x** BF16 and
**1.23x** FP8. Throughput at the ceiling: FP8 2.87x BF16, FP4 3.57x BF16.

### KV-pressure walls, refined — the peak-footprint arithmetic predicted all three exactly

```text
         predicted   measured bracket   basis
BF16            17            [17, 18]   44,688 KV tokens / 2,560
FP8             38            [38, 39]   97,888 / 2,560
FP4             47            [47, 48]   120,944 / 2,560
```

BF16's bracket reproduces pilot P2 exactly. The locked ladder returned `[32,48]` for *both* FP8 and
FP4; the refinement is what separates them, which is why it was added (D11, "Ladder resolution").

**The SLO ceiling is not the wall.** Every configuration serves past its wall before breaching the
latency bound — BF16 to 1.24x its wall, FP8 and FP4 to ~1.5x. Preemption is not SLO violation, and
the two must not be conflated when reporting.

### What may not be claimed

Below-wall throughput differences are reproducible and **concurrency-dependent**, and they are not
attributed to any mechanism. FP4-over-BF16 measures 2.43x at C=1, 2.12x at C=8, 1.93x at C=16 — quote
the concurrency with the number, never as a bandwidth or weight-residency benefit. The capacity leg
is confirmed by mechanism (KV arithmetic predicts the walls); the throughput leg is measured, not
explained.

The headline is **maximum in-flight requests** within the SLO, not "concurrent users": the driver is
closed-loop with `ignore_eos` and homogeneous prompts, with no arrival process.

```text
0. D10/D11 corrected; P1 exit criterion amended               DONE 2026-08-24
1-3. orchestrator, SKIPPED_PAST_SLO, budget/window reconciliation   DONE 2026-08-24
4. serving sweep + refinement                                 DONE 2026-08-25
5. quality run                                                 <- current step
   KL first (D13 path, proven); PPL and tasks need D14/D15
6. marginal tradeoff: quality loss vs sustainable concurrency
```

## Last session

**Session 5 — amended the P1 exit criterion, cleared the three engineering blockers, ran the sweep.**

- **P1 recorded as an informative falsification, not a gate.** Tolerances untouched, verdict still
  FAIL. The contract's exit criterion was rewritten; the post-hoc three-term traffic model in D10 is
  explicitly barred from becoming a replacement gate. The stale `NOT CLEARED` verdict was superseded
  by changing `analyze.py`'s clearance logic rather than hand-editing the artifact, and
  `cleared_for_full_serving_sweep` is now conjunctive so a passing science gate cannot read as
  permission to launch.
- **Propagated the correction** to `PROJECT_SPEC.md` and `HARDWARE_PROFILE.md`, which still carried
  the framing D11 forbids, and withdrew the "bandwidth travels better" portability claim in
  `LIMITATIONS.md` as unsupported.
- **Rebuilt the cell budget.** `wall_cap_s` was flat while the window required
  `min_periods * period_estimate_s()` recomputed every tick, so the requirement receded as
  throughput fell and the cap bit hardest on the capacity cells. Periods are now counted from tokens
  (`tokens / (out_tokens * C)`) — exact, verified against all 46 pilot cells to within 2.2% — and
  budgets derive from the period frozen at gate-fire, sized to dominate the whole close conjunction.
  Measured proof: BF16 C=32 returns a valid 5.04-period window at 908 s, past the documented 900 s
  cap that would have discarded it.
- **`SKIPPED_PAST_SLO`** on an SLO-only allowlist. Nothing else truncates a ladder. Validated twice
  on real data: BF16@C18 and FP8@C48 both preempt heavily and still meet the SLO; keying the skip on
  pressure would have understated FP8's capacity by a full ladder step.
- **`outcome_class`** (measured / infeasible / defect) added because `valid_result` was answering two
  questions. Unmapped statuses default to `defect`, never `infeasible`.
- Un-clamped the stationarity half-window (the fixed 10-sample cap broke H9 above ~90 s periods);
  made `sock_read` period-derived so starvation past a wall is not recorded as request failures;
  serialized `CellConfig`, throttle reasons and corpus-wrap counts into every record.
- Added a **`SWEEP_REFINE_SLO`** phase mid-run after noticing the original refinement bisects KV
  pressure while the deliverable is the SLO crossing — a different transition. Without it every
  headline number carried ladder granularity, understating all three by 24-31%.
- **Two bugs found by running rather than reading.** A killed orchestrator orphaned an EngineCore
  holding 22 GiB (`launch()` outside the `try/finally`), and `stop()`'s fallback
  `pkill -f 'vllm serve'` had never matched anything because v1 renames the child to
  `VLLM::EngineCore` — dead code guarding exactly that case.
- **One self-inflicted incident, recorded.** Importing `run_sweep` in a throwaway process registered
  its `atexit` handler, which killed any GPU-holding PID including the live sweep's engine. Cost one
  cell (BF16 C=32 rep1), quarantined to `cells_externally_killed.jsonl` and later re-measured at
  572.85 tok/s against the smoke's 572.4. Teardown is now process-group-scoped and `atexit` arms only
  under `__main__`.
- `scripts/pilot/` -> `scripts/harness/`, shared helpers extracted to `orchestration.py`; verified
  behaviour-free by the analyzer reproducing every pilot artifact byte-identically.
- Verification before GPU time: `selftest.py` drives the cell state machine against a stub engine
  that oscillates, collapses, stalls, starves and mis-budgets — 27/27.

**Sweep quality.** Repeatability <=0.22% throughput spread across 3 repetitions on all but one cell
(FP4@C8, 1.73%). Pre-registered H6 drift test: worst matched-cell drift rep1 vs rep3 is 0.20%
throughput and 0.76% clock, both inside the pilot baseline — **no drift signal, so H6 is measured
rather than assumed for this run**. `kv_cache_tokens` was identical across all launches per
configuration (44,688 / 97,888 / 120,944), so H10 contamination did not occur.

## Known issues / unresolved premises

- **The refined ceilings are n=1.** The 21 / 57 / 70 figures come from a repetition-1 bisection. The
  ladder points bracketing them are n=3 with <=0.22% spread, and per-cell CV is tiny, so the risk is
  low — but the headline numbers have not been replicated and must be labelled n=1 wherever they
  appear. Replicating the three bisections is cheap (~9 cells) and is the first thing to do if the
  numbers are going to be quoted.
- **`PREFILL_PROBE` inherited the decode SLO, and it cost a cell.** The 50 ms TPOT bound is a decode
  criterion; applied to a 32-output-token prefill shape it fired at C=2 for BF16, so C=8 was skipped
  and BF16 has no C=8 probe point. BF16 at C=8 would need 65,792 KV tokens against 44,688 anyway, so
  the point may be infeasible regardless — but the *reason* it is missing is a rule misapplied, not a
  measurement. The probe needs its own saturation criterion (TTFT-based) before it is re-run.
- **The below-wall gap is unattributed and that is now a settled position.** Reproducible,
  concurrency-dependent, and not separated into weight traffic / KV traffic / kernel efficiency /
  scheduling by this measurement. The post-hoc three-term model in D10 stays post-hoc.
- **Latin-square carryover is unbalanced.** FP8 follows BF16 in 2 of 3 repetitions, FP4 in 0 of 3;
  balancing carryover across three treatments needs six sequences. The thermal preflight gate and the
  drift test stand in for it, and the drift test came back clean, but the design limitation is real.
- **Queue *time* is still not recorded** — only depth. D11 asked for both; this build has no
  per-request queue-time observable.
- **`meets_slo` is survivor-biased.** Computed only over requests that both start and finish inside
  the window, so starved requests contribute nothing. Read it with `num_waiting_reqs_max` and
  `window_completed_requests`, both recorded.
- **No quality axis yet.** D14 (perplexity corpus/budget) and D15 (downstream tasks) are open, and
  chat-formatted tasks stay blocked by the `chat_template` deviation. The sweep produces one axis of
  a two-axis deliverable; nothing here locates the deployment boundary.
- **Checkpoint provenance has one open deviation.** Weights are SHA256-identical to official, but
  `tokenizer_config.json` carries a shorter `chat_template`. Within-model comparability is
  unaffected; absolute instruct-task scores would not be comparable to published Llama numbers.
- **Calibration sensitivity is untested, and applies only to FP4.** The single FP4 draw used 128
  ultrachat samples, seed 0.
- **The GPU is power-limited at 145 W** and `throttle_sw_power_cap_frac` reads 1.0 for entire
  windows, so every number is measured under a power ceiling rather than at fixed clocks.
- **Phase 1 measures one workload shape.** The arithmetic benefit of low precision is observed only
  in the bounded prefill probe; prefill-dominated deployments are out of scope.

At the end of a session, overwrite `Current focus`, `Last session`, and `Known issues / unresolved premises` in place. Git history is the changelog; this file should remain a current-state hub.
