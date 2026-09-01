"""No-GPU verification of the cell state machine against a controllable stub engine.

The pilot's 46 recorded cells are all status OK / meets_slo true, so replaying them exercises
none of the terminal branches. This drives a fake vLLM that can oscillate, collapse or stall.
"""

import asyncio
import os
import sys
import threading
import time

from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import analyze_ceiling as ac  # noqa: E402
from harness import common, driver  # noqa: E402
from harness import orchestration as orch  # noqa: E402
from harness import run_sweep as rs  # noqa: E402


class StubEngine:
    """Emits tokens at a controllable aggregate rate and serves a Prometheus-shaped /metrics."""

    def __init__(self, rate, n_out):
        self.rate = rate
        self.n_out = n_out
        self.active = 0
        self.generated = 0
        self.stalled = False
        # keyed on tokens emitted, not wall time: the gate's duration is not predictable
        # enough to land a rate change inside the measure window by clock alone
        self.plan = []          # (tokens_emitted, new_rate)
        self.t0 = time.perf_counter()

    def current_rate(self):
        r = self.rate
        for at, new in self.plan:
            if self.generated >= at:
                r = new
        return r

    async def completions(self, request):
        body = await request.json()
        n = body.get("max_tokens", self.n_out)
        n_in = len(body.get("prompt") or [])
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        self.active += 1
        try:
            for i in range(n):
                while self.stalled:
                    await asyncio.sleep(0.05)
                await asyncio.sleep(max(self.active, 1) / float(self.current_rate()))
                self.generated += 1
                fin = "length" if i == n - 1 else None
                await resp.write(
                    b'data: {"choices":[{"text":"x","finish_reason":'
                    + (b'"length"' if fin else b'null') + b'}]}\n\n')
            await resp.write(
                ('data: {"choices":[],"usage":{"completion_tokens":%d,"prompt_tokens":%d}}\n\n'
                 % (n, n_in)).encode())
            await resp.write(b'data: [DONE]\n\n')
            await resp.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass
        finally:
            self.active -= 1
        return resp

    async def metrics(self, request):
        return web.Response(text=(
            "vllm:kv_cache_usage_perc 0.5\n"
            "vllm:num_preemptions_total 0\n"
            "vllm:prompt_tokens_recomputed_total 0\n"
            "vllm:num_requests_waiting 0\n"
            "vllm:num_requests_running %d\n"
            "vllm:prefix_cache_queries_total 0\n"
            "vllm:prefix_cache_hits_total 0\n"
            "vllm:generation_tokens_total %d\n"
            "vllm:prompt_tokens_total 0\n"
            "vllm:prompt_tokens_cached_total 0\n" % (self.active, self.generated)))


class StubServer:
    """Matches the surface Cell uses: base, alive(), metrics()."""

    def __init__(self, engine, port):
        self.engine = engine
        self.port = port
        self.base = "http://127.0.0.1:%d" % port
        self._alive = True
        self.startup = {"kv_cache_tokens": 44688, "dispatch_verdict": {"ok": True}}

    def alive(self):
        return self._alive

    def wait_drained(self, timeout=600):
        return True

    def stop(self, *a, **kw):
        self._alive = False

    def metrics(self):
        import requests
        return common.engine_counters(
            common.parse_prometheus(requests.get(self.base + "/metrics", timeout=5).text))


def serve(engine, port):
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = web.Application()
        app.router.add_post("/v1/completions", engine.completions)
        app.router.add_get("/metrics", engine.metrics)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
        loop.run_forever()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(1.0)


RESULTS = []


def check(name, got, want):
    ok = got == want
    RESULTS.append((name, ok, got, want))
    print("  %-9s %-46s got=%s want=%s" % ("PASS" if ok else "FAIL", name, got, want))
    return ok


def run_case(name, n_out, conc, rate, plan=None, stalled_after=None, **cc_kw):
    port = 9700 + len(RESULTS) + hash(name) % 50
    eng = StubEngine(rate, n_out)
    eng.plan = plan or []
    serve(eng, port)
    srv = StubServer(eng, port)
    common.WORKLOADS["SELFTEST"] = {"input_tokens": 8, "output_tokens": n_out}
    prompts = [{"index": i, "token_ids": list(range(8)), "prefix_hash": "h%d" % i}
               for i in range(64)]
    kw = dict(workload="SELFTEST", concurrency=conc, repetition=1,
              min_requests_factor=4, window_floor_s=3.0, warmup_wall_cap_s=30.0,
              gate_timeout_s=30.0, hard_cap_s=90.0, warmup_lifetimes=2, min_periods=4)
    kw.update(cc_kw)
    cc = driver.CellConfig(**kw)
    if stalled_after:
        def stall():
            time.sleep(stalled_after)
            eng.stalled = True
        threading.Thread(target=stall, daemon=True).start()
    return driver.run_cell(srv, prompts, cc, tag=name)


def synth_cells(cfg, k, per_rep):
    """(rep -> {C: meets_slo or None}) into cell records the analyzer will accept."""
    out = []
    for rep, verdicts in per_rep.items():
        for C, slo in verdicts.items():
            if slo is None:
                continue
            out.append({"workload": "DECODE_PRIMARY", "configuration_id": cfg, "concurrency": C,
                        "repetition": rep, "job": "SWEEP_CEILING_REP", "status": "OK",
                        "meets_slo": slo, "tpot_ms_p95": 49.0 if slo else 51.0})
    return out


def ceiling_criterion_cases():
    k, triplet = 57, [56, 57, 58]

    def score(v):
        at = {C: {"status": "OK", "meets_slo": v[C]} for C in triplet if v.get(C) is not None}
        return ac.score_repetition(k, triplet, at)

    check("t9.confirmed", score({56: True, 57: True, 58: False}), ("CONFIRMED", 57))
    check("t9.moved_down", score({56: True, 57: False, 58: False}), ("MOVED_DOWN", 56))
    check("t9.unresolved_above", score({56: True, 57: True, 58: True}),
          ("UNRESOLVED_ABOVE", None))
    check("t9.unresolved_below", score({56: False, 57: False, 58: False}),
          ("UNRESOLVED_BELOW", None))
    check("t9.non_monotone", score({56: False, 57: True, 58: False}), ("NON_MONOTONE", None))
    check("t9.incomplete_missing_k", score({56: True, 58: False}), ("INCOMPLETE", None))
    # rep 1 never measured FP4 C=69; K and K+1 alone must still place the ceiling
    check("t9.k_minus_1_optional_when_k_passes", score({57: True, 58: False}), ("CONFIRMED", 57))
    check("t9.k_minus_1_required_when_k_fails", score({57: False, 58: False}),
          ("INCOMPLETE", None))

    cfg = "FP8_PRIMARY"
    conf = {r: {56: True, 57: True, 58: False} for r in (1, 2, 3)}
    check("t9.overall_confirmed",
          ac.analyze(synth_cells(cfg, k, conf))["configurations"][cfg]["status"], "CONFIRMED")
    unstable = dict(conf); unstable[3] = {56: True, 57: False, 58: False}
    check("t9.overall_unstable",
          ac.analyze(synth_cells(cfg, k, unstable))["configurations"][cfg]["status"], "UNSTABLE")
    moved = {r: {56: True, 57: False, 58: False} for r in (1, 2, 3)}
    check("t9.overall_moved",
          ac.analyze(synth_cells(cfg, k, moved))["configurations"][cfg]["status"], "MOVED")
    partial = {1: conf[1], 2: conf[2]}
    check("t9.overall_not_yet",
          ac.analyze(synth_cells(cfg, k, partial))["configurations"][cfg]["status"],
          "NOT_YET_REPLICATED")


def ceiling_group_case():
    """The group must emit the whole triplet -- including the point that breaches the SLO."""
    import shutil
    import tempfile

    cfg, port = "BF16_REFERENCE", 9899
    # aggregate ~50 tok/s makes TPOT ~ C*20 ms, so C=1,2 meet the 50 ms SLO and C=3 does not
    eng = StubEngine(50.0, 12)
    serve(eng, port)
    srv = StubServer(eng, port)

    tmp = tempfile.mkdtemp(prefix="ceilingtest_")
    saved_decode = common.WORKLOADS["DECODE_PRIMARY"]
    saved_k = dict(rs.CEILING_REP)
    saved_launch = rs.launch
    saved_dir = rs.OUT_DIR
    launches = []
    try:
        common.WORKLOADS["DECODE_PRIMARY"] = {"input_tokens": 8, "output_tokens": 12}
        rs.CEILING_REP[cfg] = 2
        rs.set_out_dir(tmp)
        rs.launch = lambda *a, **kw: (launches.append(a) or srv)

        prompts = [{"index": i, "token_ids": list(range(8)), "prefix_hash": "h%d" % i}
                   for i in range(64)]
        manifest = {"corpus_version": "stub", "prompt_set_hash": "stub"}
        env = {"gpu": {}, "software": {"vllm": "stub"}}
        kw = dict(min_requests_factor=4, window_floor_s=3.0, warmup_wall_cap_s=30.0,
                  gate_timeout_s=30.0, hard_cap_s=90.0, warmup_lifetimes=2, min_periods=4,
                  margin=1.5, abort_tpot_ms=None)

        rs.replicate_ceiling_group(cfg, 2, env, manifest, prompts, [], None, cell_kw=kw)
        rows = orch.read_cells(rs.CELLS)
        check("t10.triplet_emitted", [r["concurrency"] for r in rows], [1, 2, 3])
        check("t10.job_label", sorted({r["job"] for r in rows}), ["SWEEP_CEILING_REP"])
        check("t10.repetition", sorted({r["repetition"] for r in rows}), [2])
        check("t10.one_launch", len(launches), 1)
        check("t10.breaching_point_kept",
              [r["meets_slo"] for r in rows], [True, True, False])
        check("t10.ceiling_recorded", sorted({r["ceiling_under_test"] for r in rows}), [2])

        # resume: a second call must not re-run a cell already in the artifact
        rs.replicate_ceiling_group(cfg, 2, env, manifest, prompts, [], None, cell_kw=kw)
        check("t10.resume_is_idempotent", len(orch.read_cells(rs.CELLS)), 3)
        check("t10.resume_no_relaunch", len(launches), 1)

        # a failed launch must leave three explicit placeholders, not three absent cells
        def boom(*a, **kw):
            raise rs.LaunchError("stub launch failure")
        rs.launch = boom
        rs.replicate_ceiling_group(cfg, 3, env, manifest, prompts, [], None, cell_kw=kw)
        r3 = [r for r in orch.read_cells(rs.CELLS) if r["repetition"] == 3]
        check("t10.launch_failure_placeholders",
              sorted(r["status"] for r in r3), ["LAUNCH_FAILED"] * 3)
    finally:
        common.WORKLOADS["DECODE_PRIMARY"] = saved_decode
        rs.CEILING_REP.clear()
        rs.CEILING_REP.update(saved_k)
        rs.launch = saved_launch
        rs.set_out_dir(saved_dir)
        shutil.rmtree(tmp, ignore_errors=True)

    # the frozen serving spec must survive a selftest that mutates WORKLOADS
    check("t10.spec_hash_unperturbed", rs.spec_hash(), "df0f0f124d987a5c")


def main():
    common.TELEMETRY_PERIOD_S = 0.5
    driver.common.TELEMETRY_PERIOD_S = 0.5
    print("\n== T1 steady state -> OK, whole-period window ==")
    r = run_case("t1_steady", n_out=40, conc=4, rate=400)
    check("t1.status", r["status"], "OK")
    check("t1.outcome", r["outcome_class"], "measured")
    check("t1.window_complete", r["window_is_complete"], True)
    check("t1.periods>=4", r["periods_in_window"] >= 4.0, True)
    check("t1.meets_slo", r["meets_slo"], True)
    tok = r["window_streamed_tokens"] / float(40 * 4)
    check("t1.periods_are_token_exact", abs(tok - r["periods_in_window"]) < 0.05, True)

    print("\n== T2 within-SLO collapse -> NONSTATIONARY (infeasible), not a defect ==")
    r = run_case("t2_collapse", n_out=40, conc=8, rate=1600,
                 plan=[(11000, 250.0)], min_periods=8, window_floor_s=4.0)
    print("     diag:", {k: r[k] for k in ("period_frozen_s", "period_at_close_s",
                                           "period_drift_ratio", "measure_budget_s",
                                           "window_seconds", "tpot_ms_p95")})
    check("t2.status", r["status"], "NONSTATIONARY")
    check("t2.outcome", r["outcome_class"], "infeasible")
    check("t2.drift_recorded", r["period_drift_ratio"] is not None, True)

    print("\n== T3 collapse past the SLO -> SLO_VIOLATED (a result) ==")
    r = run_case("t3_slo", n_out=40, conc=8, rate=1600,
                 plan=[(11000, 100.0)], min_periods=8, window_floor_s=4.0)
    print("     diag:", {k: r[k] for k in ("period_drift_ratio", "measure_budget_s",
                                           "window_seconds", "tpot_ms_p95", "meets_slo")})
    check("t3.status", r["status"], "SLO_VIOLATED")
    check("t3.outcome", r["outcome_class"], "measured")
    check("t3.valid_result", r["valid_result"], True)

    print("\n== T4 hard stall -> CELL_HUNG (defect), and a record still exists ==")
    r = run_case("t4_stall", n_out=200, conc=4, rate=400, stalled_after=6.0,
                 stall_s=4.0, hard_cap_s=25.0)
    check("t4.status", r["status"], "CELL_HUNG")
    check("t4.outcome", r["outcome_class"], "defect")
    check("t4.record_exists", isinstance(r.get("cell_wall_seconds"), float), True)

    print("\n== T5 short-period workload (the PREFILL_PROBE arithmetic) ==")
    r = run_case("t5_prefill", n_out=8, conc=2, rate=200, window_floor_s=5.0)
    check("t5.status", r["status"], "OK")
    check("t5.budget_exceeds_floor", r["measure_budget_s"] >= 5.0, True)
    check("t5.window>=floor", r["window_seconds"] >= 5.0, True)

    print("\n== T6 budget arithmetic dominates the whole close conjunction ==")
    cc = driver.CellConfig(workload="DECODE_PRIMARY", concurrency=1, min_requests_factor=4,
                           window_floor_s=120.0, min_periods=4, margin=1.5)
    c = driver.Cell(StubServer(StubEngine(1, 1), 1), [], cc)
    check("t6.prefill_not_starved", c.compute_measure_budget_s(1.0) >= 120.0, True)
    check("t6.uses_request_term", round(c.compute_measure_budget_s(100.0)), 750)

    print("\n== T8 mis-sized budget at a stable rate -> CELL_TIMEOUT (defect), not SLO_VIOLATED ==")
    r = run_case("t8_budget", n_out=40, conc=4, rate=400, margin=0.15)
    check("t8.status", r["status"], "CELL_TIMEOUT")
    check("t8.outcome", r["outcome_class"], "defect")
    check("t8.not_laundered_as_result", r["valid_result"], False)
    check("t8.window_incomplete", r["window_is_complete"], False)

    print("\n== T7 status -> outcome_class map is total ==")
    check("t7.unknown_defaults_defect", driver.outcome_class("SOMETHING_NEW"), "defect")
    check("t7.skip_is_infeasible", driver.outcome_class("SKIPPED_PAST_SLO"), "infeasible")
    check("t7.steady_not_reached", driver.outcome_class("STEADY_STATE_NOT_REACHED"), "infeasible")

    print("\n== T9 ceiling-replication criterion ==")
    ceiling_criterion_cases()

    print("\n== T10 ceiling-replication group against the stub engine ==")
    ceiling_group_case()

    bad = [n for n, ok, _, _ in RESULTS if not ok]
    print("\n%d/%d checks passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    if bad:
        print("FAILED: %s" % ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
