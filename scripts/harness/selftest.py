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
from harness import common, driver  # noqa: E402


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

    def alive(self):
        return self._alive

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

    bad = [n for n, ok, _, _ in RESULTS if not ok]
    print("\n%d/%d checks passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    if bad:
        print("FAILED: %s" % ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
