"""Closed-loop request driver that enforces the D10/D11 contract for one serving cell.

`vllm bench serve` supplies a semaphore over a fixed prompt count and nothing else; the
warmup discard, steady-state entry, cell abort and per-point counters all live here.
"""

import asyncio
import json
import os
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilot import common  # noqa: E402

STATUS_OK = "OK"

# k intervals must span at least one oscillation period; at these rates 6 x 10 s does.
STATIONARITY_HALF_WINDOW = 6
STATIONARITY_HALF_WINDOW_MAX = 10
STATIONARITY_TOL = 0.06


class CellConfig:
    def __init__(self, workload="DECODE_PRIMARY", concurrency=1, repetition=1,
                 min_requests_factor=4, window_floor_s=120.0, wall_cap_s=common.CELL_WALL_CAP_S,
                 warmup_wall_cap_s=600.0, gate_timeout_s=180.0, abort_tpot_ms=None,
                 warmup_lifetimes=2, min_periods=4):
        self.workload = workload
        self.concurrency = concurrency
        self.repetition = repetition
        self.min_requests_factor = min_requests_factor
        self.window_floor_s = window_floor_s
        self.wall_cap_s = wall_cap_s
        self.warmup_wall_cap_s = warmup_wall_cap_s
        self.gate_timeout_s = gate_timeout_s
        self.abort_tpot_ms = abort_tpot_ms
        self.warmup_lifetimes = warmup_lifetimes
        self.min_periods = min_periods


class Cell:
    def __init__(self, server, prompts, cell_cfg, tag=""):
        self.srv = server
        self.prompts = prompts
        self.cc = cell_cfg
        self.tag = tag
        self.wl = common.WORKLOADS[cell_cfg.workload]
        self.n_out = self.wl["output_tokens"]
        self.n_in = self.wl["input_tokens"]

        self.records = []
        self.samples = []
        self.errors = []
        self.streamed_tokens = 0
        self.prompt_cursor = 0
        self.prompt_sequence = []
        self.phase = "warmup_a"
        self.t0 = None
        self.t_measure_start = None
        self.t_measure_end = None
        self.tokens_at_measure_start = None
        self.gate = {}
        self.status = STATUS_OK
        self.stop = asyncio.Event()

    def next_prompt(self):
        i = self.prompt_cursor
        self.prompt_cursor += 1
        idx = i % len(self.prompts)
        self.prompt_sequence.append(idx)
        return self.prompts[idx]

    async def one_request(self, session):
        p = self.next_prompt()
        body = {
            "model": "pilot",
            "prompt": p["token_ids"],
            "max_tokens": self.n_out,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        t_start = time.perf_counter()
        rec = {"prompt_index": p["index"], "prefix_hash": p["prefix_hash"],
               "t_start": t_start, "phase": self.phase}
        chunk_times = []
        usage = None
        finish_reason = None
        try:
            async with session.post("/v1/completions", json=body) as resp:
                if resp.status != 200:
                    rec.update({"error": f"http_{resp.status}", "body": (await resp.text())[:200]})
                    self.errors.append(rec)
                    return rec
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    ch = obj.get("choices") or []
                    if not ch:
                        continue
                    chunk_times.append(time.perf_counter())
                    self.streamed_tokens += 1
                    if ch[0].get("finish_reason"):
                        finish_reason = ch[0]["finish_reason"]
        except Exception as exc:
            rec.update({"error": f"exception:{type(exc).__name__}", "body": str(exc)[:200]})
            self.errors.append(rec)
            return rec

        t_end = time.perf_counter()
        n_usage = (usage or {}).get("completion_tokens")
        rec.update({
            "t_end": t_end,
            "e2e_s": t_end - t_start,
            "ttft_s": (chunk_times[0] - t_start) if chunk_times else None,
            "n_chunks": len(chunk_times),
            "completion_tokens": n_usage,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "finish_reason": finish_reason,
        })
        if len(chunk_times) >= 2:
            itls = [chunk_times[i] - chunk_times[i - 1] for i in range(1, len(chunk_times))]
            rec["itl_s"] = itls
            rec["tpot_s"] = (t_end - rec["ttft_s"] - t_start) / (len(chunk_times) - 1)
        self.records.append(rec)
        return rec

    async def slot(self, session):
        while not self.stop.is_set():
            await self.one_request(session)

    async def telemetry_loop(self):
        loop = asyncio.get_running_loop()
        while not self.stop.is_set():
            t = time.perf_counter()
            gpu = await loop.run_in_executor(None, common.gpu_telemetry)
            try:
                eng = await loop.run_in_executor(None, self.srv.metrics)
            except Exception as exc:
                eng = {"error": str(exc)[:120]}
            self.samples.append({
                "t": t, "phase": self.phase, "streamed_tokens": self.streamed_tokens,
                "gpu": gpu, "engine": eng,
            })
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=common.TELEMETRY_PERIOD_S)
            except asyncio.TimeoutError:
                pass

    def _window_samples(self):
        if self.t_measure_start is None:
            return []
        end = self.t_measure_end if self.t_measure_end else time.perf_counter()
        return [s for s in self.samples if self.t_measure_start <= s["t"] <= end]

    def _throughput_series(self, phases=("gate", "warmup_b")):
        s = [x for x in self.samples if x["phase"] in phases]
        out = []
        for a, b in zip(s, s[1:]):
            dt = b["t"] - a["t"]
            if dt > 0:
                out.append((b["streamed_tokens"] - a["streamed_tokens"]) / dt)
        return out

    def _throughput_windows(self, n=3):
        series = self._throughput_series()
        return series[-n:] if len(series) >= n else None

    def _half_window(self):
        """Must span at least one oscillation period, which varies with rate and concurrency."""
        per = self.period_estimate_s()
        if not per:
            return STATIONARITY_HALF_WINDOW
        need = int(per / common.TELEMETRY_PERIOD_S) + 1
        return max(STATIONARITY_HALF_WINDOW, min(STATIONARITY_HALF_WINDOW_MAX, need))

    def _stationary(self, k=None, tol=STATIONARITY_TOL):
        """Stationary, not flat.

        Phase-aligned slots make this workload periodic: every sequence starts together and
        takes the same number of steps, so KV footprint and therefore decode-attention traffic
        ramp in lockstep. Throughput oscillates with a period of one sequence lifetime around a
        flat mean. A flatness test over a fraction of that period can never pass, so compare the
        mean of the last k intervals against the mean of the k before them, with k chosen to span
        at least one period.
        """
        k = k or self._half_window()
        series = self._throughput_series()
        if len(series) < 2 * k:
            return False, {"intervals_available": len(series), "intervals_needed": 2 * k}
        recent, prior = series[-k:], series[-2 * k:-k]
        mr, mp = common.mean(recent), common.mean(prior)
        if not mr or not mp:
            return False, None
        drift = abs(mr - mp) / mp
        amp = ((max(series[-2 * k:]) - min(series[-2 * k:])) / mr) if mr else None
        return drift <= tol, {
            "mean_recent_tok_s": round(mr, 1), "mean_prior_tok_s": round(mp, 1),
            "drift_rel": round(drift, 4), "tolerance": tol,
            "half_window_samples": k,
            "period_estimate_s": round(self.period_estimate_s() or 0, 1),
            "oscillation_amplitude_rel": round(amp, 4) if amp else None,
        }

    def period_estimate_s(self):
        """One sequence lifetime at the observed rate; the oscillation period."""
        series = self._throughput_series(("gate", "warmup_b", "measure"))
        tput = common.mean(series[-6:]) if len(series) >= 3 else None
        if not tput:
            return None
        return self.n_out * self.cc.concurrency / tput

    def _kv_gate(self):
        s = [x for x in self.samples if x["phase"] in ("gate", "warmup_b")]
        vals = [x["engine"].get("kv_cache_usage") for x in s
                if isinstance(x["engine"], dict) and x["engine"].get("kv_cache_usage") is not None]
        if len(vals) < 2:
            return False, None
        a, b = vals[-2], vals[-1]
        return (b >= a) or (b >= 0.99), (a, b)

    def running_tpot_p95_ms(self):
        vals = [r["tpot_s"] * 1e3 for r in self.records
                if r.get("tpot_s") and self.t_measure_start and r["t_start"] >= self.t_measure_start]
        if len(vals) < 8:
            return None
        return common.quantiles(vals, (0.95,))["p95"]

    def in_window_records(self):
        if self.t_measure_start is None:
            return []
        end = self.t_measure_end or time.perf_counter()
        return [r for r in self.records
                if r.get("t_end") and r["t_start"] >= self.t_measure_start and r["t_end"] <= end]

    async def monitor(self):
        target_a = self.cc.concurrency * self.n_out
        target_b = self.cc.warmup_lifetimes * self.cc.concurrency * self.n_out
        t_gate_entry = None
        while not self.stop.is_set():
            await asyncio.sleep(0.5)
            now = time.perf_counter()
            if now - self.t0 > self.cc.wall_cap_s:
                self.status = "CELL_TIMEOUT"
                self.stop.set()
                return
            if not self.srv.alive():
                self.status = "SERVER_DIED"
                self.stop.set()
                return

            if self.phase == "warmup_a":
                if self.streamed_tokens >= target_a or now - self.t0 > self.cc.warmup_wall_cap_s:
                    self.phase = "warmup_b"
            elif self.phase == "warmup_b":
                if self.streamed_tokens >= target_b or now - self.t0 > self.cc.warmup_wall_cap_s:
                    self.gate["warmup_truncated"] = self.streamed_tokens < target_b
                    self.gate["warmup_tokens_target"] = target_b
                    self.gate["warmup_tokens_actual"] = self.streamed_tokens
                    self.gate["warmup_seconds"] = round(now - self.t0, 2)
                    self.phase = "gate"
                    t_gate_entry = now
            elif self.phase == "gate":
                kv_ok, kv_pair = self._kv_gate()
                stat_ok, stat = self._stationary()
                tw = self._throughput_windows()
                flat_ok = None
                if tw:
                    m = common.mean(tw)
                    flat_ok = bool(m > 0 and all(abs(v - m) / m <= 0.05 for v in tw))
                if kv_ok and stat_ok:
                    self.gate.update({
                        "fired": True,
                        "seconds_in_gate": round(now - t_gate_entry, 2),
                        "kv_usage_pair": kv_pair,
                        "stationarity": stat,
                        "flatness_diagnostic_only": flat_ok,
                        "throughput_windows_tok_s": [round(v, 1) for v in tw] if tw else None,
                    })
                    self.phase = "measure"
                    self.t_measure_start = now
                    self.tokens_at_measure_start = self.streamed_tokens
                elif now - t_gate_entry > self.cc.gate_timeout_s:
                    self.gate.update({"fired": False, "kv_ok": kv_ok, "stationary": stat_ok,
                                      "stationarity": stat,
                                      "flatness_diagnostic_only": flat_ok,
                                      "seconds_in_gate": round(now - t_gate_entry, 2),
                                      "throughput_windows_tok_s": tw})
                    self.status = "STEADY_STATE_NOT_REACHED"
                    self.stop.set()
                    return
            elif self.phase == "measure":
                if self.cc.abort_tpot_ms:
                    p95 = self.running_tpot_p95_ms()
                    if p95 and p95 > self.cc.abort_tpot_ms:
                        self.t_measure_end = now
                        self.status = "SLO_VIOLATED"
                        self.stop.set()
                        return
                elapsed = now - self.t_measure_start
                done = len(self.in_window_records())
                # whole-period averaging: the sawtooth biases a window that ends mid-period
                per = self.period_estimate_s()
                periods_ok = (per is None) or (elapsed >= self.cc.min_periods * per)
                if elapsed >= self.cc.window_floor_s and periods_ok and \
                        done >= self.cc.min_requests_factor * self.cc.concurrency:
                    self.t_measure_end = now
                    self.gate["period_estimate_s"] = round(per, 1) if per else None
                    self.gate["periods_in_window"] = round(elapsed / per, 2) if per else None
                    self.stop.set()
                    return

    async def run(self):
        self.t0 = time.perf_counter()
        m0 = self.srv.metrics()
        conn = aiohttp.TCPConnector(limit=self.cc.concurrency + 8)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=300)
        async with aiohttp.ClientSession(self.srv.base, connector=conn, timeout=timeout) as session:
            tasks = [asyncio.create_task(self.slot(session)) for _ in range(self.cc.concurrency)]
            tasks.append(asyncio.create_task(self.telemetry_loop()))
            mon = asyncio.create_task(self.monitor())
            await mon
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        m1 = self.srv.metrics()
        return self.summarise(m0, m1)

    def summarise(self, m0, m1):
        wr = self.in_window_records()
        ws = self._window_samples()
        dur = ((self.t_measure_end or time.perf_counter()) - self.t_measure_start) \
            if self.t_measure_start else None

        bad_len = [r["completion_tokens"] for r in wr if r.get("completion_tokens") != self.n_out]
        bad_in = [r["prompt_tokens"] for r in wr if r.get("prompt_tokens") != self.n_in]
        bad_finish = [r.get("finish_reason") for r in wr if r.get("finish_reason") != "length"]
        bad_chunks = [r["prompt_index"] for r in wr if r.get("n_chunks") != r.get("completion_tokens")]

        eng0 = ws[0]["engine"] if ws and isinstance(ws[0]["engine"], dict) else None
        eng1 = ws[-1]["engine"] if ws and isinstance(ws[-1]["engine"], dict) else None
        # engine counters are read at telemetry samples, which sit strictly inside the window;
        # comparing their delta against a full-window client count would compare unequal spans
        engine_span = (ws[-1]["t"] - ws[0]["t"]) if len(ws) >= 2 else None

        def delta(k):
            if not eng0 or not eng1 or eng0.get(k) is None or eng1.get(k) is None:
                return None
            return eng1[k] - eng0[k]

        kv = [s["engine"].get("kv_cache_usage") for s in ws
              if isinstance(s["engine"], dict) and s["engine"].get("kv_cache_usage") is not None]
        waiting = [s["engine"].get("num_waiting_reqs") for s in ws
                   if isinstance(s["engine"], dict) and s["engine"].get("num_waiting_reqs") is not None]
        running = [s["engine"].get("num_running_reqs") for s in ws
                   if isinstance(s["engine"], dict) and s["engine"].get("num_running_reqs") is not None]

        preempt_series = []
        for a, b in zip(ws, ws[1:]):
            if isinstance(a["engine"], dict) and isinstance(b["engine"], dict) \
                    and a["engine"].get("num_preemptions") is not None \
                    and b["engine"].get("num_preemptions") is not None:
                preempt_series.append(b["engine"]["num_preemptions"] - a["engine"]["num_preemptions"])

        tokens_window = (self.streamed_tokens - self.tokens_at_measure_start) \
            if self.tokens_at_measure_start is not None else None

        # a cell that never opened its window still carries pressure evidence; without this an
        # aborted wall-search point would read as "clean" and invert the verdict
        def whole(k):
            if m0.get(k) is None or m1.get(k) is None:
                return None
            return m1[k] - m0[k]

        all_kv = [x["engine"].get("kv_cache_usage") for x in self.samples
                  if isinstance(x["engine"], dict) and x["engine"].get("kv_cache_usage") is not None]
        all_preempt = []
        for a, b in zip(self.samples, self.samples[1:]):
            if isinstance(a["engine"], dict) and isinstance(b["engine"], dict) \
                    and a["engine"].get("num_preemptions") is not None \
                    and b["engine"].get("num_preemptions") is not None:
                all_preempt.append(b["engine"]["num_preemptions"] - a["engine"]["num_preemptions"])

        ttft = [r["ttft_s"] * 1e3 for r in wr if r.get("ttft_s") is not None]
        tpot = [r["tpot_s"] * 1e3 for r in wr if r.get("tpot_s") is not None]
        e2e = [r["e2e_s"] for r in wr if r.get("e2e_s") is not None]
        itl = [v * 1e3 for r in wr for v in r.get("itl_s", [])]

        def agg(key, fn):
            vals = [s["gpu"].get(key) for s in ws if isinstance(s["gpu"], dict)
                    and s["gpu"].get(key) is not None]
            return round(fn(vals), 2) if vals else None

        pcq, pch = delta("prefix_cache_queries"), delta("prefix_cache_hits")

        status = self.status
        invalid_reasons = []
        if bad_len:
            invalid_reasons.append(f"output_length_violation:{len(bad_len)}")
        if bad_in:
            invalid_reasons.append(f"input_length_violation:{len(bad_in)}")
        if bad_finish:
            invalid_reasons.append(f"finish_reason_not_length:{len(bad_finish)}")
        if bad_chunks:
            invalid_reasons.append(f"chunk_token_count_mismatch:{len(bad_chunks)}")
        if self.errors:
            invalid_reasons.append(f"request_failures:{len(self.errors)}")
        if pch:
            invalid_reasons.append(f"prefix_cache_hits:{pch}")
        if not ws:
            invalid_reasons.append("missing_telemetry")
        if status == STATUS_OK and invalid_reasons:
            status = "INVALID"

        return {
            "tag": self.tag,
            "workload": self.cc.workload,
            "concurrency": self.cc.concurrency,
            "repetition": self.cc.repetition,
            "status": status,
            "valid_result": status in (STATUS_OK, "SLO_VIOLATED"),
            "invalid_reasons": invalid_reasons,
            "input_tokens": self.n_in,
            "output_tokens_requested": self.n_out,

            "window_seconds": round(dur, 2) if dur else None,
            "window_completed_requests": len(wr),
            "window_streamed_tokens": tokens_window,
            "output_tokens_per_s": round(tokens_window / dur, 2) if (tokens_window and dur) else None,
            "output_tokens_per_s_completed_only": round(
                sum(r.get("completion_tokens") or 0 for r in wr) / dur, 2) if dur else None,
            "request_throughput_per_s": round(len(wr) / dur, 4) if dur else None,
            "engine_generation_tokens_delta": delta("generation_tokens"),
            "engine_sample_span_s": round(engine_span, 2) if engine_span else None,
            "engine_generation_tokens_per_s": (
                round(delta("generation_tokens") / engine_span, 2)
                if engine_span and delta("generation_tokens") is not None else None),
            # engine deltas span the first->last telemetry sample inside the window, so the client
            # side must be read at those same samples or the comparison loses up to one cadence
            # at each edge and looks like a counting error
            "client_tokens_between_window_samples": (
                (ws[-1]["streamed_tokens"] - ws[0]["streamed_tokens"]) if len(ws) >= 2 else None),
            "window_sample_span_s": (round(ws[-1]["t"] - ws[0]["t"], 2) if len(ws) >= 2 else None),

            "ttft_ms_p50": common.quantiles(ttft)["p50"], "ttft_ms_p95": common.quantiles(ttft)["p95"],
            "tpot_ms_p50": common.quantiles(tpot)["p50"], "tpot_ms_p95": common.quantiles(tpot)["p95"],
            "tpot_ms_mean": common.mean(tpot),
            "itl_ms_p50": common.quantiles(itl)["p50"], "itl_ms_p95": common.quantiles(itl)["p95"],
            "e2e_s_p50": common.quantiles(e2e)["p50"], "e2e_s_p95": common.quantiles(e2e)["p95"],
            "itl_sample_count": len(itl),
            "chunk_token_agreement": not bad_chunks,
            "meets_slo": (common.quantiles(tpot)["p95"] <= common.SLO_TPOT_MS) if tpot else None,

            "kv_cache_usage_p50": common.quantiles(kv)["p50"] if kv else None,
            "kv_cache_usage_p95": common.quantiles(kv)["p95"] if kv else None,
            "kv_cache_usage_max": max(kv) if kv else None,
            "num_preemptions_delta": delta("num_preemptions"),
            "recomputed_tokens_delta": delta("recomputed_tokens"),
            "preemption_nonzero_samples": sum(1 for v in preempt_series if v > 0),
            "preemption_sample_count": len(preempt_series),
            "num_waiting_reqs_mean": round(common.mean(waiting), 2) if waiting else None,
            "num_waiting_reqs_max": max(waiting) if waiting else None,
            "num_running_reqs_mean": round(common.mean(running), 2) if running else None,
            "prefix_cache_queries_delta": pcq,
            "prefix_cache_hits_delta": pch,
            "prompt_tokens_cached_delta": delta("prompt_tokens_cached"),

            "gpu_util_mean_pct": agg("utilization.gpu", common.mean),
            "gpu_mem_used_mib_max": agg("memory.used", max),
            "gpu_power_w_mean": agg("power.draw", common.mean),
            "sm_clock_mhz_mean": agg("clocks.sm", common.mean),
            "mem_clock_mhz_mean": agg("clocks.mem", common.mean),
            "gpu_temp_c_max": agg("temperature.gpu", max),
            "pcie_gen_mode": agg("pcie.link.gen.current", max),

            "cell_num_preemptions_delta": whole("num_preemptions"),
            "cell_recomputed_tokens_delta": whole("recomputed_tokens"),
            "cell_prefix_cache_hits_delta": whole("prefix_cache_hits"),
            "cell_kv_cache_usage_max": max(all_kv) if all_kv else None,
            "cell_preemption_nonzero_samples": sum(1 for v in all_preempt if v > 0),
            "cell_preemption_sample_count": len(all_preempt),
            "telemetry_sample_count": len(self.samples),

            "period_estimate_s": self.gate.get("period_estimate_s"),
            "periods_in_window": self.gate.get("periods_in_window"),
            "oscillation_amplitude_rel": (self.gate.get("stationarity") or {}).get(
                "oscillation_amplitude_rel"),

            "gate": self.gate,
            "errors": self.errors[:10],
            "error_count": len(self.errors),
            "total_requests_issued": self.prompt_cursor,
            "prompt_indices_in_window": sorted({r["prompt_index"] for r in wr}),
            "cell_wall_seconds": round((self.t_measure_end or time.perf_counter()) - self.t0, 2),
            "metrics_at_cell_start": m0,
            "metrics_at_cell_end": m1,
        }


def run_cell(server, prompts, cell_cfg, tag=""):
    cell = Cell(server, prompts, cell_cfg, tag)
    return asyncio.run(cell.run())
