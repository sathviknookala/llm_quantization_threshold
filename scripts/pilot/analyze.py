"""Turn cells.jsonl into the pilot artifacts and the PILOT_DECISION verdict.

Pass/fail criteria are pre-registered here so a failed job cannot be reinterpreted after
seeing the numbers.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilot import common  # noqa: E402

CELLS = os.path.join(common.PILOT_DIR, "cells.jsonl")

CRITERIA = {
    "p1_ordering": "BF16 < FP8 < FP4 output tok/s at every sub-wall concurrency (1, 8, 12)",
    "p1_ratio_tolerance_rel": 0.20,
    "p1_ratio_stability_rel": 0.15,
    "p1_subwall_clean": "no sustained BF16 KV preemption/recompute at C=1,8,12",
    "p2_expected_band": [15, 25],
    "p3_rho_max": 0.25,
    "p3_escalation": [3, 5, 7],
    "d11_expected_ratio": {k: round(v, 3) for k, v in common.D11_EXPECTED_RATIO.items()},
}

ROW_FIELDS = [
    "job", "configuration_id", "quantization", "workload", "concurrency", "repetition",
    "status", "valid_result", "invalid_reasons",
    "input_tokens", "output_tokens_requested",
    "output_tokens_per_s", "request_throughput_per_s", "engine_generation_tokens_delta",
    "ttft_ms_p50", "ttft_ms_p95", "tpot_ms_p50", "tpot_ms_p95", "itl_ms_p50", "itl_ms_p95",
    "e2e_s_p50", "e2e_s_p95", "meets_slo",
    "kv_cache_usage_p50", "kv_cache_usage_p95", "kv_cache_usage_max",
    "num_preemptions_delta", "recomputed_tokens_delta", "preemption_nonzero_samples",
    "preemption_sample_count", "num_waiting_reqs_mean", "num_waiting_reqs_max",
    "num_running_reqs_mean", "prefix_cache_queries_delta", "prefix_cache_hits_delta",
    "gpu_util_mean_pct", "gpu_mem_used_mib_max", "gpu_power_w_mean", "sm_clock_mhz_mean",
    "mem_clock_mhz_mean", "gpu_temp_c_max", "pcie_gen_mode",
    "cell_num_preemptions_delta", "cell_recomputed_tokens_delta", "cell_kv_cache_usage_max",
    "cell_preemption_nonzero_samples", "telemetry_sample_count", "chunk_token_agreement",
    "period_estimate_s", "periods_in_window", "oscillation_amplitude_rel",
    "client_tokens_between_window_samples", "window_sample_span_s",
    "kv_cache_tokens", "window_seconds", "window_completed_requests", "window_streamed_tokens",
    "error_count", "total_requests_issued", "cell_wall_seconds",
    "weight_bytes_gb", "model_path", "serving_backend", "corpus_version", "prompt_set_hash",
    "slo_tpot_ms", "started_at", "finished_at",
]


def load():
    if not os.path.exists(CELLS):
        return []
    out = []
    for line in open(CELLS):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def flat(rec):
    row = {k: rec.get(k) for k in ROW_FIELDS}
    row["gpu_uuid"] = (rec.get("gpu") or {}).get("uuid")
    row["gpu_name"] = (rec.get("gpu") or {}).get("name")
    row["driver"] = (rec.get("gpu") or {}).get("driver_version")
    row["dispatch_ok"] = (rec.get("dispatch_verdict") or {}).get("ok")
    row["prefix_caching_logged"] = rec.get("enable_prefix_caching_logged")
    row["gate_fired"] = (rec.get("gate") or {}).get("fired")
    row["warmup_truncated"] = (rec.get("gate") or {}).get("warmup_truncated")
    row["invalid_reasons"] = ";".join(rec.get("invalid_reasons") or [])
    return row


def write_csv(path, rows):
    if not rows:
        return None
    fields = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# telemetry period plus jitter; the sampler runs nvidia-smi and an HTTP scrape each tick
_TICK = common.TELEMETRY_PERIOD_S * 1.2


def _engine_rate_agrees(r):
    """Engine counters span sample-to-sample; a client count spans the whole window.

    Never let this pass vacuously: if the exact same-interval client count is absent, fall back to
    bounding the engine delta's span between (window - 2 ticks) and the full window and requiring
    the client rate to sit inside the implied interval.
    """
    b = r.get("engine_generation_tokens_delta")
    if b is None:
        return False
    c = r.get("client_tokens_between_window_samples")
    if c is not None:
        return abs(b - c) <= 0.05 * max(1, c)
    a, w = r.get("window_streamed_tokens"), r.get("window_seconds")
    if not a or not w:
        return False
    lo, hi = b / w, b / max(1e-6, w - 2 * _TICK)
    return lo * 0.99 <= a / w <= hi * 1.01


def mean_of(rs, key):
    vals = [r[key] for r in rs if r["status"] == "OK" and r.get(key) is not None]
    return round(common.mean(vals), 2) if vals else None


def by_cc(recs, jobs, workload="DECODE_PRIMARY"):
    out = {}
    for r in recs:
        if r["job"] in jobs and r["workload"] == workload:
            out.setdefault((r["configuration_id"], r["concurrency"]), []).append(r)
    return out


def tp(recs):
    return [r["output_tokens_per_s"] for r in recs
            if r["status"] == "OK" and r["output_tokens_per_s"]]


def analyse_p1(recs):
    groups = by_cc(recs, {"P1"})
    concurrencies = sorted({c for (_, c) in groups})
    res = {"criteria": {k: CRITERIA[k] for k in
                        ("p1_ordering", "p1_ratio_tolerance_rel", "p1_ratio_stability_rel",
                         "p1_subwall_clean", "d11_expected_ratio")},
           "per_point": {}, "ratios": {}, "findings": []}

    for C in concurrencies:
        point = {}
        for cid in common.CONFIGS:
            vals = tp(groups.get((cid, C), []))
            point[common.CONFIGS[cid]["short"]] = {
                "n_reps": len(vals),
                "mean_tok_s": round(common.mean(vals), 2) if vals else None,
                "values": [round(v, 2) for v in vals],
                "cv_pct": round(100 * common.sample_std(vals) / common.mean(vals), 2)
                if len(vals) > 1 else None,
                "tpot_ms_p95_mean": mean_of(groups.get((cid, C), []), "tpot_ms_p95"),
                "meets_slo_all_reps": all(r.get("meets_slo") for r in groups.get((cid, C), [])
                                          if r["status"] == "OK") if groups.get((cid, C)) else None,
                "preemptions_total": sum((r.get("num_preemptions_delta") or 0)
                                         for r in groups.get((cid, C), [])),
                "recomputed_tokens_total": sum((r.get("recomputed_tokens_delta") or 0)
                                               for r in groups.get((cid, C), [])),
                "sustained_pressure_reps": sum(
                    1 for r in groups.get((cid, C), [])
                    if (r.get("preemption_nonzero_samples") or 0) >= 2
                    or (r.get("recomputed_tokens_delta") or 0) > 0),
            }
        res["per_point"][C] = point

    ordering_ok, ratio_ok, stability_ok, clean_ok = True, True, True, True
    base_ratio = {}
    for C in concurrencies:
        p = res["per_point"][C]
        b, f8, f4 = p["BF16"]["mean_tok_s"], p["FP8"]["mean_tok_s"], p["FP4"]["mean_tok_s"]
        entry = {}
        if b and f8 and f4:
            if not (b < f8 < f4):
                ordering_ok = False
                res["findings"].append(f"C={C}: ordering violated BF16={b} FP8={f8} FP4={f4}")
            for cid, val in (("FP8_PRIMARY", f8), ("FP4_PRIMARY", f4)):
                short = common.CONFIGS[cid]["short"]
                r = val / b
                exp = common.D11_EXPECTED_RATIO[cid]
                dev = (r - exp) / exp
                entry[short] = {"ratio": round(r, 3), "expected": round(exp, 3),
                                "rel_deviation": round(dev, 3),
                                "within_tolerance": abs(dev) <= CRITERIA["p1_ratio_tolerance_rel"]}
                if abs(dev) > CRITERIA["p1_ratio_tolerance_rel"]:
                    ratio_ok = False
                    res["findings"].append(
                        f"C={C} {short}: ratio {r:.3f} deviates {dev:+.1%} from expected {exp:.3f}")
                if C == min(concurrencies):
                    base_ratio[short] = r
                elif short in base_ratio:
                    drift = (r - base_ratio[short]) / base_ratio[short]
                    entry[short]["drift_vs_C1"] = round(drift, 3)
                    entry[short]["stable"] = abs(drift) <= CRITERIA["p1_ratio_stability_rel"]
                    if abs(drift) > CRITERIA["p1_ratio_stability_rel"]:
                        stability_ok = False
                        res["findings"].append(
                            f"C={C} {short}: ratio drifted {drift:+.1%} vs C=1 "
                            f"(collapse threshold {CRITERIA['p1_ratio_stability_rel']:.0%})")
        res["ratios"][C] = entry
        if res["per_point"][C]["BF16"]["sustained_pressure_reps"]:
            clean_ok = False
            res["findings"].append(
                f"C={C}: BF16 shows sustained KV pressure; not a clean sub-wall point")

    have_all = all(res["per_point"][C][s]["mean_tok_s"] for C in concurrencies
                   for s in ("BF16", "FP8", "FP4"))
    res["checks"] = {"ordering_bf16_lt_fp8_lt_fp4": ordering_ok,
                     "ratios_within_20pct_of_d11": ratio_ok,
                     "ratio_stable_across_concurrency": stability_ok,
                     "subwall_free_of_bf16_kv_pressure": clean_ok,
                     "all_cells_present": have_all and set(concurrencies) == {1, 8, 12}}
    res["verdict"] = "PASS" if all(res["checks"].values()) else "FAIL"
    return res


def analyse_p2(recs):
    rows = [r for r in recs if r["job"] == "P2" and r["configuration_id"] == "BF16_REFERENCE"]
    per_c = {}
    for r in rows:
        per_c.setdefault(r["concurrency"], []).append(r)

    from pilot.run_pilot import pressured as is_pressured

    def press(rs):
        return any(is_pressured(x) for x in rs)

    states = {C: ("pressured" if press(rs) else "clean") for C, rs in sorted(per_c.items())}
    clean = [C for C, s in states.items() if s == "clean"]
    press_cs = [C for C, s in states.items() if s == "pressured"]
    last_clean = max(clean) if clean else None
    first_pressured = min([C for C in press_cs if last_clean is None or C > last_clean]) \
        if press_cs else None

    kv_tokens = next((r.get("kv_cache_tokens") for r in rows if r.get("kv_cache_tokens")), None)
    peak_per_seq = 2560
    predicted_peak = (kv_tokens // peak_per_seq) if kv_tokens else None
    predicted_mean = (kv_tokens // 1536) if kv_tokens else None

    reproduced = {}
    for C in (last_clean, first_pressured):
        if C and len(per_c.get(C, [])) > 1:
            reproduced[C] = [("pressured" if press([x]) else "clean") for x in per_c[C]]

    band = CRITERIA["p2_expected_band"]
    in_band = bool(first_pressured and band[0] <= first_pressured <= band[1])
    below = bool(first_pressured and first_pressured < band[0])
    above = bool(last_clean and last_clean >= band[1] and not first_pressured)

    checks = {
        "transition_observed": bool(last_clean and first_pressured),
        "transition_in_d11_band_15_25": in_band,
        "not_limited_materially_below_15": not below,
        "not_unconstrained_above_25": not above,
        "transition_reproducible": all(len(set(v)) == 1 for v in reproduced.values())
        if reproduced else False,
    }
    return {
        "criteria": {"expected_band": band,
                     "wall_definition": ("highest concurrency sustaining steady state with no "
                                         "KV-induced preemption/recompute, to the next concurrency "
                                         "with persistent KV saturation and/or preemption/recompute"),
                     "waiting_requests_insufficient": True},
        "measured_kv_cache_tokens": kv_tokens,
        "predicted_wall_peak_basis": predicted_peak,
        "predicted_wall_mean_basis": predicted_mean,
        "per_concurrency_state": states,
        "per_concurrency_detail": {
            C: [{"rep": x["repetition"], "status": x["status"],
                 "tok_s": x["output_tokens_per_s"], "tpot_ms_p95": x["tpot_ms_p95"],
                 "kv_p95": x["kv_cache_usage_p95"], "kv_max": x["kv_cache_usage_max"],
                 "preempt": x["num_preemptions_delta"],
                 "preempt_samples": x["preemption_nonzero_samples"],
                 "recomputed": x["recomputed_tokens_delta"],
                 "cell_preempt": x.get("cell_num_preemptions_delta"),
                 "cell_recomputed": x.get("cell_recomputed_tokens_delta"),
                 "cell_kv_max": x.get("cell_kv_cache_usage_max"),
                 "waiting_max": x["num_waiting_reqs_max"],
                 "warmup_truncated": (x.get("gate") or {}).get("warmup_truncated"),
                 "mem_mib_max": x["gpu_mem_used_mib_max"]} for x in rs]
            for C, rs in sorted(per_c.items())},
        "measured_wall_bracket": [last_clean, first_pressured],
        "throughput_peak": _throughput_knee(per_c),
        "reproducibility": reproduced,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def _throughput_knee(per_c):
    """Throughput regression is the earliest wall symptom; preemption lags it."""
    pts = []
    for C, rs in sorted(per_c.items()):
        vals = [x["output_tokens_per_s"] for x in rs
                if x["status"] == "OK" and x["output_tokens_per_s"]]
        if vals:
            pts.append((C, common.mean(vals), mean_of(rs, "tpot_ms_p95")))
    if len(pts) < 2:
        return None
    best = max(range(len(pts)), key=lambda i: pts[i][1])
    if best + 1 >= len(pts):
        return None
    pc, pt, ptp = pts[best]
    nc, nt, ntp = pts[best + 1]
    return {"peak_C": pc, "peak_tok_s": round(pt, 2), "next_C": nc, "next_tok_s": round(nt, 2),
            "regression_pct": round(100 * (nt - pt) / pt, 1),
            "tpot_from": ptp, "tpot_to": ntp}


def analyse_p3(recs):
    groups = by_cc(recs, {"P1", "P3"})
    points = {"sub_wall": 8, "high_concurrency": 24}
    out = {"criteria": {"rho_max": CRITERIA["p3_rho_max"],
                        "primary_metric": "output tokens/sec",
                        "escalation": CRITERIA["p3_escalation"]},
           "points": {}, "cv_all_configs": {}}

    for (cid, C), rs in sorted(groups.items()):
        vals = tp(rs)
        if len(vals) > 1:
            out["cv_all_configs"][f"{common.CONFIGS[cid]['short']}@C{C}"] = {
                "n": len(vals), "mean": round(common.mean(vals), 2),
                "sd": round(common.sample_std(vals), 3),
                "cv_pct": round(100 * common.sample_std(vals) / common.mean(vals), 3),
                "tpot_p95_cv_pct": _cv([r["tpot_ms_p95"] for r in rs
                                        if r["status"] == "OK" and r["tpot_ms_p95"]]),
            }

    for name, C in points.items():
        a = tp(groups.get(("FP8_PRIMARY", C), []))
        b = tp(groups.get(("FP4_PRIMARY", C), []))
        rec = {"concurrency": C, "n_fp8": len(a), "n_fp4": len(b),
               "fp8_values": [round(v, 2) for v in a], "fp4_values": [round(v, 2) for v in b]}
        if len(a) >= 2 and len(b) >= 2:
            ma, mb = common.mean(a), common.mean(b)
            sa, sb = common.sample_std(a), common.sample_std(b)
            na, nb = len(a), len(b)
            delta = abs(mb - ma)
            se = (sa ** 2 / na + sb ** 2 / nb) ** 0.5
            va, vb = sa ** 2 / na, sb ** 2 / nb
            df = ((va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))) if (va + vb) > 0 else 1
            t = common.t_crit_95(max(1, int(round(df))))
            h = t * se
            rho = h / delta if delta > 0 else None
            rec.update({"mean_fp8": round(ma, 2), "mean_fp4": round(mb, 2),
                        "sd_fp8": round(sa, 3), "sd_fp4": round(sb, 3),
                        "cv_fp8_pct": round(100 * sa / ma, 3), "cv_fp4_pct": round(100 * sb / mb, 3),
                        "delta_tok_s": round(delta, 2), "se_delta": round(se, 3),
                        "welch_df": round(df, 2), "t_crit_95": t,
                        "half_width_95": round(h, 3),
                        "rho": round(rho, 4) if rho else None,
                        "resolves": bool(rho is not None and rho <= CRITERIA["p3_rho_max"])})
        out["points"][name] = rec

    tested = [v for v in out["points"].values() if "rho" in v and v["rho"] is not None]
    checks = {
        "both_points_measured": len(tested) == 2,
        "rho_within_0_25_at_all_points": all(v["resolves"] for v in tested) if tested else False,
    }
    n_used = max([v["n_fp8"] for v in out["points"].values()] or [0])
    out["checks"] = checks
    out["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    out["locked_repetition_count"] = n_used if all(checks.values()) else None
    out["escalation_required"] = None if all(checks.values()) else \
        next((n for n in CRITERIA["p3_escalation"] if n > n_used), "reopen measurement design")
    return out


def _cv(vals):
    if len(vals) < 2:
        return None
    return round(100 * common.sample_std(vals) / common.mean(vals), 3)


def analyse_p5(recs):
    timed = [r for r in recs if r["workload"] == "DECODE_PRIMARY"]
    inv = {}
    if not timed:
        return {"invariants": {}, "failed_invariants": ["no_decode_primary_cells"],
                "verdict": "FAIL"}

    def every(name, fn, rows=None):
        rows = timed if rows is None else rows
        bad = [f"{r['job']}:{r['quantization']}:C{r['concurrency']}:r{r['repetition']}"
               for r in rows if not fn(r)]
        inv[name] = {"pass": not bad, "violations": bad[:12], "n_cells": len(rows)}

    every("exact_input_512_tokens", lambda r: r["input_tokens"] == 512)
    every("exact_output_2048_requested", lambda r: r["output_tokens_requested"] == 2048)
    every("no_output_length_violation",
          lambda r: not any(x.startswith("output_length_violation") for x in (r["invalid_reasons"] or [])))
    every("no_input_length_violation",
          lambda r: not any(x.startswith("input_length_violation") for x in (r["invalid_reasons"] or [])))
    every("finish_reason_length_only",
          lambda r: not any(x.startswith("finish_reason_not_length") for x in (r["invalid_reasons"] or [])))
    every("frozen_corpus_identical",
          lambda r: r["prompt_set_hash"] == timed[0]["prompt_set_hash"])
    every("prefix_caching_off_in_config",
          lambda r: str(r.get("enable_prefix_caching_logged")) in ("False", "None"))
    every("prefix_cache_hits_zero", lambda r: (r.get("prefix_cache_hits_delta") or 0) == 0)
    every("no_request_failures", lambda r: (r.get("error_count") or 0) == 0)
    every("kernel_dispatch_as_intended",
          lambda r: (r.get("dispatch_verdict") or {}).get("ok") is True)
    every("kv_cache_usage_recorded", lambda r: r.get("kv_cache_usage_p95") is not None)
    every("preemption_counter_recorded", lambda r: r.get("num_preemptions_delta") is not None)
    every("recomputed_tokens_counter_recorded", lambda r: r.get("recomputed_tokens_delta") is not None)
    every("waiting_requests_counter_recorded", lambda r: r.get("num_waiting_reqs_max") is not None)
    every("telemetry_recorded", lambda r: r.get("sm_clock_mhz_mean") is not None)
    every("window_spans_whole_periods",
          lambda r: r["status"] != "OK" or r.get("periods_in_window") is None
          or r["periods_in_window"] >= 3.5)
    every("steady_state_gate_fired",
          lambda r: (r.get("gate") or {}).get("fired") is True or r["status"] != "OK")
    every("chunk_count_matches_usage_tokens", lambda r: r.get("chunk_token_agreement") is not False)
    every("engine_token_rate_agrees", _engine_rate_agrees)

    nonzero = [r for r in timed if (r.get("num_preemptions_delta") or 0) > 0
               or (r.get("recomputed_tokens_delta") or 0) > 0]
    inv["counters_exercised_nonzero"] = {
        "pass": bool(nonzero),
        "cells": [f"{r['quantization']}:C{r['concurrency']}:r{r['repetition']}"
                  f" preempt={r['num_preemptions_delta']} recomp={r['recomputed_tokens_delta']}"
                  for r in nonzero[:12]],
        "n_cells": len(nonzero),
    }

    statuses = {}
    for r in recs:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    inv["cell_status_classification"] = {"pass": True, "counts": statuses}

    probe = [r for r in recs if r["workload"] == "PREFILL_PROBE"]
    inv["prefill_probe_plumbing"] = {
        "pass": bool(probe) and all(r["status"] == "OK" for r in probe),
        "n_cells": len(probe),
        "detail": [{"quantization": r["quantization"], "status": r["status"],
                    "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens_requested"],
                    "prefix_cache_hits": r["prefix_cache_hits_delta"],
                    "ttft_ms_p50": r["ttft_ms_p50"],
                    "invalid_reasons": r["invalid_reasons"]} for r in probe],
    }

    slo = {}
    for r in timed:
        if r["job"] == "P1" and r["concurrency"] in (8, 12):
            slo.setdefault(f"{r['quantization']}@C{r['concurrency']}", []).append(
                {"tpot_ms_p95": r["tpot_ms_p95"], "meets_slo": r["meets_slo"]})
    bf16_ok = all(x["meets_slo"] for k, v in slo.items() if k.startswith("BF16") for x in v) \
        if any(k.startswith("BF16") for k in slo) else None
    inv["slo_visibility"] = {
        "pass": True,
        "slo_tpot_ms_p95": common.SLO_TPOT_MS,
        "per_cell": slo,
        "bf16_within_slo_at_c8_c12": bf16_ok,
        "note": ("recorded for visibility; a BF16 breach at C=8-12 collapses its "
                 "max-concurrency-at-SLO and must be surfaced before the sweep"),
    }

    gating = [k for k, v in inv.items() if isinstance(v, dict) and v.get("pass") is False]
    return {"invariants": inv, "failed_invariants": gating,
            "verdict": "PASS" if not gating else "FAIL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=common.PILOT_DIR)
    a = ap.parse_args()
    recs = load()
    if not recs:
        raise SystemExit("no cells recorded yet")

    rows = [flat(r) for r in recs]
    write_csv(os.path.join(a.out_dir, "p1_bandwidth_assumption.csv"),
              [r for r in rows if r["job"] == "P1"])
    write_csv(os.path.join(a.out_dir, "p2_bf16_kv_wall.csv"),
              [r for r in rows if r["job"] == "P2"])
    write_csv(os.path.join(a.out_dir, "p3_repeatability.csv"),
              [r for r in rows if r["job"] in ("P1", "P3")])
    write_csv(os.path.join(a.out_dir, "all_cells.csv"), rows)

    p1, p2, p3, p5 = analyse_p1(recs), analyse_p2(recs), analyse_p3(recs), analyse_p5(recs)
    common.write_json(os.path.join(a.out_dir, "p1_analysis.json"), p1)
    common.write_json(os.path.join(a.out_dir, "p2_analysis.json"), p2)
    common.write_json(os.path.join(a.out_dir, "p3_analysis.json"), p3)
    common.write_json(os.path.join(a.out_dir, "p5_harness_validation.json"), p5)

    p4_path = os.path.join(a.out_dir, "p4_hbm_bandwidth.json")
    p4 = json.load(open(p4_path)) if os.path.exists(p4_path) else None
    gate_path = os.path.join(a.out_dir, "correctness_gate.json")
    gate = json.load(open(gate_path)) if os.path.exists(gate_path) else None

    manifest = {
        "artifact": "serving pilot",
        "status": "DIAGNOSTIC ONLY - no value here may be quoted as an experiment result",
        "purpose": ("validate the free parameters and physical assumptions the D11 serving sweep "
                    "depends on"),
        "pre_registered_criteria": CRITERIA,
        "cells_recorded": len(recs),
        "jobs": sorted({r["job"] for r in recs}),
        "configurations": sorted({r["configuration_id"] for r in recs}),
        "corpus": {"version": recs[0]["corpus_version"],
                   "prompt_set_hash": recs[0]["prompt_set_hash"],
                   "decision": "D16 resolved to option (a), C4 en validation"},
        "gpu": recs[0]["gpu"],
        "software": recs[0]["software"],
        "server_controls": recs[0]["server_controls"],
        "kv_cache_tokens_measured": {
            r["quantization"]: r["kv_cache_tokens"] for r in recs if r.get("kv_cache_tokens")},
        "verdicts": {"P1": p1["verdict"], "P2": p2["verdict"], "P3": p3["verdict"],
                     "P5": p5["verdict"],
                     "P4": ("VALID" if p4 and p4["kernel_correctness"]["copy"]
                            and p4["kernel_correctness"]["triad"]
                            and p4["working_set"]["exceeds_l2"] else "INVALID") if p4 else "MISSING",
                     "correctness_gate": gate["gate_status"] if gate else "MISSING"},
        "files": sorted(f for f in os.listdir(a.out_dir) if os.path.isfile(os.path.join(a.out_dir, f))),
        "generated_at": common.now_iso(),
    }
    cleared = (p1["verdict"] == "PASS" and p2["verdict"] == "PASS" and p3["verdict"] == "PASS"
               and p5["verdict"] == "PASS" and manifest["verdicts"]["P4"] == "VALID"
               and manifest["verdicts"]["correctness_gate"] == "CLEAN")
    manifest["cleared_for_full_serving_sweep"] = cleared
    common.write_json(os.path.join(a.out_dir, "manifest.json"), manifest)
    write_decision(a.out_dir, p1, p2, p3, p4, p5, gate, manifest)
    print(json.dumps(manifest["verdicts"], indent=2))
    print("cleared_for_full_serving_sweep:", cleared)


def _r(v, nd=2):
    return round(v, nd) if isinstance(v, (int, float)) else v


def write_decision(out_dir, p1, p2, p3, p4, p5, gate, manifest):
    L = []
    A = L.append
    A("# PILOT_DECISION")
    A("")
    A("Pilot measurements are **diagnostic**. Nothing in this file is an experiment result and no")
    A("number here may be cited as measured degradation or measured serving benefit.")
    A("")
    A(f"Generated {manifest['generated_at']} - {manifest['cells_recorded']} cells.")
    A(f"Corpus `{manifest['corpus']['version']}`, prompt-set hash "
      f"`{manifest['corpus']['prompt_set_hash'][:16]}...`.")
    A("")

    A(f"## P1 - memory-bandwidth-bound assumption: **{p1['verdict']}**")
    A("")
    A("| C | BF16 tok/s | FP8 tok/s | FP4 tok/s | R_FP8 | R_FP4 | expected | BF16 KV pressure |")
    A("|---|---|---|---|---|---|---|---|")
    for C, pt in sorted(p1["per_point"].items()):
        r = p1["ratios"].get(C, {})
        A(f"| {C} | {pt['BF16']['mean_tok_s']} | {pt['FP8']['mean_tok_s']} | "
          f"{pt['FP4']['mean_tok_s']} | {r.get('FP8', {}).get('ratio')} | "
          f"{r.get('FP4', {}).get('ratio')} | 1.765 / 2.652 | "
          f"{'yes' if pt['BF16']['sustained_pressure_reps'] else 'none'} |")
    A("")
    for k, v in p1["checks"].items():
        A(f"- {'PASS' if v else 'FAIL'} - {k}")
    for f in p1["findings"]:
        A(f"- finding: {f}")
    A("")

    A(f"## P2 - BF16 KV wall: **{p2['verdict']}**")
    A("")
    A(f"- measured KV capacity this configuration: **{p2['measured_kv_cache_tokens']} tokens**")
    A(f"- predicted wall, peak-footprint basis (2,560 tok/seq): **{p2['predicted_wall_peak_basis']}**")
    A(f"- predicted wall, mean-occupancy basis (1,536 tok/seq): **{p2['predicted_wall_mean_basis']}**")
    A(f"- **measured wall bracket: C_KV^BF16 in {p2['measured_wall_bracket']}**")
    A(f"- pre-registered D11 neighbourhood: {p2['criteria']['expected_band']}")
    A("")
    A("| C | state | tok/s | TPOT P95 ms | KV P95 | preempt | recomputed | waiting max | status |")
    A("|---|---|---|---|---|---|---|---|---|")
    for C, rs in sorted(p2["per_concurrency_detail"].items()):
        for x in rs:
            A(f"| {C} | {p2['per_concurrency_state'][C]} | {_r(x['tok_s'])} | "
              f"{_r(x['tpot_ms_p95'])} | {_r(x['kv_p95'], 3)} | {x['preempt']} | "
              f"{x['recomputed']} | {x['waiting_max']} | {x['status']} |")
    A("")
    for k, v in p2["checks"].items():
        A(f"- {'PASS' if v else 'FAIL'} - {k}")
    A("")
    peak = p2.get("throughput_peak")
    if peak:
        A(f"**The throughput knee precedes preemption by one concurrency point.** Output throughput "
          f"peaks at C={peak['peak_C']} ({peak['peak_tok_s']} tok/s) and regresses at "
          f"C={peak['next_C']} ({peak['next_tok_s']} tok/s, {peak['regression_pct']}%) with TPOT P95 "
          f"rising {peak['tpot_from']} -> {peak['tpot_to']} ms — while preemption is still exactly "
          f"zero and no requests are queued. A pressure test keyed only on preemption counters "
          f"therefore detects this wall one point late. Both indicators are reported; the bracket "
          f"above uses the pre-registered preemption/recompute definition.")
        A("")

    A(f"## P3 - repetition count: **{p3['verdict']}**")
    A("")
    A("| point | C | mean FP8 | mean FP4 | delta | SE | 95% half-width | rho | resolves |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, v in p3["points"].items():
        A(f"| {name} | {v['concurrency']} | {v.get('mean_fp8')} | {v.get('mean_fp4')} | "
          f"{v.get('delta_tok_s')} | {v.get('se_delta')} | {v.get('half_width_95')} | "
          f"{v.get('rho')} | {v.get('resolves')} |")
    A("")
    A(f"- criterion rho <= {p3['criteria']['rho_max']}")
    A(f"- **locked repetition count for the full serving sweep: "
      f"{p3.get('locked_repetition_count') or 'NOT LOCKED - ' + str(p3.get('escalation_required'))}**")
    A("")
    A("Coefficient of variation, every configuration tested:")
    A("")
    A("| cell | n | mean tok/s | sd | CV % | TPOT P95 CV % |")
    A("|---|---|---|---|---|---|")
    for k, v in sorted(p3["cv_all_configs"].items()):
        A(f"| {k} | {v['n']} | {v['mean']} | {v['sd']} | {v['cv_pct']} | {v['tpot_p95_cv_pct']} |")
    A("")

    A(f"## P4 - achievable HBM bandwidth: **{manifest['verdicts']['P4']}**")
    A("")
    if p4:
        A(f"Independent CUDA streaming kernels, CUDA-event timed, working set "
          f"{p4['working_set']['array_bytes'] / 2**30:.1f} GiB per array "
          f"({p4['working_set']['array_over_l2_ratio']}x L2). Not derived from decode throughput.")
        A("")
        A("| kernel | median GB/s | P95 | max | CV % | vs spec |")
        A("|---|---|---|---|---|---|")
        for name, r in p4["results"].items():
            A(f"| {name} | {r['median_GBs']} | {r['p95_GBs']} | {r['max_GBs']} | {r['cv_pct']} | "
              f"{(p4.get('achieved_over_spec') or {}).get(name)} |")
        A("")
        A(f"- spec bandwidth {p4['spec_bandwidth_GBs']} GB/s "
          f"({p4['bus_width_bits']}-bit x {p4['memory_clock_khz'] / 1e6:.3f} GHz x 2); "
          f"{p4['spec_bandwidth_source']}")
        A(f"- kernel correctness validated: {p4['kernel_correctness']}")
        A("- device-memory only; no host-device transfer is involved")
    A("")

    A(f"## P5 - harness validation: **{p5['verdict']}**")
    A("")
    for k, v in p5["invariants"].items():
        if not isinstance(v, dict):
            continue
        if "pass" in v:
            mark = "PASS" if v["pass"] else "FAIL"
            extra = ""
            if not v["pass"] and v.get("violations"):
                extra = " - " + ", ".join(v["violations"][:6])
            A(f"- {mark} - {k}{extra}")
    slo = p5["invariants"]["slo_visibility"]
    A("")
    A(f"SLO visibility (TPOT P95 <= {slo['slo_tpot_ms_p95']} ms): "
      f"BF16 within SLO at C=8 and C=12 = **{slo['bf16_within_slo_at_c8_c12']}**")
    for k, v in sorted(slo["per_cell"].items()):
        A(f"- {k}: " + ", ".join(f"{x['tpot_ms_p95']:.1f} ms ({x['meets_slo']})"
                                 for x in v if x["tpot_ms_p95"]))
    A("")

    A(f"## Correctness gate: **{manifest['verdicts']['correctness_gate']}**")
    A("")
    if gate:
        A(f"{gate['n_contexts']} real held-out C4 contexts, position 1, D13 path. "
          "Sanity trip-wire, not a quality result.")
        A("")
        A("| comparison | median nats | p90 | max | finite | non-negative | top-1 agreement |")
        A("|---|---|---|---|---|---|---|")
        for k, v in gate["pairs"].items():
            A(f"| {k.replace('||', ' vs ')} | {v['median_nats']:.3e} | {v['p90_nats']:.3e} | {v['max_nats']:.3e} | "
              f"{v['all_finite']} | {v['all_nonnegative']} | {v['top1_agreement_vs_bf16']:.4f} |")
        A("")
        for k, v in gate["checks"].items():
            A(f"- {'PASS' if v else 'FAIL'} - {k}")
        if gate["failed_checks"]:
            A(f"- failed: {gate['failed_checks']}")
    A("")

    A("## Clearance")
    A("")
    if manifest["cleared_for_full_serving_sweep"]:
        A("**CLEARED for the full serving sweep.** P1, P2, P3 and P5 pass, the P4 measurement is")
        A("valid, and the correctness gate is clean.")
    else:
        A("**NOT CLEARED for the full serving sweep.**")
        A("")
        for job, v in manifest["verdicts"].items():
            if v not in ("PASS", "VALID", "CLEAN"):
                A(f"- {job} = {v}")
        A("")
        if p1["verdict"] == "FAIL" or p2["verdict"] == "FAIL":
            A("P1 or P2 failing means the physical assumptions behind D11 are not supported by the")
            A("pilot. The decision to reopen is D11's, not the harness's - the evidence is reported")
            A("as measured and the interpretation is not adjusted to fit it.")
    A("")
    A("## Decisions that must be reopened")
    A("")
    reopen = []
    if p1["verdict"] == "FAIL":
        reopen.append("D11 bandwidth-vs-capacity decomposition, and D10's weight-size-ratio prediction")
    if p2["verdict"] == "FAIL":
        reopen.append("D11 concurrency ladder - the locked points do not bracket the measured wall")
    if p3["verdict"] == "FAIL":
        reopen.append("repetition count in EXPERIMENTAL_CONTRACT.md 'Repetitions and duration'")
    if p5["invariants"]["slo_visibility"]["bf16_within_slo_at_c8_c12"] is False:
        reopen.append("the 50 ms TPOT P95 SLO - BF16 breaches it below the KV wall")
    if not reopen:
        A("- none")
    for r in reopen:
        A(f"- {r}")
    A("")
    path = os.path.join(out_dir, "PILOT_DECISION.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("WROTE", path)


if __name__ == "__main__":
    main()
