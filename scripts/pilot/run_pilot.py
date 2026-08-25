"""Pilot orchestrator: P1, P2, P3 and P5's PREFILL_PROBE plumbing cell.

Every cell is appended to cells.jsonl with enough identity to reproduce it. Reruns skip
cells already recorded, so an interrupted pilot resumes instead of restarting.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilot import common, driver  # noqa: E402
from pilot.server import VllmServer  # noqa: E402

CELLS = os.path.join(common.PILOT_DIR, "cells.jsonl")
LOGS = os.path.join(common.PILOT_DIR, "server_logs")
PORT = 8765

P1_CONCURRENCY = [1, 8, 12]
P3_EXTRA_CONCURRENCY = 24
LATIN = [["BF16_REFERENCE", "FP8_PRIMARY", "FP4_PRIMARY"],
         ["FP8_PRIMARY", "FP4_PRIMARY", "BF16_REFERENCE"],
         ["FP4_PRIMARY", "BF16_REFERENCE", "FP8_PRIMARY"]]

P1_CELL = dict(min_requests_factor=4, window_floor_s=120.0, warmup_wall_cap_s=600.0,
               wall_cap_s=1200.0, gate_timeout_s=420.0,
               abort_tpot_ms=common.SLO_TPOT_MS * common.SLO_ABORT_MULTIPLE)
# P2 must observe the pressured regime, so the 10x-SLO abort is off and the wall cap governs.
P2_CELL = dict(min_requests_factor=2, window_floor_s=120.0, warmup_wall_cap_s=300.0,
               wall_cap_s=1500.0, gate_timeout_s=420.0, abort_tpot_ms=None)


def load_prompts(workload):
    n = common.WORKLOADS[workload]["input_tokens"]
    stem = os.path.join(common.CORPUS_DIR, f"{workload.lower()}_{n}tok")
    body = json.load(open(stem + ".json"))
    manifest = json.load(open(stem + "_manifest.json"))
    for p in body["prompts"]:
        if p["n_tokens"] != n:
            raise SystemExit(f"ABORT: corpus prompt {p['index']} has {p['n_tokens']} tokens, want {n}")
    return body["prompts"], manifest


def done_keys():
    keys = set()
    if not os.path.exists(CELLS):
        return keys
    for line in open(CELLS):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((r["job"], r["configuration_id"], r["workload"], r["concurrency"], r["repetition"]))
    return keys


def append_cell(rec):
    os.makedirs(os.path.dirname(CELLS), exist_ok=True)
    with open(CELLS, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def identity(job, config_id, srv, manifest, env):
    return {
        "job": job,
        "configuration_id": config_id,
        "quantization": common.CONFIGS[config_id]["short"],
        "model_path": common.CONFIGS[config_id]["model"],
        "weight_bytes_gb": common.CONFIGS[config_id]["weight_bytes_gb"],
        "serving_backend": "vLLM " + env["software"].get("vllm", "?"),
        "gpu": env["gpu"],
        "software": env["software"],
        "server_controls": common.SERVER_CONTROLS,
        "kv_cache_tokens": srv.startup.get("kv_cache_tokens"),
        "dispatch_verdict": srv.startup.get("dispatch_verdict"),
        "enable_prefix_caching_logged": srv.startup.get("enable_prefix_caching_logged"),
        "engine_start_seconds": srv.startup.get("engine_start_seconds"),
        "server_log": srv.startup.get("log_path"),
        "corpus_version": manifest["corpus_version"],
        "prompt_set_hash": manifest["prompt_set_hash"],
        "slo_tpot_ms": common.SLO_TPOT_MS,
    }


def run_one(job, srv, config_id, prompts, manifest, env, workload, C, rep, cell_kwargs):
    cc = driver.CellConfig(workload=workload, concurrency=C, repetition=rep, **cell_kwargs)
    tag = f"{job}:{common.CONFIGS[config_id]['short']}:{workload}:C{C}:r{rep}"
    print(f"[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
    t0 = time.time()
    res = driver.run_cell(srv, prompts, cc, tag)
    rec = identity(job, config_id, srv, manifest, env)
    rec.update(res)
    rec["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0))
    rec["finished_at"] = common.now_iso()
    append_cell(rec)
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {tag} status={res['status']} "
          f"tok/s={res['output_tokens_per_s']} tpotP95={res['tpot_ms_p95']} "
          f"kvP95={res['kv_cache_usage_p95']} preempt={res['num_preemptions_delta']} "
          f"recomp={res['recomputed_tokens_delta']} pfxhits={res['prefix_cache_hits_delta']}",
          flush=True)
    return rec


def env_identity():
    return {"gpu": common.gpu_identity(), "software": common.software_identity()}


def preflight(wait_s=240):
    """Wait for a genuinely idle GPU rather than aborting on a slow teardown."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if common.gpu_is_idle():
            return
        time.sleep(5.0)
    t = common.gpu_telemetry()
    raise SystemExit(
        f"ABORT: GPU still busy after {wait_s}s "
        f"(mem={t.get('memory.used')} MiB, util={t.get('utilization.gpu')}%); "
        "timed runs require an otherwise-idle GPU")


def job_p1_p3(reps, env, skip):
    prompts, manifest = load_prompts("DECODE_PRIMARY")
    for rep in reps:
        for config_id in LATIN[(rep - 1) % 3]:
            cs = list(P1_CONCURRENCY)
            if config_id != "BF16_REFERENCE":
                cs.append(P3_EXTRA_CONCURRENCY)
            job_of = {c: ("P1" if c in P1_CONCURRENCY else "P3") for c in cs}
            todo = [c for c in cs if (job_of[c], config_id, "DECODE_PRIMARY", c, rep) not in skip]
            if not todo:
                print(f"skip {config_id} rep{rep}: all cells present", flush=True)
                continue
            preflight()
            srv = VllmServer(config_id, PORT, os.path.join(LOGS, f"p1_{config_id}_r{rep}.log"))
            print(f"launching {config_id} rep{rep} ...", flush=True)
            info = srv.start()
            print(f"  kv_tokens={info['kv_cache_tokens']} dispatch_ok={info['dispatch_verdict']['ok']} "
                  f"prefix_caching={info['enable_prefix_caching_logged']} "
                  f"start={info['engine_start_seconds']}s", flush=True)
            try:
                for C in todo:
                    srv.wait_drained()
                    run_one(job_of[C], srv, config_id, prompts, manifest, env,
                            "DECODE_PRIMARY", C, rep, P1_CELL)
            finally:
                srv.stop()


def pressured(rec):
    """KV-capacity pressure, not scheduler queueing: preemption and recompute only.

    Falls back to whole-cell counters so a cell that never opened its timed window cannot be
    mistaken for a clean point.
    """
    if rec.get("num_preemptions_delta") is not None:
        p = rec.get("num_preemptions_delta") or 0
        n = rec.get("preemption_nonzero_samples") or 0
        r = rec.get("recomputed_tokens_delta") or 0
    else:
        p = rec.get("cell_num_preemptions_delta") or 0
        n = rec.get("cell_preemption_nonzero_samples") or 0
        r = rec.get("cell_recomputed_tokens_delta") or 0
    return (p > 0 and n >= 2) or r > 0


def job_p2(env, skip):
    prompts, manifest = load_prompts("DECODE_PRIMARY")
    config_id = "BF16_REFERENCE"
    preflight()
    srv = VllmServer(config_id, PORT, os.path.join(LOGS, "p2_BF16_REFERENCE.log"))
    print("launching BF16 for P2 wall search ...", flush=True)
    info = srv.start()
    print(f"  kv_tokens={info['kv_cache_tokens']} dispatch_ok={info['dispatch_verdict']['ok']}",
          flush=True)
    seen = {}
    try:
        def measure(C, rep=1):
            key = ("P2", config_id, "DECODE_PRIMARY", C, rep)
            if key in skip:
                for line in open(CELLS):
                    r = json.loads(line)
                    if (r["job"], r["configuration_id"], r["workload"], r["concurrency"],
                            r["repetition"]) == key:
                        seen[(C, rep)] = r
                        return r
            srv.wait_drained()
            r = run_one("P2", srv, config_id, prompts, manifest, env,
                        "DECODE_PRIMARY", C, rep, P2_CELL)
            seen[(C, rep)] = r
            return r

        measure(12)
        first = measure(15)
        if pressured(first):
            last_clean, first_pressured = None, 15
            for C in (14, 13, 12):
                r = seen.get((C, 1)) or measure(C)
                if not pressured(r):
                    last_clean = C
                    break
                first_pressured = C
        else:
            last_clean, first_pressured = 15, None
            for C in (16, 17, 18, 20, 22, 25, 28, 32):
                r = measure(C)
                if pressured(r):
                    first_pressured = C
                    break
                last_clean = C
            if first_pressured and first_pressured - last_clean > 1:
                for C in range(last_clean + 1, first_pressured):
                    r = measure(C)
                    if pressured(r):
                        first_pressured = C
                        break
                    last_clean = C
        # reproducibility of the bracket itself, not of a throughput number
        for C in [c for c in (last_clean, first_pressured) if c]:
            measure(C, rep=2)
        print(f"P2 bracket: last_clean={last_clean} first_pressured={first_pressured}", flush=True)
    finally:
        srv.stop()


def job_p5_prefill(env, skip):
    prompts, manifest = load_prompts("PREFILL_PROBE")
    for config_id in ("BF16_REFERENCE", "FP8_PRIMARY", "FP4_PRIMARY"):
        if ("P5", config_id, "PREFILL_PROBE", 1, 1) in skip:
            continue
        preflight()
        srv = VllmServer(config_id, PORT, os.path.join(LOGS, f"p5_prefill_{config_id}.log"))
        srv.start()
        try:
            cell = dict(P1_CELL)
            cell.update(min_requests_factor=4, window_floor_s=60.0, warmup_wall_cap_s=300.0,
                        wall_cap_s=600.0, warmup_lifetimes=2)
            run_one("P5", srv, config_id, prompts, manifest, env, "PREFILL_PROBE", 1, 1, cell)
        finally:
            srv.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["p1", "p2", "p3", "p5", "all"])
    ap.add_argument("--reps", default="1,2,3")
    a = ap.parse_args()
    env = env_identity()
    skip = done_keys()
    reps = [int(x) for x in a.reps.split(",") if x.strip()]
    print(f"resuming with {len(skip)} cells already recorded", flush=True)
    if a.job in ("p1", "p3", "all"):
        job_p1_p3(reps, env, skip)
        skip = done_keys()
    if a.job in ("p2", "all"):
        job_p2(env, skip)
        skip = done_keys()
    if a.job in ("p5", "all"):
        job_p5_prefill(env, skip)
    print("orchestrator finished", flush=True)


if __name__ == "__main__":
    main()
