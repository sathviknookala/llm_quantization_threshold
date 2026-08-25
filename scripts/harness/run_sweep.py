"""D11 serving sweep orchestrator.

The pilot's runner hardcodes the P1/P2/P3/P5 job structure. This walks the locked concurrency
ladder with the contract's SKIPPED_PAST_SLO rule, then refines each configuration's KV-wall
bracket, and is built to survive an unattended overnight run: every cell reaches the artifact as
either a measurement or an explicit infeasibility.
"""

import argparse
import atexit
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import common, driver, orchestration as orch  # noqa: E402
from harness.run_pilot import LATIN, pressured  # noqa: E402
from harness.server import VllmServer  # noqa: E402

OUT_DIR = common.SWEEP_DIR
CELLS = os.path.join(OUT_DIR, "cells.jsonl")
LOGS = os.path.join(OUT_DIR, "server_logs")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")


def set_out_dir(d):
    """A smoke run must not consume cells the real sweep would then run on a second engine."""
    global OUT_DIR, CELLS, LOGS, MANIFEST
    OUT_DIR = d
    CELLS = os.path.join(d, "cells.jsonl")
    LOGS = os.path.join(d, "server_logs")
    MANIFEST = os.path.join(d, "manifest.json")
PORT = 8766

DECODE_LADDER = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96]
PREFILL_LADDER = [1, 2, 4, 8]

DECODE_CELL = dict(
    min_requests_factor=4, window_floor_s=120.0, warmup_wall_cap_s=600.0,
    gate_timeout_s=420.0, warmup_lifetimes=2, min_periods=4, margin=1.5,
    hard_cap_s=1800.0,
    abort_tpot_ms=common.SLO_TPOT_MS * common.SLO_ABORT_MULTIPLE,
)
PREFILL_CELL = dict(
    min_requests_factor=4, window_floor_s=60.0, warmup_wall_cap_s=300.0,
    gate_timeout_s=300.0, warmup_lifetimes=2, min_periods=4, margin=1.5,
    hard_cap_s=900.0,
    abort_tpot_ms=None,
)

# H10: a launch begun before a previous engine released VRAM silently sizes a smaller cache. The
# contamination signature was BF16 at 40,432 against 44,688 -- 10% low, silent, and it serves.
KV_EXPECTED = {"BF16_REFERENCE": 44688, "FP8_PRIMARY": 97888, "FP4_PRIMARY": 120944}
KV_TOLERANCE = {"BF16_REFERENCE": 0.01, "FP8_PRIMARY": 0.01, "FP4_PRIMARY": 0.05}

SWEEP_SPEC = {
    "schema_version": 3,
    "decode_ladder": DECODE_LADDER,
    "prefill_ladder": PREFILL_LADDER,
    "latin": LATIN,
    "decode_cell": DECODE_CELL,
    "prefill_cell": PREFILL_CELL,
    "workloads": common.WORKLOADS,
    "server_controls": common.SERVER_CONTROLS,
    "slo_tpot_ms": common.SLO_TPOT_MS,
    "kv_expected": KV_EXPECTED,
    "kv_tolerance": KV_TOLERANCE,
    "configs": {k: {"model": v["model"], "expected_kernel_pattern": v["expected_kernel_pattern"],
                    "forbidden_kernel_patterns": v["forbidden_kernel_patterns"]}
                for k, v in common.CONFIGS.items()},
}


class LaunchError(RuntimeError):
    pass


# A launch that dies between srv.start() and the caller's try/finally orphans an EngineCore
# holding 22 GiB. Observed, not theoretical.
_LIVE = []


def _teardown_all():
    while _LIVE:
        srv = _LIVE.pop()
        try:
            srv.stop()
        except Exception:
            pass
    try:
        from harness.server import gpu_holder_pids, kill_gpu_holders
        if gpu_holder_pids():
            kill_gpu_holders()
    except Exception:
        pass


atexit.register(_teardown_all)


def spec_hash():
    return common.sha256_of_json(SWEEP_SPEC)[:16]


def cell_kwargs(workload):
    return dict(PREFILL_CELL if workload == "PREFILL_PROBE" else DECODE_CELL)


def violated(rec):
    """SLO-only allowlist.

    A gate timeout, an invalid cell or a dead engine is a harness event; letting any of them
    truncate the ladder would delete the remaining points and then report the deletion as the
    configuration's serving ceiling.
    """
    st = rec.get("status")
    if st == "SLO_VIOLATED":
        return True
    return st == "OK" and rec.get("meets_slo") is False


def violation_point(cells, rep, config_id, workload):
    hits = [r["concurrency"] for r in cells
            if r.get("repetition") == rep and r.get("configuration_id") == config_id
            and r.get("workload") == workload and violated(r)]
    return min(hits) if hits else None


def check_launch(srv, config_id):
    dv = srv.startup.get("dispatch_verdict") or {}
    if not dv.get("ok"):
        raise LaunchError(f"dispatch verification failed for {config_id}: {dv}")
    kv = srv.startup.get("kv_cache_tokens")
    want = KV_EXPECTED.get(config_id)
    tol = KV_TOLERANCE.get(config_id, 0.05)
    if want and kv is not None and kv < want * (1.0 - tol):
        raise LaunchError(
            f"KV capacity {kv} is below {want} by more than {tol:.0%} -- contaminated launch (H10)")
    return kv


def launch(config_id, rep, workload, phase, warm_ok=False):
    orch.preflight(require_cool=not warm_ok)
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"{phase}_{config_id}_r{rep}_{int(time.time())}.log")
    srv = VllmServer(config_id, PORT, log)
    _LIVE.append(srv)
    srv.start()
    srv.startup["log_path"] = log
    kv = check_launch(srv, config_id)
    print(f"  launched {config_id} rep{rep} kv_tokens={kv} "
          f"start={srv.startup.get('engine_start_seconds')}s log={os.path.basename(log)}",
          flush=True)
    return srv


def run_ladder_group(config_id, rep, workload, ladder, env, manifest, prompts, cells, args):
    """One engine process for every concurrency point of one (configuration, repetition)."""
    job = "SWEEP"
    skip = orch.done_keys(CELLS)
    todo = [C for C in ladder if (job, config_id, workload, C, rep) not in skip]
    if not todo:
        print(f"skip {config_id} rep{rep} {workload}: all cells present", flush=True)
        return
    v_at = violation_point(cells, rep, config_id, workload)
    srv = None
    try:
        srv = launch(config_id, rep, workload, phase=workload.lower())
    except (LaunchError, RuntimeError) as exc:
        print(f"LAUNCH FAILED {config_id} rep{rep}: {exc}", flush=True)
        for C in todo:
            orch.write_placeholder(CELLS, job, config_id, manifest, env, workload, C, rep,
                                   "LAUNCH_FAILED", {"launch_error": str(exc)[:300]})
        if srv is not None:
            srv.stop()
            if srv in _LIVE:
                _LIVE.remove(srv)
        return
    try:
        mono_done = False
        for C in ladder:
            if (job, config_id, workload, C, rep) in orch.done_keys(CELLS):
                continue
            if v_at is not None and C > v_at:
                # rep 1 runs exactly one point past the first violation so monotonicity is
                # tested rather than assumed; the pilot's BF16 curve is already non-monotonic
                if rep == 1 and not mono_done:
                    mono_done = True
                    print(f"  monotonicity probe at C={C} (violation was C={v_at})", flush=True)
                else:
                    orch.write_placeholder(
                        CELLS, job, config_id, manifest, env, workload, C, rep,
                        "SKIPPED_PAST_SLO", {"skipped_because_slo_violated_at_c": v_at})
                    continue
            if not srv.wait_drained():
                print(f"  drain timeout before C={C}; restarting engine", flush=True)
                srv.stop()
                if srv in _LIVE:
                    _LIVE.remove(srv)
                try:
                    srv = launch(config_id, rep, workload, phase=workload.lower(), warm_ok=True)
                except (LaunchError, RuntimeError) as exc:
                    orch.write_placeholder(CELLS, job, config_id, manifest, env, workload, C, rep,
                                           "LAUNCH_FAILED", {"launch_error": str(exc)[:300]})
                    return
            rec = orch.run_one(CELLS, job, srv, config_id, prompts, manifest, env, workload, C,
                               rep, cell_kwargs(workload),
                               extra={"sweep_config_hash": spec_hash(),
                                      "engine_kv_cache_tokens": srv.startup.get("kv_cache_tokens")})
            cells.append(rec)
            if v_at is None and violated(rec):
                v_at = C
                print(f"  SLO violated at C={C}; higher points skipped for this repetition",
                      flush=True)
    finally:
        if srv is not None:
            srv.stop()
            if srv in _LIVE:
                _LIVE.remove(srv)


def refine_group(config_id, env, manifest, prompts, cells, args):
    """Bisect the KV-wall bracket the locked ladder is too coarse to resolve.

    FP8 (~38) and FP4 (~47) both fall in the ladder's 32-48 gap, so the sweep alone returns the
    same bracket for both and cannot separate them. D11's points stay untouched; this is additive.
    """
    job = "SWEEP_REFINE"
    workload = "DECODE_PRIMARY"
    mine = [r for r in cells if r.get("configuration_id") == config_id
            and r.get("workload") == workload and r.get("job") == "SWEEP"
            and r.get("status") in ("OK", "SLO_VIOLATED")]
    clean = sorted({r["concurrency"] for r in mine if not pressured(r)})
    press = sorted({r["concurrency"] for r in mine if pressured(r)})
    if not clean or not press:
        print(f"refine {config_id}: no pressure transition in the ladder "
              f"(clean={clean} pressured={press}); nothing to bisect", flush=True)
        return
    lo, hi = max(clean), min([c for c in press if c > max(clean)] or [None])
    if hi is None or hi - lo <= 1:
        print(f"refine {config_id}: bracket already [{lo},{hi}]", flush=True)
        return
    print(f"refine {config_id}: bisecting [{lo},{hi}]", flush=True)
    srv = None
    try:
        srv = launch(config_id, 1, workload, phase="refine")
    except (LaunchError, RuntimeError) as exc:
        print(f"LAUNCH FAILED refine {config_id}: {exc}", flush=True)
        return
    try:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            key = (job, config_id, workload, mid, 1)
            done = {(r["job"], r["configuration_id"], r["workload"], r["concurrency"],
                     r["repetition"]): r for r in orch.read_cells(CELLS)}
            if key in done:
                rec = done[key]
            else:
                if not srv.wait_drained():
                    srv.stop()
                    if srv in _LIVE:
                        _LIVE.remove(srv)
                    srv = launch(config_id, 1, workload, phase="refine", warm_ok=True)
                rec = orch.run_one(CELLS, job, srv, config_id, prompts, manifest, env, workload,
                                   mid, 1, cell_kwargs(workload),
                                   extra={"sweep_config_hash": spec_hash(),
                                          "refine_bracket": [lo, hi]})
                cells.append(rec)
            if pressured(rec):
                hi = mid
            else:
                lo = mid
        print(f"refine {config_id}: wall bracket [{lo},{hi}]", flush=True)
    finally:
        if srv is not None:
            srv.stop()
            if srv in _LIVE:
                _LIVE.remove(srv)


def projected_seconds(C, tput, n_out):
    period = n_out * C / float(tput)
    return 6.5 * period + 110.0


def plan_only():
    print(f"sweep_config_hash = {spec_hash()}\n")
    est = {"BF16_REFERENCE": {1: 36, 4: 140, 8: 268, 12: 381, 16: 490, 24: 415},
           "FP8_PRIMARY": {1: 66, 4: 250, 8: 450, 12: 618, 16: 780, 24: 1000, 32: 1180,
                           48: 1400, 64: 1550, 96: 1700},
           "FP4_PRIMARY": {1: 88, 4: 320, 8: 567, 12: 764, 16: 950, 24: 1188, 32: 1400,
                           48: 1650, 64: 1800, 96: 1950}}
    total = 0.0
    n_out = common.WORKLOADS["DECODE_PRIMARY"]["output_tokens"]
    for rep in (1, 2, 3):
        for cfg in LATIN[(rep - 1) % 3]:
            for C in DECODE_LADDER:
                t = est.get(cfg, {}).get(C)
                if t is None:
                    print(f"  rep{rep} {cfg:16} C={C:<3} SKIPPED_PAST_SLO (projected)")
                    continue
                s = projected_seconds(C, t, n_out)
                total += s
                print(f"  rep{rep} {cfg:16} C={C:<3} ~{s / 60:5.1f} min")
    launches = 9
    print(f"\n  decode cells est   {total / 3600:.1f} h")
    print(f"  launches           {launches} decode + 3 prefill + up to 3 refine")
    print(f"  launch overhead    ~{launches * 240 / 3600:.1f} h (preflight + start + teardown)")
    print(f"  prefill probe      ~0.3 h")
    print(f"  refinement         ~1.5 h")
    print(f"  TOTAL              ~{total / 3600 + launches * 240 / 3600 + 1.8:.1f} h")


def guard_spec():
    os.makedirs(OUT_DIR, exist_ok=True)
    h = spec_hash()
    if os.path.exists(MANIFEST):
        prior = json.load(open(MANIFEST)).get("sweep_config_hash")
        if prior and prior != h:
            raise SystemExit(
                f"ABORT: SWEEP_SPEC changed ({prior} -> {h}) but {CELLS} holds cells from the "
                f"old spec. Move results/sweep aside or restore the spec; resume must not mix.")
    common.write_json(MANIFEST, {
        "artifact": "D11 serving sweep",
        "sweep_config_hash": h,
        "spec": SWEEP_SPEC,
        "started_at": common.now_iso(),
    })
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="all", choices=["decode", "prefill", "refine", "all"])
    ap.add_argument("--reps", default="1,2,3")
    ap.add_argument("--configs", default="")
    ap.add_argument("--ladder", default="")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    if args.plan_only:
        plan_only()
        return
    if args.out_dir:
        set_out_dir(args.out_dir)

    h = guard_spec()
    print(f"sweep_config_hash = {h}", flush=True)
    reps = [int(x) for x in args.reps.split(",") if x.strip()]
    configs = [c for c in args.configs.split(",") if c.strip()] or None
    ladder = ([int(x) for x in args.ladder.split(",")] if args.ladder else None)
    env = orch.env_identity()

    def bye(signum, frame):
        raise SystemExit(f"received signal {signum}; engines are torn down by the finally blocks")
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)

    if args.job in ("decode", "all"):
        prompts, manifest = orch.load_prompts("DECODE_PRIMARY")
        lad = ladder or DECODE_LADDER
        for rep in reps:
            for cfg in LATIN[(rep - 1) % 3]:
                if configs and cfg not in configs:
                    continue
                cells = orch.read_cells(CELLS)
                run_ladder_group(cfg, rep, "DECODE_PRIMARY", lad, env, manifest, prompts,
                                 cells, args)

    if args.job in ("prefill", "all"):
        prompts, manifest = orch.load_prompts("PREFILL_PROBE")
        for cfg in ("BF16_REFERENCE", "FP8_PRIMARY", "FP4_PRIMARY"):
            if configs and cfg not in configs:
                continue
            cells = orch.read_cells(CELLS)
            run_ladder_group(cfg, 1, "PREFILL_PROBE", ladder or PREFILL_LADDER, env, manifest,
                             prompts, cells, args)

    if args.job in ("refine", "all"):
        prompts, manifest = orch.load_prompts("DECODE_PRIMARY")
        for cfg in ("BF16_REFERENCE", "FP8_PRIMARY", "FP4_PRIMARY"):
            if configs and cfg not in configs:
                continue
            refine_group(cfg, env, manifest, prompts, orch.read_cells(CELLS), args)

    print("sweep phase complete", flush=True)


if __name__ == "__main__":
    main()
