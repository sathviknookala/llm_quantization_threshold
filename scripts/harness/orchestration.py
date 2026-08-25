"""Cell identity, resume and launch helpers shared by the pilot and the sweep.

One definition of a cell's identity is what keeps the two artifacts comparable; duplicated
copies drift, and the drift is invisible until someone tries to read them together.
"""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import common, driver  # noqa: E402


def load_prompts(workload):
    n = common.WORKLOADS[workload]["input_tokens"]
    stem = os.path.join(common.CORPUS_DIR, f"{workload.lower()}_{n}tok")
    body = json.load(open(stem + ".json"))
    manifest = json.load(open(stem + "_manifest.json"))
    for p in body["prompts"]:
        if p["n_tokens"] != n:
            raise SystemExit(
                f"ABORT: corpus prompt {p['index']} has {p['n_tokens']} tokens, want {n}")
    return body["prompts"], manifest


def read_cells(cells_path):
    out = []
    if not os.path.exists(cells_path):
        return out
    for line in open(cells_path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def done_keys(cells_path):
    return {(r["job"], r["configuration_id"], r["workload"], r["concurrency"], r["repetition"])
            for r in read_cells(cells_path)}


def append_cell(cells_path, rec):
    os.makedirs(os.path.dirname(cells_path), exist_ok=True)
    with open(cells_path, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
        # a torn final line survives resume but poisons every other reader
        fh.flush()
        os.fsync(fh.fileno())


def identity(job, config_id, srv, manifest, env):
    startup = getattr(srv, "startup", {}) or {}
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
        "kv_cache_tokens": startup.get("kv_cache_tokens"),
        "dispatch_verdict": startup.get("dispatch_verdict"),
        "enable_prefix_caching_logged": startup.get("enable_prefix_caching_logged"),
        "engine_start_seconds": startup.get("engine_start_seconds"),
        "server_log": startup.get("log_path"),
        "corpus_version": manifest["corpus_version"],
        "prompt_set_hash": manifest["prompt_set_hash"],
        "slo_tpot_ms": common.SLO_TPOT_MS,
    }


def run_one(cells_path, job, srv, config_id, prompts, manifest, env, workload, C, rep,
            cell_kwargs, extra=None):
    cc = driver.CellConfig(workload=workload, concurrency=C, repetition=rep, **cell_kwargs)
    tag = f"{job}:{common.CONFIGS[config_id]['short']}:{workload}:C{C}:r{rep}"
    print(f"[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
    t0 = time.time()
    try:
        res = driver.run_cell(srv, prompts, cc, tag)
    except Exception as exc:
        # an exception here used to propagate past append_cell and lose the cell entirely
        res = {"tag": tag, "workload": workload, "concurrency": C, "repetition": rep,
               "status": "HARNESS_ERROR", "valid_result": False,
               "outcome_class": driver.outcome_class("HARNESS_ERROR"),
               "invalid_reasons": [f"exception:{type(exc).__name__}"],
               "traceback": traceback.format_exc()[-2000:],
               "cell_config": cc.as_dict()}
    rec = identity(job, config_id, srv, manifest, env)
    rec.update(res)
    if extra:
        rec.update(extra)
    rec["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0))
    rec["finished_at"] = common.now_iso()
    append_cell(cells_path, rec)
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {tag} status={res.get('status')} "
          f"outcome={res.get('outcome_class')} tok/s={res.get('output_tokens_per_s')} "
          f"tpotP95={res.get('tpot_ms_p95')} slo={res.get('meets_slo')} "
          f"kvP95={res.get('kv_cache_usage_p95')} preempt={res.get('num_preemptions_delta')} "
          f"periods={res.get('periods_in_window')} wall={res.get('cell_wall_seconds')}s",
          flush=True)
    return rec


def write_placeholder(cells_path, job, config_id, manifest, env, workload, C, rep, status,
                      extra=None):
    """Skipped and abandoned cells are results and must be in the artifact, not absent."""
    class _NoServer:
        startup = {}
    rec = identity(job, config_id, _NoServer(), manifest, env)
    rec.update({
        "tag": f"{job}:{common.CONFIGS[config_id]['short']}:{workload}:C{C}:r{rep}",
        "workload": workload, "concurrency": C, "repetition": rep,
        "status": status, "valid_result": False,
        "outcome_class": driver.outcome_class(status),
        "invalid_reasons": [], "window_is_complete": False,
        "input_tokens": common.WORKLOADS[workload]["input_tokens"],
        "output_tokens_requested": common.WORKLOADS[workload]["output_tokens"],
        "finished_at": common.now_iso(),
    })
    if extra:
        rec.update(extra)
    append_cell(cells_path, rec)
    print(f"[{time.strftime('%H:%M:%S')}] {status} {rec['tag']} {extra or ''}", flush=True)
    return rec


def env_identity():
    return {"gpu": common.gpu_identity(), "software": common.software_identity()}


def preflight(wait_s=240, require_cool=True, max_temp_c=common.PREFLIGHT_MAX_TEMP_C):
    """Idle memory is not a comparable thermal state; a config that skips early would otherwise
    hand the next launch a cooler card than its counterbalanced position implies."""
    t0 = time.time()
    idle = cool = False
    while time.time() - t0 < wait_s:
        idle = common.gpu_is_idle()
        cool = common.gpu_is_cool(max_temp_c) if require_cool else True
        if idle and cool:
            return True
        time.sleep(5.0)
    t = common.gpu_telemetry()
    if not idle:
        raise SystemExit(
            f"ABORT: GPU still busy after {wait_s}s (mem={t.get('memory.used')} MiB, "
            f"util={t.get('utilization.gpu')}%); timed runs require an otherwise-idle GPU")
    print(f"WARN: GPU at {t.get('temperature.gpu')}C after {wait_s}s, above the {max_temp_c}C "
          f"preflight target; proceeding and recording it", flush=True)
    return False
