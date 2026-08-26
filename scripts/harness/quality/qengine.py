"""vLLM lifecycle for the quality arm: one engine per configuration, in its own process.

Serving-specific launch code is deliberately not reused -- `server.VllmServer` hardcodes
`--no-enable-prefix-caching`, which the quality rig inverts on purpose (H7). The process/VRAM
lessons from the sweep are reused by importing `harness.server`, which has no import side effects.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common, orchestration as orch, server  # noqa: E402
from harness.quality import kl_math as K, positions as P, qcommon as q  # noqa: E402

KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
# vLLM prefixes log lines with a pid and a timestamp; hashing them raw made two identical engines
# hash differently, which would defeat resume and misreport mixed engine identity
LOG_PREFIX_RE = re.compile(r"^\(\S+ pid=\d+\)\s*|^(INFO|WARNING|ERROR)\s+[\d-]+\s+[\d:]+\s*")
GRAPH_RE = re.compile(r"Graph capturing finished in", re.I)
CAPTURE_RE = re.compile(r"Capturing CUDA graphs", re.I)


def engine_kwargs(config_id, overrides=None):
    c = dict(q.QUALITY_ENGINE_CONTROLS)
    c.update(overrides or {})
    if c.get("enforce_eager") is None:
        raise SystemExit("ABORT: enforce_eager is undecided; the engine-profile gate (G9) fixes it")
    if c["max_model_len"] < P.max_context_len():
        raise SystemExit(
            f"ABORT: max_model_len {c['max_model_len']} < longest retained context "
            f"{P.max_context_len()}")
    return {
        "model": q.QUALITY_CONFIGS[config_id]["model"],
        "max_model_len": c["max_model_len"],
        "gpu_memory_utilization": c["gpu_memory_utilization"],
        "max_num_seqs": c["max_num_seqs"],
        "kv_cache_dtype": c["kv_cache_dtype"],
        "max_logprobs": c["max_logprobs"],
        "enable_prefix_caching": c["enable_prefix_caching"],
        "max_num_batched_tokens": c["max_num_batched_tokens"],
        "enforce_eager": c["enforce_eager"],
        "seed": c["seed"],
        "dtype": c["dtype"],
        # LLM() defaults this to True, which makes get_metrics() report nothing; the prefix-cache
        # counters are how cache reuse is observed rather than assumed
        "disable_log_stats": False,
    }, c


def _strip_log_prefix(line):
    prev = None
    while prev != line:
        prev = line
        line = LOG_PREFIX_RE.sub("", line).strip()
    return re.sub(r"^\[[^\]]+\]\s*", "", line)


def observed_identity(log_text, resolved, config_id):
    """Engine identity from what the engine actually did, not from what it was asked for.

    H10 is the precedent: a contaminated launch served correctly while silently sizing a smaller
    cache, so a requested-flag hash is not an identity.
    """
    kv = KV_RE.search(log_text)
    kernel_lines = sorted({ln.strip() for ln in log_text.splitlines()
                           if any(p in ln for p in server.KERNEL_PATTERNS)})[:20]
    normalized_kernel_lines = sorted({_strip_log_prefix(ln) for ln in kernel_lines})
    obs = {
        "resolved_config": resolved,
        "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "graph_capture_observed": bool(GRAPH_RE.search(log_text) or CAPTURE_RE.search(log_text)),
        "kernel_lines": kernel_lines,
        "normalized_kernel_lines": normalized_kernel_lines,
    }
    cfg = q.QUALITY_CONFIGS[config_id]
    blob = " | ".join(kernel_lines)
    want = cfg["expected_kernel_pattern"]
    bad = [p for p in cfg["forbidden_kernel_patterns"] if p in blob]
    obs["dispatch_verdict"] = {
        "expected_pattern": want,
        "expected_present": (want in blob) if want else None,
        "forbidden_present": bad,
        "ok": (bad == []) and (want is None or want in blob),
    }
    obs["engine_identity_hash"] = common.sha256_of_json({
        "configuration_id": config_id,
        "resolved_config": resolved,
        "kv_cache_tokens": obs["kv_cache_tokens"],
        "graph_capture_observed": obs["graph_capture_observed"],
        "kernel_lines": normalized_kernel_lines,
    })[:16]
    return obs


def _release_gpu(pgid):
    """Poll, then escalate, then re-poll -- the full sequence from VllmServer.stop().

    wait_for_gpu_release alone kills nothing; a stalled release is how an EngineCore once held
    22 GiB and how a later launch silently sized a 10%-small cache (H10).
    """
    if server.wait_for_gpu_release(timeout=300):
        return {"released": True, "escalated": []}
    killed = server.kill_gpu_holders(pgid)
    ok = server.wait_for_gpu_release(timeout=120)
    return {"released": bool(ok), "escalated": killed}


def run_job(job, log_path, allow_dirty=False, require_cool=True, timeout=3600):
    """Parent side: preflight, launch a child engine, harvest observed state, release VRAM."""
    q.require_clean_tree(allow_dirty, stage=f"engine:{job['config_id']}:{job['task']}")
    orch.preflight(require_cool=require_cool)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    job_path = log_path + ".job.json"
    common.write_json(job_path, job)

    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--child", job_path],
            stdout=fh, stderr=subprocess.STDOUT, env=server.CHILD_ENV, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            rc = -signal.SIGKILL
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    release = _release_gpu(pgid)

    log_text = open(log_path, errors="replace").read()
    meta_path = job["out_json"]
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    obs = observed_identity(log_text, meta.get("resolved_config", {}), job["config_id"])
    meta.update({
        "returncode": rc,
        "wall_seconds": round(time.time() - t0, 2),
        "observed": obs,
        "vram_release": release,
        "log_path": log_path,
    })
    if rc != 0:
        meta["error_tail"] = log_text[-3000:]
    common.write_json(meta_path, meta)
    if rc != 0:
        raise SystemExit(f"ABORT: engine job failed rc={rc}; see {log_path}")
    if not obs["dispatch_verdict"]["ok"]:
        raise SystemExit(f"ABORT: dispatch verification failed: {obs['dispatch_verdict']}")
    return meta


def _child():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", required=True)
    a = ap.parse_args()
    job = json.load(open(a.child))

    from vllm import LLM, SamplingParams
    kwargs, controls = engine_kwargs(job["config_id"], job.get("engine_overrides"))
    llm = LLM(**kwargs)

    vc = llm.llm_engine.vllm_config
    resolved = {
        "enforce_eager": bool(vc.model_config.enforce_eager),
        "max_num_batched_tokens": int(vc.scheduler_config.max_num_batched_tokens),
        "enable_prefix_caching": bool(vc.cache_config.enable_prefix_caching),
        "max_model_len": int(vc.model_config.max_model_len),
        "num_gpu_blocks": getattr(vc.cache_config, "num_gpu_blocks", None),
        "block_size": getattr(vc.cache_config, "block_size", None),
        "kv_cache_dtype": str(vc.cache_config.cache_dtype),
        "max_logprobs": int(vc.model_config.max_logprobs),
        "dtype": str(vc.model_config.dtype),
        "seed": getattr(vc.model_config, "seed", None),
        "requested": kwargs,
    }
    meta = {"config_id": job["config_id"], "task": job["task"], "resolved_config": resolved,
            "controls": controls}

    if job["task"] == "score":
        contexts = job["contexts"]
        # one group per trajectory, ascending length, so the shorter prefix's blocks are committed
        # before the longer prefix asks for them
        group = int(job.get("group_size") or len(contexts))
        sp = SamplingParams(**q.SCORING_SAMPLING)
        V = q.VOCAB_SIZE
        mat = np.full((len(contexts), V), -np.inf, dtype=np.float32)
        per, t0 = [], time.time()
        for start in range(0, len(contexts), group):
            chunk = contexts[start:start + group]
            if [len(c) for c in chunk] != sorted(len(c) for c in chunk):
                raise SystemExit("ABORT: scoring group is not in ascending length order")
            outs = llm.generate([{"prompt_token_ids": c} for c in chunk], sp)
            for j, o in enumerate(outs):
                i = start + j
                # vLLM sorts by request id and ids are assigned in submission order, so this holds
                # today; asserting it turns an inherited guarantee into a checked one
                if list(o.prompt_token_ids) != list(chunk[j]):
                    raise SystemExit(f"ABORT: output {j} does not correspond to its request")
                lp = o.outputs[0].logprobs[0]
                for tid, obj in lp.items():
                    mat[i, tid] = obj.logprob
                row = K.validate_distribution(mat[i], V, q.GATES["prob_mass_tolerance"])
                row["entries_returned"] = len(lp)
                row["context_len"] = len(contexts[i])
                row["top1"] = K.top1(mat[i])
                per.append(row)
        meta["generate_seconds"] = round(time.time() - t0, 2)
        meta["group_size"] = group
        np.save(job["out_npy"], mat)
        meta["per_context"] = per
        meta["storage_dtype"] = str(mat.dtype)
    elif job["task"] == "generate":
        sp = SamplingParams(**q.GENERATION_SAMPLING)
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": p} for p in job["prompts"]], sp)
        meta["generate_seconds"] = round(time.time() - t0, 2)
        # token IDs straight from the engine: decode/re-encode could change the sequence at the
        # prompt/continuation boundary
        for o, sent in zip(outs, job["prompts"]):
            # a silent prompt/continuation mispairing here would corrupt the frozen trajectory
            # artifact permanently, with nothing downstream able to detect it
            if list(o.prompt_token_ids) != list(sent):
                raise SystemExit("ABORT: generation output does not correspond to its prompt")
        meta["continuations"] = [list(o.outputs[0].token_ids) for o in outs]
        meta["continuation_lengths"] = [len(c) for c in meta["continuations"]]
    else:
        raise SystemExit(f"ABORT: unknown task {job['task']!r}")

    try:
        want = ("prefix_cache", "preemption", "prompt_tokens")
        got = {}
        for m in llm.get_metrics():
            if not any(w in m.name for w in want):
                continue
            v = getattr(m, "value", None)
            if v is None and hasattr(m, "count"):
                v = {"count": m.count, "sum": getattr(m, "sum", None)}
            got[m.name] = v
        meta["engine_metrics"] = got
    except Exception as exc:  # noqa: BLE001
        meta["engine_metrics"] = {"error": str(exc)}

    common.write_json(job["out_json"], meta)
    print("CHILD_OK", job["out_json"], flush=True)


if __name__ == "__main__":
    if "--child" in sys.argv:
        _child()
    else:
        raise SystemExit("qengine is driven by gates/collectors; --child is the engine side")
