"""Pre-sweep correctness gate: the D13 next-token-distribution path on real held-out contexts.

A sanity trip-wire for a broken checkpoint or an unintended execution path, not a quality
result. Runs before the sweep because timing a corrupted checkpoint wastes the most
expensive resource in the project.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import common  # noqa: E402

# Deliberately loose "obviously broken" trip-wires, calibrated ~2 orders of magnitude above the
# qualification feasibility values (results/qualification/kl_feasibility.json). Not quality bounds.
GATE = {
    "self_kl_median_max_nats": 1e-6,
    "self_kl_max_nats": 1e-4,
    "fp8_kl_median_max_nats": 0.1,
    "fp4_kl_median_max_nats": 0.5,
    "fp8_top1_agreement_min": 0.90,
    "fp4_top1_agreement_min": 0.75,
    "prob_mass_tolerance": 1e-3,
}
EPS = 1e-12


def collect(config_id, contexts, out_dir, reverse=False, label_suffix=""):
    from vllm import LLM, SamplingParams
    cfg = common.CONFIGS[config_id]
    llm = LLM(model=cfg["model"], max_model_len=4096, gpu_memory_utilization=0.85,
              max_logprobs=128256, enable_prefix_caching=True, enforce_eager=True,
              seed=common.SERVER_CONTROLS["seed"])
    tok = llm.get_tokenizer()
    V = len(tok)
    order = list(range(len(contexts)))[::-1] if reverse else list(range(len(contexts)))
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=128256)
    outs = llm.generate([{"prompt_token_ids": contexts[i]} for i in order], sp)

    mat = np.full((len(contexts), V), -np.inf, dtype=np.float32)
    covered, finite = [], []
    for slot, o in zip(order, outs):
        lp = o.outputs[0].logprobs[0]
        for tid, obj in lp.items():
            mat[slot, tid] = obj.logprob
        covered.append(len(lp))
        finite.append(bool(np.isfinite([v.logprob for v in lp.values()]).all()))
    p = np.exp(mat.astype(np.float64))
    label = cfg["short"] + label_suffix
    np.save(os.path.join(out_dir, f"{label}_logprobs.npy"), mat.astype(np.float16))
    meta = {
        "label": label, "configuration_id": config_id, "model": cfg["model"],
        "vocab_size": V, "n_contexts": len(contexts),
        "batch_order_reversed": reverse,
        "full_vocab_returned": all(c >= V for c in covered),
        "min_entries_returned": min(covered), "all_logprobs_finite": all(finite),
        "prob_mass_sum_per_context": [round(float(x), 6) for x in p.sum(axis=1)],
        "prob_mass_normalized": bool(
            np.all(np.abs(p.sum(axis=1) - 1.0) <= GATE["prob_mass_tolerance"])),
    }
    del llm
    return mat, meta


def kl(pa, pb):
    return (pa * (np.log(pa + EPS) - np.log(pb + EPS))).sum(axis=1)


def norm(mat):
    p = np.exp(mat.astype(np.float64))
    return p / p.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-contexts", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(common.PILOT_DIR, "correctness_gate.json"))
    ap.add_argument("--logits-dir", default=os.path.join(common.PILOT_DIR, "gate_logits"))
    a = ap.parse_args()

    stem = os.path.join(common.CORPUS_DIR, "decode_primary_512tok")
    body = json.load(open(stem + ".json"))
    manifest = json.load(open(stem + "_manifest.json"))
    contexts = [p["token_ids"] for p in body["prompts"][:a.n_contexts]]
    ctx_hash = common.sha256_of_json(contexts)
    os.makedirs(a.logits_dir, exist_ok=True)

    import subprocess
    mats, metas = {}, {}
    for config_id, rev, suf in (("BF16_REFERENCE", False, ""), ("BF16_REFERENCE", True, "_selfcheck"),
                                ("FP8_PRIMARY", False, ""), ("FP4_PRIMARY", False, "")):
        # one engine per pass in a subprocess, so VRAM is fully released between configurations
        script = os.path.abspath(__file__)
        tmp = os.path.join(a.logits_dir, f"_pass_{common.CONFIGS[config_id]['short']}{suf}.json")
        rc = subprocess.run(
            [sys.executable, script, "--single-pass", config_id, str(a.n_contexts),
             a.logits_dir, tmp, "1" if rev else "0", suf],
            env=dict(os.environ), capture_output=True, text=True)
        with open(os.path.join(a.logits_dir,
                               f"_pass_{common.CONFIGS[config_id]['short']}{suf}.log"), "w") as fh:
            fh.write(rc.stdout + "\n===STDERR===\n" + rc.stderr)
        if not os.path.exists(tmp):
            raise SystemExit(f"ABORT: pass {config_id}{suf} failed\n{rc.stdout[-3000:]}\n{rc.stderr[-3000:]}")
        metas[common.CONFIGS[config_id]["short"] + suf] = json.load(open(tmp))
        mats[common.CONFIGS[config_id]["short"] + suf] = np.load(
            os.path.join(a.logits_dir,
                         f"{common.CONFIGS[config_id]['short']}{suf}_logprobs.npy"))

    P = {k: norm(v) for k, v in mats.items()}
    ref_top1 = P["BF16"].argmax(1)
    pairs = {}
    for name, other in (("BF16||BF16_selfcheck", "BF16_selfcheck"),
                        ("BF16||FP8", "FP8"), ("BF16||FP4", "FP4")):
        d = kl(P["BF16"], P[other])
        pairs[name] = {
            "median_nats": float(np.median(d)), "mean_nats": float(d.mean()),
            "max_nats": float(d.max()), "min_nats": float(d.min()),
            "p90_nats": float(np.quantile(d, 0.9)),
            "all_finite": bool(np.isfinite(d).all()),
            "all_nonnegative": bool((d >= -1e-12).all()),
            "top1_agreement_vs_bf16": float((P[other].argmax(1) == ref_top1).mean()),
            "per_context_nats": [float(v) for v in d],
        }

    checks = {
        "all_configs_loaded": len(metas) == 4,
        "dispatch_reconfirmed": None,
        "full_vocab_all_passes": all(m["full_vocab_returned"] for m in metas.values()),
        "distributions_finite": all(m["all_logprobs_finite"] for m in metas.values()),
        "distributions_normalized": all(m["prob_mass_normalized"] for m in metas.values()),
        "bf16_self_kl_near_zero": (
            pairs["BF16||BF16_selfcheck"]["median_nats"] <= GATE["self_kl_median_max_nats"]
            and pairs["BF16||BF16_selfcheck"]["max_nats"] <= GATE["self_kl_max_nats"]),
        "fp8_kl_finite_nonnegative": pairs["BF16||FP8"]["all_finite"] and pairs["BF16||FP8"]["all_nonnegative"],
        "fp4_kl_finite_nonnegative": pairs["BF16||FP4"]["all_finite"] and pairs["BF16||FP4"]["all_nonnegative"],
        "fp8_not_pathological": pairs["BF16||FP8"]["median_nats"] <= GATE["fp8_kl_median_max_nats"],
        "fp4_not_pathological": pairs["BF16||FP4"]["median_nats"] <= GATE["fp4_kl_median_max_nats"],
        "fp8_top1_agreement_ok": pairs["BF16||FP8"]["top1_agreement_vs_bf16"] >= GATE["fp8_top1_agreement_min"],
        "fp4_top1_agreement_ok": pairs["BF16||FP4"]["top1_agreement_vs_bf16"] >= GATE["fp4_top1_agreement_min"],
    }
    dispatch = {}
    for config_id in ("BF16_REFERENCE", "FP8_PRIMARY", "FP4_PRIMARY"):
        short = common.CONFIGS[config_id]["short"]
        log = os.path.join(a.logits_dir, f"_pass_{short}.log")
        blob = open(log, errors="replace").read() if os.path.exists(log) else ""
        want = common.CONFIGS[config_id]["expected_kernel_pattern"]
        bad = [p for p in common.CONFIGS[config_id]["forbidden_kernel_patterns"] if p in blob]
        dispatch[config_id] = {
            "expected_pattern": want,
            "expected_present": (want in blob) if want else None,
            "forbidden_present": bad,
            "ok": (bad == []) and (want is None or want in blob),
            "log_available": bool(blob),
        }
    checks["dispatch_reconfirmed"] = all(v["ok"] for v in dispatch.values())

    rec = {
        "job": "correctness_gate",
        "purpose": "sanity trip-wire before timing; NOT a quality result",
        "n_contexts": a.n_contexts,
        "context_tokens": manifest["tokens_per_prompt"],
        "corpus_version": manifest["corpus_version"],
        "prompt_set_hash": manifest["prompt_set_hash"],
        "context_hash": ctx_hash,
        "context_token_identity": True,
        "position": "1 (next-token after the 512-token context)",
        "prefix_caching": "ENABLED (quality-rig exception, D13/H7)",
        "pre_registered_thresholds": GATE,
        "per_pass_meta": metas,
        "dispatch": dispatch,
        "pairs": pairs,
        "checks": checks,
        "gate_status": "CLEAN" if all(v for v in checks.values()) else "FAILED",
        "failed_checks": [k for k, v in checks.items() if not v],
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
        "not_a_quality_result": True,
    }
    common.write_json(a.out, rec)
    print(json.dumps({"gate_status": rec["gate_status"], "failed_checks": rec["failed_checks"],
                      "checks": checks,
                      "pairs": {k: {kk: vv for kk, vv in v.items() if kk != "per_context_nats"}
                                for k, v in pairs.items()}}, indent=2))
    print("WROTE", a.out)


def single_pass():
    config_id, n, logits_dir, out, rev, suf = sys.argv[2:8]
    stem = os.path.join(common.CORPUS_DIR, "decode_primary_512tok")
    body = json.load(open(stem + ".json"))
    contexts = [p["token_ids"] for p in body["prompts"][:int(n)]]
    _, meta = collect(config_id, contexts, logits_dir, reverse=(rev == "1"), label_suffix=suf)
    common.write_json(out, meta)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single-pass":
        single_pass()
    else:
        main()
