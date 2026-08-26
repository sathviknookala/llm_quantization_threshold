"""Pre-registered gates for the quality arm.

Each gate answers one question with an explicit pass/fail and writes a tracked artifact. Empirical
gates (replication floor, cache equivalence, precision, engine profile) are measurements about the
rig and are reported as such; hard invariants abort.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import kl_math as K, positions as P, qcommon as q  # noqa: E402
from harness.quality import qengine as E  # noqa: E402

LEGACY_EPS = 1e-12
GATE_LOGITS = os.path.join(common.PILOT_DIR, "gate_logits")


def _legacy_probs(mat):
    p = np.exp(np.asarray(mat, dtype=np.float64))
    return p / p.sum(axis=1, keepdims=True)


def _legacy_kl(pa, pb):
    return (pa * (np.log(pa + LEGACY_EPS) - np.log(pb + LEGACY_EPS))).sum(axis=1)


def _new_kl_rows(ma, mb):
    return np.array([K.kl_nats(ma[i], mb[i]) for i in range(ma.shape[0])])


def numerics_compat(logits_dir=GATE_LOGITS, reference="results/pilot/correctness_gate.json"):
    """G7 -- the new numerics against the historical gate, with the EPS effect isolated.

    The only reference data available is fp16, while production stores fp32, so old-vs-new would
    otherwise confound the EPS change with a storage change. Both formulas are therefore run over
    the SAME fp16 arrays; any residual beyond the isolated EPS delta is a real numerics change.
    """
    mats = {}
    for label in ("BF16", "BF16_selfcheck", "FP8", "FP4"):
        path = os.path.join(logits_dir, f"{label}_logprobs.npy")
        if not os.path.exists(path):
            return {"gate": "numerics_compat", "status": "UNAVAILABLE",
                    "reason": f"missing {path}", "passed": None}
        mats[label] = np.load(path)

    tracked = json.load(open(reference)) if os.path.exists(reference) else {}
    tracked_pairs = tracked.get("pairs", {})

    probs = {k: _legacy_probs(v) for k, v in mats.items()}
    pairs, worst_rel, worst_abs = {}, 0.0, 0.0
    for name, a, b in (("BF16||BF16_selfcheck", "BF16", "BF16_selfcheck"),
                       ("BF16||FP8", "BF16", "FP8"),
                       ("BF16||FP4", "BF16", "FP4")):
        old = _legacy_kl(probs[a], probs[b])
        new = _new_kl_rows(mats[a], mats[b])
        delta = new - old
        rel = np.abs(delta) / np.maximum(old, 1e-30)
        ref_vals = tracked_pairs.get(name, {}).get("per_context_nats")
        repro = None
        if ref_vals is not None:
            repro = float(np.abs(np.array(ref_vals) - old).max())
        pairs[name] = {
            "contexts": int(old.size),
            "old_mean_nats": float(old.mean()),
            "new_mean_nats": float(new.mean()),
            "mean_abs_delta_nats": float(np.abs(delta).mean()),
            "max_abs_delta_nats": float(np.abs(delta).max()),
            "rel_delta_p99": float(np.quantile(rel, 0.99)),
            "rel_delta_max": float(rel.max()),
            "legacy_reproduces_tracked_artifact_max_abs": repro,
        }
        if name != "BF16||BF16_selfcheck":
            worst_rel = max(worst_rel, float(rel.max()))
            worst_abs = max(worst_abs, float(np.abs(delta).mean()))

    self_new = _new_kl_rows(mats["BF16"], mats["BF16_selfcheck"])
    repro_ok = all(v["legacy_reproduces_tracked_artifact_max_abs"] is not None
                   and v["legacy_reproduces_tracked_artifact_max_abs"] <= 1e-12
                   for v in pairs.values())
    # G7 bounds the EPS effect via the max; the p99 threshold belongs to G4, which compares
    # storage representations rather than formulas
    checks = {
        "legacy_formula_reproduces_tracked_artifact": repro_ok,
        "eps_isolated_mean_abs_within_tolerance":
            worst_abs <= q.GATES["numerics_mean_abs_max_nats"],
        "eps_isolated_rel_within_tolerance": worst_rel <= q.GATES["numerics_rel_max"],
        "self_kl_still_exactly_zero_under_new_formula": bool((self_new == 0.0).all()),
    }
    return {
        "gate": "numerics_compat",
        "purpose": "G7: new float64/no-EPS numerics vs the historical EPS-floored formula, "
                   "measured on identical fp16 arrays so the EPS effect is isolated",
        "storage_of_reference_arrays": "float16",
        "production_storage_dtype": q.STORAGE_DTYPE,
        "thresholds": {"mean_abs_max_nats": q.GATES["numerics_mean_abs_max_nats"],
                       "rel_max": q.GATES["numerics_rel_max"]},
        "pairs": pairs,
        "checks": checks,
        "passed": all(checks.values()),
        "caveat": "the reference artifact was produced at git_head "
                  f"{tracked.get('software', {}).get('git_head', '?')[:12]} with git_dirty="
                  f"{tracked.get('software', {}).get('git_dirty')}; it is not independently "
                  "reproducible and is cited as historical evidence only",
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }




# G9 profiles. B is the candidate production profile; B2 repeats it under an independent launch so
# the replication floor is measured in the same pass that measures the profile differences.
PROFILES = {
    "eager_2048":  {"enforce_eager": True,  "max_num_batched_tokens": 2048},
    "graph_2048":  {"enforce_eager": False, "max_num_batched_tokens": 2048},
    "graph_2048_r2": {"enforce_eager": False, "max_num_batched_tokens": 2048},
    "eager_2048_r2": {"enforce_eager": True, "max_num_batched_tokens": 2048},
    "graph_8192":  {"enforce_eager": False, "max_num_batched_tokens": 8192},
}
# each profile carries its OWN floor: a difference is only a difference relative to the
# reproducibility of the profile it was measured under
PROFILE_COMPARISONS = (
    ("replication_floor_graph", "graph_2048", "graph_2048_r2",
     "same graph profile, independent launches -- the noise floor under CUDA graphs"),
    ("replication_floor_eager", "eager_2048", "eager_2048_r2",
     "same eager profile, independent launches -- the noise floor under eager"),
    ("eager_vs_graph", "eager_2048", "graph_2048",
     "only enforce_eager flipped"),
    ("chunked_vs_unchunked", "graph_2048", "graph_8192",
     "serving-compatible 2048 vs the offline default 8192"),
)


def _provisional_trajectories(out_dir, n_traj, allow_dirty):
    """Throwaway continuations so the profile can be settled before production trajectories exist.

    Deliberately not written to the production artifact: freezing is gated on this gate's result.
    """
    path = os.path.join(out_dir, "provisional_trajectories.json")
    prompts, manifest = q.load_prompts(n=n_traj)
    want = {"prompt_set_hash": manifest["prompt_set_hash"],
            "subset_hash": manifest["subset_hash"],
            "n_trajectories": n_traj,
            "generation": dict(q.GENERATION_SAMPLING),
            "kl_spec_hash": q.spec_hash()}
    if os.path.exists(path):
        prior = json.load(open(path))
        got = {k: prior.get(k) for k in want}
        if got != want:
            raise SystemExit(
                f"ABORT: {path} was built under a different corpus/policy/spec and must not be "
                f"reused.\n  on disk: {got}\n  active : {want}\nMove it aside to rebuild.")
        return prior
    job = {
        "config_id": q.REFERENCE_CONFIG, "task": "generate",
        "prompts": [p["token_ids"] for p in prompts],
        "out_json": os.path.join(out_dir, "_provisional_gen.json"),
        "out_npy": os.path.join(out_dir, "_unused.npy"),
        "engine_overrides": PROFILES["graph_2048"],
    }
    meta = E.run_job(job, os.path.join(out_dir, "logs", "provisional_generate.log"),
                     allow_dirty=allow_dirty)
    bad = [i for i, n in enumerate(meta["continuation_lengths"]) if n != q.GENERATION_TOKENS]
    if bad:
        raise SystemExit(f"ABORT: provisional trajectories {bad} are not exactly "
                         f"{q.GENERATION_TOKENS} tokens: {meta['continuation_lengths']}")
    rec = {
        "PROVISIONAL": True,
        "purpose": "engine-profile gate only; NOT the production trajectory artifact",
        "not_a_production_trajectory_artifact": True,
        "corpus_version": manifest["corpus_version"],
        **want,
        "engine_profile": PROFILES["graph_2048"],
        "observed": meta.get("observed"),
        "trajectories": [
            {"trajectory_index": i, "prompt_index": p["index"],
             "prompt_sha256": q.prompt_hash(p["token_ids"]),
             "prompt_token_ids": p["token_ids"],
             "continuation_token_ids": c,
             "continuation_sha256": q.prompt_hash(c)}
            for i, (p, c) in enumerate(zip(prompts, meta["continuations"]))],
    }
    common.write_json(path, rec)
    return rec


def _contexts_for(traj):
    """Flattened in (trajectory, ascending position) order; groups of 10 match production."""
    ctx, index = [], []
    for t in traj["trajectories"]:
        for cell in P.build_all(t["prompt_token_ids"], t["continuation_token_ids"]):
            ctx.append(cell["context_ids"])
            index.append({"trajectory_index": t["trajectory_index"],
                          "position_p": cell["position_p"],
                          "context_len": cell["context_len"],
                          "target_token_id": cell["target_token_id"]})
    return ctx, index


def _compare(a_mat, b_mat, index, n_positions):
    """Per-cell KL between two engine profiles, aggregated with the trajectory as the unit."""
    vals = np.array([K.kl_nats(a_mat[i], b_mat[i]) for i in range(a_mat.shape[0])])
    identical = int(sum(bool(np.array_equal(a_mat[i], b_mat[i])) for i in range(a_mat.shape[0])))
    top1_same = int(sum(int(K.top1(a_mat[i]) == K.top1(b_mat[i])) for i in range(a_mat.shape[0])))
    grid = vals.reshape(-1, n_positions)
    by_pos = {}
    for j, p in enumerate(P.RETAINED_POSITIONS):
        by_pos[str(p)] = {"mean_nats": float(grid[:, j].mean()),
                          "max_nats": float(grid[:, j].max())}
    return {
        "cells": int(vals.size),
        "bit_identical_cells": identical,
        "top1_agreement": top1_same / float(vals.size),
        "max_nats": float(vals.max()),
        "mean_nats": float(vals.mean()),
        "median_nats": float(np.median(vals)),
        "p99_nats": float(np.quantile(vals, 0.99)),
        "trajectory_means": [float(x) for x in grid.mean(axis=1)],
        "headline_nats": float(K.headline(grid)),
        "by_position": by_pos,
        "worst_cell": {**index[int(np.argmax(vals))], "kl_nats": float(vals.max())},
    }


def engine_profile(out_dir=None, configs=("BF16_REFERENCE", "FP4_PRIMARY"), n_traj=4,
                   allow_dirty=False):
    """G9 -- decide enforce_eager and the batching knob by measurement, and measure the floor.

    Runs before production trajectories are frozen: generation and scoring must share one profile.
    """
    out_dir = out_dir or os.path.join(q.QUALITY_DIR, "gates", "engine_profile")
    os.makedirs(out_dir, exist_ok=True)
    q.guard_manifest(out_dir, "G9 engine-profile gate")
    traj = _provisional_trajectories(out_dir, n_traj, allow_dirty)
    contexts, index = _contexts_for(traj)
    n_pos = len(P.RETAINED_POSITIONS)
    contexts_hash = common.sha256_of_json(contexts)
    expected_cells = n_traj * n_pos
    if len(contexts) != expected_cells:
        raise SystemExit(f"ABORT: built {len(contexts)} contexts, expected {expected_cells}")

    mats, metas = {}, {}
    for config_id in configs:
        for pname, overrides in PROFILES.items():
            key = f"{config_id}:{pname}"
            npy = os.path.join(out_dir, f"{q.QUALITY_CONFIGS[config_id]['short']}_{pname}.npy")
            js = npy.replace(".npy", ".json")
            reusable = os.path.exists(npy) and os.path.exists(js)
            if reusable:
                prior = json.load(open(js))
                if (prior.get("kl_spec_hash") != q.spec_hash()
                        or prior.get("contexts_hash") != contexts_hash):
                    raise SystemExit(
                        f"ABORT: {npy} was produced under a different spec or context set; "
                        "move results/quality/gates/engine_profile aside rather than mixing.")
            if not reusable:
                job = {"config_id": config_id, "task": "score", "contexts": contexts,
                       "group_size": n_pos, "out_npy": npy, "out_json": js,
                       "engine_overrides": overrides}
                meta = E.run_job(job, os.path.join(out_dir, "logs", f"{config_id}_{pname}.log"),
                                 allow_dirty=allow_dirty)
                meta["kl_spec_hash"] = q.spec_hash()
                meta["contexts_hash"] = contexts_hash
                meta["config_identity"] = q.config_identity(config_id)
                common.write_json(js, meta)
            mats[key] = np.load(npy)
            metas[key] = json.load(open(js))
            if mats[key].shape != (len(contexts), q.VOCAB_SIZE):
                raise SystemExit(
                    f"ABORT: {npy} has shape {mats[key].shape}, expected "
                    f"{(len(contexts), q.VOCAB_SIZE)}")
            if len(metas[key].get("per_context", [])) != len(contexts):
                raise SystemExit(
                    f"ABORT: {js} holds {len(metas[key].get('per_context', []))} validated "
                    f"contexts, expected {len(contexts)}")

    results, checks = {}, {}
    for config_id in configs:
        short = q.QUALITY_CONFIGS[config_id]["short"]
        per_cfg = {}
        for name, a, b, why in PROFILE_COMPARISONS:
            per_cfg[name] = {"why": why,
                             **_compare(mats[f"{config_id}:{a}"], mats[f"{config_id}:{b}"],
                                        index, n_pos)}
        floors = {"eager_vs_graph": max(per_cfg["replication_floor_graph"]["max_nats"],
                                        per_cfg["replication_floor_eager"]["max_nats"]),
                  "chunked_vs_unchunked": per_cfg["replication_floor_graph"]["max_nats"]}
        for name, floor in floors.items():
            m = per_cfg[name]["max_nats"]
            per_cfg[name]["replication_floor_used_nats"] = floor
            per_cfg[name]["ratio_to_replication_floor"] = (None if floor == 0.0 else m / floor)
            per_cfg[name]["above_replication_floor"] = bool(m > floor)
        results[short] = per_cfg
        checks[f"{short}_cell_count_complete"] = all(
            per_cfg[name]["cells"] == expected_cells for name, _, _, _ in PROFILE_COMPARISONS)
        checks[f"{short}_all_cells_valid"] = all(
            c["full_vocab"] and c["all_finite"] and c["normalized"]
            for k, mm in metas.items() if k.startswith(config_id)
            for c in mm["per_context"])
        checks[f"{short}_dispatch_ok"] = all(
            metas[k]["observed"]["dispatch_verdict"]["ok"]
            for k in metas if k.startswith(config_id))
        checks[f"{short}_eager_flag_observed"] = (
            metas[f"{config_id}:eager_2048"]["resolved_config"]["enforce_eager"] is True
            and metas[f"{config_id}:graph_2048"]["resolved_config"]["enforce_eager"] is False)
        checks[f"{short}_batching_observed"] = (
            metas[f"{config_id}:graph_2048"]["resolved_config"]["max_num_batched_tokens"] == 2048
            and metas[f"{config_id}:graph_8192"]["resolved_config"]["max_num_batched_tokens"] == 8192)
        checks[f"{short}_prefix_caching_observed"] = all(
            metas[k]["resolved_config"]["enable_prefix_caching"] is True
            for k in metas if k.startswith(config_id))
        checks[f"{short}_detokenize_pinned_off"] = q.SCORING_SAMPLING["detokenize"] is False

    return {
        "gate": "engine_profile",
        "purpose": "G9: decide enforce_eager and max_num_batched_tokens by measurement, and "
                   "measure the BF16 replication floor in the same pass",
        "configs": list(configs),
        "profiles": PROFILES,
        "n_trajectories": n_traj,
        "expected_cells_per_comparison": expected_cells,
        "cells_per_comparison": len(contexts),
        "contexts_hash": contexts_hash,
        "config_identity": {c: q.config_identity(c) for c in configs},
        "kl_spec_hash": q.spec_hash(),
        "git": q.git_state(),
        "provisional_trajectories": {
            "PROVISIONAL": True,
            "prompt_set_hash": traj["prompt_set_hash"],
            "continuation_sha256": [t["continuation_sha256"][:16] for t in traj["trajectories"]],
        },
        "results": results,
        "observed_engine_identity": {
            k: {"engine_identity_hash": v["observed"]["engine_identity_hash"],
                "kv_cache_tokens": v["observed"]["kv_cache_tokens"],
                "graph_capture_observed": v["observed"]["graph_capture_observed"],
                "resolved": {kk: v["resolved_config"].get(kk) for kk in
                             ("enforce_eager", "max_num_batched_tokens",
                              "enable_prefix_caching", "num_gpu_blocks")},
                "generate_seconds": v.get("generate_seconds")}
            for k, v in metas.items()},
        "checks": checks,
        "passed": all(checks.values()),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["numerics", "engine-profile"])
    ap.add_argument("--out", default="")
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--configs", default="BF16_REFERENCE,FP4_PRIMARY")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if a.gate == "numerics":
        rec = numerics_compat()
    else:
        rec = engine_profile(configs=tuple(c for c in a.configs.split(",") if c),
                             n_traj=a.n_traj, allow_dirty=a.allow_dirty)
    out = a.out or os.path.join(q.QUALITY_DIR, "gates", f"{rec['gate']}.json")
    common.write_json(out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("software",)}, indent=2)[:4000])
    print("WROTE", out)
    return 0 if rec.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
