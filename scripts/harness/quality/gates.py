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


def numerics_compat(logits_dir=GATE_LOGITS, reference="results/pilot/correctness_gate.json",
                    allow_dirty=False):
    """G7 -- the new numerics against the historical gate, with the EPS effect isolated.

    The only reference data available is fp16, while production stores fp32, so old-vs-new would
    otherwise confound the EPS change with a storage change. Both formulas are therefore run over
    the SAME fp16 arrays; any residual beyond the isolated EPS delta is a real numerics change.
    """
    git = q.require_clean_tree(allow_dirty, stage="numerics_compat")
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
        "kl_spec_hash": q.spec_hash(),
        "git": git,
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
     "enforce_eager flipped; graph capture also reserves VRAM before KV profiling, so the graph "
     "profile lands slightly fewer KV blocks -- a side effect vLLM does not let us remove"),
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
    vals = np.array([K.kl_nats(a_mat[i], b_mat[i], q.GATES["kl_negative_tolerance"])
                     for i in range(a_mat.shape[0])])
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


def engine_profile(out_dir=None, configs=None, n_traj=4, allow_dirty=False):
    """G9 -- decide enforce_eager and the batching knob by measurement, and measure the floor.

    Runs before production trajectories are frozen: generation and scoring must share one profile.
    """
    configs = tuple(configs or q.LADDER)
    out_dir = out_dir or os.path.join(q.QUALITY_DIR, "gates", "engine_profile")
    os.makedirs(out_dir, exist_ok=True)
    q.require_clean_tree(allow_dirty, stage="engine_profile:summary")
    q.guard_manifest(out_dir, "G9 engine-profile gate")
    traj = _provisional_trajectories(out_dir, n_traj, allow_dirty)
    contexts, index = _contexts_for(traj)
    n_pos = len(P.RETAINED_POSITIONS)
    contexts_hash = common.sha256_of_json(contexts)
    identity = {c: q.config_identity(c) for c in configs}
    expected_cells = n_traj * n_pos
    if len(contexts) != expected_cells:
        raise SystemExit(f"ABORT: built {len(contexts)} contexts, expected {expected_cells}")

    mats, metas = {}, {}
    for config_id in configs:
        for pname, overrides in PROFILES.items():
            key = f"{config_id}:{pname}"
            npy = os.path.join(out_dir, f"{q.QUALITY_CONFIGS[config_id]['short']}_{pname}.npy")
            js = npy.replace(".npy", ".json")
            ckpt = identity[config_id]["checkpoint_content_hash"]
            reusable = os.path.exists(npy) and os.path.exists(js)
            if reusable:
                prior = json.load(open(js))
                if (prior.get("kl_spec_hash") != q.spec_hash()
                        or prior.get("contexts_hash") != contexts_hash
                        or (prior.get("config_identity") or {})
                        .get("checkpoint_content_hash") != ckpt):
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
                meta["config_identity"] = identity[config_id]
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
            fc = K.floor_comparison(per_cfg[name]["max_nats"], floor)
            per_cfg[name]["vs_replication_floor"] = fc
            per_cfg[name]["replication_floor_used_nats"] = floor
            per_cfg[name]["ratio_to_replication_floor"] = fc["ratio_to_floor"]
            per_cfg[name]["above_replication_floor"] = fc["above_replication_floor"]
        results[short] = per_cfg
        checks[f"{short}_no_preemption"] = all(
            (metas[k].get("engine_metrics") or {}).get("vllm:num_preemptions", 0) in (0, None)
            for k in metas if k.startswith(config_id))
        checks[f"{short}_detokenize_observed_off"] = all(
            c.get("decoded_token_is_none") is True
            for k, mm in metas.items() if k.startswith(config_id)
            for c in mm["per_context"])
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
        "config_identity": identity,
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
        "integrity_passed": all(checks.values()),
        # deliberately NOT a single `passed`: every integrity check can hold while eager and graph
        # execution disagree, which is the finding this gate exists to produce
        "profile_equivalence": {
            short: {name: {"above_replication_floor": r[name]["above_replication_floor"],
                           "headline_nats": r[name]["headline_nats"],
                           "max_nats": r[name]["max_nats"],
                           "top1_agreement": r[name]["top1_agreement"]}
                    for name in ("eager_vs_graph", "chunked_vs_unchunked")}
            for short, r in results.items()},
        "profile_decision_required": True,
        "profile_differences_exceeding_floor": sorted(
            f"{short}:{name}" for short, r in results.items()
            for name in ("eager_vs_graph", "chunked_vs_unchunked")
            if r[name]["above_replication_floor"]),
        "freeze_release": "trajectory freezing requires an explicit reviewed profile decision; "
                          "this gate measures, it does not authorise",
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }


GATE_NAMES = ("numerics", "engine-profile", "replayability", "cache-equivalence",
              "replication-floor", "storage-precision")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=list(GATE_NAMES))
    ap.add_argument("--out", default="")
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--configs", default="")
    ap.add_argument("--root", default="", help="collection root for the data-dependent gates")
    ap.add_argument("--floor", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfgs = tuple(c for c in a.configs.split(",") if c)
    if a.gate == "numerics":
        rec = numerics_compat(allow_dirty=a.allow_dirty)
    elif a.gate == "engine-profile":
        rec = engine_profile(configs=cfgs or ("BF16_REFERENCE", "FP4_PRIMARY"),
                             n_traj=a.n_traj, allow_dirty=a.allow_dirty)
    elif a.gate == "replayability":
        rec = replayability(n_traj=None, allow_dirty=a.allow_dirty)
    elif a.gate == "cache-equivalence":
        rec = cache_equivalence(a.root, configs=cfgs or ("BF16_REFERENCE", "FP4_PRIMARY"),
                                n_traj=a.n_traj, allow_dirty=a.allow_dirty)
    elif a.gate == "replication-floor":
        rec = replication_floor(a.root, configs=cfgs or None, n_traj=a.n_traj,
                                allow_dirty=a.allow_dirty)
    else:
        rec = storage_precision(a.root, n_traj=a.n_traj, allow_dirty=a.allow_dirty,
                                floor_path=a.floor or None)
    out = a.out or os.path.join(q.QUALITY_DIR, "gates", f"{rec['gate']}.json")
    if os.path.exists(out) and not a.force:
        prior = json.load(open(out)).get("kl_spec_hash")
        if prior and prior != q.spec_hash():
            raise SystemExit(
                f"ABORT: {out} was written under KL_SPEC {prior} and this run is {q.spec_hash()}; "
                "pass --force only if you mean to discard the earlier report.")
    common.write_json(out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("software",)}, indent=2)[:4000])
    print("WROTE", out)
    if rec.get("verdict_is_informational"):
        return 0
    verdict = rec.get("passed")
    if verdict is None:
        verdict = rec.get("integrity_passed")
    return 0 if verdict else 1



def replayability(out_dir=None, n_traj=None, allow_dirty=False):
    """Is seeded BF16 generation reproducible across independent launches?

    Empirical, not an invariant. If it is not replayable the design still holds: the frozen token
    IDs are the evaluation contexts, and nothing downstream regenerates them.
    """
    from harness.quality import trajectories as T
    out_dir = out_dir or os.path.join(q.QUALITY_DIR, "gates", "replayability")
    os.makedirs(out_dir, exist_ok=True)
    q.require_clean_tree(allow_dirty, stage="replayability")
    frozen = T.load()
    # the whole frozen set by default: a subset would be submitted in a smaller group than the
    # freeze used, and batch composition changes numerics -- that would confound batching with
    # the launch-to-launch nondeterminism this gate exists to measure
    n = n_traj or frozen["n_trajectories"]
    if n != frozen["n_trajectories"]:
        raise SystemExit(
            f"ABORT: replay over {n} of {frozen['n_trajectories']} trajectories would submit "
            f"groups of {min(n, q.GENERATION_GROUP_SIZE)} against the freeze's "
            f"{q.GENERATION_GROUP_SIZE}; the comparison would not isolate launch nondeterminism.")
    prompts = [t["prompt_token_ids"] for t in frozen["trajectories"][:n]]

    replay_json = os.path.join(out_dir, "_gen_replay.json")
    want = {"kl_spec_hash": q.spec_hash(), "engine_profile_name": q.PROFILE_NAME,
            "trajectory_set_hash": frozen["trajectory_set_hash"], "n_trajectories": n}
    if os.path.exists(replay_json):
        meta = json.load(open(replay_json))
        got = {k: meta.get(k) for k in want}
        if got != want:
            raise SystemExit(
                f"ABORT: {replay_json} was generated under a different spec, profile or "
                f"trajectory set.\n  on disk: {got}\n  active : {want}\nMove it aside to rerun.")
    else:
        meta = T.generate(q.REFERENCE_CONFIG, prompts, out_dir, "replay", allow_dirty=allow_dirty)
        meta.update(want)
        common.write_json(replay_json, meta)

    rows, identical = [], 0
    for i, (t, c) in enumerate(zip(frozen["trajectories"][:n], meta["continuations"])):
        a = list(t["continuation_token_ids"])
        same = (a == list(c))
        identical += int(same)
        first = None
        if not same:
            first = next((j for j in range(min(len(a), len(c))) if a[j] != c[j]), min(len(a), len(c)))
        rows.append({"trajectory_index": i, "identical": same, "first_divergence": first,
                     "matching_prefix_tokens": len(a) if same else first,
                     "replay_length": len(c)})
    replayable = identical == n
    return {
        "gate": "replayability",
        "purpose": "P7: does the locked seed and profile reproduce BF16 token IDs across "
                   "independent launches?",
        "kind": "empirical measurement, not a correctness invariant",
        "n_trajectories": n,
        "engine_profile_name": q.PROFILE_NAME,
        "generation": dict(q.GENERATION_SAMPLING),
        "generation_group_size": q.GENERATION_GROUP_SIZE,
        "trajectory_set_hash": frozen["trajectory_set_hash"],
        "frozen_engine_identity": (frozen.get("observed") or {}).get("engine_identity_hash"),
        "replay_engine_identity": (meta.get("observed") or {}).get("engine_identity_hash"),
        "identical_trajectories": identical,
        "replayable": replayable,
        "passed": None,
        "verdict_is_informational": True,
        "min_matching_prefix_tokens": min(r["matching_prefix_tokens"] for r in rows),
        "per_trajectory": rows,
        "consequence": ("generation is replayable under this profile" if replayable else
                        "generation is NOT bit-reproducible across launches; the frozen token IDs "
                        "in trajectories.json are the authoritative contexts and are never "
                        "regenerated"),
        "kl_spec_hash": q.spec_hash(),
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }


def _score_contexts(config_id, contexts, out_stem, overrides, allow_dirty, contexts_hash):
    """One scoring launch into a bare .npy/.json pair, reused only under matching identity."""
    npy, js = out_stem + ".npy", out_stem + ".json"
    ident = q.config_identity(config_id)
    if os.path.exists(npy) and os.path.exists(js):
        prior = json.load(open(js))
        if (prior.get("kl_spec_hash") == q.spec_hash()
                and prior.get("contexts_hash") == contexts_hash
                and (prior.get("config_identity") or {}).get("checkpoint_content_hash")
                == ident["checkpoint_content_hash"]
                and prior.get("engine_overrides") == dict(overrides or {})):
            mat = np.load(npy)
            _assert_scored(mat, prior, contexts, npy)
            return mat, prior
        raise SystemExit(f"ABORT: {npy} was produced under a different spec, context set, "
                         "checkpoint or engine override; move it aside rather than mixing.")
    job = {"config_id": config_id, "task": "score", "contexts": contexts,
           "group_size": len(P.RETAINED_POSITIONS), "out_npy": npy, "out_json": js,
           "engine_overrides": dict(overrides or {})}
    meta = E.run_job(job, os.path.join(os.path.dirname(out_stem), "logs",
                                       os.path.basename(out_stem) + ".log"),
                     allow_dirty=allow_dirty, timeout=7200)
    meta["kl_spec_hash"] = q.spec_hash()
    meta["contexts_hash"] = contexts_hash
    meta["config_identity"] = ident
    meta["engine_overrides"] = dict(overrides or {})
    common.write_json(js, meta)
    mat = np.load(npy)
    _assert_scored(mat, meta, contexts, npy)
    return mat, meta


def _assert_scored(mat, meta, contexts, npy):
    """The completeness and validity checks collect() and engine_profile() already make.

    A short batch leaves -inf rows that would flow straight into kl_nats and corrupt the very
    floor every other number is read against.
    """
    if mat.shape != (len(contexts), q.VOCAB_SIZE):
        raise SystemExit(f"ABORT: {npy} has shape {mat.shape}, expected "
                         f"{(len(contexts), q.VOCAB_SIZE)}")
    per = meta.get("per_context") or []
    if len(per) != len(contexts):
        raise SystemExit(f"ABORT: {npy} validated {len(per)} contexts, expected {len(contexts)}")
    bad = [c for c in per if not (c["full_vocab"] and c["all_finite"] and c["normalized"])]
    if bad:
        raise SystemExit(f"ABORT: {npy} holds {len(bad)} invalid distributions; first: {bad[0]}")
    off = [c for c in per if c.get("decoded_token_is_none") is not True]
    if off:
        raise SystemExit(f"ABORT: {npy}: detokenize was not observed off on {len(off)} contexts")


def _fp8_reference_kl(root, n_traj):
    """BF16||FP8 headline over the same grid -- the scale the two pre-registered fractions are
    expressed against. Without it the floor and cache gates cannot fail on substance."""
    from harness.quality import collect_kl as C
    try:
        a, _, _ = C.load_matrix("BF16_REFERENCE", root=root, n_traj=n_traj)
        b, _, _ = C.load_matrix("FP8_PRIMARY", root=root, n_traj=n_traj)
    except (SystemExit, OSError, ValueError) as exc:
        return None, f"unavailable: {exc}"
    vals = np.array([K.kl_nats(a[i], b[i], q.GATES["kl_negative_tolerance"])
                     for i in range(a.shape[0])])
    return float(K.headline(vals.reshape(n_traj, len(P.RETAINED_POSITIONS)))), "measured"


def _grid_from(root, cfg, n):
    from harness.quality import collect_kl as C
    mat, cells, summary = C.load_matrix(cfg, root=root, n_traj=n)
    return mat, cells, summary


def cache_equivalence(root, out_dir=None, configs=("BF16_REFERENCE", "FP4_PRIMARY"), n_traj=4,
                      allow_dirty=False):
    """G3 -- prefix caching is the H7 exception; this measures what it costs numerically.

    The cached side is the production collection itself, so the comparison is against the very
    arrays the study will use rather than a re-run of them.
    """
    from harness.quality import collect_kl as C, trajectories as T
    out_dir = out_dir or os.path.join(q.QUALITY_DIR, "gates", "cache_equivalence")
    os.makedirs(out_dir, exist_ok=True)
    q.require_clean_tree(allow_dirty, stage="cache_equivalence")
    traj = C.subset(T.load(), n_traj)
    contexts, index = C.build_grid(traj)
    contexts_hash = common.sha256_of_json(contexts)

    fp8_kl, fp8_why = _fp8_reference_kl(root, n_traj)
    per_cfg, checks = {}, {}
    for cfg in configs:
        short = q.QUALITY_CONFIGS[cfg]["short"]
        cached, _, csum = _grid_from(root, cfg, n_traj)
        uncached, umeta = _score_contexts(
            cfg, contexts, os.path.join(out_dir, f"{short}_nocache"),
            {"enable_prefix_caching": False}, allow_dirty, contexts_hash)
        if umeta["resolved_config"]["enable_prefix_caching"] is not False:
            raise SystemExit(f"ABORT: {cfg} uncached run still reports prefix caching on")
        cmp = _compare(cached, uncached, index, len(P.RETAINED_POSITIONS))
        cmp["vs_replication_floor_note"] = ("compare against the production-profile floor in "
                                            "replication_floor.json")
        cmp["cached_engine_identity"] = csum["engine_identity_hash"]
        cmp["uncached_engine_identity"] = umeta["observed"]["engine_identity_hash"]
        cmp["cached_prefix_cache_metrics"] = {
            k: v for k, v in (csum.get("engine_metrics") or {}).items() if "prefix" in k}
        cmp["uncached_prefix_cache_metrics"] = {
            k: v for k, v in (umeta.get("engine_metrics") or {}).items() if "prefix" in k}
        per_cfg[short] = cmp
        checks[f"{short}_cells_complete"] = cmp["cells"] == n_traj * len(P.RETAINED_POSITIONS)
        if fp8_kl is not None:
            bound = q.GATES["cache_equivalence_max_frac_of_fp8"] * fp8_kl
            cmp["pre_registered_bound_nats"] = bound
            cmp["fraction_of_bf16_fp8_kl"] = (cmp["headline_nats"] / fp8_kl) if fp8_kl else None
            checks[f"{short}_within_pre_registered_fraction_of_fp8"] = (
                cmp["headline_nats"] <= bound)
        else:
            checks[f"{short}_fp8_reference_available"] = False
    return {
        "gate": "cache_equivalence",
        "purpose": "G3: prefix caching on (the production setting) vs off, same contexts",
        "kind": "empirical measurement with one pre-registered bound",
        "bf16_fp8_reference_kl_nats": fp8_kl,
        "bf16_fp8_reference_status": fp8_why,
        "pre_registered_fraction": q.GATES["cache_equivalence_max_frac_of_fp8"],
        "low_sample": bool(n_traj < q.N_TRAJECTORIES),
        "configs": list(configs),
        "n_trajectories": n_traj,
        "contexts_hash": contexts_hash,
        "results": per_cfg,
        "checks": checks,
        "integrity_passed": all(checks.values()),
        "kl_spec_hash": q.spec_hash(),
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }


def replication_floor(root, out_dir=None, configs=None, n_traj=4, allow_dirty=False):
    """G2 -- the noise floor of the production profile, from an independent second launch.

    Measured against the production collection itself, per configuration, so every KL reported
    later has a same-profile floor to be read beside.
    """
    from harness.quality import collect_kl as C, trajectories as T
    configs = tuple(configs or q.LADDER)
    out_dir = out_dir or os.path.join(q.QUALITY_DIR, "gates", "replication_floor")
    os.makedirs(out_dir, exist_ok=True)
    q.require_clean_tree(allow_dirty, stage="replication_floor")
    traj = C.subset(T.load(), n_traj)
    contexts, index = C.build_grid(traj)
    contexts_hash = common.sha256_of_json(contexts)
    n_pos = len(P.RETAINED_POSITIONS)

    fp8_kl, fp8_why = _fp8_reference_kl(root, n_traj)
    per_cfg, checks = {}, {}
    for cfg in configs:
        short = q.QUALITY_CONFIGS[cfg]["short"]
        a, _, asum = _grid_from(root, cfg, n_traj)
        b, bmeta = _score_contexts(cfg, contexts, os.path.join(out_dir, f"{short}_r2"),
                                   {}, allow_dirty, contexts_hash)
        if asum["engine_identity_hash"] == bmeta["observed"]["engine_identity_hash"]:
            same_engine = True
        else:
            same_engine = False
        cmp = _compare(a, b, index, n_pos)
        cmp["engine_identity_a"] = asum["engine_identity_hash"]
        cmp["engine_identity_b"] = bmeta["observed"]["engine_identity_hash"]
        cmp["same_observed_engine_identity"] = same_engine
        per_cfg[short] = cmp
        checks[f"{short}_independent_launches_same_profile"] = same_engine
        checks[f"{short}_cells_complete"] = cmp["cells"] == n_traj * n_pos
        if fp8_kl is not None:
            bound = q.GATES["replication_floor_max_frac_of_fp8"] * fp8_kl
            cmp["pre_registered_bound_nats"] = bound
            cmp["fraction_of_bf16_fp8_kl"] = (cmp["headline_nats"] / fp8_kl) if fp8_kl else None
            checks[f"{short}_floor_within_pre_registered_fraction_of_fp8"] = (
                cmp["headline_nats"] <= bound)
        else:
            checks[f"{short}_fp8_reference_available"] = False
    return {
        "gate": "replication_floor",
        "purpose": "G2: same configuration, same graph_2048 profile, two independent launches",
        "kind": "empirical measurement -- the resolution limit of every KL reported here",
        "engine_profile_name": q.PROFILE_NAME,
        "bf16_fp8_reference_kl_nats": fp8_kl,
        "bf16_fp8_reference_status": fp8_why,
        "pre_registered_fraction": q.GATES["replication_floor_max_frac_of_fp8"],
        "low_sample": bool(n_traj < q.N_TRAJECTORIES),
        "configs": list(configs),
        "n_trajectories": n_traj,
        "contexts_hash": contexts_hash,
        "per_config": {k: {"headline_nats": v["headline_nats"], "max_nats": v["max_nats"],
                           "cells": v["cells"], "n_trajectories": n_traj,
                           "mean_nats": v["mean_nats"], "p99_nats": v["p99_nats"],
                           "top1_agreement": v["top1_agreement"],
                           "bit_identical_cells": v["bit_identical_cells"],
                           "by_position": v["by_position"]}
                       for k, v in per_cfg.items()},
        "results": per_cfg,
        "checks": checks,
        "integrity_passed": all(checks.values()),
        "kl_spec_hash": q.spec_hash(),
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }


def storage_precision(root, n_traj=4, allow_dirty=False, floor_path=None):
    """G4 -- fp16 vs fp32 STORAGE, derived from the same fp32 arrays.

    Both sides come from one set of model outputs, so the only variable is the stored
    representation. Comparing separately generated fp16 and fp32 runs would confound storage with
    the launch-to-launch floor.
    """
    from harness.quality import collect_kl as C, trajectories as T
    q.require_clean_tree(allow_dirty, stage="storage_precision")
    traj = C.subset(T.load(), n_traj)
    n = traj["n_trajectories"]
    mats = {}
    for cfg in q.LADDER:
        mat, cells, summary = C.load_matrix(cfg, root=root, n_traj=n)
        if str(mat.dtype) != "float32":
            raise SystemExit(f"ABORT: {cfg} is stored as {mat.dtype}; G4 needs the fp32 arrays")
        mats[cfg] = mat
    floor = json.load(open(floor_path)) if floor_path and os.path.exists(floor_path) else None

    idx = K.bootstrap_indices(n, q.BOOTSTRAP["draws"], q.BOOTSTRAP["seed"])
    pairs, worst_rel_p99, worst_rel_max, worst_abs = {}, 0.0, 0.0, 0.0
    for a, b in q.KL_PAIRS:
        label = f"{q.QUALITY_CONFIGS[a]['short']}||{q.QUALITY_CONFIGS[b]['short']}"
        f32 = np.array([K.kl_nats(mats[a][i], mats[b][i], q.GATES["kl_negative_tolerance"])
                        for i in range(mats[a].shape[0])])
        ha = mats[a].astype(np.float16).astype(np.float32)
        hb = mats[b].astype(np.float16).astype(np.float32)
        f16 = np.array([K.kl_nats(ha[i], hb[i], q.GATES["kl_negative_tolerance"])
                        for i in range(ha.shape[0])])
        d = f16 - f32
        # AND-exclude: the relative bound is only meaningful where the true KL is above the
        # absolute floor, otherwise a 1e-12 baseline manufactures enormous relative errors
        keep = f32 >= q.GATES["precision_abs_floor_nats"]
        rel = np.abs(d[keep]) / f32[keep] if keep.any() else np.zeros(0)
        delta_draws = K.bootstrap_headline(d.reshape(n, -1), idx)
        rec = {
            "cells": int(f32.size),
            "cells_above_abs_floor": int(keep.sum()),
            "headline_delta_bootstrap": K.bootstrap_summary(
                float(K.headline(d.reshape(n, -1))), delta_draws, ci=tuple(q.BOOTSTRAP["ci"])),
            "abs_floor_nats": q.GATES["precision_abs_floor_nats"],
            "fp32_headline_nats": float(K.headline(f32.reshape(n, -1))),
            "fp16_headline_nats": float(K.headline(f16.reshape(n, -1))),
            "headline_delta_nats": float(K.headline(f16.reshape(n, -1))
                                         - K.headline(f32.reshape(n, -1))),
            "mean_abs_delta_nats": float(np.abs(d).mean()),
            "max_abs_delta_nats": float(np.abs(d).max()),
            "rel_p99": float(np.quantile(rel, 0.99)) if rel.size else None,
            "rel_max": float(rel.max()) if rel.size else None,
            # fp16 reaches -65504, and a log-softmax entry over this vocab bottoms out near -80,
            # so nothing underflows to -inf. The real storage risk is mantissa precision, ~3
            # decimal digits, which is what these two measure.
            "max_abs_logprob_representation_error": float(np.abs(ha - mats[a]).max()),
            "entries_changed_by_fp16_frac": float((ha != mats[a]).mean()),
            "min_logprob_seen": float(mats[a][np.isfinite(mats[a])].min()),
        }
        if floor:
            fl = ((floor.get("per_config") or {}).get(q.QUALITY_CONFIGS[a]["short"]) or {})
            if fl.get("headline_nats") is not None:
                rec["headline_delta_vs_floor"] = K.floor_comparison(
                    abs(rec["headline_delta_nats"]), fl["headline_nats"])
        pairs[label] = rec
        if rec["rel_p99"] is not None:
            worst_rel_p99 = max(worst_rel_p99, rec["rel_p99"])
            worst_rel_max = max(worst_rel_max, rec["rel_max"])
        worst_abs = max(worst_abs, rec["max_abs_delta_nats"])

    tested = sum(r["cells_above_abs_floor"] for r in pairs.values())
    total = sum(r["cells"] for r in pairs.values())
    checks = {
        # without this, a run whose every cell fell below the absolute floor would leave both
        # relative bounds at their 0.0 init and "pass" a gate that evaluated nothing
        "enough_cells_above_abs_floor_to_test": tested >= max(10, total // 10),
        "rel_p99_within_tolerance": worst_rel_p99 <= q.GATES["precision_rel_p99_max"],
        "rel_max_within_tolerance": worst_rel_max <= q.GATES["precision_rel_max"],
    }
    return {
        "gate": "storage_precision",
        "purpose": "G4: fp16 vs fp32 storage of the SAME model outputs",
        "kind": "empirical measurement with pre-registered thresholds",
        "thresholds": {k: q.GATES[k] for k in
                       ("precision_rel_p99_max", "precision_rel_max", "precision_abs_floor_nats")},
        "n_trajectories": n,
        "low_sample": bool(n < q.N_TRAJECTORIES),
        "cells_above_abs_floor_total": tested,
        "cells_total": total,
        "pairs": pairs,
        "checks": checks,
        "passed": all(checks.values()),
        "recommended_storage_dtype": "float16" if all(checks.values()) else "float32",
        "kl_spec_hash": q.spec_hash(),
        "git": q.git_state(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }

if __name__ == "__main__":
    sys.exit(main())
