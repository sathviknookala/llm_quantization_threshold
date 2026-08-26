"""P12 -- production preflight for the 64-trajectory KL run.

Answers one question: would P13 collect the right thing? Everything here is either a structural
check over the FULL 640-cell grid, a guard exercised until it actually fires, or a cheap engine
probe. It collects no production data and writes nothing into the production root.
"""

import argparse
import copy
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import kl_math as K, positions as P, qcommon as q  # noqa: E402
from harness.quality import analyze_kl as A, collect_kl as C, trajectories as T  # noqa: E402
from harness.quality import qengine as E  # noqa: E402

PREFLIGHT_DIR = os.path.join(q.QUALITY_DIR, "preflight")
PRODUCTION_ROOT = q.KL_DIR
N_POS = len(P.RETAINED_POSITIONS)


def _rec(name, ok, detail):
    return {"check": name, "ok": bool(ok), "detail": detail}


def structural():
    """The full production grid, re-derived cell by cell. No GPU."""
    out = []
    traj = T.load()
    n = traj["n_trajectories"]
    out.append(_rec("frozen_trajectories_load_and_verify", n == q.N_TRAJECTORIES,
                    {"n": n, "trajectory_set_hash": traj["trajectory_set_hash"]}))

    contexts, index = C.build_grid(traj)
    out.append(_rec("full_grid_rederived", len(contexts) == n * N_POS,
                    {"cells": len(contexts), "expected": n * N_POS,
                     "every_cell_rederive_checked": True}))

    order = [(c["trajectory_index"], c["position_p"]) for c in index]
    out.append(_rec("canonical_order", order == P.grid_order(n),
                    {"trajectory_major_position_ascending": order == P.grid_order(n)}))

    lens = [len(c) for c in contexts]
    want_lens = [P.context_len(p) for _, p in P.grid_order(n)]
    out.append(_rec("context_lengths_match_contract", lens == want_lens,
                    {"min": min(lens), "max": max(lens),
                     "longest_is_2559": max(lens) == 2559}))

    targets_ok = all(
        c["target_token_id"] == traj["trajectories"][c["trajectory_index"]]
        ["continuation_token_ids"][c["position_p"] - 1] for c in index)
    out.append(_rec("targets_match_continuation", targets_ok, {"cells_checked": len(index)}))

    plan = C.shard_plan(n, "BF16", root=PRODUCTION_ROOT)
    tiles = [(p["start"], p["stop"]) for p in plan]
    contiguous = all(tiles[i][1] == tiles[i + 1][0] for i in range(len(tiles) - 1))
    out.append(_rec("shard_plan_tiles_grid",
                    contiguous and tiles[0][0] == 0 and tiles[-1][1] == n * N_POS
                    and all(s % N_POS == 0 and e % N_POS == 0 for s, e in tiles),
                    {"shards": len(plan), "tiles": tiles}))

    ctrl = dict(q.QUALITY_ENGINE_CONTROLS)
    out.append(_rec("max_model_len_covers_longest_context",
                    ctrl["max_model_len"] >= P.max_context_len(),
                    {"max_model_len": ctrl["max_model_len"], "longest": P.max_context_len()}))
    out.append(_rec("engine_profile_is_locked",
                    ctrl["enforce_eager"] is False and ctrl["max_num_batched_tokens"] == 2048
                    and ctrl["enable_prefix_caching"] is True,
                    {"profile": q.PROFILE_NAME,
                     "enforce_eager": ctrl["enforce_eager"],
                     "max_num_batched_tokens": ctrl["max_num_batched_tokens"],
                     "enable_prefix_caching": ctrl["enable_prefix_caching"],
                     "detokenize": q.SCORING_SAMPLING["detokenize"]}))

    ident = {c: q.config_identity(c) for c in q.LADDER}
    out.append(_rec("checkpoint_identity_resolves", all(
        v["checkpoint_content_hash"] for v in ident.values()),
        {c: v["checkpoint_content_hash"][:16] for c, v in ident.items()}))

    need = n * N_POS * q.VOCAB_SIZE * 4 * len(q.LADDER)
    free = shutil.disk_usage(common.REPO).free
    out.append(_rec("disk_space_for_full_run", free > need * 3,
                    {"required_gib": round(need / 2**30, 2),
                     "free_gib": round(free / 2**30, 1)}))

    claimed = os.path.exists(os.path.join(PRODUCTION_ROOT, "manifest.json"))
    out.append(_rec("production_root_unclaimed", not claimed,
                    {"root": os.path.relpath(PRODUCTION_ROOT, common.REPO),
                     "distinct_from_smoke": True}))
    return out, traj, contexts, index, ident


def _expect_reject(name, fn, out, want_substring=None):
    """A guard that never fires is not a guard."""
    try:
        fn()
    except SystemExit as exc:
        hit = want_substring is None or want_substring in str(exc)
        return out.append(_rec(name, hit, {"raised": str(exc)[:180]}))
    except (P.PositionContractError, ValueError) as exc:
        return out.append(_rec(name, True, {"raised": f"{type(exc).__name__}: {str(exc)[:150]}"}))
    out.append(_rec(name, False, {"raised": None, "problem": "guard did not fire"}))


def rejection_tests(traj, index):
    """Exercise every resume/provenance guard against real artifacts until it aborts."""
    out = []
    sandbox = os.path.join(PREFLIGHT_DIR, "sandbox")
    shutil.rmtree(sandbox, ignore_errors=True)
    os.makedirs(os.path.join(sandbox, "dist", "BF16"), exist_ok=True)
    os.makedirs(os.path.join(sandbox, "shards", "BF16"), exist_ok=True)

    n_s = 2
    rows = n_s * N_POS
    arr = np.random.default_rng(0).normal(-10, 1, size=(rows, q.VOCAB_SIZE)).astype(np.float32)
    npy = os.path.join(sandbox, "dist", "BF16", "shard_000.npy")
    js = os.path.join(sandbox, "shards", "BF16", "shard_000.json")
    np.save(npy, arr)
    prov = {"config_id": "BF16_REFERENCE", "kl_spec_hash": q.spec_hash(),
            "trajectory_set_hash": traj["trajectory_set_hash"],
            "subset_n": n_s, "prompt_subset_hash": traj["prompt_subset_hash"],
            "contexts_hash": "x", "checkpoint_content_hash": "ck",
            "tokenizer_identity": {}, "engine_profile_name": q.PROFILE_NAME,
            "storage_dtype": "float32", "retained_positions": list(P.RETAINED_POSITIONS)}
    meta = {"start": 0, "stop": rows, "npy": npy, "rows": rows,
            "engine_identity_hash": "eng000", "provenance": prov,
            "per_context": [{"full_vocab": True, "all_finite": True, "normalized": True,
                             "decoded_token_is_none": True} for _ in range(rows)],
            "index": index[:rows]}
    common.write_json(js, meta)
    sh = {"shard": 0, "start": 0, "stop": rows, "npy": npy, "json": js}

    ok, why = C._shard_reusable(sh, prov, rows)
    out.append(_rec("faithful_shard_is_reusable", ok, {"reason": why}))

    for label, key, val in (("spec", "kl_spec_hash", "deadbeefdeadbeef"),
                            ("trajectory_set", "trajectory_set_hash", "0" * 64),
                            ("checkpoint", "checkpoint_content_hash", "other"),
                            ("subset_n", "subset_n", 99)):
        bad = dict(prov); bad[key] = val
        ok2, why2 = C._shard_reusable(sh, bad, rows)
        out.append(_rec(f"resume_refuses_mismatched_{label}", not ok2, {"reason": why2}))

    short = copy.deepcopy(meta); short["per_context"] = short["per_context"][:-1]
    common.write_json(js, short)
    ok3, why3 = C._shard_reusable(sh, prov, rows)
    out.append(_rec("resume_refuses_incomplete_shard", not ok3, {"reason": why3}))
    common.write_json(js, meta)

    def _summary(shards, eid="eng000"):
        return {"n_trajectories": n_s, "engine_identity_hash": eid, "shards": shards,
                "provenance": prov}

    rel_npy = os.path.relpath(npy, common.REPO)
    rel_js = os.path.relpath(js, common.REPO)
    root_summary = os.path.join(sandbox, "collection_BF16.json")
    common.write_json(root_summary, _summary(
        [{"shard": 0, "start": 0, "stop": rows, "npy": rel_npy, "json": rel_js}]))
    mat, cells, _ = C.load_matrix("BF16_REFERENCE", root=sandbox, n_traj=n_s)
    out.append(_rec("sandbox_matrix_loads", mat.shape == (rows, q.VOCAB_SIZE),
                    {"shape": list(mat.shape)}))

    common.write_json(root_summary, _summary(
        [{"shard": 0, "start": 0, "stop": rows, "npy": rel_npy, "json": rel_js}], eid="different"))
    _expect_reject("load_refuses_summary_identity_disagreement",
                   lambda: C.load_matrix("BF16_REFERENCE", root=sandbox, n_traj=n_s), out,
                   "claims")

    scrambled = copy.deepcopy(meta)
    scrambled["index"] = list(reversed(scrambled["index"]))
    common.write_json(js, scrambled)
    common.write_json(root_summary, _summary(
        [{"shard": 0, "start": 0, "stop": rows, "npy": rel_npy, "json": rel_js}]))
    _expect_reject("load_refuses_out_of_order_grid",
                   lambda: C.load_matrix("BF16_REFERENCE", root=sandbox, n_traj=n_s), out,
                   "canonical order")
    common.write_json(js, meta)

    view = C.subset(traj, n_s)
    good_cells = [dict(c) for c in index[:rows]]
    out.append(_rec("verify_cells_accepts_faithful_grid",
                    A.verify_cells(good_cells, view) == rows, {"cells": rows}))

    swapped = [dict(c) for c in good_cells]
    swapped[1]["position_p"], swapped[2]["position_p"] = (swapped[2]["position_p"],
                                                          swapped[1]["position_p"])
    _expect_reject("verify_cells_refuses_position_scramble",
                   lambda: A.verify_cells(swapped, view), out)

    dropped = [c for c in good_cells if c["position_p"] != 2048]
    _expect_reject("verify_cells_refuses_partial_trajectory",
                   lambda: A.verify_cells(dropped, view), out, "not 10/10")

    nohash = [dict(c) for c in good_cells]
    nohash[0].pop("context_sha256", None)
    _expect_reject("verify_cells_refuses_missing_context_hash",
                   lambda: A.verify_cells(nohash, view), out, "context_sha256")

    tampered = copy.deepcopy(view)
    tampered["trajectories"] = [dict(t) for t in tampered["trajectories"]]
    flipped = list(tampered["trajectories"][0]["continuation_token_ids"])
    flipped[1000] += 1
    tampered["trajectories"][0] = {**tampered["trajectories"][0],
                                   "continuation_token_ids": flipped}
    _expect_reject("verify_cells_detects_tampered_trajectory",
                   lambda: A.verify_cells(good_cells, tampered), out)

    _expect_reject("freeze_refuses_to_overwrite", lambda: T.freeze(), out, "already exists")

    bad_manifest_dir = os.path.join(sandbox, "mguard")
    q.guard_manifest(bad_manifest_dir, "test", extra={"trajectory_set_hash": "aaa"})
    _expect_reject("manifest_guard_refuses_changed_trajectory_set",
                   lambda: q.guard_manifest(bad_manifest_dir, "test",
                                            extra={"trajectory_set_hash": "bbb"}), out)

    _expect_reject("engine_refuses_undecided_enforce_eager",
                   lambda: E.engine_kwargs("BF16_REFERENCE", {"enforce_eager": None}), out)

    _expect_reject("kl_refuses_infinite_divergence",
                   lambda: K.kl_nats(np.array([0.0, -1.0]), np.array([0.0, -np.inf])), out)

    shutil.rmtree(sandbox, ignore_errors=True)
    return out


def engine_probe(traj, allow_dirty=False, reference=None):
    """One trajectory's ten contexts per configuration: do the engines still come up identically?"""
    out, observed = [], {}
    os.makedirs(PREFLIGHT_DIR, exist_ok=True)
    view = C.subset(traj, 1)
    contexts, _ = C.build_grid(view)
    chash = common.sha256_of_json(contexts)
    from harness.quality import gates as G
    for cfg in q.LADDER:
        short = q.QUALITY_CONFIGS[cfg]["short"]
        _, meta = G._score_contexts(cfg, contexts, os.path.join(PREFLIGHT_DIR, f"{short}_probe"),
                                    {}, allow_dirty, chash)
        obs = meta["observed"]
        observed[short] = {"engine_identity_hash": obs["engine_identity_hash"],
                           "kv_cache_tokens": obs["kv_cache_tokens"],
                           "dispatch_ok": obs["dispatch_verdict"]["ok"],
                           "graph_capture_observed": obs["graph_capture_observed"],
                           "resolved": {k: meta["resolved_config"][k] for k in
                                        ("enforce_eager", "max_num_batched_tokens",
                                         "enable_prefix_caching")},
                           "prefix_cache": {k: v for k, v in
                                            (meta.get("engine_metrics") or {}).items()
                                            if "prefix_cache" in k and "external" not in k},
                           "seconds": meta.get("generate_seconds")}
        out.append(_rec(f"{short}_dispatch_verified", obs["dispatch_verdict"]["ok"],
                        obs["dispatch_verdict"]))
        if reference and short in reference:
            same = obs["engine_identity_hash"] == reference[short]["engine_identity_hash"]
            out.append(_rec(f"{short}_engine_identity_matches_smoke", same,
                            {"now": obs["engine_identity_hash"],
                             "smoke": reference[short]["engine_identity_hash"],
                             "kv_now": obs["kv_cache_tokens"],
                             "kv_smoke": reference[short]["kv_cache_tokens"]}))
    return out, observed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(q.QUALITY_DIR, "preflight.json"))
    ap.add_argument("--skip-engine", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    git = q.require_clean_tree(a.allow_dirty, stage="preflight")
    checks, traj, contexts, index, ident = structural()
    checks += rejection_tests(traj, index)

    reference = {}
    smoke_root = os.path.join(q.QUALITY_DIR, "smoke")
    for cfg in q.LADDER:
        short = q.QUALITY_CONFIGS[cfg]["short"]
        path = os.path.join(smoke_root, f"collection_{short}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            reference[short] = {"engine_identity_hash": d["engine_identity_hash"],
                                "kv_cache_tokens": d["observed"]["kv_cache_tokens"],
                                "seconds_per_context": d["seconds"] / d["cells"]}

    observed = {}
    if not a.skip_engine:
        ec, observed = engine_probe(traj, allow_dirty=a.allow_dirty, reference=reference)
        checks += ec

    per_ctx = [r["seconds_per_context"] for r in reference.values()] or [0.15]
    cells = q.N_TRAJECTORIES * N_POS
    projection = {
        "cells_per_config": cells,
        "configs": len(q.LADDER),
        "seconds_per_context_measured": {k: round(v["seconds_per_context"], 4)
                                         for k, v in reference.items()},
        "projected_scoring_minutes_total": round(sum(per_ctx) * cells / 60.0, 1),
        "projected_bytes_gib": round(cells * q.VOCAB_SIZE * 4 * len(q.LADDER) / 2**30, 2),
        "shards_per_config": len(C.shard_plan(q.N_TRAJECTORIES, "BF16", root=PRODUCTION_ROOT)),
    }

    failed = [c["check"] for c in checks if not c["ok"]]
    rec = {
        "artifact": "P12 production preflight",
        "purpose": "prove the 64-trajectory run would collect the right thing; collects none of it",
        "n_trajectories": q.N_TRAJECTORIES,
        "cells_per_config": cells,
        "trajectory_set_hash": traj["trajectory_set_hash"],
        "kl_spec_hash": q.spec_hash(),
        "engine_profile_name": q.PROFILE_NAME,
        "config_identity": ident,
        "smoke_reference": reference,
        "engine_probe": observed,
        "projection": projection,
        "checks": checks,
        "checks_total": len(checks),
        "checks_failed": failed,
        "passed": not failed,
        "authorises": "nothing on its own; P13 remains a separate decision and the BF16 "
                      "replication-floor question is not settled by this artifact",
        "git": git,
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }
    common.write_json(a.out, rec)
    for c in checks:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} preflight checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    print("WROTE", a.out)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
