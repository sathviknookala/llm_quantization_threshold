"""P8 -- collect full-vocabulary next-token distributions for one configuration.

One engine launch per configuration scores the whole 64x10 grid in trajectory-sized groups, writing
shards as they complete. Resume reuses a shard only when every identity it was produced under still
matches; a partial shard is rewritten, never merged.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import positions as P, qcommon as q  # noqa: E402
from harness.quality import qengine as E, trajectories as T  # noqa: E402

SHARD_TRAJECTORIES = 8
N_POS = len(P.RETAINED_POSITIONS)


def run_dir(root=None):
    return root or q.KL_DIR


def dist_dir(short, root=None):
    return os.path.join(run_dir(root), "dist", short)


def shard_dir(short, root=None):
    return os.path.join(run_dir(root), "shards", short)


def subset(traj, n):
    """First-n view of the frozen set, for the smoke. The trajectories themselves are unchanged;
    the subset size travels in provenance so a smoke can never be read as the production grid."""
    if n is None or n == traj["n_trajectories"]:
        return traj
    if n > traj["n_trajectories"]:
        raise SystemExit(f"ABORT: asked for {n} trajectories, the frozen set holds "
                         f"{traj['n_trajectories']}")
    view = dict(traj)
    view["trajectories"] = traj["trajectories"][:n]
    view["n_trajectories"] = n
    view["subset_of_trajectory_set_hash"] = traj["trajectory_set_hash"]
    return view


def build_grid(traj):
    """Contexts in canonical (trajectory, ascending position) order, re-derived and checked.

    `rederive_and_check` runs here on the production path, not only in analysis: a context that
    fails the contract must never reach the GPU, let alone a stored artifact.
    """
    contexts, index = [], []
    for t in traj["trajectories"]:
        cells = P.build_all(t["prompt_token_ids"], t["continuation_token_ids"])
        for cell in cells:
            ok, problems = P.rederive_and_check(
                t["prompt_token_ids"], t["continuation_token_ids"], cell["position_p"],
                cell["context_len"], cell["target_token_id"], cell["context_ids"])
            if not ok:
                raise SystemExit(f"ABORT: trajectory {t['trajectory_index']} position "
                                 f"{cell['position_p']} fails re-derivation: {problems}")
            contexts.append(cell["context_ids"])
            index.append({"trajectory_index": t["trajectory_index"],
                          "prompt_index": t["prompt_index"],
                          "position_p": cell["position_p"],
                          "context_len": cell["context_len"],
                          "target_token_id": cell["target_token_id"],
                          "context_sha256": q.prompt_hash(cell["context_ids"])})
    P.assert_complete_grid(index, len(traj["trajectories"]))
    order = [(c["trajectory_index"], c["position_p"]) for c in index]
    if order != P.grid_order(len(traj["trajectories"])):
        raise SystemExit("ABORT: grid is not in canonical (trajectory, ascending position) order; "
                         "stored row indices would not mean what analysis assumes")
    return contexts, index


def shard_plan(n_traj, short, root=None, shard_trajectories=SHARD_TRAJECTORIES):
    plan = []
    for k, start_t in enumerate(range(0, n_traj, shard_trajectories)):
        stop_t = min(start_t + shard_trajectories, n_traj)
        plan.append({
            "shard": k,
            "trajectory_start": start_t,
            "trajectory_stop": stop_t,
            "start": start_t * N_POS,
            "stop": stop_t * N_POS,
            "npy": os.path.join(dist_dir(short, root), f"shard_{k:03d}.npy"),
            "json": os.path.join(shard_dir(short, root), f"shard_{k:03d}.json"),
        })
    return plan


def _provenance(config_id, traj, contexts_hash, ident):
    return {
        "config_id": config_id,
        "kl_spec_hash": q.spec_hash(),
        "trajectory_set_hash": traj["trajectory_set_hash"],
        "subset_n": traj["n_trajectories"],
        "prompt_subset_hash": traj["prompt_subset_hash"],
        "contexts_hash": contexts_hash,
        "checkpoint_content_hash": ident["checkpoint_content_hash"],
        "tokenizer_identity": ident["tokenizer_identity"],
        "engine_profile_name": q.PROFILE_NAME,
        "storage_dtype": q.STORAGE_DTYPE,
        "retained_positions": list(P.RETAINED_POSITIONS),
    }


def _shard_reusable(sh, prov, expect_rows):
    """Resume only under exact matching identity. Anything else is a rewrite, not a merge."""
    if not (os.path.exists(sh["npy"]) and os.path.exists(sh["json"])):
        return False, "absent"
    try:
        meta = json.load(open(sh["json"]))
    except (ValueError, OSError) as exc:
        return False, f"unreadable metadata: {exc}"
    got = meta.get("provenance") or {}
    for k, want in prov.items():
        if got.get(k) != want:
            return False, f"provenance mismatch on {k}"
    if meta.get("start") != sh["start"] or meta.get("stop") != sh["stop"]:
        return False, "shard boundaries differ"
    if len(meta.get("per_context") or []) != expect_rows:
        return False, "incomplete per-context validation"
    try:
        arr = np.load(sh["npy"], mmap_mode="r")
    except (ValueError, OSError) as exc:
        return False, f"unreadable array: {exc}"
    if arr.shape != (expect_rows, q.VOCAB_SIZE):
        return False, f"array shape {arr.shape}"
    if str(arr.dtype) != q.STORAGE_DTYPE:
        return False, f"array dtype {arr.dtype}"
    return True, "reused"


def collect(config_id, root=None, allow_dirty=False, n_traj=None,
            shard_trajectories=SHARD_TRAJECTORIES, traj=None, require_cool=True):
    root = run_dir(root)
    short = q.QUALITY_CONFIGS[config_id]["short"]
    q.require_clean_tree(allow_dirty, stage=f"collect_kl:{config_id}")
    traj = traj if traj is not None else subset(T.load(), n_traj)
    n = traj["n_trajectories"]
    os.makedirs(dist_dir(short, root), exist_ok=True)
    os.makedirs(shard_dir(short, root), exist_ok=True)
    q.guard_manifest(root, "KL collection",
                     extra={"trajectory_set_hash": traj["trajectory_set_hash"]})

    contexts, index = build_grid(traj)
    contexts_hash = common.sha256_of_json(contexts)
    ident = q.config_identity(config_id)
    prov = _provenance(config_id, traj, contexts_hash, ident)

    plan = shard_plan(n, short, root, shard_trajectories)
    todo, reused, reasons = [], [], {}
    for sh in plan:
        ok, why = _shard_reusable(sh, prov, sh["stop"] - sh["start"])
        reasons[sh["shard"]] = why
        (reused if ok else todo).append(sh)

    prior_ids = sorted({(json.load(open(s["json"])).get("engine_identity_hash"))
                        for s in reused if os.path.exists(s["json"])} - {None})
    meta = None
    if todo:
        job = {"config_id": config_id, "task": "score_shards", "contexts": contexts,
               "group_size": N_POS, "shards": todo, "provenance": prov,
               "out_npy": os.path.join(dist_dir(short, root), "_unused.npy"),
               "out_json": os.path.join(shard_dir(short, root), "_engine.json")}
        meta = E.run_job(job, os.path.join(run_dir(root), "logs", f"collect_{short}.log"),
                         allow_dirty=allow_dirty, require_cool=require_cool, timeout=7200)
        eid = meta["observed"]["engine_identity_hash"]
        if prior_ids and prior_ids != [eid]:
            raise SystemExit(
                f"ABORT: reused shards were scored under engine identity {prior_ids} but this "
                f"launch is {eid}; a configuration must not be assembled from mixed engines")
        for sh in todo:
            m = json.load(open(sh["json"]))
            m["engine_identity_hash"] = eid
            m["observed"] = meta["observed"]
            m["engine_metrics"] = meta.get("engine_metrics")
            m["git"] = q.git_state()
            m["software"] = common.software_identity()
            m["index"] = index[sh["start"]:sh["stop"]]
            common.write_json(sh["json"], m)

    # re-validated after the fact from what is on disk, so a reused shard is held to the same bar
    final, identities = [], set()
    for sh in plan:
        ok, why = _shard_reusable(sh, prov, sh["stop"] - sh["start"])
        if not ok:
            raise SystemExit(f"ABORT: shard {sh['shard']} is not usable after collection: {why}")
        m = json.load(open(sh["json"]))
        identities.add(m.get("engine_identity_hash"))
        final.append(m)
    if len(identities) != 1 or None in identities:
        raise SystemExit(f"ABORT: shards span engine identities {sorted(identities)}")

    cells = [c for m in final for c in m["index"]]
    P.assert_complete_grid(cells, n)
    bad = [c for m in final for c in m["per_context"]
           if not (c["full_vocab"] and c["all_finite"] and c["normalized"])]
    if bad:
        raise SystemExit(f"ABORT: {len(bad)} stored distributions are not full-vocab, finite and "
                         f"normalized; first: {bad[0]}")
    not_off = [c for m in final for c in m["per_context"] if c.get("decoded_token_is_none") is not True]
    if not_off:
        raise SystemExit(f"ABORT: detokenize was not observed off on {len(not_off)} contexts")

    rec = {
        "artifact": "KL collection",
        "config_id": config_id,
        "quantization": short,
        "n_trajectories": n,
        "positions": list(P.RETAINED_POSITIONS),
        "cells": n * N_POS,
        "provenance": prov,
        "config_identity": ident,
        "engine_identity_hash": identities.pop(),
        "shards": [{"shard": s["shard"], "start": s["start"], "stop": s["stop"],
                    "npy": os.path.relpath(s["npy"], common.REPO),
                    "json": os.path.relpath(s["json"], common.REPO),
                    "reused": reasons[s["shard"]] == "reused"} for s in plan],
        "shard_reuse": reasons,
        "launched": bool(todo),
        "seconds": (meta or {}).get("generate_seconds"),
        "wall_seconds": (meta or {}).get("wall_seconds"),
        "engine_metrics": (meta or {}).get("engine_metrics"),
        "observed": (meta or {}).get("observed"),
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }
    common.write_json(os.path.join(run_dir(root), f"collection_{short}.json"), rec)
    return rec


def load_matrix(config_id, root=None, n_traj=None):
    """The (n_trajectories * 10, vocab) array, assembled from shards in canonical order."""
    short = q.QUALITY_CONFIGS[config_id]["short"]
    summary = json.load(open(os.path.join(run_dir(root), f"collection_{short}.json")))
    n = n_traj or summary["n_trajectories"]
    blocks, cells, identities = [], [], set()
    for s in sorted(summary["shards"], key=lambda x: x["start"]):
        arr = np.load(os.path.join(common.REPO, s["npy"]))
        m = json.load(open(os.path.join(common.REPO, s["json"])))
        if arr.shape[0] != s["stop"] - s["start"]:
            raise SystemExit(f"ABORT: {s['npy']} holds {arr.shape[0]} rows, expected "
                             f"{s['stop'] - s['start']}")
        identities.add(m.get("engine_identity_hash"))
        blocks.append(arr)
        cells.extend(m["index"])
    mat = np.concatenate(blocks, axis=0)
    P.assert_complete_grid(cells, n)
    # completeness is a set property; the reshape in analysis depends on the ORDER
    order = [(c["trajectory_index"], c["position_p"]) for c in cells]
    if order != P.grid_order(n):
        raise SystemExit(
            f"ABORT: {short} shards reassemble out of canonical order; the (trajectory, position) "
            "reshape in analysis would average the wrong ten rows together")
    if len(identities) != 1 or None in identities:
        raise SystemExit(
            f"ABORT: {short} shards span engine identities {sorted(identities)}; a configuration "
            "must not be assembled from mixed engines")
    if identities != {summary.get("engine_identity_hash")}:
        raise SystemExit(f"ABORT: {short} shards were scored under {sorted(identities)} but "
                         f"collection_{short}.json claims {summary.get('engine_identity_hash')}")
    if mat.shape != (n * N_POS, q.VOCAB_SIZE):
        raise SystemExit(f"ABORT: assembled matrix has shape {mat.shape}")
    return mat, cells, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="comma-separated; default the whole ladder")
    ap.add_argument("--root", default="")
    ap.add_argument("--n-traj", type=int, default=0)
    ap.add_argument("--shard-trajectories", type=int, default=SHARD_TRAJECTORIES)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    configs = [c for c in (a.config.split(",") if a.config else list(q.LADDER)) if c]
    for c in configs:
        rec = collect(c, root=a.root or None, allow_dirty=a.allow_dirty,
                      n_traj=a.n_traj or None, shard_trajectories=a.shard_trajectories)
        print(json.dumps({k: v for k, v in rec.items()
                          if k in ("config_id", "cells", "launched", "seconds",
                                   "engine_identity_hash", "shard_reuse")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
