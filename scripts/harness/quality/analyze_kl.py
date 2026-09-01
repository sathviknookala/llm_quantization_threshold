"""P9 -- turn stored distributions into the KL result.

Every stored cell is re-derived from the frozen trajectories before it is used: the collector's own
metadata is treated as a claim to be checked, not as provenance to be trusted. Refuses to summarise
an incomplete grid.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import kl_math as K, positions as P, qcommon as q  # noqa: E402
from harness.quality import collect_kl as C, trajectories as T  # noqa: E402

N_POS = len(P.RETAINED_POSITIONS)


def verify_cells(cells, traj):
    """Independent reconstruction of every context and target from the frozen trajectory set."""
    by_index = {t["trajectory_index"]: t for t in traj["trajectories"]}
    checked = 0
    for c in cells:
        t = by_index.get(c["trajectory_index"])
        if t is None:
            raise SystemExit(f"ABORT: stored cell names trajectory {c['trajectory_index']}, which "
                             "is not in the frozen set")
        context, target = P.build_context(
            t["prompt_token_ids"], t["continuation_token_ids"], c["position_p"])
        if c["context_len"] != len(context):
            raise SystemExit(f"ABORT: trajectory {c['trajectory_index']} position "
                             f"{c['position_p']}: stored context_len {c['context_len']} differs "
                             f"from re-derivation {len(context)}")
        # length and target are necessary but not sufficient: an interior token can be flipped
        # while both still match, and this hash is the only check that catches it
        if not c.get("context_sha256"):
            raise SystemExit(f"ABORT: trajectory {c['trajectory_index']} position "
                             f"{c['position_p']} carries no context_sha256; interior content "
                             "would be unverifiable")
        if c["context_sha256"] != q.prompt_hash(context):
            raise SystemExit(f"ABORT: trajectory {c['trajectory_index']} position "
                             f"{c['position_p']}: stored context hash differs from re-derivation")
        if c["target_token_id"] != target:
            raise SystemExit(f"ABORT: trajectory {c['trajectory_index']} position "
                             f"{c['position_p']}: stored target differs from re-derivation")
        checked += 1
    try:
        P.assert_complete_grid(cells, traj["n_trajectories"])
    except P.GridIncompleteError as exc:
        raise SystemExit(f"ABORT: {exc}. A partial trajectory must not be averaged: the headline "
                         "is a mean of per-trajectory means.") from exc
    return checked


def pair_grid(ma, mb, n_traj):
    vals = np.array([K.kl_nats(ma[i], mb[i], q.GATES["kl_negative_tolerance"])
                     for i in range(ma.shape[0])])
    return vals.reshape(n_traj, N_POS)


def summarise(grid, idx, cells, label):
    traj_means = K.trajectory_means(grid)
    point = float(traj_means.mean())
    boot = K.bootstrap_summary(point, K.bootstrap_headline(grid, idx),
                               ci=tuple(q.BOOTSTRAP["ci"]))
    by_pos = {}
    pos_draws = K.bootstrap_positions(grid, idx)
    for j, p in enumerate(P.RETAINED_POSITIONS):
        col = grid[:, j]
        by_pos[str(p)] = {
            "mean_nats": float(col.mean()),
            "median_nats": float(np.median(col)),
            "p95_nats": float(np.quantile(col, 0.95)),
            "max_nats": float(col.max()),
            **K.bootstrap_summary(float(col.mean()), pos_draws[:, j],
                                  ci=tuple(q.BOOTSTRAP["ci"])),
        }
    flat = grid.reshape(-1)
    worst = int(np.argmax(flat))
    return {
        "pair": label,
        "n_trajectories": int(grid.shape[0]),
        "n_positions": int(grid.shape[1]),
        "cells": int(flat.size),
        "headline_nats": point,
        "headline_bootstrap": boot,
        "coverage_caveat": "percentile bootstrap at n=64 on a right-skewed non-negative statistic; "
                           "95% is nominal, not guaranteed. Read bias and std_error beside it.",
        "trajectory_means_nats": [float(x) for x in traj_means],
        "per_cell": {
            "mean_nats": float(flat.mean()),
            "median_nats": float(np.median(flat)),
            "p95_nats": float(np.quantile(flat, 0.95)),
            "p99_nats": float(np.quantile(flat, 0.99)),
            "max_nats": float(flat.max()),
            "worst_cell": {**cells[worst], "kl_nats": float(flat[worst])},
        },
        "by_position": by_pos,
    }


def analyze(root=None, out=None, n_traj=None, floor_path=None, allow_dirty=False,
            configs=None):
    root = C.run_dir(root)
    q.require_clean_tree(allow_dirty, stage="analyze_kl")
    traj = C.subset(T.load(), n_traj)
    n = traj["n_trajectories"]
    ladder = list(configs or q.LADDER)

    mats, cells_by_cfg, summaries = {}, {}, {}
    for cfg in ladder:
        mat, cells, summary = C.load_matrix(cfg, root=root, n_traj=n)
        checked = verify_cells(cells, traj)
        if summary["provenance"].get("subset_n") != n:
            raise SystemExit(f"ABORT: {cfg} was collected over "
                             f"{summary['provenance'].get('subset_n')} trajectories, analysing {n}")
        if summary["provenance"]["trajectory_set_hash"] != traj["trajectory_set_hash"]:
            raise SystemExit(f"ABORT: {cfg} was collected against trajectory set "
                             f"{summary['provenance']['trajectory_set_hash'][:16]}, not "
                             f"{traj['trajectory_set_hash'][:16]}")
        if summary["provenance"]["kl_spec_hash"] != q.spec_hash():
            raise SystemExit(f"ABORT: {cfg} was collected under KL_SPEC "
                             f"{summary['provenance']['kl_spec_hash']}, now {q.spec_hash()}")
        mats[cfg], cells_by_cfg[cfg], summaries[cfg] = mat, cells, summary
        summaries[cfg]["cells_reverified"] = checked

    ref_cells = cells_by_cfg[ladder[0]]
    for cfg in ladder[1:]:
        if cells_by_cfg[cfg] != ref_cells:
            raise SystemExit(f"ABORT: {cfg}'s cell grid differs from {ladder[0]}'s; the pairs "
                             "would not be scored on the same contexts")

    idx = K.bootstrap_indices(n, q.BOOTSTRAP["draws"], q.BOOTSTRAP["seed"])
    floor_per_config, floor_desc = (q.load_floor(floor_path, traj["trajectory_set_hash"])
                                    if floor_path else (None, None))

    pairs = {}
    for a, b in q.KL_PAIRS:
        if a not in mats or b not in mats:
            continue
        label = f"{q.QUALITY_CONFIGS[a]['short']}||{q.QUALITY_CONFIGS[b]['short']}"
        grid = pair_grid(mats[a], mats[b], n)
        rec = summarise(grid, idx, ref_cells, label)
        rec["reference_config"], rec["comparison_config"] = a, b
        if floor_per_config:
            fa = floor_per_config.get(q.QUALITY_CONFIGS[a]["short"]) or {}
            fb = floor_per_config.get(q.QUALITY_CONFIGS[b]["short"]) or {}
            have = [f for f in (fa, fb) if f.get("headline_nats") is not None]
            if have:
                # both sides contribute launch-to-launch noise, so the binding floor is the larger
                fh = max(f["headline_nats"] for f in have)
                vs = {
                    "floor_source": os.path.relpath(floor_path, common.REPO),
                    "floor_configs": [q.QUALITY_CONFIGS[c]["short"] for c, f in
                                      ((a, fa), (b, fb)) if f.get("headline_nats") is not None],
                    "floor_rule": "max over both sides of the pair",
                    "headline": K.floor_comparison(rec["headline_nats"], fh),
                }
                floor_cells = [f.get("cells") for f in have]
                if all(c == rec["cells"] for c in floor_cells):
                    vs["worst_cell"] = K.floor_comparison(
                        rec["per_cell"]["max_nats"], max(f["max_nats"] for f in have))
                else:
                    # max-of-640 exceeds max-of-40 in ~95% of draws under an identical null, so
                    # comparing maxima across different cell counts measures sample size, not signal
                    vs["worst_cell"] = None
                    vs["worst_cell_not_comparable"] = (
                        f"the floor was measured over {floor_cells} cells and this pair over "
                        f"{rec['cells']}; a maximum is an extreme-value statistic and the two are "
                        "not comparable at different sample sizes. Re-run the floor at the same "
                        "trajectory count to enable this comparison.")
                rec["vs_replication_floor"] = vs
            else:
                # a floor covering neither side must leave a record: an absent key reads as an
                # unremarkable omission, which is the same silence this whole path had
                rec["vs_replication_floor"] = None
                rec["replication_floor_unavailable_for"] = [
                    q.QUALITY_CONFIGS[c]["short"] for c in (a, b)]
                rec["replication_floor_unavailable_because"] = (
                    f"{floor_desc['floor_path']} covers {floor_desc['floor_configs']} only; "
                    "backfilling from a floor of different provenance would mix designs")
        pairs[label] = rec

    # KL is not additive; the direct FP8||FP4 measurement is reported, and the difference of the
    # two BF16-anchored values is recorded only to show it is not the same quantity
    if "FP8||FP4" in pairs and "BF16||FP8" in pairs and "BF16||FP4" in pairs:
        direct = pairs["FP8||FP4"]["headline_nats"]
        naive = pairs["BF16||FP4"]["headline_nats"] - pairs["BF16||FP8"]["headline_nats"]
        # the denominator is a difference of two KLs: no guaranteed sign, no guaranteed magnitude
        usable = abs(naive) >= 1e-9 and naive > 0
        pairs["FP8||FP4"]["subtraction_proxy_would_have_said"] = {
            "value_nats": naive,
            "direct_nats": direct,
            "proxy_is_negative": bool(naive < 0),
            "ratio_direct_over_proxy": (direct / naive) if usable else None,
            "ratio_omitted_because": None if usable else
                ("the proxy is negative, which a divergence cannot be" if naive < 0
                 else "the proxy is within 1e-9 nats of zero and the ratio is unbounded"),
            "note": "reported to document the size of the error the barred proxy makes; "
                    "the proxy is never used as a result",
        }

    rec = {
        "artifact": "KL analysis",
        "n_trajectories": n,
        "positions": list(P.RETAINED_POSITIONS),
        "statistical_unit": "trajectory",
        "cells_per_config": n * N_POS,
        "cells_reverified_per_config": {c: summaries[c]["cells_reverified"] for c in ladder},
        "trajectory_set_hash": traj["trajectory_set_hash"],
        "kl_spec_hash": q.spec_hash(),
        "bootstrap": dict(q.BOOTSTRAP),
        "engine_profile_name": q.PROFILE_NAME,
        "engine_identity": {c: summaries[c]["engine_identity_hash"] for c in ladder},
        "checkpoint_content_hash": {
            c: summaries[c]["provenance"]["checkpoint_content_hash"] for c in ladder},
        "replication_floor": floor_per_config,
        "replication_floor_source": floor_desc,
        "replication_floor_omitted": floor_path is None,
        "pairs": pairs,
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }
    out = out or os.path.join(root, "kl_summary.json")
    common.write_json(out, rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-traj", type=int, default=0)
    ap.add_argument("--floor", default="")
    ap.add_argument("--configs", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    rec = analyze(root=a.root or None, out=a.out or None, n_traj=a.n_traj or None,
                  floor_path=a.floor or None, allow_dirty=a.allow_dirty,
                  configs=[c for c in a.configs.split(",") if c] or None)
    for label, p in rec["pairs"].items():
        b = p["headline_bootstrap"]
        print(f"{label:12s} headline {p['headline_nats']:.6e} nats  "
              f"95% CI [{b['ci_low']:.6e}, {b['ci_high']:.6e}]  "
              f"worst cell {p['per_cell']['max_nats']:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
