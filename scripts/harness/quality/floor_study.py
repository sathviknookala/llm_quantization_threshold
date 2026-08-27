"""Production-scale BF16 replication floor: three independent launches over the frozen 64.

The n=4 smoke put the BF16 floor at 2.98e-04 nats from a single launch pair. That is one pair on
one tenth of the grid, and the whole quality axis is read against it. This scores all 640 retained
positions on three fresh launches and reports every ordered pair, so the floor carries a spread
rather than a point.

Collects nothing into the production root; P13 is untouched.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import kl_math as K, positions as P, qcommon as q  # noqa: E402
from harness.quality import analyze_kl as A, collect_kl as C, trajectories as T  # noqa: E402

STUDY_DIR = os.path.join(q.QUALITY_DIR, "floor64")
CONFIG = q.REFERENCE_CONFIG
SHORT = q.QUALITY_CONFIGS[CONFIG]["short"]
N_POS = len(P.RETAINED_POSITIONS)
SMOKE_HEADLINE = 2.9835710748370010e-04
SMOKE_N_TRAJ = 4


def launch_root(i, root=None):
    return os.path.join(root or STUDY_DIR, f"launch{i}")


def collect(n_launches, root=None, allow_dirty=False):
    """One independent engine launch per repetition, each scoring the full grid."""
    out = []
    for i in range(1, n_launches + 1):
        rec = C.collect(CONFIG, root=launch_root(i, root), allow_dirty=allow_dirty)
        out.append({"launch": i,
                    "engine_identity_hash": rec["engine_identity_hash"],
                    "kv_cache_tokens": rec["observed"]["kv_cache_tokens"],
                    "cells": rec["cells"],
                    "seconds": rec["seconds"],
                    "wall_seconds": rec["wall_seconds"],
                    "prefix_cache": {k: v for k, v in (rec.get("engine_metrics") or {}).items()
                                     if "prefix_cache" in k and "external" not in k},
                    "root": os.path.relpath(launch_root(i, root), common.REPO)})
        print(f"launch {i}: engine={rec['engine_identity_hash']} "
              f"kv={rec['observed']['kv_cache_tokens']} cells={rec['cells']} "
              f"scoring={rec['seconds']}s", flush=True)
    return out


def _pair_stats(grid, idx, cells, label):
    traj_means = K.trajectory_means(grid)
    point = float(traj_means.mean())
    flat = grid.reshape(-1)
    worst = int(np.argmax(flat))
    by_pos = {}
    for j, p in enumerate(P.RETAINED_POSITIONS):
        col = grid[:, j]
        by_pos[str(p)] = {"mean_nats": float(col.mean()),
                          "median_nats": float(np.median(col)),
                          "max_nats": float(col.max())}
    return {
        "pair": label,
        "headline_nats": point,
        "headline_bootstrap": K.bootstrap_summary(
            point, K.bootstrap_headline(grid, idx), ci=tuple(q.BOOTSTRAP["ci"])),
        "per_cell": {"mean_nats": float(flat.mean()),
                     "median_nats": float(np.median(flat)),
                     "p95_nats": float(np.quantile(flat, 0.95)),
                     "p99_nats": float(np.quantile(flat, 0.99)),
                     "max_nats": float(flat.max()),
                     "worst_cell": {**cells[worst], "kl_nats": float(flat[worst])}},
        "bit_identical_cells": int((flat == 0.0).sum()),
        "by_position": by_pos,
        # the same first-4 trajectories the smoke measured, so the two are directly comparable
        "first4_headline_nats": float(K.trajectory_means(grid[:SMOKE_N_TRAJ]).mean()),
        "first4_max_nats": float(grid[:SMOKE_N_TRAJ].reshape(-1).max()),
    }


def analyze(n_launches, root=None, allow_dirty=False, out=None):
    q.require_clean_tree(allow_dirty, stage="floor_study")
    traj = T.load()
    n = traj["n_trajectories"]
    idx = K.bootstrap_indices(n, q.BOOTSTRAP["draws"], q.BOOTSTRAP["seed"])

    mats, summaries, ref_cells = {}, {}, None
    for i in range(1, n_launches + 1):
        mat, cells, summary = C.load_matrix(CONFIG, root=launch_root(i, root), n_traj=n)
        A.verify_cells(cells, traj)
        if summary["provenance"]["trajectory_set_hash"] != traj["trajectory_set_hash"]:
            raise SystemExit(f"ABORT: launch {i} was collected against a different trajectory set")
        if summary["provenance"]["kl_spec_hash"] != q.spec_hash():
            raise SystemExit(f"ABORT: launch {i} was collected under a different KL_SPEC")
        if ref_cells is None:
            ref_cells = cells
        elif cells != ref_cells:
            raise SystemExit(f"ABORT: launch {i}'s cell grid differs from launch 1's")
        mats[i], summaries[i] = mat, summary

    identities = {i: s["engine_identity_hash"] for i, s in summaries.items()}
    ckpts = {i: s["provenance"]["checkpoint_content_hash"] for i, s in summaries.items()}
    if len(set(identities.values())) != 1:
        raise SystemExit(f"ABORT: launches span engine identities {identities}; these are not "
                         "repetitions of one configuration")
    if len(set(ckpts.values())) != 1:
        raise SystemExit(f"ABORT: launches span checkpoints {ckpts}")

    pairs = {}
    for a, b in itertools.permutations(range(1, n_launches + 1), 2):
        grid = np.array([K.kl_nats(mats[a][r], mats[b][r], q.GATES["kl_negative_tolerance"])
                         for r in range(mats[a].shape[0])]).reshape(n, N_POS)
        pairs[f"L{a}||L{b}"] = _pair_stats(grid, idx, ref_cells, f"L{a}||L{b}")
        print(f"  {a}||{b} headline={pairs[f'L{a}||L{b}']['headline_nats']:.4e}", flush=True)

    heads = np.array([p["headline_nats"] for p in pairs.values()])
    worsts = np.array([p["per_cell"]["max_nats"] for p in pairs.values()])
    first4 = np.array([p["first4_headline_nats"] for p in pairs.values()])

    pos_agg = {}
    for p in P.RETAINED_POSITIONS:
        vals = np.array([pr["by_position"][str(p)]["mean_nats"] for pr in pairs.values()])
        pos_agg[str(p)] = {"mean_across_pairs_nats": float(vals.mean()),
                           "min_nats": float(vals.min()), "max_nats": float(vals.max()),
                           "spread_ratio": float(vals.max() / vals.min()) if vals.min() > 0 else None}

    # exploratory only: what the floor would be if the reference were an average of launches.
    # Adopting it would change the pre-registered design and would need its own registration;
    # it is NOT used to evaluate the 1% bound anywhere in this artifact.
    mean_logits = np.mean([mats[i] for i in mats], axis=0)
    avg_grid = np.array([K.kl_nats(mean_logits[r], mats[1][r], q.GATES["kl_negative_tolerance"])
                         for r in range(mean_logits.shape[0])]).reshape(n, N_POS)

    smoke_ratio = float(heads.mean() / SMOKE_HEADLINE)
    within = bool(heads.min() <= SMOKE_HEADLINE <= heads.max())
    rec = {
        "artifact": "production-scale BF16 replication floor",
        "purpose": "three independent BF16 launches over the frozen 64 trajectories, all 640 "
                   "retained positions, every ordered launch pair",
        "config": CONFIG,
        "engine_profile_name": q.PROFILE_NAME,
        "n_launches": n_launches,
        "n_trajectories": n,
        "cells_per_launch": n * N_POS,
        "ordered_pairs": len(pairs),
        "trajectory_set_hash": traj["trajectory_set_hash"],
        "kl_spec_hash": q.spec_hash(),
        "engine_identity_hash": sorted(set(identities.values()))[0],
        "checkpoint_content_hash": sorted(set(ckpts.values()))[0],
        "launch_identity": identities,
        "headline": {
            "per_pair_nats": {k: v["headline_nats"] for k, v in pairs.items()},
            "mean_nats": float(heads.mean()),
            "min_nats": float(heads.min()),
            "max_nats": float(heads.max()),
            "std_nats": float(heads.std(ddof=1)),
            "spread_ratio_max_over_min": float(heads.max() / heads.min()),
        },
        "worst_cell": {
            "per_pair_nats": {k: v["per_cell"]["max_nats"] for k, v in pairs.items()},
            "mean_nats": float(worsts.mean()),
            "min_nats": float(worsts.min()),
            "max_nats": float(worsts.max()),
            "cells_per_pair": n * N_POS,
        },
        "per_position": pos_agg,
        "vs_smoke_estimate": {
            "smoke_headline_nats": SMOKE_HEADLINE,
            "smoke_n_trajectories": SMOKE_N_TRAJ,
            "smoke_cells": SMOKE_N_TRAJ * N_POS,
            "smoke_pairs": 1,
            "production_mean_headline_nats": float(heads.mean()),
            "production_over_smoke": smoke_ratio,
            "smoke_value_inside_production_pair_range": within,
            "same_first4_trajectories_headline_nats": {
                "per_pair": {k: v["first4_headline_nats"] for k, v in pairs.items()},
                "mean": float(first4.mean()), "min": float(first4.min()),
                "max": float(first4.max()),
            },
            "note": "headline is a mean and is comparable across grid sizes; the worst cell is an "
                    "extreme-value statistic and the 640-cell and 40-cell maxima are not",
        },
        "exploratory_averaged_reference": {
            "ADOPTED": False,
            "headline_nats": float(K.headline(avg_grid)),
            "max_nats": float(avg_grid.reshape(-1).max()),
            "definition": "D_KL(mean of the three stored logprob matrices || launch 1)",
            "warning": "reported for the design decision only. Averaging launches changes the "
                       "pre-registered reference and would need its own registration; it is not "
                       "used to evaluate the 1% bound in this artifact and must not be quoted as "
                       "a pass.",
        },
        "pre_registered_bound": {
            "fraction_of_bf16_fp8_kl": q.GATES["replication_floor_max_frac_of_fp8"],
            "threshold_unchanged": True,
            "bf16_fp8_kl_required_to_pass_nats": float(
                heads.mean() / q.GATES["replication_floor_max_frac_of_fp8"]),
            "note": "a production BF16||FP8 KL does not exist yet -- that is P13. The smoke's "
                    "n=4 value is quoted in the report for scale only.",
        },
        "pairs": pairs,
        "launches": [{"launch": i, "engine_identity_hash": identities[i],
                      "root": os.path.relpath(launch_root(i, root), common.REPO)}
                     for i in sorted(mats)],
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }
    out = out or os.path.join(q.QUALITY_DIR, "gates", "replication_floor_production.json")
    common.write_json(out, rec)
    return rec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--root", default="")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    root = a.root or None
    if not a.analyze_only:
        collect(a.launches, root=root, allow_dirty=a.allow_dirty)
    if a.collect_only:
        return 0
    rec, out = analyze(a.launches, root=root, allow_dirty=a.allow_dirty, out=a.out or None)
    h = rec["headline"]
    print(f"\nheadline floor: mean {h['mean_nats']:.4e} over {rec['ordered_pairs']} ordered pairs, "
          f"range [{h['min_nats']:.4e}, {h['max_nats']:.4e}], spread {h['spread_ratio_max_over_min']:.2f}x")
    print(f"smoke estimate {SMOKE_HEADLINE:.4e} -> production/smoke = "
          f"{rec['vs_smoke_estimate']['production_over_smoke']:.3f}x")
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
