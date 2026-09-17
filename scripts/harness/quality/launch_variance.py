"""BF16 launch identity as a nuisance variance component. Pure analysis; no engine, no GPU.

Supplements `analyze_kl.py`; supersedes nothing. G2/G2' and their thresholds stand as written --
this does not re-adjudicate them, and it cannot: see `does_not_address_G2_prime` below.

No BF16->BF16 floor is subtracted from any reported KL. That is a rule this module PROPOSES, not
one it inherits -- the only subtraction the tracked docs bar is the FP8->FP4 proxy (D13,
2026-08-25). Its reasoning is its own: a floor-subtracted KL is not a KL, the floor is a spread
rather than a bias, and the two are different functionals of the same perturbation (below).
Comparing the MAGNITUDES of two separately measured divergences is neither barred subtraction nor
a derived divergence, and that -- not "it is ordinal" -- is why the comparisons here are allowed.

The reference is not a fixed distribution: an independent BF16 launch produces a different one, and
the pre-registered analysis conditions on whichever launch happened to run. Launch identity is
treated as a crossed repeated-measure factor over the same frozen cells, so every reported KL is an
average of per-launch KLs against REAL launches. This is not the barred averaged reference: KL is
convex in its first argument, so KL(mean_r B_r || Q) <= mean_r KL(B_r || Q) by Jensen -- the
averaged reference is provably smaller, and the gap is exactly the launch diversity it deletes.
`floor_study.exploratory_averaged_reference` measures that gap at 0.335x and stays unadopted.

The model, stated so it can be checked
-------------------------------------
    Y_{r,t} = mu + A_r + S_t + E_{rt},   Y_{r,t} = mean over the ten retained positions

    E[MS_A] = sigma2_E + T*sigma2_A     df R-1
    E[MS_S] = sigma2_E + R*sigma2_S     df T-1
    E[MS_E] = sigma2_E                  df (R-1)(T-1)

POSITION IS A FIXED FACTOR -- ten pre-registered strided levels, not a sample from a population --
so the within-cell position spread is not an error term and is never used as one. There is no pure
replicate error at all: scoring is deterministic given a launch. What MS_E estimates is therefore

    sigma2_E = sigma2_{A x S} + sigma2_{A x S x P} / P

the launch x trajectory interaction plus a tenth of the three-way term, not measurement noise.

What A_r does NOT cover
-----------------------
- **The generating launch.** The frozen trajectories came from one BF16 launch that is documented
  non-replayable. Its identity is perfectly confounded with S_t and is inestimable at any R, so
  every number here is conditional on it.
- **The quantized launch.** One FP8 and one FP4 collection exist, so the estimand is
  E_r[KL(B_r || Q_1)], not E_{r,q}[.]. Defensible because FP8 and FP4 replicate to ~1e-11 under
  CUDA graphs (`gates/engine_profile.json`), but assumed rather than measured.
- **Between-session drift.** The floor64 launches were collected consecutively inside one
  `floor_study.collect()` call. sigma2_A is a WITHIN-SESSION variance and a lower bound on what a
  differently-dated deployment launch would show.

Why R is the binding problem
----------------------------
sigma2_A_hat is proportional to chi2_{R-1}/(R-1). At R=3 its 95% multiplicative interval is
[0.27, 39.5]; SE_launch is consistent with anything from half to six times the point value, and
under sigma2_A = 0 the moment estimate is negative about 63% of the time. A truncated point
estimate would therefore report 0.0 more often than not and read as "no launch effect". So the
primary launch summary here is MS_A, its F ratio, and a ONE-SIDED 95% UPPER BOUND; the moment point
estimate is reported beside them, untruncated, and never alone.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import kl_math as K, positions as P, qcommon as q  # noqa: E402
from harness.quality import analyze_kl as A, collect_kl as C, trajectories as T  # noqa: E402

N_POS = len(P.RETAINED_POSITIONS)

# BF16 realizations of the frozen grid. `n_trajectories` is what each root actually collected; the
# smoke root collected 4, so any pooled analysis is capped there until P13 supplies more.
BF16_LAUNCH_SOURCES = (
    {"launch": "F1", "root": "results/quality/floor64/launch1", "n_trajectories": 64,
     "provenance": "production replication floor, launch 1"},
    {"launch": "F2", "root": "results/quality/floor64/launch2", "n_trajectories": 64,
     "provenance": "production replication floor, launch 2"},
    {"launch": "F3", "root": "results/quality/floor64/launch3", "n_trajectories": 64,
     "provenance": "production replication floor, launch 3"},
    {"launch": "S", "root": "results/quality/smoke", "n_trajectories": 4,
     "provenance": "P10 smoke; the only BF16 launch collected on the same 40-context grid as "
                   "the FP8/FP4 distributions"},
)
SMOKE_ROOT = "results/quality/smoke"

# Student t, two-sided 0.975, by degrees of freedom. scipy is not a dependency of this repo and a
# launch-inflated interval must not silently fall back to a normal quantile at 2 df.
_T975 = {1: 12.7062047364, 2: 4.30265272991, 3: 3.18244630528, 4: 2.77644510520,
         5: 2.57058183563, 6: 2.44691185114, 7: 2.36462425159, 8: 2.30600413520,
         9: 2.26215716280, 10: 2.22813885196, 12: 2.17881282966, 15: 2.13144954556,
         20: 2.08596344727, 30: 2.04227245630, 60: 2.00029782106}


def t_crit_975(df):
    """Conservative: an unlisted df takes the next SMALLER tabulated df, never a larger one."""
    if df is None or not np.isfinite(df) or df < 1:
        return None
    if df >= 60:
        return 1.95996398454  # the normal limit; the table's tail is flat well before here
    usable = [k for k in sorted(_T975) if k <= df]
    return _T975[usable[-1]]


# chi-square lower 5% quantiles by df, for the one-sided UPPER bound on a variance component.
_CHI2_005 = {1: 0.0039321400, 2: 0.1025865887, 3: 0.3518463061, 4: 0.7107230214,
             5: 1.1454762260, 6: 1.6353828943, 7: 2.1673499092, 8: 2.7326367934,
             9: 3.3251128430, 10: 3.9403004024, 12: 5.2260295285, 15: 7.2609438385,
             20: 10.8508007464, 23: 13.0905141653, 30: 18.4926790000}


def chi2_lower_005(df):
    """Conservative: an unlisted df takes the next SMALLER tabulated df, which widens the bound."""
    usable = [k for k in sorted(_CHI2_005) if k <= df]
    return _CHI2_005[usable[-1]] if usable else None


# Two-sided chi-square quantiles, for a 95% interval on an OBSERVED sd. Used to say whether a
# pre-registered prediction is consistent with what R launches showed -- and, at 2 df, how little
# that test can exclude.
_CHI2_025 = {1: 0.00098207, 2: 0.05063562, 3: 0.21579528, 4: 0.48441857, 5: 0.83121164,
             6: 1.23734674, 7: 1.68986950, 8: 2.17972817, 9: 2.70039052, 10: 3.24697004}
_CHI2_975 = {1: 5.02388619, 2: 7.37775891, 3: 9.34840360, 4: 11.14328678, 5: 12.83250175,
             6: 14.44937578, 7: 16.01276190, 8: 17.53454614, 9: 19.02276780, 10: 20.48317735}


def sd_ci_chi2(sd, df):
    """95% interval for a normal-theory sd on `df` degrees of freedom.

    At the R=3 this design supports, df=2 and the interval spans a factor of 12. Reported so that
    "the prediction is consistent with the data" cannot be mistaken for "the prediction is
    confirmed": almost nothing is excluded at this df.
    """
    if sd is None or df is None or df not in _CHI2_025:
        return None
    lo = float(sd * np.sqrt(df / _CHI2_975[df]))
    hi = float(sd * np.sqrt(df / _CHI2_025[df]))
    return {"sd": float(sd), "df": int(df), "ci_95": [lo, hi],
            "width_ratio": float(hi / lo) if lo > 0 else None,
            "method": "chi-square interval on a normal-theory sd"}


def pooled_sigma_proj(comparisons):
    """One coupling constant shared across the BF16-anchored comparisons.

    The first-order argument makes sigma_proj a property of the launch perturbation, not of the
    configuration, so pooling it buys degrees of freedom that no per-configuration estimate at R=3
    can. Pooled in quadrature with equal weights -- the configurations contribute the same 2 df.
    """
    vals = {k: (c.get("scaling_law") or {}).get("sigma_proj")
            for k, c in comparisons.items()}
    have = {k: v for k, v in vals.items() if v}
    if not have:
        return None
    pooled = float(np.sqrt(np.mean([v ** 2 for v in have.values()])))
    return {
        "sigma_proj_pooled": pooled,
        "per_comparison": have,
        "spread_ratio": float(max(have.values()) / min(have.values())) if len(have) > 1 else None,
        "rule": "root-mean-square over the BF16-anchored comparisons, equal weights",
        "why_pooled": "sigma_proj is a property of the launch perturbation rather than of the "
                      "configuration; agreement across configurations whose signals differ by "
                      "nearly an order of magnitude is the evidence for pooling it",
    }


class LaunchDesignError(ValueError):
    pass


def _rel(path):
    return os.path.relpath(path, common.REPO)


def load_launch_matrices(sources=BF16_LAUNCH_SOURCES, n_traj=4, traj=None):
    """BF16 logprob matrices, one per launch, over the first `n_traj` trajectories.

    `load_matrix` reads only the shards covering those trajectories, so a 4-trajectory design
    touches one 8-trajectory shard per launch rather than all eight. Loading the full grid and
    slicing cost 1.34 GB peak to retain 82 MB.
    """
    traj = traj if traj is not None else C.subset(T.load(), None)
    mats, meta = {}, []
    for src in sources:
        if src["n_trajectories"] < n_traj:
            raise LaunchDesignError(
                f"launch {src['launch']} collected {src['n_trajectories']} trajectories, "
                f"the requested design needs {n_traj}")
        root = os.path.join(common.REPO, src["root"])
        mat, cells, summary = C.load_matrix(q.REFERENCE_CONFIG, root=root, n_traj=n_traj)
        A.verify_cells(cells, C.subset(traj, n_traj))
        mats[src["launch"]] = mat
        meta.append({**src, "cells": cells, "summary": summary})
    return mats, meta


def matrix_digest(mat):
    """SHA-256 of the exact bytes consumed. `dist/*.npy` is gitignored and a BF16 launch provably
    does not regenerate itself, so this digest is the only durable record of which launch produced
    a number here."""
    return hashlib.sha256(np.ascontiguousarray(mat).tobytes()).hexdigest()


def check_exchangeability(meta, n_traj, mats=None):
    """Preconditions for treating these launches as repetitions of ONE configuration.

    The launches were not all collected on the same grid: floor64 scored 640 contexts and the smoke
    40, so batch composition and prefix-cache structure could differ. That is checked numerically
    by `regime_check`; this checks identity, which must match exactly.

    `contexts_hash` and `subset_n` are deliberately NOT in `must_match`: they cannot match across
    grid sizes by construction. Per-cell `context_sha256` equality over the shared cells is the
    substitute, and it is the same authority `analyze_kl.verify_cells` uses.

    `engine_identity_hash` is a CONFIGURATION identity, not a launch nonce -- `qengine`
    hashes only deterministic properties of the configuration, so it can neither distinguish two
    launches nor fail to match across them. It is checked because a mismatch would be
    disqualifying, not because matching is evidence of independence. `content_distinct` is what
    actually establishes that these are four different runs.
    """
    must_match = {
        "kl_spec_hash": lambda m: m["summary"]["provenance"]["kl_spec_hash"],
        "trajectory_set_hash": lambda m: m["summary"]["provenance"]["trajectory_set_hash"],
        "checkpoint_content_hash": lambda m: m["summary"]["provenance"]["checkpoint_content_hash"],
        "tokenizer_identity": lambda m: m["summary"]["provenance"]["tokenizer_identity"],
        "engine_profile_name": lambda m: m["summary"]["provenance"]["engine_profile_name"],
        "storage_dtype": lambda m: m["summary"]["provenance"]["storage_dtype"],
        "retained_positions": lambda m: m["summary"]["provenance"]["retained_positions"],
        "engine_identity_hash": lambda m: m["summary"]["engine_identity_hash"],
        "config_id": lambda m: m["summary"]["config_id"],
    }
    checked = {}
    for key, get in must_match.items():
        vals = {m["launch"]: get(m) for m in meta}
        distinct = {json.dumps(v, sort_keys=True) for v in vals.values()}
        if len(distinct) != 1:
            raise LaunchDesignError(
                f"BF16 launches disagree on {key}: {vals}. These are not repetitions of one "
                "configuration and must not be pooled as launch-level replicates.")
        checked[key] = next(iter(vals.values()))

    ref_cells = meta[0]["cells"]
    for m in meta[1:]:
        if m["cells"] != ref_cells:
            raise LaunchDesignError(
                f"launch {m['launch']}'s first {n_traj * N_POS} cells differ from "
                f"{meta[0]['launch']}'s; the launches would not be scored on the same contexts")

    # A launch that did not launch may be the previous launch's data under a new directory name --
    # or it may be a finished launch whose root was merely RE-ASSEMBLED. `launched` cannot tell
    # them apart, because it describes the last call rather than the cells. `cells_scored_by_launch`
    # can: it comes from the per-shard `observed` block, which only an engine launch writes. The
    # byte-distinctness check below is what actually rules out duplicated data, and it is unaffected
    # by either.
    not_launched = [m["launch"] for m in meta
                    if not (m["summary"].get("launched")
                            or m["summary"].get("cells_scored_by_launch"))]
    if not_launched:
        raise LaunchDesignError(
            f"launches {not_launched} record neither launched=true nor cells_scored_by_launch: "
            "their shards were reused rather than rescored, so they are not independent "
            "realizations")
    # the first collection's timestamp, not the re-assembly's, for the same reason
    stamps = {m["launch"]: (m["summary"].get("cells_first_collected_timestamp")
                            or m["summary"]["timestamp"]) for m in meta}
    if len(set(stamps.values())) != len(stamps):
        raise LaunchDesignError(f"BF16 launches share a collection timestamp: {stamps}")

    digests = {m["launch"]: matrix_digest(mats[m["launch"]]) for m in meta} if mats else {}
    if digests and len(set(digests.values())) != len(digests):
        raise LaunchDesignError(
            f"two BF16 launches consumed byte-identical matrices: {digests}. They are the same "
            "run's data under two directory names, not independent realizations.")

    heads = {m["launch"]: m["summary"]["git"]["git_head"][:8] for m in meta}
    return {
        "n_launches": len(meta),
        "launches": [m["launch"] for m in meta],
        "identities_verified": sorted(must_match),
        "matrix_sha256": digests,
        "content_distinct": bool(digests and len(set(digests.values())) == len(digests)),
        "keys_that_cannot_match_by_construction": {
            "contexts_hash": {m["launch"]: m["summary"]["provenance"]["contexts_hash"][:16]
                              for m in meta},
            "subset_n": {m["launch"]: m["summary"]["provenance"]["subset_n"] for m in meta},
            "substitute": "per-cell context_sha256 equality over the shared cells, asserted above",
        },
        "engine_identity_hash_is_not_a_launch_nonce": (
            "it hashes deterministic configuration properties only; it cannot distinguish "
            "launches. matrix_sha256 is what establishes independence."),
        "collection_git_head": heads,
        "collection_git_head_note": (
            "the launches were collected at different commits. Only new files and a 2-line "
            "gates.py edit separate them; collect_kl.py, qengine.py, positions.py, kl_math.py and "
            "qcommon.py -- the whole collection path -- are byte-identical across the two."
            if len(set(heads.values())) > 1 else "all launches collected at one commit"),
        "session_confound": _session_confound(stamps),
        "shared_identity": {k: v for k, v in checked.items() if k != "tokenizer_identity"},
        "cells_identical_across_launches": n_traj * N_POS,
        "distinct_collection_timestamps": stamps,
        "all_launches_actually_launched": True,
        "grid_regime": {m["launch"]: m["summary"]["n_trajectories"] * N_POS for m in meta},
        "prefix_cache": {
            m["launch"]: _cache_ratio(m["summary"].get("engine_metrics") or {}) for m in meta},
    }


SAME_SITTING_SECONDS = 3600


def _session_confound(stamps):
    """Whether launch identity is confounded with SESSION, decided from the timestamps.

    Both readings are real and they point OPPOSITE ways, so the artifact must say which one
    applies: launches spread over sittings absorb between-session drift (a larger, more
    representative nuisance), launches inside one sitting do not (a within-session LOWER bound on
    what a differently-dated deployment launch would show).
    """
    import datetime as _dt
    try:
        times = sorted(_dt.datetime.fromisoformat(v) for v in stamps.values() if v)
    except (TypeError, ValueError):
        return {"decided_from_timestamps": False,
                "note": "collection timestamps are unparseable; treat the launch component as a "
                        "within-session lower bound, which is the conservative reading"}
    span = (times[-1] - times[0]).total_seconds() if len(times) > 1 else 0.0
    one_sitting = span <= SAME_SITTING_SECONDS
    return {
        "decided_from_timestamps": True,
        "span_seconds": span,
        "one_sitting": one_sitting,
        "note": ("all launches were collected inside one sitting, so sigma2_A is a WITHIN-SESSION "
                 "variance and a lower bound on what a differently-dated deployment launch would "
                 "show" if one_sitting else
                 "the launches span sittings, so launch identity is confounded with session and "
                 "the component absorbs between-session drift as well"),
    }


def _cache_ratio(metrics):
    qy = metrics.get("vllm:prefix_cache_queries")
    hi = metrics.get("vllm:prefix_cache_hits")
    if not qy:
        return None
    return {"queries": qy, "hits": hi, "hit_rate": (hi / qy) if hi is not None else None}


def pair_grid(ma, mb, n_traj):
    return A.pair_grid(ma, mb, n_traj)


def crossed_anova(Y):
    """Balanced two-way crossed decomposition of Y[level, trajectory].

    Returns mean squares and degrees of freedom as the primary output. The moment variance
    components are reported UNTRUNCATED beside them -- a negative value is information about the
    design, and clamping it to zero manufactures a "no effect" reading the data does not support.
    """
    Y = np.asarray(Y, dtype=K.WORKING_DTYPE)
    if Y.ndim != 2:
        raise LaunchDesignError(f"expected a 2-D (level, trajectory) grid, got {Y.shape}")
    R, Tn = Y.shape
    if R < 2 or Tn < 2:
        raise LaunchDesignError(f"decomposition needs >=2 levels and >=2 trajectories, got {Y.shape}")
    gm = Y.mean()
    lvl, traj = Y.mean(axis=1), Y.mean(axis=0)
    ms_a = float(Tn * ((lvl - gm) ** 2).sum() / (R - 1))
    ms_s = float(R * ((traj - gm) ** 2).sum() / (Tn - 1))
    resid = Y - lvl[:, None] - traj[None, :] + gm
    ms_e = float((resid ** 2).sum() / ((R - 1) * (Tn - 1)))

    # Var(theta) = sA/R + sS/T + sE/(RT); substituting the moment estimators collapses to this.
    # One expression with one Satterthwaite df, rather than three truncated pieces added together.
    var_theta = (ms_a + ms_s - ms_e) / (R * Tn)
    den = (ms_a ** 2 / (R - 1) + ms_s ** 2 / (Tn - 1)
           + ms_e ** 2 / ((R - 1) * (Tn - 1)))
    df_theta = float((ms_a + ms_s - ms_e) ** 2 / den) if den > 0 else None

    chi = chi2_lower_005(R - 1)
    var_a_upper = float(ms_a * (R - 1) / (Tn * chi)) if chi else None
    return {
        "levels": int(R),
        "n_trajectories": int(Tn),
        "grand_mean_nats": float(gm),
        "level_means_nats": [float(x) for x in lvl],
        "ms_level": ms_a, "ms_trajectory": ms_s, "ms_residual": ms_e,
        "df_level": int(R - 1), "df_trajectory": int(Tn - 1),
        "df_residual": int((R - 1) * (Tn - 1)),
        "expected_mean_squares": {
            "E[MS_level]": "sigma2_E + T*sigma2_A",
            "E[MS_trajectory]": "sigma2_E + R*sigma2_S",
            "E[MS_residual]": "sigma2_E = sigma2_{AxS} + sigma2_{AxSxP}/P, NOT measurement noise",
        },
        "var_level_moment": float((ms_a - ms_e) / Tn),
        "var_trajectory_moment": float((ms_s - ms_e) / R),
        "var_residual": ms_e,
        "moment_components_may_be_negative": True,
        "var_level_upper_95": var_a_upper,
        "var_level_upper_95_rule": "sigma2_A <= MS_A*(R-1)/(T*chi2_{0.05,R-1}), using "
                                   "sigma2_E >= 0. One-sided; the only honest summary at this df.",
        "f_level": (ms_a / ms_e) if ms_e > 0 else None,
        "f_level_note": "MS_level/MS_residual on (df_level, df_residual). Tests whether the level "
                        "effect is separable from the level x trajectory interaction at all.",
        "var_theta_anova": float(var_theta),
        "se_theta_anova_nats": float(np.sqrt(var_theta)) if var_theta > 0 else None,
        "df_theta_satterthwaite": df_theta,
        "var_theta_rule": "(MS_level + MS_trajectory - MS_residual)/(R*T), the ANOVA-unbiased "
                          "Var(mean); its Satterthwaite df is the honest df of the whole estimate",
    }


# F, upper 5%, for the small (df1, df2) this design produces. scipy is not a dependency.
# Rounded UP from the exact quantile, never nearest: an entry below the true critical value calls
# a borderline effect separable when it is not. The (2,126) entry actually used is 3.0681 exact.
_F95 = {(1, 3): 10.129, (1, 9): 5.118, (2, 6): 5.144, (2, 9): 4.257, (2, 63): 3.143,
        (2, 126): 3.069, (2, 189): 3.044, (3, 9): 3.863, (3, 12): 3.491, (3, 63): 2.751,
        (3, 189): 2.653, (5, 105): 2.301, (5, 315): 2.243}


def _f_exceeds_95(f, df1, df2):
    """Unlisted (df1, df2) falls back to the largest tabulated critical value for that df1, which
    is the conservative direction: it can only make `separable` harder to claim."""
    if f is None:
        return None
    if (df1, df2) in _F95:
        return bool(f > _F95[(df1, df2)])
    same = [v for (a, _), v in _F95.items() if a == df1]
    return bool(f > max(same)) if same else None


def _launch_component(anova, R):
    """The launch summary: F ratio and a one-sided upper bound first, point estimate beside them."""
    var_pt = anova["var_level_moment"] / R
    var_up = (anova["var_level_upper_95"] / R
              if anova["var_level_upper_95"] is not None else None)
    return {
        "primary": "F ratio and the one-sided 95% upper bound; the point estimate is reported "
                   "beside them and must not be quoted alone at this df",
        "ms_level": anova["ms_level"],
        "df_level": anova["df_level"],
        "df_residual": anova["df_residual"],
        "f_launch": anova["f_level"],
        "launch_effect_separable_at_0_05": _f_exceeds_95(
            anova["f_level"], anova["df_level"], anova["df_residual"]),
        "sd_launch_identity_nats_point": (float(np.sqrt(anova["var_level_moment"]))
                                          if anova["var_level_moment"] > 0 else None),
        "sd_launch_identity_nats_point_is_negative_moment": bool(anova["var_level_moment"] <= 0),
        "sd_launch_identity_nats_upper_95": (float(np.sqrt(anova["var_level_upper_95"]))
                                             if anova["var_level_upper_95"] is not None else None),
        "var_launch_point": float(var_pt),
        "var_launch_upper_95": var_up,
        "se_launch_nats_point": float(np.sqrt(var_pt)) if var_pt > 0 else None,
        "se_launch_nats_upper_95": float(np.sqrt(var_up)) if var_up else None,
        "df_caveat": f"sigma2_A carries {anova['df_level']} df. Its 95% multiplicative interval is "
                     "[0.27, 39.5] at 2 df and [0.32, 13.9] at 3 df; reaching +/-30% on "
                     "SE_launch would need R=37, which is not reachable here.",
    }


def scaling_law(theta, sd_launch_headlines):
    """First-order coupling between the launch perturbation and the signal.

    Write B_r = B* + delta_r and Q = B* + Delta. In the Fisher metric
        KL(B_a||B_b) ~ (1/2)||delta_a - delta_b||^2            SECOND order in delta  (the floor)
        KL(B_r||Q)   ~ (1/2)||Delta||^2 - <delta_r, Delta>     FIRST order in delta   (the signal)
    so the launch-induced SD of the signal scales as sqrt(2*signal), while the floor does not scale
    with the signal at all. The coupling constant sigma_proj = SD/sqrt(2*signal) is the quantity
    that is shared across configurations, and sharing it is what buys degrees of freedom the
    per-configuration 2-3 df cannot.

    This is also why averaging launches cannot rescue G2': averaging shrinks the FIRST-order
    nuisance by sqrt(R) and leaves the SECOND-order floor untouched. They are different functionals
    of the same delta.
    """
    if theta <= 0 or sd_launch_headlines is None:
        return None
    return {
        "sd_launch_headlines_nats": float(sd_launch_headlines),
        "sigma_proj": float(sd_launch_headlines / np.sqrt(2.0 * theta)),
        "definition": "sigma_proj = SD(per-launch headlines) / sqrt(2 * headline)",
        "prediction": "SD_launch(config) = sigma_proj * sqrt(2 * signal(config)); "
                      "CV_launch falls as 1/sqrt(signal)",
        "caution": "SD(per-launch headlines) is sqrt(sigma2_A + sigma2_E/T), NOT sigma_A. It is "
                   "the right input here because the law predicts the observable spread, but it "
                   "must not be quoted as the launch variance component.",
    }


def decompose(cell_stack, idx, label, floor_nats=None):
    """Full launch/trajectory decomposition of one BF16->X comparison.

    cell_stack: (n_launches, n_trajectories, n_positions) of per-cell KL in nats.
    """
    D = np.asarray(cell_stack, dtype=K.WORKING_DTYPE)
    if D.ndim != 3:
        raise LaunchDesignError(f"expected (launch, trajectory, position), got {D.shape}")
    R, Tn, Pn = D.shape
    Y = D.mean(axis=2)                       # (launch, trajectory)
    per_traj = Y.mean(axis=0)                # launch-averaged trajectory means
    theta = float(per_traj.mean())

    # R=1 is the pre-registered design: the estimate is the existing headline and the launch
    # component is not estimable. Reported as absent, never as zero -- zero would read as measured.
    anova = crossed_anova(Y) if R >= 2 else None
    launch = _launch_component(anova, R) if anova else {
        "primary": "not estimable", "f_launch": None,
        "se_launch_nats_point": None, "se_launch_nats_upper_95": None,
        "launch_effect_separable_at_0_05": None,
        "not_estimable_because": "one BF16 launch; launch variance has no degrees of freedom. "
                                 "This is the pre-registered single-reference design, and at R=1 "
                                 "the POINT estimate generalises but the uncertainty "
                                 "decomposition does not exist.",
    }

    draws = per_traj[idx].mean(axis=1)
    boot = K.bootstrap_summary(theta, draws, ci=tuple(q.BOOTSTRAP["ci"]))
    var_boot = float(boot["std_error"] ** 2)
    var_launch_pt = max(0.0, launch.get("var_launch_point") or 0.0)
    var_launch_up = launch.get("var_launch_upper_95")

    # the interval is SCALED, not rebuilt symmetrically: the percentile bootstrap was chosen for a
    # right-skewed non-negative statistic, and theta +/- z*SE can cross zero at the long positions
    infl_pt = np.sqrt(1.0 + var_launch_pt / var_boot) if var_boot > 0 else None
    infl_up = (np.sqrt(1.0 + var_launch_up / var_boot)
               if var_boot > 0 and var_launch_up else None)

    rec = {
        "comparison": label,
        "n_launches": int(R),
        "n_trajectories": int(Tn),
        "n_positions": int(Pn),
        "cells_per_launch": int(Tn * Pn),
        "estimator": "mean over BF16 launches of the per-launch headline; each per-launch "
                     "headline is the pre-registered mean of per-trajectory means",
        "NOT_THE_HEADLINE": "EVALUATION_RIG.md A.1 locks headline_KL as the mean of 64 "
                            "per-trajectory means under ONE reference launch, and that remains "
                            "the point estimate. This is a supplementary expected value over "
                            "launch identity; the per-launch headlines are reported beside it and "
                            "this number must never be rendered as `the BF16->Q KL`.",
        "floor_subtracted": False,
        "expected_kl_over_launches_nats": theta,
        "per_launch_headline_nats": {},
        "per_trajectory_launch_averaged_nats": [float(x) for x in per_traj],
        "variance_components": anova,
        "variance_components_unavailable_because": (
            None if anova else "a decomposition needs >= 2 BF16 launches"),
        "launch_component": launch,
        "trajectory_component": {
            "method": "pre-registered percentile bootstrap over trajectories, applied to the "
                      "launch-averaged per-trajectory means",
            "bootstrap": boot,
            "se_trajectory_nats": float(boot["std_error"]),
            "estimates": "(sigma2_S + sigma2_E/R)/T, the non-launch part of Var(theta)",
            "different_estimand_from_committed_artifacts": (
                "the committed single-launch bootstrap estimates (sigma2_S + sigma2_E)/T; this "
                "one estimates (sigma2_S + sigma2_E/R)/T. They differ by sigma2_E(1-1/R)/T and "
                "are NOT the same interval, so no comparability with earlier artifacts is "
                "claimed" if R >= 2 else None),
            "finite_sample_note": f"the nonparametric bootstrap of a mean carries a (T-1)/T "
                                  f"divisor, so at T={Tn} it understates the SD by "
                                  f"{1 - np.sqrt((Tn - 1) / Tn):.1%}",
        },
        "combined": {
            "launch_variance_included": anova is not None,
            "anova_unbiased": {
                "var_theta": anova["var_theta_anova"] if anova else None,
                "se_theta_nats": anova["se_theta_anova_nats"] if anova else None,
                "df_satterthwaite": anova["df_theta_satterthwaite"] if anova else None,
                "rule": anova["var_theta_rule"] if anova else None,
                "preferred": "this is the single-expression total with one trackable df; the "
                             "bootstrap-plus-launch form below is reported because the bootstrap "
                             "is the pre-registered method",
            },
            "bootstrap_plus_launch": {
                "se_point_nats": float(np.sqrt(var_boot + var_launch_pt)),
                "se_upper_95_nats": (float(np.sqrt(var_boot + var_launch_up))
                                     if var_launch_up else None),
                "inflation_factor_point": float(infl_pt) if infl_pt else None,
                "inflation_factor_upper_95": float(infl_up) if infl_up else None,
                "ci_point": _scale_ci(boot, theta, infl_pt),
                "ci_upper_95": _scale_ci(boot, theta, infl_up),
                "rule": "the percentile interval is SCALED about the point estimate by "
                        "sqrt(1 + var_launch/var_bootstrap); the launch component is thereby "
                        "treated as symmetric while the trajectory skew is preserved",
                "orthogonality": "the two variances add because A_r is a single constant shared "
                                 "by every trajectory, so no trajectory resampling can see it. "
                                 "This holds by the assumed independence of A from (S,E), not by "
                                 "construction. Note the naive s^2 over the R launch headlines "
                                 "would double-count sigma2_E/(RT) against the bootstrap term; "
                                 "(MS_A - MS_E)/T is what avoids that.",
            },
        },
        "relative": {
            "cv_launch_point": (launch["se_launch_nats_point"] / theta
                                if theta > 0 and launch.get("se_launch_nats_point") else None),
            "cv_launch_upper_95": (launch["se_launch_nats_upper_95"] / theta
                                   if theta > 0 and launch.get("se_launch_nats_upper_95")
                                   else None),
            "cv_trajectory": (float(boot["std_error"]) / theta) if theta > 0 else None,
        },
    }
    if anova:
        rec["scaling_law"] = scaling_law(
            theta, float(np.std(anova["level_means_nats"], ddof=1)))
    if floor_nats is not None:
        rec["vs_replication_floor"] = {
            **K.floor_comparison(theta, floor_nats),
            "n_bf16_launches_averaged": int(R),
            "ratio_drifts_with_R": "averaging R launches shrinks the launch noise in the "
                                   "numerator by sqrt(R) while the denominator stays a "
                                   "two-launch quantity, so this ratio moves with a design "
                                   "parameter. Read R beside it.",
            "launch_sd_vs_floor": (
                float(rec["scaling_law"]["sd_launch_headlines_nats"] / floor_nats)
                if anova and floor_nats else None),
            "note": "the floor is reported beside the estimate and is never subtracted from it; "
                    "G2/G2' adjudicate the floor and are unchanged by this analysis",
        }
    return rec, Y, per_traj


def _scale_ci(boot, theta, factor):
    """Scale a percentile interval about the point estimate, preserving its asymmetry."""
    if not factor:
        return None
    return [float(theta + (boot["ci_low"] - theta) * factor),
            float(theta + (boot["ci_high"] - theta) * factor)]


def _satterthwaite(v1, df1, v2, df2):
    """Effective df of v1 + v2. Returns None when a component is zero-variance (no df to pool)."""
    if v1 <= 0 and v2 <= 0:
        return None
    num = (v1 + v2) ** 2
    den = 0.0
    if v1 > 0 and df1 > 0:
        den += v1 ** 2 / df1
    if v2 > 0 and df2 > 0:
        den += v2 ** 2 / df2
    return float(num / den) if den > 0 else None


def by_position(cell_stack, idx, floor_by_position=None, sigma_proj=None):
    """Per-position means, with NO variance components.

    A per-position MS_E on raw cells has a nominal df of (R-1)(T-1) but an effective df of about
    4.6% of that: the raw KL cells have kurtosis ~45, and matching Var(MS) to a chi-square gives
    df_eff = 2n/(kurtosis-1). Per-position components and any interval built from them are not
    supportable, and the moment estimate truncates to exactly zero at most positions anyway.

    What IS reported per position: the launch-averaged mean, the per-launch spread as a plain
    descriptive range, the production floor beside it, and -- where a pooled coupling constant is
    available -- the launch SD the first-order scaling law PREDICTS. The prediction borrows df from
    every configuration at once, which is the only way to say anything per position at this R.

    The ten columns are strongly dependent (they are repeated measures on one trajectory), so no
    statement here may be read as ten independent tests.
    """
    D = np.asarray(cell_stack, dtype=K.WORKING_DTYPE)
    out = {}
    for j, p in enumerate(P.RETAINED_POSITIONS):
        col = D[:, :, j]                                  # (launch, trajectory)
        per_launch = col.mean(axis=1)
        theta = float(per_launch.mean())
        per_traj = col.mean(axis=0)
        draws = per_traj[idx].mean(axis=1)
        boot = K.bootstrap_summary(theta, draws, ci=tuple(q.BOOTSTRAP["ci"]))
        floor = (floor_by_position or {}).get(str(p))
        rec = {
            "expected_kl_over_launches_nats": theta,
            "per_launch_nats": [float(x) for x in per_launch],
            "launch_spread_nats": {
                "min": float(per_launch.min()), "max": float(per_launch.max()),
                "sd_descriptive": float(per_launch.std(ddof=1)) if per_launch.size > 1 else None,
                "sd_is_not_sigma_A": "this is sqrt(sigma2_A + sigma2_E/T) -- a descriptive spread, not sigma_A",
            },
            "trajectory_bootstrap": boot,
            "variance_components": None,
            "variance_components_omitted_because": "effective df on raw cells is ~5% of nominal "
                                                   "(kurtosis ~45); the components are not "
                                                   "estimable per position",
        }
        if sigma_proj and theta > 0:
            pred = sigma_proj * np.sqrt(2.0 * theta)
            rec["scaling_law_predicted_launch_sd_nats"] = float(pred)
            rec["scaling_law_predicted_cv_launch"] = float(pred / theta)
        if floor is not None:
            rec["replication_floor_nats"] = floor
            rec["ratio_to_floor"] = (theta / floor) if floor else None
            rec["floor_source"] = "production per-position floor (64 trajectories, 6 ordered "
            rec["floor_source"] += "pairs); the 4-trajectory per-position floor swings by an "
            rec["floor_source"] += "order of magnitude and is not used"
        out[str(p)] = rec
    return out


REGIME_GROSS_FACTOR = 2.0


def regime_check(mats, launches, n_traj, odd="S", gross_factor=REGIME_GROSS_FACTOR):
    """Is the smoke launch exchangeable with the floor64 launches, or is its 40-context grid a
    separate regime?

    The FP8/FP4 distributions exist only on the smoke's grid, so a regime effect would confound
    grid size with launch identity. The statistic is each launch's LEVERAGE -- its mean divergence
    to every other launch -- because under exchangeability the launch labels are arbitrary and the
    leverages are exchangeable too.

    Deliberately NOT a containment test on the pair ranges: with R(R-1) pairs from four launches the
    extremes bounce, and requiring the cross-regime range to sit inside the within-regime range
    fails about half the time under exchangeability itself. What survives at this R is a rank and a
    gross trip-wire, and the rank cannot reach p<0.25 with four launches -- stated, not hidden.
    """
    if odd not in launches or len(launches) < 3:
        return None
    within = [l for l in launches if l != odd]
    heads = {}
    for a, b in itertools.permutations(launches, 2):
        heads[(a, b)] = float(K.headline(pair_grid(mats[a], mats[b], n_traj)))
    leverage = {l: float(np.mean([v for (a, b), v in heads.items() if l in (a, b)]))
                for l in launches}
    ordered = sorted(leverage, key=lambda l: leverage[l], reverse=True)
    rank = ordered.index(odd) + 1

    within_pairs = {f"{a}||{b}": v for (a, b), v in heads.items() if a != odd and b != odd}
    cross_pairs = {f"{a}||{b}": v for (a, b), v in heads.items() if odd in (a, b)}
    wmax = max(within_pairs.values())
    cmean = float(np.mean(list(cross_pairs.values())))
    gross = bool(cmean > gross_factor * wmax)

    return {
        "question": "does the smoke's 40-context grid put its BF16 launch in a different regime "
                    "from the 640-context floor64 launches?",
        "structural_argument": (
            "trajectories 0-3 are scored first in both regimes, from an identical cold engine, in "
            "identical submission order (one trajectory of ten nested contexts per generate call), "
            "with identical num_gpu_blocks and kv_cache_tokens and zero preemptions. KV eviction "
            "in the 640-context grid begins far later than trajectory 3 and cannot propagate "
            "backwards, so the regime difference is causally downstream of the cells used here."),
        "statistic": "launch leverage -- the mean divergence from one launch to every other",
        "leverage_nats": leverage,
        "leverage_rank_of_odd_launch": rank,
        "n_launches": len(launches),
        "rank_p_value_floor": 1.0 / len(launches),
        "power_caveat": f"with {len(launches)} launches the smallest attainable rank p-value is "
                        f"{1.0 / len(launches):.2f}, so this test CANNOT reject exchangeability at "
                        "0.05 and a moderate regime effect would go undetected. It rules out a "
                        "gross effect, nothing finer.",
        "within_regime_pairs_nats": within_pairs,
        "cross_regime_pairs_nats": cross_pairs,
        "within_regime_range_nats": [float(min(within_pairs.values())), float(wmax)],
        "cross_regime_mean_nats": cmean,
        "gross_regime_factor": gross_factor,
        "gross_regime_effect": gross,
        "verdict": (f"cross-regime mean {cmean:.3e} exceeds {gross_factor}x the largest "
                    f"within-regime pair {wmax:.3e}; DO NOT pool the {odd} launch" if gross else
                    f"no gross regime effect: cross-regime mean {cmean:.3e} against a largest "
                    f"within-regime pair of {wmax:.3e}, and {odd} ranks {rank} of "
                    f"{len(launches)} on leverage. The launches are pooled."),
        "what_this_cannot_rule_out": "a regime effect common to all launches shifts every BF16 "
                                     "realization together and is invisible to any between-launch "
                                     "comparison; and a moderate effect is beyond this test's "
                                     "power. Only a floor64-scale FP8/FP4 collection (P13) closes "
                                     "either.",
    }


def bf16_to_bf16(mats, launches, n_traj, idx):
    """The BF16->BF16 run-to-run quantity, over all ordered launch pairs.

    Deliberately NOT decomposed with `crossed_anova` over pairs: the R(R-1) ordered pairs share
    launches (L1||L2 and L1||L3 both carry launch 1), so they are not independent levels of a
    random factor and an ANOVA over them has neither R(R-1)-1 df nor an interpretable variance
    component. Reported descriptively, with each launch's leverage.
    """
    grids, heads = {}, {}
    for a, b in itertools.permutations(launches, 2):
        g = pair_grid(mats[a], mats[b], n_traj)
        grids[f"{a}||{b}"] = g
        heads[f"{a}||{b}"] = float(K.headline(g))
    vals = np.array(list(heads.values()))
    full = float(vals.mean())
    # Phi is a U-statistic over launches: each launch appears in 2(R-1) of the R(R-1) ordered
    # pairs, so the pairs are not independent levels and sd/sqrt(R(R-1)) understates the spread.
    # Delete-one-LAUNCH jackknife is the variance the design actually supports.
    pseudo, Rn = [], len(launches)
    for l in launches:
        keep = [v for k, v in heads.items() if l not in k.split("||")]
        pseudo.append(Rn * full - (Rn - 1) * float(np.mean(keep)))
    pseudo = np.array(pseudo)
    se_jk = float(pseudo.std(ddof=1) / np.sqrt(Rn))
    naive = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    tc = t_crit_975(Rn - 1)
    stack = np.array([grids[k] for k in sorted(grids)])
    per_traj = stack.mean(axis=(0, 2))
    draws = per_traj[idx].mean(axis=1)
    leverage = {a: float(np.mean([v for k, v in heads.items() if a in k.split("||")]))
                for a in launches}
    by_pos = {}
    for j, p in enumerate(P.RETAINED_POSITIONS):
        per_pair = {k: float(g[:, j].mean()) for k, g in grids.items()}
        col = np.array(list(per_pair.values()))
        by_pos[str(p)] = {"mean_over_ordered_pairs_nats": float(col.mean()),
                          "min_nats": float(col.min()), "max_nats": float(col.max()),
                          "sd_over_ordered_pairs_nats": float(col.std(ddof=1)),
                          # the ordered pairs share launches, so an interval on this mean has to
                          # come from a delete-one-LAUNCH jackknife, not from these six values
                          "per_ordered_pair_nats": per_pair,
                          "launch_level_uncertainty": floor_launch_ci(per_pair, len(launches))}
    return {
        "quantity": "D_KL(B_a || B_b) over ordered BF16 launch pairs -- the replication floor",
        "n_launches": len(launches),
        "ordered_pairs": len(heads),
        "n_trajectories": n_traj,
        "per_pair_headline_nats": heads,
        "mean_nats": float(vals.mean()),
        "min_nats": float(vals.min()),
        "max_nats": float(vals.max()),
        "sd_over_ordered_pairs_nats": float(vals.std(ddof=1)),
        "launch_level_uncertainty": {
            "method": "delete-one-launch jackknife; the ordered pairs are a U-statistic over "
                      "launches, not R(R-1) independent levels",
            "pseudo_values_nats": [float(x) for x in pseudo],
            "se_nats": se_jk,
            "naive_se_over_ordered_pairs_nats": naive,
            "naive_understates_by": (se_jk / naive) if naive > 0 else None,
            "df": Rn - 1,
            **_nonneg_ci(full, se_jk, tc),
            "why": "treating the ordered pairs as independent levels would report df=R(R-1)-1 "
                   "where the design has R-1, and understate the floor's own uncertainty",
        },
        "launch_leverage_nats": leverage,
        "trajectory_bootstrap": K.bootstrap_summary(float(per_traj.mean()), draws,
                                                    ci=tuple(q.BOOTSTRAP["ci"])),
        "by_position": by_pos,
        "no_variance_component_reported_because": (
            "the ordered pairs share launches, so they are not independent levels of a random "
            "factor; a crossed ANOVA over them would report a df it does not have"),
        "relationship_to_G2_prime": (
            "this recomputes the same quantity G2' adjudicated. G2' and its 1% bound are "
            "historical and unchanged; nothing here re-adjudicates them"),
    }


def resolvability(theta_rec, floor_mean, floor_ci, per_traj_q, per_traj_phi, idx):
    """Is the quantization effect still resolvable once BF16 reference variance is carried?

    An INTERVAL COMPARISON, with the uncertainty on both sides made explicit. The earlier design
    here was a dominance count -- the fraction of trajectories with Y^Q_t > Phi_t -- and it was
    dropped on review for three reasons, all of which stand: it is identically the sign test on the
    contrast it claimed not to compute; it is 4/4 wherever it is defined and would be 64/64 at
    production scale, while sitting at ~0.5 exactly at the long positions where the question bites;
    and its null ("an FP8 deployment is no further from BF16 than another BF16 launch is") is not a
    proposition anyone holds.

    The per-trajectory RATIO replaces it. A ratio is not a difference, does not degenerate, and
    keeps the pairing that makes the two arms comparable.
    """
    theta = theta_rec["expected_kl_over_launches_nats"]
    boot = theta_rec["trajectory_component"]["bootstrap"]
    lc = theta_rec["launch_component"]
    combined = theta_rec["combined"]["bootstrap_plus_launch"]
    # At production scale some trajectories reproduce BIT-IDENTICALLY across every launch pair, so
    # their floor is exactly 0.0 and their ratio is +inf. The mean and its bootstrap are then
    # genuinely undefined, not merely awkward -- and dropping those trajectories would select on
    # the denominator, removing exactly the ones where the reference is most reproducible. The
    # median keeps all T trajectories (the infinities order at the top), and the ratio of means is
    # reported beside it as a defined alternative. The n=4 smoke could not surface this: all four
    # of its trajectories had a non-zero floor.
    zero_floor = int((per_traj_phi == 0.0).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_t = per_traj_q / per_traj_phi
        ratio_draws = ratio_t[idx].mean(axis=1)
    defined = zero_floor == 0
    phi_mean = float(per_traj_phi.mean())
    return {
        "form": "interval comparison; no floor is subtracted anywhere",
        "signal": {
            "expected_kl_over_launches_nats": theta,
            "trajectory_ci": [boot["ci_low"], boot["ci_high"]],
            "launch_inflated_ci_point": combined["ci_point"],
            "launch_inflated_ci_upper_95": combined["ci_upper_95"],
            "se_launch_upper_95_nats": lc.get("se_launch_nats_upper_95"),
            "survives_worst_case_launch_variance": (
                bool(combined["ci_upper_95"] and combined["ci_upper_95"][0] > floor_mean)
                if combined.get("ci_upper_95") else None),
            "criterion": "the lower end of the launch-inflated interval, taken at the ONE-SIDED "
                         "95% UPPER bound on the launch variance, still exceeds the floor",
        },
        "floor": {
            "replication_floor_nats": floor_mean,
            "launch_level_ci_95": floor_ci,
            "phi_used": "the PRODUCTION floor headline (64 trajectories, 3 launches), with the "
                        "delete-one-launch jackknife interval. Means are comparable across grid "
                        "sizes; no worst-cell comparison is reported anywhere here, because a "
                        "maximum is an extreme-value statistic and 40-cell and 640-cell maxima "
                        "are not comparable.",
        },
        "per_trajectory_ratio": {
            "definition": "per trajectory, (BF16->Q KL) / (BF16->BF16 KL) on the SAME trajectory "
                          "-- a ratio, not a difference, and not a subtraction",
            "trajectories_with_zero_floor": zero_floor,
            "mean": (float(ratio_t.mean()) if defined else None),
            "mean_undefined_because": (
                None if defined else
                f"{zero_floor} of {per_traj_phi.size} trajectories reproduce bit-identically "
                "across every BF16 launch pair, so their floor is exactly 0.0 and their ratio is "
                "infinite. The mean and its bootstrap do not exist; they are reported as null "
                "rather than computed over a denominator-selected subset."),
            "median": (float(np.median(ratio_t)) if np.isfinite(np.median(ratio_t)) else None),
            "min": (float(ratio_t.min()) if np.isfinite(ratio_t.min()) else None),
            "ratio_of_means": (float(per_traj_q.mean() / phi_mean) if phi_mean > 0 else None),
            "ratio_of_means_note": "a different estimand from the mean of ratios, and defined "
                                   "whenever the pooled floor is non-zero",
            "max": (float(ratio_t.max()) if defined else None),
            "bootstrap_ci": (list(K.percentile_ci(ratio_draws, tuple(q.BOOTSTRAP["ci"])))
                             if defined else None),
            "phi_used": "the launch-pair-averaged BF16->BF16 KL on THESE trajectories, a "
                        "different quantity from the production floor headline above",
            "launch_coverage": "none -- the same BF16 launches enter both arms, so this is "
                               "conditional on the launch set and its interval covers trajectory "
                               "sampling only",
        },
        "does_not_address_G2_prime": (
            "averaging launches shrinks the FIRST-order launch nuisance by sqrt(R) and leaves the "
            "SECOND-order floor untouched -- they are different functionals of the same "
            "perturbation. No value of R moves the G2' ratio in the direction of a pass, and this "
            "analysis is not an argument that it should."),
    }


def _pooling_evidence(per_launch, odd="S"):
    """Where the odd-grid launch ranks on the quantity pooling actually affects.

    `regime_check` compares BF16<->BF16 divergences, which are symmetric in the two launches and
    say nothing about whether the odd launch's KL TO FP8/FP4 is exchangeable with the others. This
    reports that directly: under exchangeability the odd launch's rank is uniform on 1..R.
    """
    if odd not in per_launch or len(per_launch) < 2:
        return None
    order = sorted(per_launch, key=lambda k: per_launch[k])
    rank = order.index(odd) + 1
    R = len(per_launch)
    return {
        "statistic": "rank of the odd-grid launch's per-launch headline among all launches",
        "odd_launch": odd,
        "rank": rank,
        "of": R,
        "extreme": bool(rank in (1, R)),
        "p_extreme_under_exchangeability": 2.0 / R,
        "caveat": "the odd launch's 40-context grid is perfectly confounded with launch identity "
                  "for that one level, so an extreme rank cannot be attributed. With R=4 this "
                  "statistic cannot reach p<0.5 for a two-sided extreme; it is disclosure, not a "
                  "test. R=3 -- the launches that share a grid regime -- is the reported variant "
                  "for this reason.",
    }


def _check_committed(label, s_headline, smoke_root, single_launch_grid=None,
                     reference_launch="the P10 smoke's BF16 collection"):
    """Recover the committed single-launch headline, and where possible re-derive it.

    The committed value is always read -- it is the baseline this estimator has to be shown to move
    in a stated direction, and the smoke's BF16 launch is not in the launch set by default. When it
    IS in the set, its recomputed headline must match the artifact exactly: that is the end-to-end
    check that this path is the one that produced the tracked number.
    """
    path = os.path.join(common.REPO, smoke_root, "kl_summary.json")
    if not os.path.exists(path):
        raise LaunchDesignError(
            f"{_rel(path)} is absent; the committed single-launch headline cannot be recovered "
            "and the direction this estimator moves G2' could not be stated")
    committed = json.load(open(path))["pairs"].get(label, {}).get("headline_nats")
    if committed is None:
        raise LaunchDesignError(f"{_rel(path)} records no headline for {label}")
    rec = {"artifact": _rel(path), "committed_headline_nats": committed,
           "reference_launch": reference_launch}
    if s_headline is None:
        rec["recomputed_bit_identical"] = None
        rec["not_recomputed_because"] = ("the committed reference launch is not in this launch "
                                         "set; the value is read from the artifact")
        return rec
    rel = abs(s_headline - committed) / committed if committed else None
    if rel is not None and rel > RECOMPUTE_REL_TOL:
        raise LaunchDesignError(
            f"{label}: the smoke launch recomputes to {s_headline!r} but "
            f"{_rel(path)} records {committed!r} -- relative difference {rel:.2e}, above the "
            f"{RECOMPUTE_REL_TOL:.0e} bar. That is larger than interpreter drift and the "
            "recomputation path differs from the one that produced the tracked artifact.")
    rec.update({
        "recomputed_nats": float(s_headline),
        "recomputed_bit_identical": bool(s_headline == committed),
        "relative_difference": rel,
        "tolerance": RECOMPUTE_REL_TOL,
        "why_not_bit_identical": (
            None if s_headline == committed else
            "the interpreter that produced results/quality/ is not the one running now (the "
            "artifact records python 3.12.13 with torch and vLLM importable; neither is true "
            "here), so np.log/np.exp inside kl_nats differ in their last bits. An exact fsum over "
            "the 40 cells reproduces THIS value, not the committed one, so the difference is in "
            "the per-cell KL values rather than in the reduction order."),
    })
    return rec


# Bit-equality is the wrong bar across interpreters: the committed quality artifacts were produced
# under python 3.12 with torch/vLLM present, and np.log/np.exp move in their last bits between
# builds. The measured drift is 3.3e-14 (FP8) and 3.3e-15 (FP4); this sits ~300x above it and ~10
# orders of magnitude below anything that could change a reading.
RECOMPUTE_REL_TOL = 1e-11

PRODUCTION_FLOOR = os.path.join(q.QUALITY_DIR, "gates", "replication_floor_production.json")
DEFAULT_OUT = os.path.join(q.QUALITY_DIR, "launch_variance.json")


def _nonneg_ci(point, se, tc):
    """A symmetric t interval on a non-negative quantity can put its lower end below zero, which is
    not an attainable value for a divergence. Both are reported: the raw interval says what the
    arithmetic gave, the clamped one says what is attainable."""
    if not tc:
        return {"ci_95": None, "ci_95_raw": None, "ci_95_lower_clamped": None}
    lo, hi = point - tc * se, point + tc * se
    return {
        "ci_95": [max(0.0, lo), hi],
        "ci_95_raw": [lo, hi],
        "ci_95_lower_clamped": bool(lo < 0.0),
        "ci_95_note": ("the symmetric interval's lower end fell below zero, which a divergence "
                       "cannot reach; reported clamped, with the raw value beside it"
                       if lo < 0.0 else None),
    }


def floor_launch_ci(per_pair_nats, n_launches):
    """Delete-one-launch jackknife interval for a floor recorded as per-ordered-pair headlines.

    The production artifact reports the mean over six ordered pairs and their spread, but six
    pairs from three launches carry R-1 = 2 df, not 5. Recomputed here rather than read, because
    the artifact does not carry a launch-level interval.
    """
    heads = {tuple(k.split("||")): v for k, v in per_pair_nats.items()}
    launches = sorted({x for k in heads for x in k})
    if len(launches) != n_launches:
        raise LaunchDesignError(
            f"floor names {launches} but claims {n_launches} launches")
    full = float(np.mean(list(heads.values())))
    pseudo = np.array([n_launches * full - (n_launches - 1)
                       * float(np.mean([v for k, v in heads.items() if l not in k]))
                       for l in launches])
    se = float(pseudo.std(ddof=1) / np.sqrt(n_launches))
    tc = t_crit_975(n_launches - 1)
    return {
        "point_nats": full,
        "se_launch_level_nats": se,
        "df": n_launches - 1,
        **_nonneg_ci(full, se, tc),
        "naive_se_over_ordered_pairs_nats": float(
            np.std(list(heads.values()), ddof=1) / np.sqrt(len(heads))),
        "method": "delete-one-launch jackknife over the ordered-pair headlines",
    }


def production_floor_positions(path=PRODUCTION_FLOOR):
    """Per-position floors from the production artifact, not from the 4-trajectory subset.

    The 4-trajectory per-position floor is a mean of 4 cells and swings by an order of magnitude;
    the comparison column has to come from the 640-cell measurement.
    """
    rec = json.load(open(path))
    if rec.get("kl_spec_hash") != q.spec_hash():
        raise LaunchDesignError(
            f"{_rel(path)} was written under KL_SPEC {rec.get('kl_spec_hash')}, "
            f"the current spec is {q.spec_hash()}")
    return ({p: v["mean_across_pairs_nats"] for p, v in rec["per_position"].items()},
            rec["headline"]["mean_nats"], rec,
            floor_launch_ci(rec["headline"]["per_pair_nats"], rec["n_launches"]))


def analyze(n_traj=4, sources=BF16_LAUNCH_SOURCES, quantized=("FP8_PRIMARY", "FP4_PRIMARY"),
            smoke_root=SMOKE_ROOT, out=None, allow_dirty=False, floor_path=PRODUCTION_FLOOR,
            committed_launch="S", own_outputs=()):
    q.require_clean_tree(allow_dirty, stage="launch_variance", own_outputs=own_outputs)
    traj_full = T.load()
    traj = C.subset(traj_full, n_traj)

    mats, meta = load_launch_matrices(sources, n_traj=n_traj, traj=traj_full)
    launches = [m["launch"] for m in meta]
    pre = check_exchangeability(meta, n_traj, mats=mats)
    regime = regime_check(mats, launches, n_traj)
    if regime is not None and regime["gross_regime_effect"]:
        raise LaunchDesignError(
            "the smoke launch is outside the floor64 launch-to-launch range; pooling it would "
            f"confound grid regime with launch identity. {regime['verdict']}")

    idx = K.bootstrap_indices(n_traj, q.BOOTSTRAP["draws"], q.BOOTSTRAP["seed"])
    floor_pos, floor_mean, floor_rec, floor_jk = production_floor_positions(floor_path)
    floor_ci = floor_jk["ci_95"]

    phi = bf16_to_bf16(mats, launches, n_traj, idx)
    phi["kl_cells_nats"] = {
        "axes": ["ordered_launch_pair", "trajectory", "position"],
        "ordered_pair_order": [f"{a}||{b}" for a, b in itertools.permutations(launches, 2)],
        "position_order": list(P.RETAINED_POSITIONS),
        "values": [[[float(v) for v in row] for row in lay]
                   for lay in [pair_grid(mats[a], mats[b], n_traj)
                               for a, b in itertools.permutations(launches, 2)]],
    }
    phi_stack = np.array([pair_grid(mats[a], mats[b], n_traj)
                          for a, b in itertools.permutations(launches, 2)])
    per_traj_phi = phi_stack.mean(axis=(0, 2))

    ref_cells = meta[0]["cells"]
    comparisons = {}
    for cfg in quantized:
        short = q.QUALITY_CONFIGS[cfg]["short"]
        qmat, qcells, qsummary = C.load_matrix(cfg, root=os.path.join(common.REPO, smoke_root),
                                               n_traj=n_traj)
        A.verify_cells(qcells, C.subset(traj_full, n_traj))
        if qcells != ref_cells:
            raise LaunchDesignError(
                f"{short}'s cells differ from the BF16 launches'; the comparison would not be "
                "scored on the same contexts")
        for key in ("kl_spec_hash", "trajectory_set_hash", "engine_profile_name"):
            if qsummary["provenance"][key] != meta[0]["summary"]["provenance"][key]:
                raise LaunchDesignError(
                    f"{short} disagrees with the BF16 launches on {key}")

        stack = np.array([pair_grid(mats[r], qmat, n_traj) for r in launches])
        label = f"BF16||{short}"
        rec, Y, per_traj_q = decompose(stack, idx, label, floor_nats=floor_mean)
        rec["per_launch_headline_nats"] = {r: float(Y[i].mean()) for i, r in enumerate(launches)}
        rec["pooling_evidence"] = _pooling_evidence(rec["per_launch_headline_nats"])
        # the launch the committed single-reference headline was computed against: "S" for the
        # smoke supplement, the designated reference launch for P13. When it IS in the launch set
        # the recomputation is the end-to-end check that this path produced the tracked number.
        rec["reproduces_committed_headline"] = _check_committed(
            label, rec["per_launch_headline_nats"].get(committed_launch), smoke_root,
            single_launch_grid=pair_grid(mats[launches[0]], qmat, n_traj),
            reference_launch=f"the reference launch {committed_launch!r} of "
                             f"{_rel(os.path.join(common.REPO, smoke_root))}")
        rec["quantized_source"] = {
            "config_id": cfg,
            "root": smoke_root,
            "collected_n_trajectories": qsummary["n_trajectories"],
            "engine_identity_hash": qsummary["engine_identity_hash"],
            "checkpoint_content_hash": qsummary["provenance"]["checkpoint_content_hash"],
            "single_realization": True,
            "note": "one FP8/FP4 launch only; this analysis carries BF16 reference variance and "
                    "says nothing about quantized-side launch variance",
        }
        rec["by_position"] = by_position(
            stack, idx, floor_by_position=floor_pos,
            sigma_proj=(rec.get("scaling_law") or {}).get("sigma_proj"))
        rec["resolvability"] = resolvability(rec, floor_mean, floor_ci, per_traj_q,
                                             per_traj_phi, idx)
        # the inputs are gitignored and a BF16 launch does not regenerate itself, so the per-cell
        # stack travels in the artifact or the numbers become unverifiable
        rec["kl_cells_nats"] = {
            "axes": ["bf16_launch", "trajectory", "position"],
            "bf16_launch_order": list(launches),
            "position_order": list(P.RETAINED_POSITIONS),
            "values": [[[float(v) for v in row] for row in lay] for lay in stack],
        }
        comparisons[label] = rec

    # the estimator moves the failed bound; which direction it moves it is the load-bearing fact
    g2 = {}
    for label, c in comparisons.items():
        theta = c["expected_kl_over_launches_nats"]
        # read from the tracked artifact, not from per_launch_headline_nats: the committed
        # reference launch is not in the launch set at all under the default R=3
        committed = ((c.get("reproduces_committed_headline") or {})
                     .get("committed_headline_nats"))
        g2[label] = {
            "committed_single_launch_nats": committed,
            "floor_fraction_at_committed_single_launch": (floor_mean / committed
                                                          if committed else None),
            "floor_fraction_at_expected_over_launches": floor_mean / theta,
            "bound": q.GATES["replication_floor_max_frac_of_fp8"],
            "direction": (None if not committed else
                          "worse -- the floor is a LARGER fraction of the signal under this "
                          "estimator" if theta < committed else
                          "better -- the floor is a smaller fraction of the signal"),
        }
    unresolved = [k for k, v in g2.items() if v["direction"] is None]
    if unresolved:
        raise LaunchDesignError(
            f"cannot establish which direction this estimator moves the G2' bound for "
            f"{unresolved}: the committed single-launch headline was not recovered. An estimator "
            "that touches a failed pre-registered bound must say which way it moves it.")
    any_rescued = any(v["floor_fraction_at_committed_single_launch"] > v["bound"]
                      >= v["floor_fraction_at_expected_over_launches"] for v in g2.values())

    rec = {
        "artifact": "BF16 launch-variance supplement to the KL analysis",
        "supplements": os.path.join(smoke_root, "kl_summary.json"),
        "supersedes": None,
        "question": "what is the expected BF16->FP8 / BF16->FP4 KL over BF16 launch identity, how "
                    "much of its uncertainty is launch-to-launch, and does the effect stay "
                    "resolvable once that is carried?",
        "contract_position": {
            "floor_subtracted_from_reported_kl": False,
            "bf16_launch_treated_as": "nuisance variance component / crossed repeated measure",
            "reference_definition_changed": False,
            "logit_matrices_averaged": False,
            "G2_G2prime_status": "historical outcomes and thresholds preserved unchanged; this "
                                 "analysis supplements them and re-adjudicates nothing",
            "pre_registered_bootstrap_unchanged": True,
            "no_failure_converted_to_a_pass": not any_rescued,
            "no_failure_converted_to_a_pass_evidence": g2,
            "against": "EXPERIMENTAL_CONTRACT.md quality-run validity: `no result was averaged or "
                       "re-referenced to convert a failure into a pass`",
        },
        "marginal_step_is_unaffected": {
            "pair": "FP8||FP4",
            "launch_component": None,
            "why": "the project's marginal quantity is measured with FP8 as the reference and "
                   "contains no BF16 distribution at all, so BF16 launch identity cannot enter "
                   "it. The whole nuisance-variance problem is confined to the BF16-anchored "
                   "pairs.",
        },
        "other_fp8_fp4_distributions_considered_and_rejected": {
            "results/quality/gates/engine_profile/": "4 traj x 10 pos x 2 launches per config "
                "under graph_2048 -- the tempting near-miss. Barred twice over: its manifest "
                "carries KL_SPEC 4ef13273db16d285, not the current spec, and its "
                "provisional_trajectories.json holds different continuations from the frozen set, "
                "so the contexts are not the same contexts.",
            "results/quality/gates/cache_equivalence/": "prefix caching off -- a different engine "
                "profile; and no FP8 arm.",
            "results/qualification/logits/": "fp16 storage (fails G4), single position, "
                "historical prototype path.",
            "results/pilot/gate_logits/": "fp16 storage, position 1 only, self-described as a "
                "pre-timing trip-wire and not a quality result.",
        },
        "n_trajectories": n_traj,
        "n_positions": N_POS,
        "positions": list(P.RETAINED_POSITIONS),
        "statistical_unit": "trajectory; BF16 launch is a crossed repeated-measure factor",
        "trajectory_set_hash": traj["trajectory_set_hash"],
        "kl_spec_hash": q.spec_hash(),
        "bootstrap": dict(q.BOOTSTRAP),
        "engine_profile_name": q.PROFILE_NAME,
        "bf16_launch_sources": [{k: v for k, v in m.items() if k not in ("cells", "summary")}
                                for m in meta],
        "exchangeability_preconditions": pre,
        "grid_regime_check": regime,
        "bf16_to_bf16": phi,
        "replication_floor_source": {
            "path": _rel(floor_path),
            "headline_nats": floor_mean,
            "n_trajectories": floor_rec["n_trajectories"],
            "n_launches": floor_rec["n_launches"],
            "launch_level_uncertainty": floor_jk,
            "used_for": "the side-by-side floor column only; never subtracted",
            "n4_figure_inside_launch_interval": (
                bool(floor_ci and floor_ci[0] <= 2.9835710748370010e-04 <= floor_ci[1])
                if floor_ci else None),
            "n4_figure_note": "the superseded n=4 floor estimate of 2.984e-04 was called a high "
                              "draw from a noisy sample. At the launch level it sits INSIDE the "
                              "production floor's own 95% interval, so `5.6% of the FP8 signal` "
                              "is really a range, not a point.",
        },
        "comparisons": comparisons,
        "scale_limit": {
            "bf16_to_bf16_scale": f"{n_traj} trajectories x {N_POS} positions x "
                                  f"{len(launches)} launches",
            "bf16_to_quantized_scale": f"{n_traj} trajectories x {N_POS} positions x "
                                       f"{len(launches)} BF16 launches",
            "quantized_root": smoke_root,
            "is_production_scale": bool(n_traj == q.N_TRAJECTORIES),
            # derived, never a literal: this block previously carried smoke-era text that declared
            # a 64-trajectory production artifact "NOT a result" from inside that artifact
            "consequence": (
                f"production scale: {n_traj} of {q.N_TRAJECTORIES} trajectories"
                if n_traj == q.N_TRAJECTORIES else
                f"{n_traj} of {q.N_TRAJECTORIES} trajectories -- a subset. Every BF16->FP8 / "
                "BF16->FP4 number here is n=%d in trajectories and is NOT a result; the "
                "estimator and its behaviour on real artifacts are what it delivers." % n_traj),
        },
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }
    out = out or DEFAULT_OUT
    common.write_json(out, rec)
    return rec, out


def project_to_scale(rec, n_traj_target=q.N_TRAJECTORIES, launches=(1, 3, 4, 8)):
    """What BF16 launch identity would contribute to a P13-scale headline.

    sigma_A does not shrink with trajectory count; the trajectory term does. This is the only
    pre-P13 answer to "how many BF16 launches does P13 need?", and it is an extrapolation from
    components measured at n=4 with 2 df on the launch term, not a measurement.

    Both the point and the one-sided 95% upper bound are projected: at this df the upper bound is
    the number a design decision should be taken against.
    """
    out = {}
    for label, c in rec["comparisons"].items():
        vc = c["variance_components"]
        theta = c["expected_kl_over_launches_nats"]
        var_s = vc["var_trajectory_moment"]
        var_e = vc["var_residual"]
        rows = {}
        for R in launches:
            var_traj = max(0.0, var_s) / n_traj_target + var_e / (R * n_traj_target)
            row = {"se_trajectory_nats": float(np.sqrt(var_traj))}
            for name, va in (("point", max(0.0, vc["var_level_moment"])),
                             ("upper_95", vc["var_level_upper_95"])):
                vl = (va or 0.0) / R
                tot = var_traj + vl
                row[f"se_launch_nats_{name}"] = float(np.sqrt(vl))
                row[f"se_total_nats_{name}"] = float(np.sqrt(tot))
                row[f"launch_share_of_variance_{name}"] = float(vl / tot) if tot > 0 else None
                row[f"se_total_over_theta_{name}"] = float(np.sqrt(tot) / theta) if theta > 0 else None
            rows[f"R={R}"] = row
        out[label] = {
            "projected_to_n_trajectories": n_traj_target,
            "expected_kl_over_launches_nats_assumed": theta,
            "by_n_bf16_launches": rows,
            "caveat": f"components estimated at n={c['n_trajectories']} trajectories with "
                      f"{vc['df_level']} df on the launch term and {vc['df_trajectory']} on the "
                      "trajectory term; indicative, not a measurement",
            "pre_registerable_prediction": (
                "the scaling law predicts the BF16 launch CV at production scale from sigma_proj "
                "pooled across configurations: SD_launch = sigma_proj*sqrt(2*signal). Registering "
                "that before P13 turns P13 into a test of the nuisance model rather than a first "
                "look at it."),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--smoke-root", default=SMOKE_ROOT)
    ap.add_argument("--out", default="")
    ap.add_argument("--floor", default=PRODUCTION_FLOOR)
    ap.add_argument("--include-smoke-launch", action="store_true",
                    help="sensitivity check: ADD the smoke BF16 launch, giving R=4. Its 40-context "
                         "grid is perfectly confounded with launch for that one level, so R=3 -- "
                         "the three launches that share a grid regime -- is the reported variant.")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    sources = tuple(s for s in BF16_LAUNCH_SOURCES
                    if a.include_smoke_launch or s["launch"] != "S")
    rec, out = analyze(n_traj=a.n_traj, sources=sources, smoke_root=a.smoke_root,
                       out=a.out or None, allow_dirty=a.allow_dirty, floor_path=a.floor)
    rec["projection"] = project_to_scale(rec)
    common.write_json(out, rec)

    print(f"BF16 launches: {[s['launch'] for s in sources]}  "
          f"trajectories: {rec['n_trajectories']}")
    f = rec["bf16_to_bf16"]
    jk = f["launch_level_uncertainty"]
    print(f"BF16||BF16 over {f['ordered_pairs']} ordered pairs: {f['mean_nats']:.4e} nats, "
          f"launch-level 95% CI [{jk['ci_95'][0]:.4e}, {jk['ci_95'][1]:.4e}] "
          f"({jk['df']} df; the naive pair SE understates by {jk['naive_understates_by']:.1f}x)")
    for label, c in rec["comparisons"].items():
        lc, vc = c["launch_component"], c["variance_components"]
        comb, rel = c["combined"], c["relative"]
        print(f"\n{label}")
        print(f"  E_r[KL]              {c['expected_kl_over_launches_nats']:.6e} nats  "
              f"(floor NOT subtracted; NOT the locked headline)")
        print(f"  per launch           " + "  ".join(
            f"{k}={v:.4e}" for k, v in c["per_launch_headline_nats"].items()))
        print(f"  launch effect        F={lc['f_launch']:.2f} on "
              f"({vc['df_level']},{vc['df_residual']}) df -> separable="
              f"{lc['launch_effect_separable_at_0_05']}")
        pt = lc.get("se_launch_nats_point")
        print(f"  SE launch            point {pt:.3e} ({rel['cv_launch_point']:.2%})"
              if pt else "  SE launch            point estimate is a negative moment")
        print(f"                       95% upper {lc['se_launch_nats_upper_95']:.3e} "
              f"({rel['cv_launch_upper_95']:.2%})")
        print(f"  SE trajectory        {c['trajectory_component']['se_trajectory_nats']:.3e}  "
              f"({rel['cv_trajectory']:.2%})  [pre-registered bootstrap]")
        an = comb["anova_unbiased"]
        print(f"  SE total (ANOVA)     {an['se_theta_nats']:.3e}  "
              f"Satterthwaite df {an['df_satterthwaite']:.1f}")
        bp = comb["bootstrap_plus_launch"]
        print(f"  SE total (boot+lnch) {bp['se_point_nats']:.3e} point, "
              f"{bp['se_upper_95_nats']:.3e} at the 95% launch upper bound")
        sl = c.get("scaling_law") or {}
        if sl:
            print(f"  scaling law          sigma_proj {sl['sigma_proj']:.3e}  "
                  f"(SD_launch = sigma_proj*sqrt(2*signal))")
        r = c["resolvability"]
        print(f"  resolvable           ratio to floor "
              f"{c['vs_replication_floor']['ratio_to_floor']:.1f}x (R={c['n_launches']}), "
              f"survives worst-case launch variance="
              f"{r['signal']['survives_worst_case_launch_variance']}")
    print("\nWROTE", _rel(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
