"""P13 -- the 64-trajectory production KL run, under the registered R=3 BF16 design.

WHAT IS REGISTERED HERE IS REGISTERED BEFORE ANY PRODUCTION FP8/FP4 CELL EXISTS. The constants in
`REGISTERED` below -- the launch count, the collection order, the resolution rule and the falsifiable
nuisance-model prediction -- are fixed in the tree and committed before `collect_all()` is allowed to
run, and `require_registration_committed()` enforces exactly that.

What P13 does NOT change
------------------------
- **G2 and G2' stay failed.** The 1% bound is untouched, no threshold moves, and nothing here
  re-adjudicates them. The floor is reported beside every number and subtracted from none.
- **No pooled or averaged BF16 distribution is ever constructed.** Launches are averaged at the
  KL-VALUE level, against real launches. Averaging the logit matrices is convex-barred (Jensen) and
  stays barred.
- **The locked headline is still the locked headline.** `EVALUATION_RIG.md` A.1 fixes it as the mean
  of 64 per-trajectory means under ONE reference launch; that is `analyze_kl.py`'s output against
  launch 1 and it remains the point estimate. The launch-averaged value is reported beside it.
- **`FP8||FP4` is untouched.** It contains no BF16 distribution, so BF16 launch identity cannot
  enter it, and it carries no launch component at all.

The revised estimator, in one line: BF16 run-to-run variability is carried as a NUISANCE VARIANCE
COMPONENT on a crossed repeated-measure factor, not removed, not subtracted, and not averaged away.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import positions as P, qcommon as q  # noqa: E402
from harness.quality import analyze_kl as A, collect_kl as C  # noqa: E402
from harness.quality import launch_variance as L  # noqa: E402

ROOT = q.KL_DIR
N_POS = len(P.RETAINED_POSITIONS)
SUMMARY = os.path.join(ROOT, "p13_summary.json")

# The designated single reference for the LOCKED headline is launch 1, and it is the main root's
# own collection so that `analyze_kl.py --root results/quality/kl` reproduces the locked contract
# without any P13-specific path handling.
BF16_LAUNCH_ROOTS = {1: ROOT,
                     2: os.path.join(ROOT, "bf16_launch2"),
                     3: os.path.join(ROOT, "bf16_launch3")}

REGISTERED = {
    "registered_on": "2026-09-16",
    "registered_before": "any production FP8 or FP4 cell was collected",
    "design": {
        "n_trajectories": 64,
        "n_positions": 10,
        "n_bf16_launches": 3,
        "engine_profile": "graph_2048 (G9, locked)",
        "storage_dtype": "float32 (G4, locked)",
        "designated_reference_launch_for_locked_headline": 1,
        "collection_order": ["BF16 launch 1", "FP8", "BF16 launch 2", "FP4", "BF16 launch 3"],
        "why_interleaved": (
            "the three floor64 launches ran consecutively inside one call, so their variance "
            "component is a WITHIN-SESSION one and a documented lower bound on what a "
            "differently-dated deployment launch would show. Interleaving the BF16 launches "
            "around the quantized collections makes any within-run drift land ON the launch "
            "factor instead of hiding underneath it. It cannot be removed -- launch identity and "
            "launch order are the same thing at R=3 -- so it is made conservative instead."),
    },
    "estimator": {
        "reported_headline": "the locked single-reference mean of 64 per-trajectory means "
                             "(analyze_kl.py, reference launch 1) -- unchanged",
        "supplementary_estimand": "E_r[KL(B_r || Q)], the divergence a randomly drawn BF16 launch "
                                  "of the locked profile exhibits",
        "model": "Y_{r,t} = mu + A_r + S_t + E_{rt}, Y = mean over the ten retained positions",
        "launch_summary": "F ratio and a one-sided 95% upper bound on sigma2_A, with the "
                          "untruncated moment estimate beside them. At 2 df a truncated point "
                          "estimate reports exactly 0.0 about 63% of the time under a null.",
        "trajectory_summary": "the pre-registered percentile bootstrap, unchanged in draws, seed, "
                              "unit and shared index matrix",
        "floor_treatment": "reported beside, never subtracted; a spread, not a bias",
        "fp8_fp4": "measured directly with FP8 as reference; unchanged and carries no launch term",
    },
    # The falsifiable prediction. Registered from the n=4 supplement, tested at n=64.
    "nuisance_model_prediction": {
        "law": "SD_launch(config) = sigma_proj * sqrt(2 * signal(config))",
        "sigma_proj_pooled_nats": 2.1e-03,
        "sigma_proj_measured_at_n4": {"BF16||FP8": 2.05e-03, "BF16||FP4": 2.11e-03},
        "predicted_launch_cv": {"BF16||FP8": 0.05, "BF16||FP4": 0.018},
        "predicted_quantity": "SD of the three per-launch headlines, i.e. "
                              "sqrt(sigma2_A + sigma2_E/T) -- NOT sigma_A",
        "falsified_if": "the registered sigma_proj falls outside the 95% chi-square interval of "
                        "the observed per-launch-headline SD, for both comparisons",
        "power_caveat": "at R=3 that interval spans a factor of ~12. The test can embarrass the "
                        "model but cannot confirm it, and this is stated before the run rather "
                        "than after.",
    },
    "resolution_rule": {
        "applies_to": "the BF16-anchored comparisons, at the headline and at every position",
        "classes": ["resolved", "noise_limited"],
        "rule": "RESOLVED iff the lower end of the launch-inflated 95% interval on the signal "
                "exceeds the upper end of the 95% launch-level interval on the BF16<->BF16 "
                "replication floor, both evaluated at the same position",
        "signal_interval": "the pre-registered percentile bootstrap interval, SCALED about the "
                           "point estimate by sqrt(1 + var_launch/var_bootstrap). Scaled rather "
                           "than rebuilt symmetrically: the percentile form was chosen for a "
                           "right-skewed non-negative statistic and theta +/- z*SE can cross zero "
                           "at the long positions.",
        "var_launch_per_position": "from the pooled scaling law, var = (sigma_proj * "
                                   "sqrt(2*theta_p))^2. Per-position variance COMPONENTS are not "
                                   "estimable -- raw KL cells have kurtosis ~45, an effective df "
                                   "near 5% of nominal -- so the pooled coupling constant is what "
                                   "makes a per-position statement possible at all.",
        "sigma_proj_used": "this run's own pooled estimate; the registered 2.1e-03 is applied as "
                           "a stated sensitivity check, never as the primary",
        "floor_interval": "delete-one-launch jackknife over the ordered pairs of THIS run's three "
                          "BF16 launches; six ordered pairs are a U-statistic carrying R-1 = 2 df, "
                          "not five",
        "every_position_is_reported": True,
        "no_position_is_dropped": "classification is a label on a reported number. Positions are "
                                  "never selected, filtered or reordered by outcome, and the "
                                  "headline is computed over all ten regardless of class.",
        "why_not_post_hoc_selection": "the rule is fixed here, before the FP8/FP4 cells exist. "
                                      "Which positions it lands on is data; the rule is not.",
    },
    "what_is_not_changed": {
        "G2_G2prime": "recorded failures; thresholds unchanged; not re-adjudicated",
        "floor_subtracted_from_any_reported_kl": False,
        "averaged_or_pooled_bf16_distribution": False,
        "locked_headline_definition": False,
        "bootstrap_draws_seed_unit": False,
        "fp8_fp4_direct_measurement": False,
    },
}


def registration_hash():
    return common.sha256_of_json(REGISTERED)[:16]


def require_registration_committed():
    """The registration must be in a commit before it can govern a collection.

    A pre-registration that lives only in the working tree is not one: it could be edited after
    seeing the first cells with nothing recording that it had been.
    """
    rel = os.path.relpath(os.path.abspath(__file__), common.REPO)
    if rel in q.dirty_paths():
        raise SystemExit(
            f"ABORT: {rel} carries uncommitted changes. The P13 registration governs this run and "
            "must be committed before any cell is collected.")
    return {"registration_file": rel, "registration_hash": registration_hash(),
            "committed": True, **q.git_state()}


def launch_sources():
    return tuple({"launch": f"L{i}", "root": os.path.relpath(r, common.REPO),
                  "n_trajectories": q.N_TRAJECTORIES,
                  "provenance": f"P13 production BF16 launch {i}"
                                + (" (designated reference for the locked headline)"
                                   if i == 1 else "")}
                 for i, r in sorted(BF16_LAUNCH_ROOTS.items()))


def collect_all(allow_dirty=False, n_traj=None, require_cool=True):
    """The registered collection order, one engine launch per entry."""
    reg = require_registration_committed()
    scope = [os.path.relpath(ROOT, common.REPO)]
    head = q.require_clean_tree(allow_dirty, stage="p13:start")["git_head"]
    plan = [("BF16_REFERENCE", BF16_LAUNCH_ROOTS[1], "BF16 launch 1"),
            ("FP8_PRIMARY", ROOT, "FP8"),
            ("BF16_REFERENCE", BF16_LAUNCH_ROOTS[2], "BF16 launch 2"),
            ("FP4_PRIMARY", ROOT, "FP4"),
            ("BF16_REFERENCE", BF16_LAUNCH_ROOTS[3], "BF16 launch 3")]
    out = []
    for cfg, root, label in plan:
        rec = C.collect(cfg, root=root, allow_dirty=allow_dirty, n_traj=n_traj,
                        require_cool=require_cool, own_outputs=scope, head=head)
        obs = rec.get("observed") or {}
        ev = rec.get("dispatch_evidence") or {}
        if ev.get("ok") is not True:
            raise SystemExit(f"ABORT: {label} carries no passing positive dispatch evidence: {ev}")
        out.append({"step": label, "config_id": cfg,
                    "root": os.path.relpath(root, common.REPO),
                    "engine_identity_hash": rec["engine_identity_hash"],
                    "kv_cache_tokens": obs.get("kv_cache_tokens"),
                    "cells": rec["cells"], "seconds": rec["seconds"],
                    "wall_seconds": rec["wall_seconds"],
                    "dispatch_evidence_ok": (rec.get("dispatch_evidence") or {}).get("ok"),
                    "launched": rec["launched"]})
        print(f"{label:15s} engine={rec['engine_identity_hash']} kv={obs.get('kv_cache_tokens')} "
              f"cells={rec['cells']} scoring={rec['seconds']}s", flush=True)
    return {"registration": reg, "steps": out}


def _scaled_lower(boot, theta, var_launch):
    """Lower end of the pre-registered percentile interval, inflated for launch variance."""
    var_boot = float(boot["std_error"]) ** 2
    if var_boot <= 0:
        return None, None
    infl = float(np.sqrt(1.0 + var_launch / var_boot))
    return float(theta + (boot["ci_low"] - theta) * infl), infl


def classify(theta, boot, sigma_proj, floor_point, floor_hi):
    """The registered resolution rule, at one position or at the headline."""
    var_launch = (sigma_proj ** 2) * 2.0 * theta if (sigma_proj and theta > 0) else 0.0
    lo, infl = _scaled_lower(boot, theta, var_launch)
    resolved = bool(lo is not None and floor_hi is not None and lo > floor_hi)
    return {
        "expected_kl_over_launches_nats": theta,
        "predicted_launch_sd_nats": float(np.sqrt(var_launch)) if var_launch else None,
        "trajectory_ci": [boot["ci_low"], boot["ci_high"]],
        "launch_inflated_ci_low_nats": lo,
        "inflation_factor": infl,
        "replication_floor_nats": floor_point,
        "replication_floor_ci_high_nats": floor_hi,
        "class": "resolved" if resolved else "noise_limited",
        "margin_nats": (lo - floor_hi) if (lo is not None and floor_hi is not None) else None,
        "ratio_signal_low_to_floor_high": (lo / floor_hi) if (lo and floor_hi) else None,
    }


def resolution(rec, sigma_proj, registered_sigma_proj):
    """Per-position and headline classification for every BF16-anchored comparison."""
    phi = rec["bf16_to_bf16"]
    floor_hi_head = phi["launch_level_uncertainty"]["ci_95"][1]
    out = {}
    for label, c in rec["comparisons"].items():
        head = classify(c["expected_kl_over_launches_nats"],
                        c["trajectory_component"]["bootstrap"], sigma_proj,
                        phi["mean_nats"], floor_hi_head)
        head_sens = classify(c["expected_kl_over_launches_nats"],
                             c["trajectory_component"]["bootstrap"], registered_sigma_proj,
                             phi["mean_nats"], floor_hi_head)
        positions = {}
        for p, pos in c["by_position"].items():
            f = phi["by_position"][p]
            fh = f["launch_level_uncertainty"]["ci_95"][1]
            positions[p] = {
                **classify(pos["expected_kl_over_launches_nats"], pos["trajectory_bootstrap"],
                           sigma_proj, f["mean_over_ordered_pairs_nats"], fh),
                "class_under_registered_sigma_proj": classify(
                    pos["expected_kl_over_launches_nats"], pos["trajectory_bootstrap"],
                    registered_sigma_proj, f["mean_over_ordered_pairs_nats"], fh)["class"],
                "per_launch_nats": pos["per_launch_nats"],
            }
        klass = {p: v["class"] for p, v in positions.items()}
        out[label] = {
            "rule": REGISTERED["resolution_rule"]["rule"],
            "sigma_proj_used": sigma_proj,
            "headline": head,
            "headline_under_registered_sigma_proj": head_sens,
            "by_position": positions,
            "resolved_positions": [p for p, k in klass.items() if k == "resolved"],
            "noise_limited_positions": [p for p, k in klass.items() if k == "noise_limited"],
            "n_resolved": sum(1 for k in klass.values() if k == "resolved"),
            "n_positions": len(klass),
            "all_positions_reported": len(klass) == N_POS,
            "reading": "a noise-limited position is one where this rig cannot separate the "
                       "quantization effect from BF16 relaunch noise. It is a statement about "
                       "resolution, not about the effect being absent, and the position still "
                       "contributes to the headline.",
        }
    return out


def prediction_test(rec, pooled):
    """Observed nuisance model against what was registered before the run."""
    pred = REGISTERED["nuisance_model_prediction"]
    reg_sigma = pred["sigma_proj_pooled_nats"]
    per = {}
    for label, c in rec["comparisons"].items():
        sl = c.get("scaling_law") or {}
        vc = c["variance_components"]
        sd_obs = sl.get("sd_launch_headlines_nats")
        theta = c["expected_kl_over_launches_nats"]
        sd_pred = reg_sigma * np.sqrt(2.0 * theta)
        ci = L.sd_ci_chi2(sd_obs, vc["df_level"]) if sd_obs else None
        inside = (bool(ci["ci_95"][0] <= sd_pred <= ci["ci_95"][1]) if ci else None)
        per[label] = {
            "observed_sd_launch_headlines_nats": sd_obs,
            "observed_sd_ci_95": ci["ci_95"] if ci else None,
            "observed_sigma_proj": sl.get("sigma_proj"),
            "registered_sigma_proj": reg_sigma,
            "predicted_sd_launch_nats": float(sd_pred),
            "observed_cv_launch": (sd_obs / theta) if (sd_obs and theta) else None,
            "predicted_cv_launch_registered": pred["predicted_launch_cv"].get(label),
            "predicted_sd_inside_observed_ci": inside,
            "ratio_observed_over_predicted": (sd_obs / sd_pred) if (sd_obs and sd_pred) else None,
        }
    inside_all = [v["predicted_sd_inside_observed_ci"] for v in per.values()]
    return {
        "registered": pred,
        "per_comparison": per,
        "pooled_observed": pooled,
        "pooled_ratio_observed_over_registered": (pooled["sigma_proj_pooled"] / reg_sigma
                                                  if pooled else None),
        "verdict": ("CONSISTENT" if all(x is True for x in inside_all) else
                    "FALSIFIED" if all(x is False for x in inside_all) else
                    "MIXED" if any(x is not None for x in inside_all) else "NOT_EVALUABLE"),
        "verdict_meaning": {
            "CONSISTENT": "the registered sigma_proj sits inside the observed interval for every "
                          "comparison. At 2 df that interval spans a factor of ~12, so this is a "
                          "failure to embarrass the model, not a confirmation of it.",
            "FALSIFIED": "the registered value is outside the observed interval for every "
                         "comparison; the nuisance model as registered is wrong",
            "MIXED": "the comparisons disagree; the pooled coupling constant is not supported",
        },
    }


def analyze_all(n_traj=None, allow_dirty=False, out=None, floor_path=L.PRODUCTION_FLOOR):
    """The locked headline, the launch-variance supplement, the prediction test, the rule."""
    n = n_traj or q.N_TRAJECTORIES
    scope = [os.path.relpath(ROOT, common.REPO)]
    q.require_clean_tree(allow_dirty, stage="p13:analyze", own_outputs=scope)

    locked = A.analyze(root=ROOT, n_traj=n, floor_path=floor_path, allow_dirty=allow_dirty,
                       own_outputs=scope)
    lv, _ = L.analyze(n_traj=n, sources=launch_sources(),
                      smoke_root=os.path.relpath(ROOT, common.REPO),
                      out=os.path.join(ROOT, "launch_variance_p13.json"),
                      allow_dirty=allow_dirty, floor_path=floor_path,
                      committed_launch="L1", own_outputs=scope)
    pooled = L.pooled_sigma_proj(lv["comparisons"])
    sigma = pooled["sigma_proj_pooled"] if pooled else None
    pred = prediction_test(lv, pooled)
    res = resolution(lv, sigma, REGISTERED["nuisance_model_prediction"]["sigma_proj_pooled_nats"])

    rec = {
        "artifact": "P13 production KL result",
        "registration": {"hash": registration_hash(), **REGISTERED},
        "n_trajectories": n,
        "n_positions": N_POS,
        "n_bf16_launches": len(BF16_LAUNCH_ROOTS),
        "trajectory_set_hash": locked["trajectory_set_hash"],
        "kl_spec_hash": q.spec_hash(),
        "engine_profile_name": q.PROFILE_NAME,
        "locked_headline": {
            "definition": "EVALUATION_RIG.md A.1: mean of per-trajectory means, single reference "
                          "launch (launch 1). This is THE headline; everything else supplements.",
            "artifact": os.path.relpath(os.path.join(ROOT, "kl_summary.json"), common.REPO),
            "pairs": {label: {"headline_nats": p["headline_nats"],
                              "ci_95": [p["headline_bootstrap"]["ci_low"],
                                        p["headline_bootstrap"]["ci_high"]],
                              "std_error_nats": p["headline_bootstrap"]["std_error"],
                              "worst_cell_nats": p["per_cell"]["max_nats"],
                              "vs_replication_floor": p.get("vs_replication_floor")}
                      for label, p in locked["pairs"].items()},
        },
        "launch_variance_supplement": os.path.relpath(
            os.path.join(ROOT, "launch_variance_p13.json"), common.REPO),
        "expected_over_launches": {
            label: {
                "expected_kl_over_launches_nats": c["expected_kl_over_launches_nats"],
                "per_launch_headline_nats": c["per_launch_headline_nats"],
                "se_launch_point_nats": c["launch_component"].get("se_launch_nats_point"),
                "se_launch_upper_95_nats": c["launch_component"].get("se_launch_nats_upper_95"),
                "f_launch": c["launch_component"].get("f_launch"),
                "launch_effect_separable_at_0_05":
                    c["launch_component"].get("launch_effect_separable_at_0_05"),
                "se_trajectory_nats": c["trajectory_component"]["se_trajectory_nats"],
                "se_total_anova_nats": c["combined"]["anova_unbiased"]["se_theta_nats"],
                "ci_launch_inflated_point": c["combined"]["bootstrap_plus_launch"]["ci_point"],
                "ci_launch_inflated_upper_95":
                    c["combined"]["bootstrap_plus_launch"]["ci_upper_95"],
                "launch_share_of_variance": (
                    None if not c["combined"]["bootstrap_plus_launch"]["se_point_nats"] else
                    float(max(0.0, c["launch_component"].get("var_launch_point") or 0.0)
                          / c["combined"]["bootstrap_plus_launch"]["se_point_nats"] ** 2)),
            } for label, c in lv["comparisons"].items()},
        "marginal_step": {
            "pair": "FP8||FP4",
            "headline_nats": locked["pairs"].get("FP8||FP4", {}).get("headline_nats"),
            "ci_95": [locked["pairs"]["FP8||FP4"]["headline_bootstrap"]["ci_low"],
                      locked["pairs"]["FP8||FP4"]["headline_bootstrap"]["ci_high"]]
                     if "FP8||FP4" in locked["pairs"] else None,
            "launch_component": None,
            "why_no_launch_component": "FP8 is the reference; no BF16 distribution enters it",
        },
        "replication_floor_this_run": {
            "mean_nats": lv["bf16_to_bf16"]["mean_nats"],
            "per_pair_nats": lv["bf16_to_bf16"]["per_pair_headline_nats"],
            "launch_level_ci_95": lv["bf16_to_bf16"]["launch_level_uncertainty"]["ci_95"],
            "subtracted_from_anything": False,
        },
        "nuisance_model_prediction_test": pred,
        "resolution": res,
        "G2_G2prime": {
            "status": "RECORDED FAILURES; unchanged by this run",
            "bound": q.GATES["replication_floor_max_frac_of_fp8"],
            "threshold_relaxed": False,
            "floor_subtracted": False,
            "averaged_reference_used": False,
            # BF16-anchored pairs only: FP8||FP4 has no BF16 reference, and FP8/FP4 replicate to
            # ~1e-11 under CUDA graphs, so quoting the BF16 floor against it invents a bound
            "floor_fraction_of_signal": {
                label: (lv["bf16_to_bf16"]["mean_nats"] / p["headline_nats"])
                for label, p in locked["pairs"].items()
                if p["headline_nats"] > 0 and label.startswith("BF16||")},
            "floor_not_applicable_to": ["FP8||FP4"],
            "floor_not_applicable_because": "the BF16 replication floor bounds BF16-anchored "
                                            "comparisons. FP8||FP4 uses FP8 as its reference, "
                                            "whose own replication floor is ~1e-11 nats.",
        },
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }
    out = out or SUMMARY
    common.write_json(out, rec)
    return rec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n-traj", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-cool-check", action="store_true")
    a = ap.parse_args()
    if not (a.collect or a.analyze):
        raise SystemExit("give --collect, --analyze, or both")
    if a.collect:
        rep = collect_all(allow_dirty=a.allow_dirty, n_traj=a.n_traj or None,
                          require_cool=not a.no_cool_check)
        print(json.dumps(rep["registration"], indent=2))
    if not a.analyze:
        return 0
    rec, out = analyze_all(n_traj=a.n_traj or None, allow_dirty=a.allow_dirty,
                           out=a.out or None)
    print(f"\nregistration {rec['registration']['hash']}   "
          f"{rec['n_trajectories']} trajectories x {rec['n_positions']} positions x "
          f"{rec['n_bf16_launches']} BF16 launches\n")
    print("LOCKED HEADLINE (single reference launch 1)")
    for label, p in rec["locked_headline"]["pairs"].items():
        print(f"  {label:12s} {p['headline_nats']:.6e} nats  "
              f"95% CI [{p['ci_95'][0]:.6e}, {p['ci_95'][1]:.6e}]")
    print("\nEXPECTED OVER BF16 LAUNCHES (supplement; floor never subtracted)")
    for label, c in rec["expected_over_launches"].items():
        print(f"  {label:12s} {c['expected_kl_over_launches_nats']:.6e} nats   "
              f"SE_launch {c['se_launch_upper_95_nats']:.2e} (95% upper)  "
              f"SE_traj {c['se_trajectory_nats']:.2e}  "
              f"launch share {c['launch_share_of_variance']:.1%}"
              if c["se_launch_upper_95_nats"] else f"  {label}: launch SE not estimable")
    f = rec["replication_floor_this_run"]
    print(f"\nBF16<->BF16 floor, this run: {f['mean_nats']:.4e} nats  "
          f"95% [{f['launch_level_ci_95'][0]:.4e}, {f['launch_level_ci_95'][1]:.4e}]")
    pt = rec["nuisance_model_prediction_test"]
    print(f"nuisance model vs registration: {pt['verdict']}  "
          f"(pooled sigma_proj {pt['pooled_observed']['sigma_proj_pooled']:.3e} against "
          f"{pt['registered']['sigma_proj_pooled_nats']:.3e} registered)")
    print("\nRESOLUTION (all positions reported; class is a label, not a filter)")
    for label, r in rec["resolution"].items():
        print(f"  {label:12s} headline {r['headline']['class']:14s} "
              f"{r['n_resolved']}/{r['n_positions']} positions resolved")
        if r["noise_limited_positions"]:
            print(f"               noise-limited at p = "
                  f"{', '.join(r['noise_limited_positions'])}")
    print("\nG2/G2' remain recorded failures; no threshold moved, no floor subtracted.")
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
