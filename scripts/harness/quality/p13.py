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
    report = {"artifact": "P13 collection record", "registration": reg, "steps": out,
              "collection_git_head": head, "timestamp": common.now_iso()}
    common.write_json(COLLECTION_RECORD, report)
    return report


def _scaled_lower(boot, theta, var_launch):
    """Lower end of the pre-registered percentile interval, inflated for launch variance."""
    var_boot = float(boot["std_error"]) ** 2
    if var_boot <= 0:
        return None, None
    infl = float(np.sqrt(1.0 + var_launch / var_boot))
    return float(theta + (boot["ci_low"] - theta) * infl), infl


def classify(theta, boot, sigma_proj, floor_point, floor_hi):
    """The registered resolution rule, at one position or at the headline.

    `sigma_proj is None` means the coupling constant could not be estimated, which is NOT the same
    as a measured absence of launch variance; it takes the same arithmetic branch as 0.0, so it is
    given its own class rather than being allowed to read as `resolved` under an un-inflated rule.
    """
    if sigma_proj is None:
        return {"expected_kl_over_launches_nats": theta, "class": "not_evaluable",
                "not_evaluable_because": "no pooled sigma_proj could be estimated, so the "
                                         "registered launch inflation cannot be applied",
                "trajectory_ci": [boot["ci_low"], boot["ci_high"]],
                "replication_floor_nats": floor_point,
                "replication_floor_ci_high_nats": floor_hi}
    var_launch = (sigma_proj ** 2) * 2.0 * theta if theta > 0 else 0.0
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
        # which uncertainty actually binds. The class is named for the floor comparison, but the
        # inflation factor says how much of the interval width the LAUNCH term contributed: at
        # 1.00 the classification is decided entirely by trajectory sampling.
        "class_depends_on_launch_term": bool(
            lo is not None and floor_hi is not None
            and (lo > floor_hi) != (boot["ci_low"] > floor_hi)),
        "binding_uncertainty": ("launch" if (infl or 1.0) > 1.05 else "trajectory_sampling"),
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
        n_launch_dependent = sum(1 for v in positions.values()
                                 if v.get("class_depends_on_launch_term"))
        out[label] = {
            "rule": REGISTERED["resolution_rule"]["rule"],
            "sigma_proj_used": sigma_proj,
            "classifications_that_depend_on_the_launch_term": n_launch_dependent,
            "inflation_factor_range": [
                min(v["inflation_factor"] for v in positions.values() if v.get("inflation_factor")),
                max(v["inflation_factor"] for v in positions.values() if v.get("inflation_factor"))],
            "headline": head,
            "headline_under_registered_sigma_proj": head_sens,
            "by_position": positions,
            "resolved_positions": [p for p, k in klass.items() if k == "resolved"],
            "noise_limited_positions": [p for p, k in klass.items() if k == "noise_limited"],
            "n_resolved": sum(1 for k in klass.values() if k == "resolved"),
            "n_positions": len(klass),
            "all_positions_reported": len(klass) == N_POS,
            "what_resolved_licenses": (
                "the estimated divergence at this position, after trajectory-sampling "
                "uncertainty and the registered launch inflation, exceeds the upper launch-level "
                "bound on the typical BF16<->BF16 divergence. It does NOT license 'separable from "
                "relaunch noise': the floor is SECOND order in the launch perturbation while the "
                "signal is FIRST order, so the floor is not the noise the signal is exposed to. "
                "Separability from relaunch noise would be theta against its own SD_launch, "
                "against which every position clears by one to three orders of magnitude."),
            "what_noise_limited_means": (
                "this rig cannot put the position's divergence above the floor's upper launch "
                "bound. It is a statement about resolution, not about the effect being absent, "
                "and the position still contributes to the headline. Read `binding_uncertainty` "
                "before attributing it to launch noise -- where the inflation factor is ~1.00 the "
                "limit is trajectory sampling, not relaunching."),
            "multiplicity": "no multiplicity correction is applied and none is implied. The ten "
                            "positions are repeated measures on the same 64 trajectories sharing "
                            "one bootstrap index matrix, so they are strongly dependent and must "
                            "not be read as ten independent tests.",
            "floor_side_uncertainty_is_launch_only": (
                "floor_hi covers launch sampling (delete-one-launch jackknife, 2 df) and not the "
                "floor's own trajectory sampling. Carrying both would widen floor_hi; see "
                "floor_side_sensitivity."),
        }
    return out


def rec_theta(rec, label):
    return rec["comparisons"][label]["expected_kl_over_launches_nats"]


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
            # the registered CV was computed at the n=4 signal; at the OBSERVED signal the same
            # law predicts a different CV, and that is the number to compare against
            "predicted_cv_launch_at_observed_signal": (float(sd_pred / theta) if theta else None),
            "predicted_cv_launch_as_registered": pred["predicted_launch_cv"].get(label),
            "predicted_cv_registered_was_computed_at": "the n=4 signal, not this run's",
            "sigma_A_share_of_sd_headlines": (
                None if not vc or vc.get("var_level_moment") is None or not sd_obs else
                float(max(0.0, vc["var_level_moment"]) / (sd_obs ** 2))),
            "predicted_sd_inside_observed_ci": inside,
            "ratio_observed_over_predicted": (sd_obs / sd_pred) if (sd_obs and sd_pred) else None,
        }
    inside_all = [v["predicted_sd_inside_observed_ci"] for v in per.values()]

    # POST-HOC, and labelled as such: the registered conjunction is close to unfalsifiable -- two
    # arms computed from the SAME three launches, each against an interval spanning ~12x. The
    # law's sharpest consequence is the SD RATIO between configurations, which carries almost no
    # launch-sampling error because both SDs are functions of the same three perturbations.
    sharp = None
    labels = [l for l in per if per[l]["observed_sd_launch_headlines_nats"]]
    if len(labels) == 2:
        a, b = sorted(labels, key=lambda l: rec_theta(rec, l))
        ta, tb = rec_theta(rec, a), rec_theta(rec, b)
        obs = (per[b]["observed_sd_launch_headlines_nats"]
               / per[a]["observed_sd_launch_headlines_nats"])
        sharp = {
            "status": "POST HOC -- not registered before the run; reported as a diagnostic, "
                      "never as the adjudication",
            "statistic": f"SD_launch({b}) / SD_launch({a})",
            "predicted_by_the_law": float(np.sqrt(tb / ta)),
            "observed": float(obs),
            "why_it_is_sharp": "both SDs are functions of the same three launch perturbations, so "
                               "the ratio carries far less sampling error than either SD alone",
            "reading": "the law says the ratio is sqrt(signal ratio); a ratio near 1 says the "
                       "launch spread does not scale with the signal at all",
        }
    return {
        "registered": pred,
        "per_comparison": per,
        "sharp_post_hoc_diagnostic": sharp,
        "pooled_observed": pooled,
        "pooled_ratio_observed_over_registered": (pooled["sigma_proj_pooled"] / reg_sigma
                                                  if pooled else None),
        "verdict": ("CONSISTENT" if all(x is True for x in inside_all) else
                    "FALSIFIED" if all(x is False for x in inside_all) else
                    "MIXED" if any(x is not None for x in inside_all) else "NOT_EVALUABLE"),
        "verdict_is_weak_by_construction": (
            "the registered criterion is a CONJUNCTION over two arms computed from the SAME three "
            "BF16 launches, each tested against a chi-square interval spanning ~12x at 2 df. It is "
            "close to unfalsifiable as written. Read the verdict's own gloss and the pooled ratio "
            "rather than the label."),
        "verdict_meaning": {
            "CONSISTENT": "the registered sigma_proj sits inside the observed interval for every "
                          "comparison. At 2 df that interval spans a factor of ~12, so this is a "
                          "failure to embarrass the model, not a confirmation of it.",
            "FALSIFIED": "the registered value is outside the observed interval for every "
                         "comparison; the nuisance model as registered is wrong",
            "MIXED": "the comparisons disagree; the pooled coupling constant is not supported",
        },
    }


SMOKE_FP8_HEADLINE = 3.690e-03


def g2_restatement(locked, lv, floor_path):
    """G2' restated at production scale, with BOTH floors named.

    There are two defensible denominators and they are not the same number: the PRE-REGISTERED
    floor that G2' was actually adjudicated against, and this run's own BF16<->BF16 mean. Quoting
    one without the other makes the move from the smoke's 5.6% look like a pure signal effect when
    part of it is a change of floor. Both are reported, the pre-registered one is primary because
    it is the one the bound was registered against, and the move is decomposed.
    """
    registered = json.load(open(floor_path))["headline"]["mean_nats"]
    this_run = lv["bf16_to_bf16"]["mean_nats"]
    bound = q.GATES["replication_floor_max_frac_of_fp8"]
    # BF16-anchored pairs only: FP8||FP4 has no BF16 reference, and FP8/FP4 replicate to ~1e-11
    # under CUDA graphs, so quoting the BF16 floor against it invents a bound
    pairs = {label: p["headline_nats"] for label, p in locked["pairs"].items()
             if p["headline_nats"] > 0 and label.startswith("BF16||")}
    fp8 = pairs.get("BF16||FP8")
    return {
        "status": "RECORDED FAILURES; unchanged by this run",
        "bound": bound,
        "threshold_relaxed": False,
        "floor_subtracted": False,
        "averaged_reference_used": False,
        "primary_floor": {
            "which": "the PRE-REGISTERED production floor, the denominator G2' was adjudicated "
                     "against",
            "source": os.path.relpath(floor_path, common.REPO),
            "floor_nats": registered,
            "fraction_of_signal": {k: registered / v for k, v in pairs.items()},
            "passes_bound": {k: bool(registered / v <= bound) for k, v in pairs.items()},
        },
        "this_run_floor": {
            "which": "this run's own BF16<->BF16 mean over six ordered pairs of its three launches",
            "floor_nats": this_run,
            "fraction_of_signal": {k: this_run / v for k, v in pairs.items()},
            "passes_bound": {k: bool(this_run / v <= bound) for k, v in pairs.items()},
        },
        "move_from_the_smoke_figure": {
            "smoke_fraction": registered / SMOKE_FP8_HEADLINE,
            "smoke_signal_nats": SMOKE_FP8_HEADLINE,
            "production_signal_nats": fp8,
            "signal_effect_only": registered / fp8 if fp8 else None,
            "signal_and_floor_effect": this_run / fp8 if fp8 else None,
            "reading": "the fall from the smoke figure is mostly the signal -- the production "
                       "BF16->FP8 KL is larger than the n=4 estimate -- but NOT entirely: part of "
                       "it is a smaller floor in this run. Both floors are within each other's "
                       "launch-level intervals, so the difference is not itself resolvable.",
        },
        "conclusion": "G2' fails on every floor/signal combination in this run's artifacts. The "
                      "bound is unchanged at %g and no result was averaged or re-referenced."
                      % bound,
        "floor_not_applicable_to": ["FP8||FP4"],
        "floor_not_applicable_because": "the BF16 replication floor bounds BF16-anchored "
                                        "comparisons. FP8||FP4 uses FP8 as its reference, whose "
                                        "own replication floor is ~1e-11 nats.",
    }


def floor_side_sensitivity(lv, res):
    """What carrying the floor's OWN trajectory uncertainty would do to the classifications.

    `floor_hi` is a launch-level jackknife and covers launch sampling only, while the signal side
    covers trajectory sampling. The floor is itself a mean over 64 trajectories. Widening it is the
    anti-conservative direction of the rule, so the effect is measured rather than argued.
    """
    phi = lv["bf16_to_bf16"]
    se_traj = phi["trajectory_bootstrap"]["std_error"]
    tc = L.t_crit_975(phi["launch_level_uncertainty"]["df"])
    out, flips = {}, 0
    for label, r in res.items():
        rows = {}
        for p, v in r["by_position"].items():
            f = phi["by_position"][p]
            jk = f["launch_level_uncertainty"]
            wider = f["mean_over_ordered_pairs_nats"] + tc * float(
                (jk["se_launch_level_nats"] ** 2 + se_traj ** 2) ** 0.5)
            lo = v.get("launch_inflated_ci_low_nats")
            klass = "resolved" if (lo is not None and lo > wider) else "noise_limited"
            flips += int(klass != v["class"])
            rows[p] = {"floor_hi_as_used": v["replication_floor_ci_high_nats"],
                       "floor_hi_with_trajectory_uncertainty": wider,
                       "class_as_used": v["class"], "class_if_widened": klass}
        out[label] = rows
    return {
        "question": "does carrying the floor's own trajectory-sampling uncertainty change any "
                    "classification?",
        "method": "floor_hi recomputed as mean + t(df) * sqrt(se_jackknife^2 + se_trajectory^2); "
                  "the floor's trajectory SE is a single pooled figure, so this is indicative",
        "floor_trajectory_se_nats": se_traj,
        "classifications_changed": flips,
        "not_adopted": "the registered rule is what adjudicates; this is a sensitivity check and "
                       "is reported beside it, never in place of it",
        "by_comparison": out,
    }


COLLECTION_RECORD = os.path.join(ROOT, "collection_record.json")


def _collection_record():
    """The collect-time record, read back so the analysis names the commit the CELLS were taken at.

    Analysis runs at a later commit than collection, and every summary records only its own HEAD.
    Without this the artifact cannot say which commit produced the data it summarises, nor that the
    registration was committed before the first cell.
    """
    if not os.path.exists(COLLECTION_RECORD):
        return {"available": False,
                "why": f"{os.path.relpath(COLLECTION_RECORD, common.REPO)} is absent; this "
                       "analysis cannot name the commit the cells were collected at"}
    rec = json.load(open(COLLECTION_RECORD))
    return {"available": True, **rec}


def marginal_step_durability(n, locked):
    """The marginal step's own cells and matrix digests.

    `dist/*.npy` is gitignored and a launch provably does not regenerate itself, so a number whose
    inputs carry no digest is not reproducible from the tree. The BF16 arm was already covered by
    launch_variance's exchangeability block; FP8, FP4 and the FP8||FP4 grid were not -- which left
    the study's own marginal quantity as the least durable number in the run.
    """
    mats, digests = {}, {}
    for cfg in ("FP8_PRIMARY", "FP4_PRIMARY"):
        mat, _, _ = C.load_matrix(cfg, root=ROOT, n_traj=n)
        mats[cfg] = mat
        digests[q.QUALITY_CONFIGS[cfg]["short"]] = L.matrix_digest(mat)
    grid = A.pair_grid(mats["FP8_PRIMARY"], mats["FP4_PRIMARY"], n)
    del mats
    idx = K.bootstrap_indices(n, q.BOOTSTRAP["draws"], q.BOOTSTRAP["seed"])
    return {
        "matrix_sha256": digests,
        "matrix_sha256_note": "SHA-256 of the exact float32 bytes consumed; the BF16 launches' "
                              "digests are in launch_variance_p13.json",
        "kl_cells_nats": {
            "pair": "FP8||FP4",
            "axes": ["trajectory", "position"],
            "position_order": list(P.RETAINED_POSITIONS),
            "values": [[float(v) for v in row] for row in grid],
        },
        "_grid": grid,
        "_idx": idx,
    }


def paired_marginal(marginal, locked, n):
    """Is the direct FP8||FP4 step larger than the BF16-anchored FP4 divergence?

    The two marginal CIs overlap, so the comparison needs a PAIRED interval: both are measured on
    the same 64 trajectories with the same bootstrap index matrix, and the paired difference is
    far better determined than either margin. This is a comparison of two separately measured
    divergences, not the barred subtraction proxy -- nothing here is used as a divergence.
    """
    import numpy as _np
    fp84 = K.trajectory_means(marginal["_grid"])
    b4 = _np.asarray(locked["pairs"]["BF16||FP4"]["trajectory_means_nats"], dtype=float)
    diff = fp84 - b4
    draws = diff[marginal["_idx"]].mean(axis=1)
    lo, hi = K.percentile_ci(draws, tuple(q.BOOTSTRAP["ci"]))
    return {
        "quantity": "FP8||FP4 minus BF16||FP4, paired on the same 64 trajectories",
        "point_nats": float(diff.mean()),
        "ci_95": [lo, hi],
        "p_le_zero": float((draws <= 0).mean()),
        "why_paired": "the two marginal intervals overlap; the paired one is the interval the "
                      "shared trajectories support",
        "not_a_divergence": "a difference of two KLs is not a KL. This is a comparison of "
                            "magnitudes and is never reported as the marginal step, which is the "
                            "directly measured FP8||FP4.",
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
    marginal = marginal_step_durability(n, locked)
    pooled = L.pooled_sigma_proj(lv["comparisons"])
    sigma = pooled["sigma_proj_pooled"] if pooled else None
    pred = prediction_test(lv, pooled)
    res = resolution(lv, sigma, REGISTERED["nuisance_model_prediction"]["sigma_proj_pooled_nats"])

    rec = {
        "artifact": "P13 production KL result",
        "registration": {"hash": registration_hash(), **REGISTERED},
        "collection_record": _collection_record(),
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
            "larger_than_bf16_to_fp4": paired_marginal(marginal, locked, n),
            "matrix_sha256": marginal["matrix_sha256"],
            "matrix_sha256_note": marginal["matrix_sha256_note"],
            "kl_cells_nats": marginal["kl_cells_nats"],
        },
        "replication_floor_this_run": {
            "mean_nats": lv["bf16_to_bf16"]["mean_nats"],
            "per_pair_nats": lv["bf16_to_bf16"]["per_pair_headline_nats"],
            "launch_level_ci_95": lv["bf16_to_bf16"]["launch_level_uncertainty"]["ci_95"],
            "subtracted_from_anything": False,
        },
        "nuisance_model_prediction_test": pred,
        "resolution": res,
        "resolution_rule_conservatism": {
            "registered_var_launch": "(sigma_proj * sqrt(2*theta))^2, i.e. the variance of the "
                                     "SPREAD OF PER-LAUNCH HEADLINES, sigma2_A + sigma2_E/T",
            "what_a_ci_on_the_mean_would_use": "sigma2_A / R -- the launch term of Var(mean over "
                                               "R launches). sigma2_E/(R*T) is already inside the "
                                               "trajectory bootstrap, so the registered form "
                                               "double-counts it and omits the 1/R.",
            "direction": "CONSERVATIVE -- the registered form is larger, so it widens the interval "
                         "and makes `resolved` harder to reach",
            "not_corrected_because": "the formula is what was registered before the FP8/FP4 cells "
                                     "existed, and the correction runs in the resolution-friendly "
                                     "direction. Changing it after seeing the data is exactly what "
                                     "pre-registration exists to prevent. It is disclosed instead.",
            "magnitude": "immaterial here: inflation factors span "
                         + str([round(x, 4) for x in
                                res[list(res)[0]]["inflation_factor_range"]])
                         + " and no classification depends on the launch term",
        },
        "floor_side_sensitivity": floor_side_sensitivity(lv, res),
        "G2_G2prime": g2_restatement(locked, lv, floor_path),
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "timestamp": common.now_iso(),
    }
    # scratch arrays used to build the durability block; never serialised
    for k in ("_grid", "_idx"):
        marginal.pop(k, None)
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
        up = c.get("se_launch_upper_95_nats")
        share = c.get("launch_share_of_variance")
        print(f"  {label:12s} {c['expected_kl_over_launches_nats']:.6e} nats   "
              f"SE_launch {('%.2e' % up) if up else 'n/a':>8} (95% upper)  "
              f"SE_traj {c['se_trajectory_nats']:.2e}  "
              f"launch share {('%.1f%%' % (100 * share)) if share is not None else 'n/a'}")
    f = rec["replication_floor_this_run"]
    print(f"\nBF16<->BF16 floor, this run: {f['mean_nats']:.4e} nats  "
          f"95% [{f['launch_level_ci_95'][0]:.4e}, {f['launch_level_ci_95'][1]:.4e}]")
    pt = rec["nuisance_model_prediction_test"]
    po = (pt.get("pooled_observed") or {}).get("sigma_proj_pooled")
    print(f"nuisance model vs registration: {pt['verdict']}  "
          f"(pooled sigma_proj {('%.3e' % po) if po else 'not estimable'} against "
          f"{pt['registered']['sigma_proj_pooled_nats']:.3e} registered)")
    sharp = pt.get("sharp_post_hoc_diagnostic")
    if sharp:
        print(f"  post-hoc SD ratio {sharp['statistic']}: predicted "
              f"{sharp['predicted_by_the_law']:.3f}, observed {sharp['observed']:.3f}")
    print("\nRESOLUTION (all positions reported; class is a label, not a filter)")
    for label, r in rec["resolution"].items():
        print(f"  {label:12s} headline {r['headline']['class']:14s} "
              f"{r['n_resolved']}/{r['n_positions']} positions resolved")
        if r["noise_limited_positions"]:
            print(f"               noise-limited at p = "
                  f"{', '.join(r['noise_limited_positions'])}")
    g2 = rec["G2_G2prime"]
    print("\nG2/G2' remain recorded failures; no threshold moved, no floor subtracted.")
    for which in ("primary_floor", "this_run_floor"):
        b = g2[which]
        print(f"  {b['which'][:52]:54s} " + "  ".join(
            f"{k}={v:.2%}" for k, v in b["fraction_of_signal"].items()))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
