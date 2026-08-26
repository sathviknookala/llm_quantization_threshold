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
from harness.quality import kl_math as K, qcommon as q  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["numerics"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rec = {"numerics": numerics_compat}[a.gate]()
    out = a.out or os.path.join(q.QUALITY_DIR, "gates", f"{rec['gate']}.json")
    common.write_json(out, rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("software",)}, indent=2)[:4000])
    print("WROTE", out)
    return 0 if rec.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
