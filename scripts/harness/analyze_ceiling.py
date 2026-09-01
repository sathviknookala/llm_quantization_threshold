"""Adjudicate the SLO-ceiling replication against its pre-registered criterion.

Kept out of run_sweep.py for the same reason analyze.py is kept out of run_pilot.py: a runner
that scores its own cells can quietly acquire a criterion it was never given.

The criterion is `docs/EXPERIMENTAL_CONTRACT.md`, "Ceiling replication pass". It is restated in
CRITERION and embedded in the output so a stored verdict names the rule that produced it.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import common, orchestration as orch  # noqa: E402
from harness.run_sweep import CEILING_REP, CEILING_REPS, ceiling_triplet  # noqa: E402

WORKLOAD = "DECODE_PRIMARY"
TARGET_REPS = (1,) + tuple(CEILING_REPS)

CRITERION = {
    "source": "docs/EXPERIMENTAL_CONTRACT.md -- Ceiling replication pass",
    "observed_ceiling": "largest C in the triplet with meets_slo true, verdicts monotone",
    "cell_admissible": "status OK and meets_slo is not None; no post-hoc exclusion",
    "per_repetition": {
        "CONFIRMED": "K meets the SLO and K+1 does not",
        "MOVED_DOWN": "K fails, K-1 meets the SLO -- ceiling is K-1 in this repetition",
        "UNRESOLVED_ABOVE": "K+1 meets the SLO; the ceiling is above the triplet. Recorded and "
                            "stopped -- widening the probe set after seeing the data is a "
                            "retroactive re-spacing",
        "UNRESOLVED_BELOW": "K and K-1 both fail; the ceiling is below the triplet",
        "NON_MONOTONE": "verdicts have more than one pass->fail transition",
        "INCOMPLETE": "K or the point needed to place the ceiling was not measured",
    },
    "overall": {
        "CONFIRMED": "every repetition returns the repetition-1 ceiling K",
        "MOVED": "every repetition agrees on some K' != K; K' supersedes K",
        "UNSTABLE": "repetitions disagree; report the observed range, drop the point estimate",
        "NOT_YET_REPLICATED": "fewer than the target repetitions have usable triplets",
    },
    "target_repetitions": list(TARGET_REPS),
}


def cell_index(cells):
    """(config, C, rep) -> cell, flagging any duplicate the job label would otherwise hide."""
    idx, dupes = {}, []
    for r in cells:
        if r.get("workload") != WORKLOAD:
            continue
        key = (r.get("configuration_id"), r.get("concurrency"), r.get("repetition"))
        if key in idx:
            dupes.append((key, idx[key].get("job"), r.get("job")))
        idx[key] = r
    return idx, dupes


def cell_view(rec):
    if rec is None:
        return None
    p95 = rec.get("tpot_ms_p95")
    return {
        "job": rec.get("job"),
        "status": rec.get("status"),
        "meets_slo": rec.get("meets_slo"),
        "tpot_ms_p95": p95,
        "margin_ms": round(common.SLO_TPOT_MS - p95, 3) if p95 is not None else None,
        "output_tokens_per_s": rec.get("output_tokens_per_s"),
        "num_preemptions_delta": rec.get("num_preemptions_delta"),
        "kv_cache_tokens": rec.get("kv_cache_tokens"),
    }


def admissible(rec):
    return rec is not None and rec.get("status") == "OK" and rec.get("meets_slo") is not None


def score_repetition(k, triplet, cells_at):
    """Apply the per-repetition criterion. Returns (verdict, observed_ceiling)."""
    v = {C: cells_at[C].get("meets_slo") for C in triplet if admissible(cells_at.get(C))}
    if k not in v or (k + 1) not in v:
        return "INCOMPLETE", None
    # a pass above a fail is the one shape the triplet cannot summarise as a ceiling
    seq = [v[C] for C in triplet if C in v]
    if any(seq[i] is False and seq[i + 1] is True for i in range(len(seq) - 1)):
        return "NON_MONOTONE", None
    if v[k + 1]:
        return "UNRESOLVED_ABOVE", None
    if v[k]:
        return "CONFIRMED", k
    if (k - 1) not in v:
        return "INCOMPLETE", None
    if v[k - 1]:
        return "MOVED_DOWN", k - 1
    return "UNRESOLVED_BELOW", None


def spread(views):
    vals = [x["tpot_ms_p95"] for x in views if x and x["tpot_ms_p95"] is not None]
    if len(vals) < 2:
        return None
    lo, hi = min(vals), max(vals)
    return {"n": len(vals), "min_ms": lo, "max_ms": hi, "range_ms": round(hi - lo, 3),
            "range_pct": round(100.0 * (hi - lo) / lo, 3)}


def analyze(cells):
    idx, dupes = cell_index(cells)
    out = {}
    for cfg, k in CEILING_REP.items():
        triplet = ceiling_triplet(cfg)
        reps, ceilings = {}, {}
        for rep in TARGET_REPS:
            at = {C: idx.get((cfg, C, rep)) for C in triplet}
            verdict, obs = score_repetition(k, triplet, at)
            reps[str(rep)] = {"verdict": verdict, "observed_ceiling": obs,
                              "cells": {str(C): cell_view(at[C]) for C in triplet}}
            if obs is not None:
                ceilings[rep] = obs
        blocking = [r for r in TARGET_REPS
                    if reps[str(r)]["verdict"] in ("NON_MONOTONE", "UNRESOLVED_ABOVE",
                                                   "UNRESOLVED_BELOW")]
        if blocking:
            status = reps[str(blocking[0])]["verdict"]
        elif len(ceilings) < len(TARGET_REPS):
            status = "NOT_YET_REPLICATED"
        elif set(ceilings.values()) == {k}:
            status = "CONFIRMED"
        elif len(set(ceilings.values())) == 1:
            status = "MOVED"
        else:
            status = "UNSTABLE"
        out[cfg] = {
            "ceiling_rep1": k,
            "triplet": triplet,
            "status": status,
            "n": len(ceilings),
            "observed_ceilings": {str(r): c for r, c in sorted(ceilings.items())},
            "observed_range": ([min(ceilings.values()), max(ceilings.values())]
                               if ceilings else None),
            "repetitions": reps,
            "matched_cell_tpot_spread": {
                str(C): spread([reps[str(r)]["cells"][str(C)] for r in TARGET_REPS])
                for C in triplet},
        }
    order = ["UNSTABLE", "NON_MONOTONE", "UNRESOLVED_ABOVE", "UNRESOLVED_BELOW",
             "NOT_YET_REPLICATED", "MOVED", "CONFIRMED"]
    overall = next((s for s in order if any(v["status"] == s for v in out.values())), "CONFIRMED")
    return {
        "artifact": "SLO ceiling replication",
        "generated_at": common.now_iso(),
        "slo_tpot_ms": common.SLO_TPOT_MS,
        "criterion": CRITERION,
        "duplicate_cells": [{"key": list(k_), "jobs": [a, b]} for k_, a, b in dupes],
        "overall_status": overall,
        "configurations": out,
    }


def report(res):
    print(f"SLO ceiling replication -- overall {res['overall_status']}")
    print(f"SLO: TPOT P95 <= {res['slo_tpot_ms']} ms\n")
    if res["duplicate_cells"]:
        print(f"WARNING: {len(res['duplicate_cells'])} duplicate (config, C, rep) keys across "
              f"job labels; the later cell was used")
        for d in res["duplicate_cells"]:
            print(f"  {d['key']} jobs={d['jobs']}")
        print()
    for cfg, v in res["configurations"].items():
        print(f"{cfg}  K(rep1)={v['ceiling_rep1']}  triplet={v['triplet']}  "
              f"n={v['n']}  {v['status']}")
        print(f"  {'rep':>4}{'verdict':>19}{'K_r':>5}   " +
              "".join(f"{'C=' + str(C):>26}" for C in v["triplet"]))
        for rep in TARGET_REPS:
            r = v["repetitions"][str(rep)]
            row = f"  {rep:>4}{r['verdict']:>19}{str(r['observed_ceiling']):>5}   "
            for C in v["triplet"]:
                c = r["cells"][str(C)]
                if c is None or c["tpot_ms_p95"] is None:
                    row += f"{'-':>26}"
                else:
                    row += (f"{c['tpot_ms_p95']:>10.2f} ms "
                            f"{c['margin_ms']:>+7.2f} {str(c['meets_slo']):>6}")
            print(row)
        sp = {C: v["matched_cell_tpot_spread"][str(C)] for C in v["triplet"]}
        parts = [f"C={C}: {s['range_ms']:.2f} ms over n={s['n']}"
                 for C, s in sp.items() if s]
        print("  matched-cell TPOT spread   " + ("; ".join(parts) if parts else "n<2 everywhere"))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=common.SWEEP_DIR)
    ap.add_argument("--write", action="store_true",
                    help="persist the verdict to <dir>/ceiling_replication.json")
    a = ap.parse_args()
    cells = orch.read_cells(os.path.join(a.dir, "cells.jsonl"))
    if not cells:
        raise SystemExit(f"no cells in {a.dir}")
    res = analyze(cells)
    report(res)
    if a.write:
        path = os.path.join(a.dir, "ceiling_replication.json")
        common.write_json(path, res)
        print(f"wrote {path}")
    else:
        print("(not written; pass --write to persist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
