"""Minimal inventory of a sweep artifact. Not the analyzer -- no criteria, no verdicts.

Exists so a run in progress or just finished can be read without inventing conclusions.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import common, orchestration as orch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=common.SWEEP_DIR)
    a = ap.parse_args()
    cells = orch.read_cells(os.path.join(a.dir, "cells.jsonl"))
    if not cells:
        print("no cells recorded yet")
        return
    print(f"{len(cells)} cells in {a.dir}\n")

    by_status = {}
    for c in cells:
        by_status.setdefault(c.get("status"), []).append(c)
    print("status".ljust(26) + "outcome".ljust(13) + "n")
    for st in sorted(by_status, key=lambda k: -len(by_status[k])):
        oc = by_status[st][0].get("outcome_class")
        print(f"  {str(st):24}{str(oc):13}{len(by_status[st])}")

    print("\nDECODE_PRIMARY cells")
    hdr = (f"{'cfg':5}{'C':>4}{'rep':>4}{'status':>18}{'outcome':>12}{'tok/s':>9}"
           f"{'tpotP95':>9}{'slo':>6}{'kvmax':>7}{'pre':>5}{'per':>6}{'wall':>7}{'kvtok':>8}")
    print(hdr)
    rows = [c for c in cells if c.get("workload") == "DECODE_PRIMARY"]
    for c in sorted(rows, key=lambda r: (r.get("configuration_id", ""), r.get("repetition", 0),
                                         r.get("concurrency", 0))):
        def g(k, f="{}"):
            v = c.get(k)
            return "-" if v is None else f.format(v)
        print(f"{c.get('quantization',''):5}{c.get('concurrency',0):>4}{c.get('repetition',0):>4}"
              f"{str(c.get('status'))[:18]:>18}{str(c.get('outcome_class'))[:12]:>12}"
              f"{g('output_tokens_per_s','{:.1f}') if c.get('output_tokens_per_s') else '-':>9}"
              f"{g('tpot_ms_p95','{:.1f}') if c.get('tpot_ms_p95') else '-':>9}"
              f"{str(c.get('meets_slo')):>6}"
              f"{g('kv_cache_usage_max','{:.3f}') if c.get('kv_cache_usage_max') else '-':>7}"
              f"{g('num_preemptions_delta','{:.0f}') if c.get('num_preemptions_delta') is not None else '-':>5}"
              f"{g('periods_in_window','{:.2f}') if c.get('periods_in_window') else '-':>6}"
              f"{g('cell_wall_seconds','{:.0f}') if c.get('cell_wall_seconds') else '-':>7}"
              f"{g('kv_cache_tokens'):>8}")

    defects = [c for c in cells if c.get("outcome_class") == "defect"]
    if defects:
        print(f"\n{len(defects)} DEFECT cells -- these are harness faults, not findings:")
        for c in defects:
            print(f"  {c.get('tag')} {c.get('status')} {c.get('invalid_reasons')}")

    ic = [c for c in cells if c.get("outcome_class") == "infeasible"]
    if ic:
        print(f"\n{len(ic)} infeasible cells:")
        for c in sorted(ic, key=lambda r: (r.get("quantization", ""), r.get("concurrency", 0))):
            why = c.get("skipped_because_slo_violated_at_c")
            print(f"  {c.get('tag')} {c.get('status')}"
                  + (f" (slo violated at C={why})" if why else ""))


if __name__ == "__main__":
    main()
