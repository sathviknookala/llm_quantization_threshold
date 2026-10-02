"""README highlight figures, drawn only from committed result artifacts.

Needs matplotlib and nothing else; it deliberately does not import the harness, so it can run
outside the pinned measurement environments.

    python scripts/plot_readme_figures.py
"""

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = os.path.join(ROOT, "results/sweep/cells.jsonl")
KL_SUMMARY = os.path.join(ROOT, "results/quality/kl/kl_summary.json")
P13_SUMMARY = os.path.join(ROOT, "results/quality/kl/p13_summary.json")
OUT_DIR = os.path.join(ROOT, "docs/figures")

CONFIGS = [("BF16_REFERENCE", "BF16"), ("FP8_PRIMARY", "FP8"), ("FP4_PRIMARY", "NVFP4")]
# The ceiling replication is partial; the README's ceilings come from the original sweep only.
SWEEP_JOBS = {"SWEEP", "SWEEP_REFINE", "SWEEP_REFINE_SLO"}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984",
        "grid": "#e4e3df", "series": {"BF16": "#2a78d6", "FP8": "#eb6834", "NVFP4": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "muted": "#8f8e86",
        "grid": "#33332f", "series": {"BF16": "#3987e5", "FP8": "#d95926", "NVFP4": "#199e70"},
    },
}


def load_cells():
    with open(CELLS) as f:
        cells = [json.loads(line) for line in f if line.strip()]
    return [c for c in cells if c.get("job") in SWEEP_JOBS and c.get("workload") == "DECODE_PRIMARY"
            and c.get("valid_result") and c.get("tpot_ms_p95") is not None]


def slo_ceilings(cells):
    out = {}
    for cid, label in CONFIGS:
        refine = {c["concurrency"]: c["meets_slo"] for c in cells
                  if c["job"] == "SWEEP_REFINE_SLO" and c["configuration_id"] == cid}
        ceiling = max(k for k, ok in refine.items() if ok)
        if refine.get(ceiling + 1) is not False:
            raise SystemExit(f"{label}: no recorded SLO breach at C={ceiling + 1}")
        out[label] = ceiling
    return out


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["muted"])
    ax.tick_params(colors=t["text2"], labelsize=9)
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["text2"])
    ax.yaxis.label.set_color(t["text2"])


def new_fig(t, size=(8, 4.6)):
    fig, ax = plt.subplots(figsize=size, dpi=100)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    return fig, ax


def title(fig, t, head, sub):
    fig.text(0.015, 0.965, head, color=t["text"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.015, 0.905, sub, color=t["text2"], fontsize=9.5, va="top")


def save(fig, name, theme):
    path = os.path.join(OUT_DIR, f"{name}_{theme}.svg")
    fig.savefig(path, facecolor=fig.get_facecolor(), metadata={"Date": None})
    plt.close(fig)
    return path


def nats(v, _pos=None):
    return f"{v:g}"


def fig_tradeoff(t, ceilings, pairs, floor, theme):
    fig, ax = new_fig(t, (8, 5))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.12)
    title(fig, t, "FP8 captures most of the capacity; NVFP4 adds most of the drift",
          "Serving capacity under a 50 ms p95 TPOT SLO vs. next-token KL divergence from BF16")

    pts = {"BF16": (ceilings["BF16"], floor["mean_nats"], *floor["launch_level_ci_95"])}
    for label, key in (("FP8", "BF16||FP8"), ("NVFP4", "BF16||FP4")):
        p = pairs[key]
        b = p["headline_bootstrap"]
        pts[label] = (ceilings[label], p["headline_nats"], b["ci_low"], b["ci_high"])

    for a, b in (("BF16", "FP8"), ("FP8", "NVFP4")):
        ax.annotate("", xy=pts[b][:2], xytext=pts[a][:2],
                    arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.4,
                                    shrinkA=9, shrinkB=9))

    for label, (x, y, lo, hi) in pts.items():
        c = t["series"][label]
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, elinewidth=2,
                    capsize=4, capthick=2, zorder=3)
        ax.plot([x], [y], "o", ms=10, color=c, mec=t["surface"], mew=2, zorder=4)

    bx, by = pts["BF16"][:2]
    ax.annotate(f"BF16  ·  {bx} requests\nreference; self-KL across\nrelaunches = {by:.2g} nats",
                (bx, by), xytext=(12, 4), textcoords="offset points", color=t["text"], fontsize=9,
                va="center")
    fx, fy = pts["FP8"][:2]
    ax.annotate(f"FP8  ·  {fx} requests ({fx / bx:.2f}× BF16)\nKL {fy:.4f} nats",
                (fx, fy), xytext=(-12, 10), textcoords="offset points", color=t["text"],
                fontsize=9, ha="right", va="bottom")
    nx, ny = pts["NVFP4"][:2]
    ax.annotate(f"NVFP4  ·  {nx} requests\n({nx / bx:.2f}× BF16)\nKL {ny:.4f} nats",
                (nx, ny), xytext=(-12, 0), textcoords="offset points", color=t["text"],
                fontsize=9, ha="right", va="center")

    step = pairs["FP8||FP4"]["headline_nats"]
    ratio = step / pairs["BF16||FP8"]["headline_nats"]
    mx, my = (fx + nx) / 2, (fy * ny) ** 0.5
    ax.annotate(f"FP8 → NVFP4 step\n{nx / fx:.2f}× capacity\n{ratio:.1f}× the KL of the\n"
                f"BF16 → FP8 step\n(direct KL {step:.4f})",
                (mx, my), xytext=(14, -6), textcoords="offset points", color=t["text2"],
                fontsize=8.5, va="top")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(nats))
    ax.set_ylim(5e-5, 0.4)
    ax.set_xlim(0, 90)
    ax.set_xlabel("Max in-flight requests within SLO (refined bisection, n=1)")
    ax.set_ylabel("Mean KL vs. BF16, nats (log)")
    fig.text(0.015, 0.02, "Bars: 95% trajectory-bootstrap CI (BF16: launch-level CI). "
             "KL is distributional shift, not task accuracy.", color=t["muted"], fontsize=8)
    return save(fig, "tradeoff", theme)


def fig_latency(t, cells, ceilings, theme):
    fig, ax = new_fig(t)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.8, bottom=0.13)
    title(fig, t, "Each step down in precision moves the SLO wall to the right",
          "p95 time per output token vs. concurrency  ·  512 input → 2,048 output tokens")

    for cid, label in CONFIGS:
        c = t["series"][label]
        by_c = defaultdict(list)
        for cell in cells:
            if cell["configuration_id"] == cid:
                by_c[cell["concurrency"]].append(cell["tpot_ms_p95"])
        xs = sorted(by_c)
        med = [sorted(by_c[x])[len(by_c[x]) // 2] for x in xs]
        ax.plot(xs, med, "-", color=c, lw=2, label=label, zorder=3)
        ax.plot([x for x in xs for _ in by_c[x]], [v for x in xs for v in by_c[x]], "o", ms=3.5,
                color=c, alpha=0.55, mew=0, zorder=3)
        cx = ceilings[label]
        cy = sorted(by_c[cx])[len(by_c[cx]) // 2]
        ax.plot([cx], [cy], "o", ms=9, color=c, mec=t["surface"], mew=2, zorder=4)
        ax.annotate(f"{label} max {cx}", (cx, cy), xytext=(8, -14), textcoords="offset points",
                    color=t["text"], fontsize=9, ha="left")

    ax.axhline(50, color=t["text2"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(99, 51.5, "50 ms SLO", color=t["text2"], fontsize=9, ha="right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.set_xlabel("In-flight requests (closed loop)")
    ax.set_ylabel("TPOT p95, ms")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=t["text"])
    leg.set_zorder(5)
    fig.text(0.015, 0.02, "Line: median across repetitions; dots: individual cells. "
             "Coarse ladder n=3, refined boundary cells n=1.", color=t["muted"], fontsize=8)
    return save(fig, "latency", theme)


def fig_positions(t, pairs, floor, theme):
    fp8, fp4 = pairs["BF16||FP8"]["by_position"], pairs["BF16||FP4"]["by_position"]
    ratios = [fp4[p]["point"] / fp8[p]["point"] for p in fp8]
    fig, ax = new_fig(t)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.13)
    lo, hi = min(ratios), max(ratios)
    title(fig, t, f"NVFP4 drifts {lo:.0f}–{hi:.0f}× more than FP8 at every retained position",
          "Mean next-token KL by generation position  ·  64 frozen BF16 continuations")

    series = [("BF16||FP8", "BF16 → FP8", t["series"]["FP8"], "-"),
              ("BF16||FP4", "BF16 → NVFP4", t["series"]["NVFP4"], "-"),
              ("FP8||FP4", "FP8 → NVFP4 (direct)", t["text2"], (0, (4, 2)))]
    for key, label, c, ls in series:
        bp = pairs[key]["by_position"]
        pos = sorted(bp, key=int)
        xs = [int(p) for p in pos]
        ax.fill_between(xs, [bp[p]["ci_low"] for p in pos], [bp[p]["ci_high"] for p in pos],
                        color=c, alpha=0.14, lw=0, zorder=2)
        ax.plot(xs, [bp[p]["point"] for p in pos], ls=ls, color=c, lw=2, marker="o", ms=4.5,
                label=label, zorder=3)

    ax.axhline(floor["mean_nats"], color=t["muted"], lw=1.2, ls=(0, (1, 2)), zorder=2)
    ax.text(2048, floor["mean_nats"] * 0.7, "BF16 self-KL across relaunches (mean)",
            color=t["muted"], fontsize=8.5, ha="right", va="top")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v)}"))
    ticks = [int(p) for p in pairs["BF16||FP8"]["by_position"]]
    # 1536 sits too close to 1024 and 2048 on a log axis to carry its own label.
    ax.set_xticks([p for p in ticks if p != 1536])
    ax.set_xticks([p for p in ticks if p == 1536], minor=True)
    ax.tick_params(axis="x", which="minor", colors=t["muted"], length=3)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))
    ax.set_yscale("log")
    ax.set_ylim(3e-5, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(nats))
    ax.set_xlabel("Generated-token position (log scale)")
    ax.set_ylabel("Mean KL, nats (log)")
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=t["text"])
    fig.text(0.015, 0.02, "Bands: 95% bootstrap CI over whole trajectories. "
             "Arrow direction: left-hand configuration is the reference.",
             color=t["muted"], fontsize=8)
    return save(fig, "kl_by_position", theme)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "readme-figures",
                         "svg.fonttype": "path"})
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = load_cells()
    ceilings = slo_ceilings(cells)
    with open(KL_SUMMARY) as f:
        pairs = json.load(f)["pairs"]
    with open(P13_SUMMARY) as f:
        floor = json.load(f)["replication_floor_this_run"]
    for theme, t in THEMES.items():
        for path in (fig_tradeoff(t, ceilings, pairs, floor, theme),
                     fig_latency(t, cells, ceilings, theme),
                     fig_positions(t, pairs, floor, theme)):
            print(os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
