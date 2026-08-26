"""No-GPU test suite for the quality arm: unit oracles, invariants and failure injection.

Mirrors harness/selftest.py's role -- prove the contract mechanically before spending GPU time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from harness.quality import kl_math as K, positions as P  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    ok = got == want
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"   got={got!r} want={want!r}"))
    return ok


def raises(name, fn, exc=P.PositionContractError):
    try:
        fn()
    except exc:
        return check(name, "raised", "raised")
    except Exception as e:  # noqa: BLE001
        return check(name, f"raised {type(e).__name__}", "raised")
    return check(name, "no raise", "raised")


def synthetic():
    """Distinct token IDs so an off-by-one is visible in the value, not just the length."""
    prompt = [1_000_000 + i for i in range(P.PROMPT_TOKENS)]
    cont = list(range(P.CONTINUATION_TOKENS))
    return prompt, cont


def test_positions():
    print("positions: contract")
    prompt, cont = synthetic()

    for p in P.RETAINED_POSITIONS:
        check(f"context_len({p}) == 511+p", P.context_len(p), 511 + p)

    for p, want_len in ((1, 512), (8, 519), (512, 1023), (2048, 2559)):
        ctx, tgt = P.build_context(prompt, cont, p)
        check(f"p={p} context length", len(ctx), want_len)
        check(f"p={p} target is continuation[p-1]", tgt, p - 1)
        check(f"p={p} context starts with the whole prompt", ctx[:512], prompt)
        if p == 1:
            check("p=1 context is exactly the prompt", ctx, prompt)
        else:
            check(f"p={p} context tail is continuation[p-2]", ctx[-1], p - 2)
        check(f"p={p} target is not the context tail", ctx[-1] == tgt, False)

    cells = P.build_all(prompt, cont)
    check("build_all yields one cell per retained position", len(cells), 10)
    check("build_all nesting invariant", P.assert_nesting(cells), True)
    check("build_all targets", [c["target_token_id"] for c in cells],
          [p - 1 for p in P.RETAINED_POSITIONS])
    check("build_all lengths", [c["context_len"] for c in cells],
          [511 + p for p in P.RETAINED_POSITIONS])

    print("positions: input validation")
    raises("short prompt rejected", lambda: P.build_context(prompt[:-1], cont, 8))
    raises("long prompt rejected", lambda: P.build_context(prompt + [7], cont, 8))
    raises("short continuation rejected", lambda: P.build_context(prompt, cont[:-1], 8))
    raises("long continuation rejected", lambda: P.build_context(prompt, cont + [7], 8))
    raises("non-retained position rejected", lambda: P.build_context(prompt, cont, 9))
    raises("position 0 rejected", lambda: P.build_context(prompt, cont, 0))
    raises("position 2049 rejected", lambda: P.build_context(prompt, cont, 2049))

    print("positions: negative controls (must fail loudly)")
    for shift in (+1, -1):
        bad, tested = [], 0
        for c in P.build_all(prompt, cont):
            p = c["position_p"]
            q = p + shift
            if not 1 <= q <= P.CONTINUATION_TOKENS:
                continue
            ctx = list(prompt) + list(cont[:q - 1])
            tgt = cont[q - 1]
            ok, _ = P.rederive_and_check(prompt, cont, p, len(ctx), tgt, ctx)
            bad.append(ok)
            tested += 1
        check(f"shift {shift:+d} exercised at 9+ positions", tested >= 9, True)
        check(f"shift {shift:+d} caught at every position", any(bad), False)

    scrambled = P.build_all(prompt, cont)
    labels = [c["position_p"] for c in scrambled]
    rotated = labels[1:] + labels[:1]
    ok_any = False
    for c, lab in zip(scrambled, rotated):
        ok, _ = P.rederive_and_check(prompt, cont, lab, c["context_len"], c["target_token_id"],
                                     c["context_ids"])
        ok_any = ok_any or ok
    check("position-label scramble caught", ok_any, False)

    other_cont = [50_000 + i for i in range(P.CONTINUATION_TOKENS)]
    ok_any = False
    for c in P.build_all(prompt, cont):
        _, foreign_target = P.build_context(prompt, other_cont, c["position_p"])
        ok, _ = P.rederive_and_check(prompt, cont, c["position_p"], c["context_len"],
                                     foreign_target, c["context_ids"])
        ok_any = ok_any or ok
    check("cross-trajectory target swap caught", ok_any, False)

    broken = P.build_all(prompt, cont)
    broken[3]["context_ids"] = broken[3]["context_ids"][:-1] + [999999]
    raises("nesting invariant catches a mutated context",
           lambda: P.assert_nesting(broken))

    print("positions: re-derivation accepts the truth")
    allgood = all(P.rederive_and_check(prompt, cont, c["position_p"], c["context_len"],
                                       c["target_token_id"], c["context_ids"])[0]
                  for c in P.build_all(prompt, cont))
    check("correct cells re-derive cleanly", allgood, True)


def close(name, got, want, tol):
    ok = abs(got - want) <= tol
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"   got={got!r} want={want!r}"))
    return ok


def test_kl_math():
    print("kl_math: numerics")
    a = np.log(np.array([0.5, 0.3, 0.2]))
    b = np.log(np.array([0.4, 0.4, 0.2]))
    analytic = float(sum(p * np.log(p / qq) for p, qq in zip([.5, .3, .2], [.4, .4, .2])))
    close("analytic 3-symbol KL", K.kl_nats(a, b), analytic, 1e-15)
    check("KL of identical distributions is exactly zero", K.kl_nats(a, a), 0.0)
    check("KL is asymmetric", K.kl_nats(a, b) == K.kl_nats(b, a), False)
    close("KL is invariant to unnormalised shifts", K.kl_nats(a + 7.5, b - 3.25), analytic, 1e-12)
    close("entropy of a uniform 4-symbol distribution",
          K.entropy_nats(np.log(np.full(4, 0.25))), float(np.log(4)), 1e-15)
    check("top1 picks the argmax", K.top1(a), 0)

    print("kl_math: domain failures (no silent flooring)")
    zeroed = np.log(np.array([0.5, 0.5, 0.0]))
    raises("zero comparison mass where reference has mass raises",
           lambda: K.kl_nats(a, zeroed), K.KLDomainError)
    check("reference may itself have zero-mass tokens", K.kl_nats(zeroed, a) > 0.0, True)
    raises("all-(-inf) distribution raises",
           lambda: K.kl_nats(np.full(3, -np.inf), a), K.KLDomainError)
    raises("non-finite grid raises",
           lambda: K.headline(np.array([[0.1, np.nan], [0.2, 0.3]])), K.KLDomainError)
    raises("1-D grid raises", lambda: K.headline(np.array([0.1, 0.2])), K.KLDomainError)

    print("kl_math: aggregation")
    rng = np.random.default_rng(7)
    grid = rng.lognormal(-5, 1, size=(64, 10))
    close("balanced grid: mean-of-means equals pooled mean",
          K.headline(grid), float(grid.mean()), 1e-12)
    ragged = [np.array([1.0, 3.0, 5.0]), np.array([9.0])]
    grouped = float(np.mean([g.mean() for g in ragged]))
    pooled = float(np.concatenate(ragged).mean())
    check("unbalanced groups: the two orders genuinely differ", abs(grouped - pooled) > 0.5, True)
    check("trajectory_means returns one value per trajectory",
          K.trajectory_means(grid).shape, (64,))


def test_bootstrap():
    print("bootstrap: determinism and unit")
    rng = np.random.default_rng(11)
    grid = rng.lognormal(-5, 1, size=(64, 10))

    i1 = K.bootstrap_indices(64, 10000, 20260825)
    i2 = K.bootstrap_indices(64, 10000, 20260825)
    check("same seed reproduces the index matrix byte-identically",
          bool((i1 == i2).all()), True)
    check("index matrix shape", i1.shape, (10000, 64))
    check("indices stay in range", bool((i1 >= 0).all() and (i1 < 64).all()), True)
    check("a different seed gives different draws",
          bool((K.bootstrap_indices(64, 10000, 1) == i1).all()), False)

    d1 = K.bootstrap_headline(grid, i1)
    d2 = K.bootstrap_headline(grid, i2)
    check("headline draws reproduce byte-identically", bool((d1 == d2).all()), True)
    check("headline draw count", d1.shape, (10000,))
    ci1 = K.percentile_ci(d1)
    ci2 = K.percentile_ci(d2)
    check("percentile CI reproduces byte-identically", ci1, ci2)
    lo, hi = ci1
    check("point estimate lies inside its CI", lo <= K.headline(grid) <= hi, True)

    pos = K.bootstrap_positions(grid, i1)
    check("position draws shape", pos.shape, (10000, 10))
    close("position draws average to the headline draws",
          float(np.abs(pos.mean(axis=1) - d1).max()), 0.0, 1e-12)

    print("bootstrap: the resampling unit is the trajectory")
    # positions perfectly correlated within a trajectory: cell-level resampling would understate
    # the interval by sqrt(10) because it treats 10 repeats of one context as 10 contexts
    base = rng.lognormal(-5, 1, size=64)
    corr = np.repeat(base[:, None], 10, axis=1)
    traj_draws = K.bootstrap_headline(corr, i1)
    tlo, thi = K.percentile_ci(traj_draws)
    flat = corr.reshape(-1)
    cell_idx = np.random.default_rng(20260825).integers(0, flat.size, size=(10000, flat.size))
    cell_draws = flat[cell_idx].mean(axis=1)
    clo, chi = K.percentile_ci(cell_draws)
    ratio = (thi - tlo) / (chi - clo)
    check("trajectory-unit CI is ~sqrt(10) wider than cell-unit CI", 2.5 < ratio < 4.0, True)
    print(f"       (measured width ratio {ratio:.2f}, sqrt(10) = 3.16)")

    print("bootstrap: diagnostics")
    summ = K.bootstrap_summary(K.headline(grid), d1)
    check("summary reports bias", "bias" in summ, True)
    check("summary reports standard error", "std_error" in summ, True)
    close("bias is small for a well-behaved grid", summ["bias"], 0.0, 1e-3)
    check("ci_half_width is consistent",
          round(summ["ci_half_width"], 12),
          round((summ["ci_high"] - summ["ci_low"]) / 2.0, 12))


def main():
    tests = [test_positions, test_kl_math, test_bootstrap]
    if os.environ.get("QSELFTEST_ONLY"):
        want = os.environ["QSELFTEST_ONLY"]
        tests = [t for t in tests if want in t.__name__]
    for t in tests:
        t()
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
