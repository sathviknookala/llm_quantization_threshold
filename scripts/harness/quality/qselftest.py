"""No-GPU test suite for the quality arm: unit oracles, invariants and failure injection.

Mirrors harness/selftest.py's role -- prove the contract mechanically before spending GPU time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from harness.quality import kl_math as K, positions as P, qcommon as q  # noqa: E402

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

    raises("re-derivation refuses to run without the context tokens",
           lambda: P.rederive_and_check(prompt, cont, 8, 519, 7), TypeError)

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
    raises("+inf in the reference raises",
           lambda: K.kl_nats(np.array([np.inf, -1.0, -2.0]), a), K.KLDomainError)
    raises("NaN in the reference raises",
           lambda: K.kl_nats(np.array([np.nan, -1.0, -2.0]), a), K.KLDomainError)
    raises("+inf in the comparison raises",
           lambda: K.kl_nats(a, np.array([np.inf, -1.0, -2.0])), K.KLDomainError)
    raises("NaN in the comparison raises",
           lambda: K.kl_nats(a, np.array([np.nan, -1.0, -2.0])), K.KLDomainError)
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


def test_completeness():
    print("completeness: the ten-position rule")
    full = [{"trajectory_index": t, "position_p": pp} for t, pp in P.grid_order(3)]
    check("a complete 3x10 grid passes", P.assert_complete_grid(full, 3), True)

    raises("a trajectory missing one position fails",
           lambda: P.assert_complete_grid([c for c in full if not
                                           (c["trajectory_index"] == 1 and c["position_p"] == 256)],
                                          3),
           P.GridIncompleteError)
    raises("a duplicated cell fails",
           lambda: P.assert_complete_grid(full + [full[5]], 3), P.GridIncompleteError)
    raises("an unretained position fails",
           lambda: P.assert_complete_grid(full + [{"trajectory_index": 0, "position_p": 777}], 3),
           P.GridIncompleteError)
    raises("a trajectory index outside the range fails",
           lambda: P.assert_complete_grid(
               full + [{"trajectory_index": 9, "position_p": pp} for pp in P.RETAINED_POSITIONS],
               3),
           P.GridIncompleteError)
    raises("an entirely absent trajectory fails",
           lambda: P.assert_complete_grid([c for c in full if c["trajectory_index"] != 2], 3),
           P.GridIncompleteError)
    mislabeled = [dict(c) for c in full]
    mislabeled[3]["position_p"] = mislabeled[2]["position_p"]
    raises("a mislabeled position fails as a duplicate-and-gap",
           lambda: P.assert_complete_grid(mislabeled, 3), P.GridIncompleteError)
    check("canonical order is trajectory-major, position-ascending",
          P.grid_order(2)[:11],
          [(0, p) for p in P.RETAINED_POSITIONS] + [(1, 1)])


def test_floor_reporting():
    print("floor: absolute magnitude travels with the ratio")
    fc = K.floor_comparison(6.148e-09, 3.92e-10)
    check("a large ratio over a near-zero floor still reports its absolute value",
          fc["value_nats"] < 1e-8 and fc["ratio_to_floor"] > 10.0, True)
    check("above_replication_floor is reported", fc["above_replication_floor"], True)
    check("excess is absolute, not relative",
          round(fc["excess_over_floor_nats"], 15), round(6.148e-09 - 3.92e-10, 15))
    check("a zero floor yields no ratio rather than an infinity",
          K.floor_comparison(1e-3, 0.0)["ratio_to_floor"], None)
    check("a value at the floor is not above it",
          K.floor_comparison(1e-3, 1e-3)["above_replication_floor"], False)
    check("no materiality verdict is emitted",
          any(k in fc for k in ("material", "significant", "meaningful")), False)


def test_analysis_verification():
    print("analysis: independent re-derivation of stored cells")
    from harness.quality import analyze_kl as A
    prompt, cont = synthetic()
    traj = {"n_trajectories": 2,
            "trajectories": [{"trajectory_index": i, "prompt_index": i,
                              "prompt_token_ids": [x + i for x in prompt],
                              "continuation_token_ids": [x + i for x in cont]}
                             for i in range(2)]}

    def cells_for(traj_rec):
        out = []
        for t in traj_rec["trajectories"]:
            for cell in P.build_all(t["prompt_token_ids"], t["continuation_token_ids"]):
                out.append({"trajectory_index": t["trajectory_index"],
                            "position_p": cell["position_p"],
                            "context_len": cell["context_len"],
                            "target_token_id": cell["target_token_id"],
                            "context_sha256": q.prompt_hash(cell["context_ids"])})
        return out

    good = cells_for(traj)
    check("a faithful grid verifies", A.verify_cells(good, traj), 20)

    shifted = [dict(c) for c in good]
    shifted[4]["target_token_id"] += 1
    raises("a corrupted target is caught", lambda: A.verify_cells(shifted, traj), SystemExit)

    badlen = [dict(c) for c in good]
    badlen[3]["context_len"] += 1
    raises("a corrupted context_len is caught",
           lambda: A.verify_cells(badlen, traj), SystemExit)

    swapped = [dict(c) for c in good]
    swapped[0]["trajectory_index"], swapped[10]["trajectory_index"] = 1, 0
    raises("a cross-trajectory swap is caught", lambda: A.verify_cells(swapped, traj), SystemExit)

    scrambled = [dict(c) for c in good]
    scrambled[1]["position_p"], scrambled[2]["position_p"] = (scrambled[2]["position_p"],
                                                              scrambled[1]["position_p"])
    raises("a position-label scramble is caught",
           lambda: A.verify_cells(scrambled, traj), SystemExit)

    partial = [c for c in good if c["position_p"] != 2048]
    raises("a 9/10 grid is refused rather than averaged",
           lambda: A.verify_cells(partial, traj), SystemExit)

    unknown = [dict(c) for c in good]
    unknown[0]["trajectory_index"] = 7
    raises("a cell naming an unknown trajectory is caught",
           lambda: A.verify_cells(unknown, traj), SystemExit)

    # length and target both still match; only the hash can see this
    interior = [dict(c) for c in good]
    victim = next(c for c in interior if c["position_p"] == 2048)
    t0 = traj["trajectories"][0]
    corrupted = list(t0["prompt_token_ids"]) + list(t0["continuation_token_ids"][:2047])
    corrupted[1000] += 1
    check("an interior flip preserves length and target",
          len(corrupted) == victim["context_len"], True)
    victim["context_sha256"] = q.prompt_hash(corrupted)
    raises("an interior-token corruption is caught by the context hash",
           lambda: A.verify_cells(interior, traj), SystemExit)

    missing_hash = [dict(c) for c in good]
    del missing_hash[0]["context_sha256"]
    raises("a cell with no context hash is refused, not silently skipped",
           lambda: A.verify_cells(missing_hash, traj), SystemExit)


def test_collection_contract():
    print("collection: grid construction and shard planning")
    from harness.quality import collect_kl as C
    prompt, cont = synthetic()
    traj = {"n_trajectories": 2, "trajectory_set_hash": "x", "prompt_subset_hash": "y",
            "trajectories": [{"trajectory_index": i, "prompt_index": i,
                              "prompt_token_ids": [x + i for x in prompt],
                              "continuation_token_ids": [x + i for x in cont]}
                             for i in range(2)]}
    contexts, index = C.build_grid(traj)
    check("grid holds n x 10 contexts", len(contexts), 20)
    check("contexts are emitted in canonical order",
          [(c["trajectory_index"], c["position_p"]) for c in index], P.grid_order(2))
    check("each trajectory's ten contexts ascend in length",
          all(len(contexts[i]) < len(contexts[i + 1]) for i in range(0, 9))
          and all(len(contexts[i]) < len(contexts[i + 1]) for i in range(10, 19)), True)
    check("the trajectory boundary resets the length",
          len(contexts[10]) < len(contexts[9]), True)

    bad = {**traj, "trajectories": [dict(traj["trajectories"][0]), traj["trajectories"][1]]}
    bad["trajectories"][0] = {**bad["trajectories"][0],
                              "continuation_token_ids": cont[:-1]}
    raises("a short continuation is refused before the GPU is touched",
           lambda: C.build_grid(bad), P.PositionContractError)

    plan = C.shard_plan(64, "BF16", root="/tmp/x", shard_trajectories=8)
    check("64 trajectories plan into 8 shards", len(plan), 8)
    check("shards tile the grid with no gap or overlap",
          [(p["start"], p["stop"]) for p in plan],
          [(k * 80, (k + 1) * 80) for k in range(8)])
    check("shard boundaries fall on trajectory boundaries",
          all(p["start"] % 10 == 0 and p["stop"] % 10 == 0 for p in plan), True)
    ragged = C.shard_plan(4, "BF16", root="/tmp/x", shard_trajectories=3)
    check("a ragged tail is still covered", [(p["start"], p["stop"]) for p in ragged],
          [(0, 30), (30, 40)])


def test_resume_provenance():
    print("collection: provenance across a no-launch resume")
    from harness.quality import collect_kl as C
    plan = C.shard_plan(4, "BF16", root="/tmp/x", shard_trajectories=2)
    old_obs = {"engine_identity_hash": "eid_old", "kv_cache_tokens": 44688}
    old_metrics = {"vllm:num_preemptions": 0, "vllm:prefix_cache_hits": 11}
    new_obs = {"engine_identity_hash": "eid_new", "kv_cache_tokens": 44688}
    new_metrics = {"vllm:num_preemptions": 0, "vllm:prefix_cache_hits": 22}
    reused = [{"seconds": 5.5, "observed": old_obs, "engine_metrics": old_metrics},
              {"seconds": 4.5, "observed": old_obs, "engine_metrics": old_metrics}]

    allr = C.resume_provenance(None, reused, plan)
    check("an all-reused collection still records observed", allr["observed"], old_obs)
    check("an all-reused collection still records engine_metrics", allr["engine_metrics"],
          old_metrics)
    check("an all-reused collection sums the shard seconds", allr["seconds"], 10.0)
    check("wall_seconds has no per-shard analogue and stays null", allr["wall_seconds"], None)
    check("all-reused provenance is sourced per field", allr["provenance_source"],
          {"observed": "shards", "engine_metrics": "shards", "engine_metrics_shard": 0,
           "wall_seconds": "unavailable", "seconds": "shards"})

    meta = {"observed": new_obs, "engine_metrics": new_metrics, "wall_seconds": 24.5,
            "generate_seconds": 4.5}
    mixed_final = [reused[0], {"seconds": 4.5, "observed": new_obs,
                               "engine_metrics": new_metrics}]
    mixed = C.resume_provenance(meta, mixed_final, plan)
    check("a mixed run takes the counters from the launch that ran, never merged",
          mixed["engine_metrics"], new_metrics)
    check("a mixed run sums seconds over the whole plan, not just the launch",
          mixed["seconds"], 10.0)
    check("mixed provenance is sourced per field", mixed["provenance_source"],
          {"observed": "launch", "engine_metrics": "launch", "wall_seconds": "launch",
           "seconds": "shards"})

    raises("a no-launch resume with no shard observed block aborts by name",
           lambda: C.resume_provenance(None, [{"seconds": 1.0}, {"seconds": 1.0}], plan),
           SystemExit)
    partial = C.resume_provenance(
        None, [{"seconds": None, "observed": old_obs}, {"seconds": 1.0, "observed": old_obs}],
        plan)
    check("a shard with no seconds leaves the sum null rather than short",
          (partial["seconds"], partial["provenance_source"]["seconds"]), (None, "unavailable"))


PRODUCTION_FLOOR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "results", "quality", "gates",
    "replication_floor_production.json")
G2_FLOOR = os.path.join(os.path.dirname(PRODUCTION_FLOOR), "replication_floor.json")


def _write_json(path, rec):
    import json
    with open(path, "w") as fh:
        json.dump(rec, fh)


def test_floor_loading(tmp):
    print("floor: both schemas load, an unusable one refuses")
    import json
    if not (os.path.exists(PRODUCTION_FLOOR) and os.path.exists(G2_FLOOR)):
        check("floor artifacts are present", False, True)
        return None
    g2, g2d = q.load_floor(G2_FLOOR)
    raw = json.load(open(G2_FLOOR))
    check("the G2 per_config shape passes through unchanged", g2, raw["per_config"])
    check("the G2 shape is labelled", g2d["floor_schema"], "per_config")

    per, desc = q.load_floor(PRODUCTION_FLOOR)
    check("the production floor adapts to one config", sorted(per), ["BF16"])
    check("production headline is the mean over ordered pairs",
          per["BF16"]["headline_nats"], 2.0840602957329884e-04)
    # the whole point of R1: worst_cell.max_nats (7.486e-03) would silently inflate the floor
    check("production worst cell is worst_cell.mean_nats, not worst_cell.max_nats",
          per["BF16"]["max_nats"], 6.425088922453055e-03)
    check("production floor carries its cell count", per["BF16"]["cells"], 640)
    check("the descriptor names the aggregation actually used",
          desc["aggregation"], "mean over ordered pairs, for both the headline and the worst cell")
    check("the descriptor carries the per-pair ranges",
          [desc["headline_range_nats"], desc["worst_cell_range_nats"]],
          [[1.8961296933297275e-04, 2.2853151354409792e-04],
           [4.600694044508615e-03, 7.486238142231847e-03]])
    check("the descriptor records the pair count", desc["floor_pairs"], 6)

    raises("a nonexistent floor path refuses rather than reading back as no floor",
           lambda: q.load_floor(os.path.join(tmp, "absent.json")), SystemExit)
    neither = os.path.join(tmp, "neither.json")
    _write_json(neither, {"kl_spec_hash": q.spec_hash(), "headline_nats": 1e-4})
    raises("a floor of neither shape refuses", lambda: q.load_floor(neither), SystemExit)
    stale = os.path.join(tmp, "stale.json")
    _write_json(stale, {**json.load(open(PRODUCTION_FLOOR)), "kl_spec_hash": "deadbeefdeadbeef"})
    raises("a floor from another KL_SPEC refuses", lambda: q.load_floor(stale), SystemExit)
    foreign = os.path.join(tmp, "foreign.json")
    _write_json(foreign, {**json.load(open(PRODUCTION_FLOOR)), "engine_profile_name": "eager_2048"})
    raises("a floor from another engine profile refuses",
           lambda: q.load_floor(foreign), SystemExit)
    mismatched = os.path.join(tmp, "mismatched.json")
    _write_json(mismatched, {**json.load(open(PRODUCTION_FLOOR)),
                             "trajectory_set_hash": "0" * 64})
    raises("a floor from another trajectory set refuses when the caller names one",
           lambda: q.load_floor(mismatched, "1" * 64), SystemExit)
    return per


def _stub_analysis_root(n, vocab=32, seed=3):
    from harness.quality import collect_kl as C
    prompt, cont = synthetic()
    traj = {"n_trajectories": n,
            "trajectory_set_hash": "synthetic-trajectory-set",
            "prompt_subset_hash": "synthetic-prompt-subset",
            "trajectories": [{"trajectory_index": i, "prompt_index": i,
                              "prompt_token_ids": [x + i for x in prompt],
                              "continuation_token_ids": [x + i for x in cont]}
                             for i in range(n)]}
    _, index = C.build_grid(traj)
    rng = np.random.default_rng(seed)
    mats = {}
    for k, cfg in enumerate(q.LADDER):
        x = rng.normal(0, 1, size=(n * 10, vocab)) + k * 0.01
        mats[cfg] = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    return traj, index, mats


def test_analysis_floor_plumbing(tmp):
    print("analysis: the production floor reaches the summary")
    from harness.quality import analyze_kl as A
    from harness.quality import collect_kl as C, trajectories as T
    import json
    traj, index, mats = _stub_analysis_root(64)
    # the real floor's numbers against a synthetic grid: only its trajectory_set_hash is restated,
    # and only because the stub grid is not the frozen set
    floor_path = os.path.join(tmp, "floor_for_stub.json")
    _write_json(floor_path, {**json.load(open(PRODUCTION_FLOOR)),
                             "trajectory_set_hash": traj["trajectory_set_hash"]})

    def fake_load_matrix(config_id, root=None, n_traj=None):
        return (mats[config_id], [dict(c) for c in index],
                {"provenance": {"subset_n": traj["n_trajectories"],
                                "trajectory_set_hash": traj["trajectory_set_hash"],
                                "kl_spec_hash": q.spec_hash(),
                                "checkpoint_content_hash": "synthetic-checkpoint"},
                 "engine_identity_hash": "synthetic-engine"})

    real_load_matrix, real_load, real_soft = C.load_matrix, T.load, A.common.software_identity
    C.load_matrix, T.load = fake_load_matrix, lambda: traj
    A.common.software_identity = lambda: {"synthetic": True}
    try:
        rec = A.analyze(root=os.path.join(tmp, "root"),
                        out=os.path.join(tmp, "kl_summary.json"),
                        floor_path=floor_path, allow_dirty=True)
        raises("a nonexistent --floor path aborts the analysis",
               lambda: A.analyze(root=os.path.join(tmp, "root"),
                                 out=os.path.join(tmp, "kl_summary_absent.json"),
                                 floor_path=os.path.join(tmp, "absent.json"), allow_dirty=True),
               SystemExit)
    finally:
        C.load_matrix, T.load = real_load_matrix, real_load
        A.common.software_identity = real_soft

    check("the top-level replication_floor is the adapted map, not the raw per_config",
          (rec["replication_floor"] or {}).get("BF16", {}).get("max_nats"),
          6.425088922453055e-03)
    check("the summary names where the floor came from",
          rec["replication_floor_source"]["floor_schema"], q.PRODUCTION_FLOOR_ARTIFACT)
    check("a summary written with a floor says so", rec["replication_floor_omitted"], False)

    vs = rec["pairs"]["BF16||FP8"]["vs_replication_floor"]
    check("a pair with a floor on one side is compared", vs["floor_configs"], ["BF16"])
    check("640 cells against a 640-cell floor enables the worst-cell comparison",
          vs["worst_cell"] is not None and "ratio_to_floor" in vs["worst_cell"], True)
    check("the enabled worst-cell branch leaves no not-comparable note",
          "worst_cell_not_comparable" in vs, False)

    fp = rec["pairs"]["FP8||FP4"]
    # .get, not [...]: an absent key is the regression under test, not a crash
    check("a pair with no floor on either side records an explicit unavailable",
          fp.get("vs_replication_floor", "key absent"), None)
    check("the unavailable record names both sides",
          fp.get("replication_floor_unavailable_for"), ["FP8", "FP4"])


def test_spec_hash():
    print("spec: the pre-registered KL_SPEC hash")
    check("KL_SPEC hash is the pre-registered value", q.spec_hash(), "5565ff73dbe5e36a")


def _kernel_log(pids):
    """25 distinct kernel lines; the pid decides only the raw prefix, never the content.

    The last body sorts after the other 24 and carries a forbidden pattern, so it lands past the
    20-line truncation window.
    """
    bodies = [f"Selected CutlassFP8ScaledMMLinearKernel for layer {i:02d}" for i in range(24)]
    bodies.append("Selected MarlinLinearKernel for layer 24")
    return "\n".join(f"(EngineCore pid={pid}) INFO 08-26 00:27:19 [__init__.py:261] {b}"
                      for pid, b in zip(pids, bodies))


def _pre_change_identity(log_text, resolved, config_id):
    """The truncate-then-normalise order finding 4 replaced. Kept only as a negative control."""
    from harness import common, server
    from harness.quality import qengine as E
    kernel_lines = sorted({ln.strip() for ln in log_text.splitlines()
                           if any(p in ln for p in server.KERNEL_PATTERNS)})[:20]
    normalized = sorted({E._strip_log_prefix(ln) for ln in kernel_lines})
    return common.sha256_of_json({
        "configuration_id": config_id,
        "resolved_config": resolved,
        "kv_cache_tokens": None,
        "graph_capture_observed": False,
        "kernel_lines": normalized,
    })[:16]


def test_observed_identity():
    print("qengine: observed identity survives log-prefix reordering")
    from harness.quality import qengine as E

    one_pid = _kernel_log(["1000"] * 25)
    two_pids = _kernel_log(["2000"] * 13 + ["1000"] * 12)
    resolved, cid = {"dtype": "auto"}, "FP8_PRIMARY"

    a = E.observed_identity(one_pid, resolved, cid)
    b = E.observed_identity(two_pids, resolved, cid)
    check("a pid split leaves engine_identity_hash unchanged",
          a["engine_identity_hash"], b["engine_identity_hash"])
    check("the pre-change order moved the hash on the same pair",
          _pre_change_identity(one_pid, resolved, cid)
          == _pre_change_identity(two_pids, resolved, cid), False)
    check("truncation still caps the normalized lines at 20",
          len(a["normalized_kernel_lines"]), 20)
    check("a forbidden pattern past position 20 is still caught",
          a["dispatch_verdict"]["forbidden_present"], ["Marlin"])
    check("the pre-change blob missed it",
          "Marlin" in " | ".join(sorted({ln.strip() for ln in one_pid.splitlines()})[:20]), False)


def test_freeze_guard(tmp):
    print("preflight: the freeze rejection test cannot generate")
    from harness.quality import preflight as PF, trajectories as T

    existing = os.path.join(tmp, "already_here.json")
    open(existing, "w").write("{}")
    raises("freeze refuses a path that already exists",
           lambda: T.freeze(path=existing), SystemExit)

    real = T.PATH
    T.PATH = os.path.join(tmp, "absent.json")
    try:
        raises("rejection_tests aborts rather than freezing a new production set",
               lambda: PF.rejection_tests(None, None), SystemExit)
    finally:
        T.PATH = real


REAL_MANIFEST_ROOTS = ("smoke", "floor64/launch1", "floor64/launch2", "floor64/launch3",
                       "gates/engine_profile")


def _raises_msg(name, fn, sub):
    try:
        fn()
    except SystemExit as e:
        return check(f"{name} [{sub}]", sub in str(e), True)
    except Exception as e:  # noqa: BLE001
        return check(name, f"raised {type(e).__name__}", "SystemExit")
    return check(name, "no raise", "SystemExit")


def _fingerprint(path):
    return open(path, "rb").read(), os.stat(path).st_mtime_ns


def test_manifest_adoption(tmp):
    print("manifest guard: subset_n adoption is allowlisted, in memory, and arms once present")
    import json
    import shutil
    sandbox = os.path.join(tmp, "mguard")

    def copy_real(rel):
        src = os.path.join(q.QUALITY_DIR, rel, "manifest.json")
        dst = os.path.join(sandbox, rel.replace("/", "_"))
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(src, os.path.join(dst, "manifest.json"))
        return dst

    missing = [r for r in REAL_MANIFEST_ROOTS
               if not os.path.exists(os.path.join(q.QUALITY_DIR, r, "manifest.json"))]
    if missing:
        check(f"the real manifests are present ({missing})", False, True)
        return

    smoke = copy_real("smoke")
    smoke_path = os.path.join(smoke, "manifest.json")
    tsh = json.load(open(smoke_path))["trajectory_set_hash"]
    before = _fingerprint(smoke_path)

    # (a)
    _, rec = q.guard_manifest(smoke, "KL collection",
                              extra={"trajectory_set_hash": tsh, "subset_n": 4},
                              adopt_if_absent={"subset_n"})
    check("an absent allowlisted key is accepted", rec["subset_n"], 4)
    check("the adopted key is named in the returned record",
          rec["manifest_keys_adopted"], ["subset_n"])
    check("adoption does not rewrite the manifest", _fingerprint(smoke_path), before)
    check("adoption is in memory only", "subset_n" in json.load(open(smoke_path)), False)

    # (b)
    pinned = os.path.join(sandbox, "pinned")
    os.makedirs(pinned)
    stored = json.load(open(smoke_path))
    stored["subset_n"] = 64
    _write_json(os.path.join(pinned, "manifest.json"), stored)
    _raises_msg("a manifest carrying a different subset_n aborts",
                lambda: q.guard_manifest(pinned, "KL collection",
                                         extra={"trajectory_set_hash": tsh, "subset_n": 4},
                                         adopt_if_absent={"subset_n"}), "subset_n")

    # (e) then (c)
    fresh = os.path.join(sandbox, "fresh")
    _, frec = q.guard_manifest(fresh, "KL collection",
                               extra={"trajectory_set_hash": tsh, "subset_n": 4},
                               adopt_if_absent={"subset_n"})
    check("a new manifest is written with subset_n",
          json.load(open(os.path.join(fresh, "manifest.json"))).get("subset_n"), 4)
    check("nothing is adopted when the manifest is created",
          frec["manifest_keys_adopted"], [])
    fresh_before = _fingerprint(os.path.join(fresh, "manifest.json"))
    _raises_msg("the pin arms once subset_n is present",
                lambda: q.guard_manifest(fresh, "KL collection",
                                         extra={"trajectory_set_hash": tsh, "subset_n": 64},
                                         adopt_if_absent={"subset_n"}), "subset_n")

    # (d)
    q.guard_manifest(fresh, "KL collection",
                     extra={"trajectory_set_hash": tsh, "subset_n": 4},
                     adopt_if_absent={"subset_n"})
    check("a fully matching manifest is not rewritten",
          _fingerprint(os.path.join(fresh, "manifest.json")), fresh_before)

    # (f), and item 8: every real manifest through with its intended outcome
    for rel in REAL_MANIFEST_ROOTS:
        d = copy_real(rel)
        path = os.path.join(d, "manifest.json")
        fp = _fingerprint(path)
        call = (lambda d=d: q.guard_manifest(d, "KL collection",
                                             extra={"trajectory_set_hash": tsh, "subset_n": 4},
                                             adopt_if_absent={"subset_n"}))
        if "trajectory_set_hash" in json.load(open(path)):
            _, r = call()
            check(f"{rel} is accepted with subset_n adopted", r["manifest_keys_adopted"],
                  ["subset_n"])
        else:
            # the G9 root also predates the current KL_SPEC, and that guard fires first
            _raises_msg(f"{rel} aborts", call, "KL_SPEC changed")
        check(f"{rel} is left untouched", _fingerprint(path), fp)

    # (f) proper: same manifest, spec hash refreshed so the absent-key branch is the one reached
    prof = copy_real("gates/engine_profile")
    prof_path = os.path.join(prof, "manifest.json")
    _write_json(prof_path, {**json.load(open(prof_path)), "kl_spec_hash": q.spec_hash()})
    _raises_msg("an absent NON-allowlisted key still aborts",
                lambda: q.guard_manifest(prof, "KL collection",
                                         extra={"trajectory_set_hash": tsh, "subset_n": 4},
                                         adopt_if_absent={"subset_n"}), "trajectory_set_hash")

    # item 9: gates.py's no-extra call is unchanged by the new parameter
    _, grec = q.guard_manifest(prof, "G9 engine-profile gate")
    check("the no-extra gate call still passes", grec["manifest_keys_adopted"], [])
    check("the no-extra gate call adopts nothing into the record",
          "trajectory_set_hash" in grec, False)


def test_collect_manifest_plumbing(tmp):
    print("collection: the adopted manifest keys reach the collection record")
    from harness.quality import collect_kl as C, qengine as E
    prompt, cont = synthetic()
    traj = {"n_trajectories": 1, "trajectory_set_hash": "tsh", "prompt_subset_hash": "psh",
            "trajectories": [{"trajectory_index": 0, "prompt_index": 0,
                              "prompt_token_ids": prompt, "continuation_token_ids": cont}]}

    def fake_run_job(job, log, **kw):
        for sh in job["shards"]:
            rows = sh["stop"] - sh["start"]
            np.save(sh["npy"], np.zeros((rows, q.VOCAB_SIZE), dtype=q.STORAGE_DTYPE))
            _write_json(sh["json"], {
                "start": sh["start"], "stop": sh["stop"], "rows": rows,
                "provenance": job["provenance"], "seconds": 1.0,
                "per_context": [{"row": i, "full_vocab": True, "all_finite": True,
                                 "normalized": True, "decoded_token_is_none": True}
                                for i in range(rows)]})
        return {"observed": {"engine_identity_hash": "eid_stub"}, "engine_metrics": {},
                "wall_seconds": 1.0}

    def run(root):
        real_job, real_ident = E.run_job, q.config_identity
        E.run_job = fake_run_job
        q.config_identity = lambda cid: {"checkpoint_content_hash": "cch",
                                         "tokenizer_identity": "tok"}
        try:
            return C.collect("BF16_REFERENCE", root=root, allow_dirty=True, traj=traj)
        finally:
            E.run_job, q.config_identity = real_job, real_ident

    adopted_root = os.path.join(tmp, "collect_adopted")
    q.guard_manifest(adopted_root, "KL collection", extra={"trajectory_set_hash": "tsh"})
    check("a collection into a pre-subset_n root records the adoption",
          run(adopted_root)["manifest_keys_adopted"], ["subset_n"])

    fresh_root = os.path.join(tmp, "collect_fresh")
    check("a collection into a fresh root records the empty list, not nothing",
          run(fresh_root)["manifest_keys_adopted"], [])


def main():
    import tempfile
    tests = [test_positions, test_kl_math, test_bootstrap, test_completeness,
             test_floor_reporting, test_analysis_verification, test_collection_contract,
             test_resume_provenance, test_floor_loading, test_analysis_floor_plumbing,
             test_spec_hash, test_observed_identity, test_freeze_guard,
             test_manifest_adoption, test_collect_manifest_plumbing]
    if os.environ.get("QSELFTEST_ONLY"):
        want = os.environ["QSELFTEST_ONLY"]
        tests = [t for t in tests if want in t.__name__]
    needs_tmp = {"test_floor_loading", "test_analysis_floor_plumbing", "test_freeze_guard",
                 "test_manifest_adoption", "test_collect_manifest_plumbing"}
    with tempfile.TemporaryDirectory(prefix="qselftest-") as tmp:
        for t in tests:
            t(tmp) if t.__name__ in needs_tmp else t()
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
