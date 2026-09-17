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

    def fake_load_matrix(config_id, root=None, n_traj=None, evidence_ok=True):
        return (mats[config_id], [dict(c) for c in index],
                {"provenance": {"subset_n": traj["n_trajectories"],
                                "trajectory_set_hash": traj["trajectory_set_hash"],
                                "kl_spec_hash": q.spec_hash(),
                                "checkpoint_content_hash": "synthetic-checkpoint"},
                 "dispatch_evidence": {"ok": evidence_ok},
                 "engine_identity_hash": "synthetic-engine"})

    real_load_matrix, real_load, real_soft = C.load_matrix, T.load, A.common.software_identity
    C.load_matrix, T.load = fake_load_matrix, lambda: traj
    A.common.software_identity = lambda: {"synthetic": True}
    try:
        C.load_matrix = lambda *a, **k: fake_load_matrix(*a, **k, evidence_ok=None)
        _raises_msg("analysis refuses an UNAVAILABLE dispatch verdict",
                    lambda: A.analyze(root=os.path.join(tmp, "root"),
                                      out=os.path.join(tmp, "kl_unavailable.json"),
                                      floor_path=floor_path, allow_dirty=True),
                    "positive dispatch evidence")
        C.load_matrix = lambda *a, **k: fake_load_matrix(*a, **k, evidence_ok=False)
        _raises_msg("analysis refuses a FAILED dispatch verdict",
                    lambda: A.analyze(root=os.path.join(tmp, "root"),
                                      out=os.path.join(tmp, "kl_failed.json"),
                                      floor_path=floor_path, allow_dirty=True),
                    "positive dispatch evidence")
        C.load_matrix = fake_load_matrix
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


def _raises_msg(name, fn, sub, exc=SystemExit):
    try:
        fn()
    except exc as e:
        return check(f"{name} [{sub}]", sub in str(e), True)
    except Exception as e:  # noqa: BLE001
        return check(name, f"raised {type(e).__name__}", exc.__name__)
    return check(name, "no raise", exc.__name__)


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


def _lv_panel(R, T, sd_a, sd_s, sd_e, seed, base=1e-3):
    """A (launch, trajectory, 1) panel with KNOWN variance components."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, sd_a, size=R)[:, None]
    st = rng.normal(0, sd_s, size=T)[None, :]
    e = rng.normal(0, sd_e, size=(R, T))
    return (base + a + st + e)[:, :, None]


def test_launch_variance_math():
    print("launch variance: the decomposition")
    from harness.quality import launch_variance as LV

    # hand oracle: a pure launch effect, no trajectory or residual structure
    Y = np.array([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]])
    a = LV.crossed_anova(Y)
    check("a pure level effect leaves zero residual", round(a["ms_residual"], 15), 0.0)
    check("a pure level effect leaves zero trajectory variance", a["var_trajectory_moment"], 0.0)
    close("MS_level oracle: T*sum((lvl-gm)^2)/(R-1) = 4*(1+1)/1", a["ms_level"], 8.0, 1e-12)
    close("sigma2_level oracle: (MS_A - MS_E)/T = 8/4", a["var_level_moment"], 2.0, 1e-12)
    close("the ANOVA-unbiased Var(theta) is (MS_A+MS_S-MS_E)/(RT)",
          a["var_theta_anova"], (a["ms_level"] + a["ms_trajectory"] - a["ms_residual"]) / 8.0,
          1e-15)

    Y = np.array([[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]])
    a = LV.crossed_anova(Y)
    check("a pure trajectory effect leaves zero level variance", a["var_level_moment"], 0.0)
    close("a pure trajectory effect recovers the trajectory variance",
          a["var_trajectory_moment"], float(np.var([1.0, 3.0, 5.0], ddof=1)), 1e-12)

    print("launch variance: components recovered from a known panel")
    R, T = 6, 40
    ests = []
    for seed in range(60):
        a = LV.crossed_anova(_lv_panel(R, T, 3e-4, 8e-4, 2e-4, seed).mean(axis=2))
        ests.append([a["var_level_moment"], a["var_trajectory_moment"], a["var_residual"]])
    m = np.array(ests).mean(axis=0)
    for name, got, want in (("sigma2_launch", m[0], 3e-4 ** 2),
                            ("sigma2_trajectory", m[1], 8e-4 ** 2),
                            ("sigma2_residual", m[2], 2e-4 ** 2)):
        rel = abs(got - want) / want
        check(f"{name} is recovered unbiased over 60 panels (rel err {rel:.1%})", rel < 0.15, True)

    print("launch variance: Var(theta) = var_launch + var_bootstrap, measured")
    # the orthogonality claim: the spread of theta over independent panels must equal the sum the
    # estimator reports. A double-count or an omission shows up here as a ratio away from 1.
    idx = K.bootstrap_indices(T, 4000, 20260825)
    thetas, reported, launch_only = [], [], []
    for seed in range(120):
        panel = _lv_panel(R, T, 3e-4, 8e-4, 2e-4, 1000 + seed)
        rec, _, _ = LV.decompose(panel, idx, "sim")
        thetas.append(rec["expected_kl_over_launches_nats"])
        reported.append(rec["combined"]["bootstrap_plus_launch"]["se_point_nats"] ** 2)
        launch_only.append(max(0.0, rec["launch_component"]["var_launch_point"]))
    empirical = float(np.var(thetas, ddof=1))
    ratio = float(np.mean(reported) / empirical)
    check(f"reported Var(theta) matches the empirical spread (ratio {ratio:.2f})",
          0.75 < ratio < 1.35, True)
    # the launch term is load-bearing: the bootstrap alone understates the spread
    boot_only = float(np.mean(reported)) - float(np.mean(launch_only))
    check(f"the trajectory bootstrap alone understates Var(theta) "
          f"(ratio {boot_only / empirical:.2f})", boot_only / empirical < 0.85, True)

    print("launch variance: no component is truncated, and the bound is one-sided")
    # under sigma2_A = 0 the moment estimate is negative ~63% of the time; a max(0,.) point
    # estimate would report 0.0 more often than not and read as `no launch effect`
    neg = sum(int(LV.crossed_anova(_lv_panel(3, 64, 0.0, 5e-4, 5e-4, s).mean(axis=2))
                  ["var_level_moment"] < 0) for s in range(200))
    check(f"a null launch effect gives a negative moment estimate in {neg}/200 panels",
          0.5 < neg / 200 < 0.75, True)
    a = LV.crossed_anova(_lv_panel(3, 64, 0.0, 5e-4, 5e-4, 5).mean(axis=2))
    check("the moment component is reported untruncated", "var_level_moment" in a, True)
    check("negativity is flagged rather than clamped away",
          a["moment_components_may_be_negative"], True)
    check("the one-sided upper bound is positive even when the moment estimate is not",
          a["var_level_upper_95"] > 0, True)
    close("upper bound oracle: MS_A*(R-1)/(T*chi2_{0.05,2}) with chi2 = 0.1025865887",
          a["var_level_upper_95"], a["ms_level"] * 2 / (64 * 0.1025865887), 1e-15)
    check("the upper bound always dominates the moment estimate",
          a["var_level_upper_95"] > a["var_level_moment"], True)
    check("chi2 lookup takes the next smaller df (widens the bound)",
          LV.chi2_lower_005(2.9), LV.chi2_lower_005(2))

    print("launch variance: a non-negative quantity never reports a negative bound")
    ci = LV._nonneg_ci(1e-4, 5e-4, 4.30265272991)
    check("a lower end below zero is clamped", ci["ci_95"][0], 0.0)
    check("the raw value is kept beside it", ci["ci_95_raw"][0] < 0, True)
    check("the clamp is flagged", ci["ci_95_lower_clamped"], True)
    check("an interval that stays positive is not clamped",
          LV._nonneg_ci(1e-3, 1e-5, 2.0)["ci_95_lower_clamped"], False)

    print("launch variance: degrees of freedom and critical values")
    check("t crit at 2 df is the Student value, not 1.96",
          round(LV.t_crit_975(2), 6), 4.302653)
    check("a fractional df takes the next SMALLER tabulated df (conservative)",
          LV.t_crit_975(2.9), LV.t_crit_975(2))
    check("t crit is monotone decreasing in df", LV.t_crit_975(3) < LV.t_crit_975(2), True)
    check("large df reaches the normal limit", round(LV.t_crit_975(500), 3), 1.96)
    check("df below 1 has no critical value", LV.t_crit_975(0.5), None)
    close("Satterthwaite of one component returns that component's df",
          LV._satterthwaite(4.0, 7, 0.0, 3), 7.0, 1e-12)
    check("Satterthwaite of two zero components is undefined",
          LV._satterthwaite(0.0, 7, 0.0, 3), None)

    print("launch variance: the estimator is a strict generalisation")
    panel = _lv_panel(1, 8, 0.0, 8e-4, 2e-4, 9)
    rec, _, _ = LV.decompose(panel, K.bootstrap_indices(8, 2000, 20260825), "R=1")
    close("R=1 reproduces the pre-registered headline exactly",
          rec["expected_kl_over_launches_nats"], K.headline(panel[0]), 1e-18)
    check("R=1 reports the launch component as absent, not as zero",
          rec["launch_component"]["se_launch_nats_point"], None)
    check("R=1 says the launch variance is not included",
          rec["combined"]["launch_variance_included"], False)
    check("R=1 says the uncertainty decomposition does not exist, not that it is zero",
          "does not exist" in rec["launch_component"]["not_estimable_because"], True)
    close("R=1 total SE is exactly the bootstrap SE",
          rec["combined"]["bootstrap_plus_launch"]["se_point_nats"],
          rec["trajectory_component"]["se_trajectory_nats"], 1e-18)
    raises("a decomposition of one level refuses",
           lambda: LV.crossed_anova(np.array([[1.0, 2.0, 3.0]])), LV.LaunchDesignError)
    raises("a decomposition of one trajectory refuses",
           lambda: LV.crossed_anova(np.array([[1.0], [2.0]])), LV.LaunchDesignError)
    raises("a non-3D cell stack refuses",
           lambda: LV.decompose(np.zeros((3, 4)), idx, "bad"), LV.LaunchDesignError)

    print("launch variance: no floor is subtracted from the estimate")
    panel = _lv_panel(4, 8, 3e-4, 8e-4, 2e-4, 21)
    i8 = K.bootstrap_indices(8, 2000, 20260825)
    plain, _, _ = LV.decompose(panel, i8, "plain")
    withfloor, _, _ = LV.decompose(panel, i8, "withfloor", floor_nats=5e-4)
    check("passing a floor does not move the point estimate",
          plain["expected_kl_over_launches_nats"], withfloor["expected_kl_over_launches_nats"])
    check("passing a floor does not move the total SE",
          plain["combined"]["bootstrap_plus_launch"]["se_point_nats"],
          withfloor["combined"]["bootstrap_plus_launch"]["se_point_nats"])
    check("the record states the floor was not subtracted", withfloor["floor_subtracted"], False)
    close("the estimate is the mean of the per-launch headlines",
          plain["expected_kl_over_launches_nats"],
          float(np.mean([K.headline(panel[r]) for r in range(4)])), 1e-15)


def test_launch_variance_estimators(tmp):
    print("launch variance: the ANOVA-unbiased total and its df")
    from harness.quality import launch_variance as LV
    R, T = 6, 40
    idx = K.bootstrap_indices(T, 4000, 20260825)
    panel = _lv_panel(R, T, 3e-4, 8e-4, 2e-4, 77)
    a = LV.crossed_anova(panel.mean(axis=2))
    close("Var(theta) collapses to (MS_A+MS_S-MS_E)/(RT)", a["var_theta_anova"],
          (a["ms_level"] + a["ms_trajectory"] - a["ms_residual"]) / (R * T), 1e-18)
    close("and equals sA/R + sS/T + sE/(RT) from the moment components",
          a["var_theta_anova"],
          a["var_level_moment"] / R + a["var_trajectory_moment"] / T + a["var_residual"] / (R * T),
          1e-18)
    check("the Satterthwaite df is finite and no larger than the total observations",
          0 < a["df_theta_satterthwaite"] <= R * T, True)
    check("the expected mean squares are stated in the record",
          "sigma2_{AxS}" in a["expected_mean_squares"]["E[MS_residual]"], True)

    print("launch variance: the interval is scaled, not rebuilt symmetrically")
    rec, _, _ = LV.decompose(panel, idx, "scale")
    bp = rec["combined"]["bootstrap_plus_launch"]
    b = rec["trajectory_component"]["bootstrap"]
    theta = rec["expected_kl_over_launches_nats"]
    lo_asym = (theta - b["ci_low"]) / (b["ci_high"] - theta)
    lo_asym2 = (theta - bp["ci_point"][0]) / (bp["ci_point"][1] - theta)
    close("scaling preserves the percentile interval's asymmetry", lo_asym2, lo_asym, 1e-9)
    check("the scaled interval is wider than the bootstrap interval",
          bp["ci_point"][1] - bp["ci_point"][0] > b["ci_high"] - b["ci_low"], True)
    check("the 95%-upper-bound interval is wider still",
          bp["ci_upper_95"][1] - bp["ci_upper_95"][0] > bp["ci_point"][1] - bp["ci_point"][0], True)
    check("no comparability with the committed single-launch bootstrap is claimed",
          "NOT the same interval" in
          rec["trajectory_component"]["different_estimand_from_committed_artifacts"], True)

    print("launch variance: the first-order scaling law")
    # SD_launch = sigma_proj * sqrt(2*signal): a signal 100x larger must move the SD 10x, not 100x
    sl_small = LV.scaling_law(1e-3, 2.0e-3 * np.sqrt(2e-3))
    sl_big = LV.scaling_law(1e-1, 2.0e-3 * np.sqrt(2e-1))
    close("sigma_proj is invariant to the signal it was measured at",
          sl_big["sigma_proj"], sl_small["sigma_proj"], 1e-12)
    close("a 100x signal predicts a 10x launch SD, not a 100x one",
          sl_big["sd_launch_headlines_nats"] / sl_small["sd_launch_headlines_nats"], 10.0, 1e-9)
    check("the law records that its input is not sigma_A",
          "NOT sigma_A" in sl_small["caution"], True)
    check("a zero signal yields no law", LV.scaling_law(0.0, 1e-4), None)

    print("launch variance: per position reports no variance components")
    bypos = LV.by_position(panel[:, :, :1].repeat(10, axis=2), K.bootstrap_indices(T, 500, 1),
                           floor_by_position={str(p): 1e-4 for p in P.RETAINED_POSITIONS},
                           sigma_proj=2.0e-3)
    check("every retained position is present", sorted(bypos, key=int),
          [str(p) for p in P.RETAINED_POSITIONS])
    one = bypos["512"]
    check("no per-position variance components are reported", one["variance_components"], None)
    check("the omission carries its reason",
          "not estimable" in one["variance_components_omitted_because"], True)
    check("the per-launch spread is labelled as not sigma_A",
          "not sigma_A" in one["launch_spread_nats"]["sd_is_not_sigma_A"], True)
    close("the predicted launch SD follows the pooled law",
          one["scaling_law_predicted_launch_sd_nats"],
          2.0e-3 * np.sqrt(2 * one["expected_kl_over_launches_nats"]), 1e-15)
    check("the per-position floor is the production one, and says so",
          "production" in one["floor_source"], True)

    print("launch variance: the direction it moves a failed bound must be establishable")
    import json as _json
    smoke_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "results", "quality", "smoke")
    got = LV._check_committed("BF16||FP8", None, smoke_dir)
    check("the committed headline is read even when its launch is not in the set",
          got["committed_headline_nats"], 3.6897120485158315e-03)
    check("and the record says it was not recomputed", got["recomputed_bit_identical"], None)
    check("a recomputation that matches is marked bit-identical",
          LV._check_committed("BF16||FP8", 3.6897120485158315e-03,
                              smoke_dir)["recomputed_bit_identical"], True)
    # the committed artifacts were produced under a different interpreter; np.log/np.exp move in
    # their last bits, so bit-equality is the wrong bar and a tolerance is the contract
    drift = 3.6897120485158315e-03 * (1 + 3e-14)
    got = LV._check_committed("BF16||FP8", drift, smoke_dir)
    check("interpreter-scale drift passes but is not called bit-identical",
          got["recomputed_bit_identical"], False)
    check("and the drift is quantified rather than hidden", got["relative_difference"] < 1e-13, True)
    check("with its cause named", "last bits" in got["why_not_bit_identical"], True)
    _raises_msg("a recomputation beyond interpreter drift aborts",
                lambda: LV._check_committed("BF16||FP8", 3.7e-03, smoke_dir),
                "larger than interpreter drift", LV.LaunchDesignError)
    _raises_msg("a drift just above the bar aborts",
                lambda: LV._check_committed(
                    "BF16||FP8", 3.6897120485158315e-03 * (1 + 2e-11), smoke_dir),
                "above the", LV.LaunchDesignError)
    _raises_msg("an absent committed artifact aborts rather than skipping the direction",
                lambda: LV._check_committed("BF16||FP8", None, os.path.join(tmp, "nope")),
                "could not be stated", LV.LaunchDesignError)

    print("launch variance: the floor's launch-level jackknife")
    # three launches, one of them displaced: the pair spread understates the launch spread
    per_pair = {"L1||L2": 1.0e-4, "L2||L1": 1.0e-4, "L1||L3": 3.0e-4,
                "L3||L1": 3.0e-4, "L2||L3": 3.0e-4, "L3||L2": 3.0e-4}
    jk = LV.floor_launch_ci(per_pair, 3)
    close("the point is the mean over ordered pairs", jk["point_nats"],
          float(np.mean(list(per_pair.values()))), 1e-15)
    check("the jackknife carries R-1 df, not R(R-1)-1", jk["df"], 2)
    check("the jackknife SE exceeds the naive over-pairs SE",
          jk["se_launch_level_nats"] > jk["naive_se_over_ordered_pairs_nats"], True)
    raises("a floor whose pairs name a different launch count aborts",
           lambda: LV.floor_launch_ci(per_pair, 4), LV.LaunchDesignError)


def test_partial_matrix_load():
    print("collect_kl: n_traj reads a prefix and loads only the shards covering it")
    import numpy as np
    from harness.quality import collect_kl as C
    from harness import common as _c
    root = os.path.join(_c.REPO, "results", "quality", "floor64", "launch1")
    if not os.path.exists(os.path.join(root, "collection_BF16.json")):
        check("floor64 launch1 is present", False, True)
        return
    full, cells_full, summary = C.load_matrix("BF16_REFERENCE", root=root)
    check("the full grid still loads", full.shape[0], summary["n_trajectories"] * 10)
    for n in (4, 8, 16):
        mat, cells, _ = C.load_matrix("BF16_REFERENCE", root=root, n_traj=n)
        check(f"n_traj={n} returns exactly {n * 10} rows", mat.shape[0], n * 10)
        check(f"n_traj={n} is bit-identical to the full load's prefix",
              bool(np.array_equal(mat, full[:n * 10])), True)
        check(f"n_traj={n} returns the matching cells", cells, cells_full[:n * 10])
    # the regression this guards: n_traj below the collected count used to load every shard and
    # then assert the grid for n, raising on the trajectories it had just read
    raises("n_traj above the collected count still aborts",
           lambda: C.load_matrix("BF16_REFERENCE", root=root, n_traj=65), SystemExit)


def test_launch_variance_guards(tmp):
    print("launch variance: exchangeability preconditions must abort")
    from harness.quality import launch_variance as LV

    def meta(n=4, **over):
        base = []
        for i, name in enumerate(("A", "B", "C")):
            rec = {"launch": name, "root": f"root{i}", "n_trajectories": 64,
                   "cells": [{"trajectory_index": 0, "position_p": 1}],
                   "summary": {"config_id": "BF16_REFERENCE", "launched": True,
                               "timestamp": f"2026-08-26T00:0{i}:00-0400",
                               "git": {"git_head": f"{i}" * 40, "git_dirty": False},
                               "engine_identity_hash": "eng", "n_trajectories": 64,
                               "engine_metrics": {"vllm:prefix_cache_queries": 100,
                                                  "vllm:prefix_cache_hits": 75},
                               "provenance": {"kl_spec_hash": "spec", "trajectory_set_hash": "tsh",
                                              "contexts_hash": "ctx" + str(i) + "0" * 16,
                                              "subset_n": 64,
                                              "checkpoint_content_hash": "ckpt",
                                              "tokenizer_identity": {"tokenizer.json": "tok"},
                                              "engine_profile_name": "graph_2048",
                                              "storage_dtype": "float32",
                                              "retained_positions": list(P.RETAINED_POSITIONS)}}}
            base.append(rec)
        for k, v in over.items():
            target, field = k.split(":", 1)
            m = next(r for r in base if r["launch"] == target)
            if field in m["summary"]["provenance"]:
                m["summary"]["provenance"][field] = v
            else:
                m["summary"][field] = v
        return base

    ok = LV.check_exchangeability(meta(), 4)
    check("a clean set of launches passes", ok["n_launches"], 3)
    check("the precondition record names what it verified",
          "trajectory_set_hash" in ok["identities_verified"], True)
    check("the precondition record carries the prefix-cache hit rate",
          ok["prefix_cache"]["A"]["hit_rate"], 0.75)
    check("a set of launches with no matrices claims no content distinctness",
          ok["content_distinct"], False)
    check("the record names the keys that cannot match by construction",
          sorted(ok["keys_that_cannot_match_by_construction"]),
          ["contexts_hash", "subset_n", "substitute"])
    check("the record says engine_identity_hash is not a launch nonce",
          "cannot distinguish" in ok["engine_identity_hash_is_not_a_launch_nonce"], True)
    sc = ok["session_confound"]
    check("the session confound is decided from the timestamps", sc["decided_from_timestamps"], True)
    # the stub stamps are one minute apart, so this run is one sitting and the caveat must say so
    check("one-sitting launches are reported as a within-session LOWER bound",
          sc["one_sitting"] and "lower bound" in sc["note"], True)
    from harness.quality import launch_variance as _L
    spread = _L._session_confound({"A": "2026-08-26T00:00:00-0400",
                                   "B": "2026-08-27T00:00:00-0400"})
    check("launches spanning sittings are reported as confounded with session",
          (not spread["one_sitting"]) and "between-session" in spread["note"], True)
    check("unparseable stamps fall back to the conservative reading",
          _L._session_confound({"A": "not-a-time"})["decided_from_timestamps"], False)

    print("launch variance: byte-identical launches are the same run twice")
    same = np.zeros((4, 3))
    raises("two launches with byte-identical matrices abort",
           lambda: LV.check_exchangeability(
               meta(), 4, mats={"A": same, "B": same, "C": same + 1.0}),
           LV.LaunchDesignError)
    distinct = LV.check_exchangeability(
        meta(), 4, mats={"A": same, "B": same + 1.0, "C": same + 2.0})
    check("distinct matrices pass and record their digests",
          [distinct["content_distinct"], len(set(distinct["matrix_sha256"].values()))], [True, 3])

    for field, bad in (("kl_spec_hash", "other-spec"),
                       ("trajectory_set_hash", "other-set"),
                       ("checkpoint_content_hash", "other-ckpt"),
                       ("engine_profile_name", "eager_2048"),
                       ("storage_dtype", "float16"),
                       ("engine_identity_hash", "other-eng")):
        raises(f"a launch differing on {field} aborts",
               lambda f=field, b=bad: LV.check_exchangeability(meta(**{f"C:{f}": b}), 4),
               LV.LaunchDesignError)
    raises("a launch that reused its shards rather than rescoring aborts",
           lambda: LV.check_exchangeability(meta(**{"C:launched": False}), 4),
           LV.LaunchDesignError)
    raises("two launches sharing a collection timestamp abort",
           lambda: LV.check_exchangeability(
               meta(**{"C:timestamp": "2026-08-26T00:00:00-0400"}), 4),
           LV.LaunchDesignError)
    bad_cells = meta()
    bad_cells[2]["cells"] = [{"trajectory_index": 1, "position_p": 1}]
    raises("a launch scored on different cells aborts",
           lambda: LV.check_exchangeability(bad_cells, 4), LV.LaunchDesignError)

    print("launch variance: the grid-regime check refuses to pool when it should")
    rng = np.random.default_rng(4)
    vocab = 24
    def norm(x):
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    base = rng.normal(0, 1, size=(40, vocab))
    mats = {"F1": norm(base + rng.normal(0, 1e-3, base.shape)),
            "F2": norm(base + rng.normal(0, 1e-3, base.shape)),
            "F3": norm(base + rng.normal(0, 1e-3, base.shape))}
    mats["S"] = norm(base + rng.normal(0, 1e-3, base.shape))
    # the criterion must ACCEPT under its own null: an exchangeable smoke launch, over many draws
    pooled = 0
    for seed in range(40):
        r2 = np.random.default_rng(100 + seed)
        b2 = r2.normal(0, 1, size=(40, vocab))
        m2 = {k: norm(b2 + r2.normal(0, 1e-3, b2.shape)) for k in ("F1", "F2", "F3", "S")}
        pooled += int(not LV.regime_check(m2, ["F1", "F2", "F3", "S"], 4)["gross_regime_effect"])
    check(f"an exchangeable smoke launch pools in {pooled}/40 draws (the null must not be rejected)",
          pooled >= 38, True)
    mats["S"] = norm(base + rng.normal(0, 5e-2, base.shape))
    bad = LV.regime_check(mats, ["F1", "F2", "F3", "S"], 4)
    check("a smoke launch far outside the within-regime spread does NOT pool",
          bad["gross_regime_effect"], True)
    check("the refusing verdict says not to pool", "DO NOT pool" in bad["verdict"], True)
    check("the odd launch ranks first on leverage when it is genuinely odd",
          bad["leverage_rank_of_odd_launch"], 1)
    check("the record states the rank test cannot reach 0.05 at four launches",
          bad["rank_p_value_floor"], 0.25)
    check("no regime check without a smoke launch",
          LV.regime_check(mats, ["F1", "F2", "F3"], 4), None)

    print("launch variance: BF16->BF16 reports no launch variance component")
    phi = LV.bf16_to_bf16(mats, ["F1", "F2", "F3"], 4,
                          K.bootstrap_indices(4, 500, 20260825))
    check("all ordered pairs are reported", phi["ordered_pairs"], 6)
    check("no variance component is claimed over dependent pairs",
          "no_variance_component_reported_because" in phi, True)
    check("each launch's leverage is reported", sorted(phi["launch_leverage_nats"]),
          ["F1", "F2", "F3"])
    check("G2' is named as unchanged", "unchanged" in phi["relationship_to_G2_prime"], True)




def test_run_scoped_clean_tree():
    """The multi-launch clean-tree defect: own outputs excused, everything else still fatal."""
    print("run-scoped clean tree")
    check("a path under an own-output root is matched",
          q._under("results/quality/kl/collection_BF16.json", ["results/quality/kl"]), True)
    check("the root itself is matched", q._under("results/quality/kl", ["results/quality/kl"]), True)
    # the bug a bare startswith() would introduce: a sibling directory sharing a prefix
    check("a prefix-sharing sibling is NOT matched",
          q._under("results/quality/kl_other/x.json", ["results/quality/kl"]), False)
    check("an unrelated path is not matched",
          q._under("scripts/harness/quality/collect_kl.py", ["results/quality/kl"]), False)

    real = q.dirty_paths
    try:
        q.dirty_paths = lambda: ["results/quality/kl/collection_BF16.json"]
        st = q.require_clean_tree(False, "t", own_outputs=["results/quality/kl"])
        check("own output does not abort", st["dirty_paths_outside_run_outputs"], [])
        check("the dirt is still recorded", st["dirty_paths"],
              ["results/quality/kl/collection_BF16.json"])
        check("git_dirty still reports the raw fact", st["git_dirty"], True)
        q.dirty_paths = lambda: ["results/quality/kl/collection_BF16.json",
                                 "scripts/harness/quality/collect_kl.py"]
        _raises_msg("a source edit still aborts under a scope",
                    lambda: q.require_clean_tree(False, "t", own_outputs=["results/quality/kl"]),
                    "scripts/harness")
        q.dirty_paths = lambda: []
        _raises_msg("a moved HEAD aborts even on a clean tree",
                    lambda: q.require_clean_tree(False, "t", head="0" * 40), "HEAD moved")
    finally:
        q.dirty_paths = real

    # the guard must fire on the real defect: launch 1 writes a TRACKED summary, launch 2 checks
    try:
        q.dirty_paths = lambda: ["results/quality/floor64/launch1/collection_BF16.json"]
        _raises_msg("unscoped, the launch-1 artifact aborts launch 2",
                    lambda: q.require_clean_tree(False, "collect_kl:launch2"), "dirty")
        st = q.require_clean_tree(False, "collect_kl:launch2",
                                  own_outputs=["results/quality/floor64"])
        check("scoped, launch 2 proceeds", st["run_scoped"], True)
    finally:
        q.dirty_paths = real


def test_dispatch_evidence():
    """Positive evidence, including the two cases the inherited verdict cannot see."""
    print("dispatch evidence (additive verifier)")
    from harness.quality import dispatch_verify as DV

    bf16 = ("INFO 09-16 00:00:00 [core.py:105] Initializing a V1 LLM engine (v0.19.1) with "
            "config: model='x', dtype=torch.bfloat16, quantization=None, seed=0\n"
            "INFO 09-16 00:00:01 [cuda.py:334] Using FLASH_ATTN attention backend\n"
            "INFO 09-16 00:00:02 [backends.py:1111] Dynamo bytecode transform time: 1.05 s\n")
    r = DV.verify(bf16, "BF16_REFERENCE")
    check("BF16 gets positive evidence, not silence", r["ok"], True)
    check("BF16 evidence names the resolved quantization method",
          r["required_evidence"]["quantization_method_is_none"]["present"], True)
    check("BF16 evidence carries the matched line",
          bool(r["required_evidence"]["attention_backend_chosen"]["evidence"]), True)

    # the gap this verifier exists to close: an empty log satisfies the INHERITED verdict
    from harness.quality import qengine as E
    inherited = E.observed_identity("", {}, "BF16_REFERENCE")["dispatch_verdict"]
    check("the inherited BF16 verdict is ok on an EMPTY log", inherited["ok"], True)
    check("the additive verifier is not", DV.verify("", "BF16_REFERENCE")["ok"], False)

    fp4 = ("INFO [core.py:105] Initializing a V1 LLM engine with config: quantization="
           "compressed-tensors, dtype=torch.bfloat16\n"
           "INFO [cuda.py:334] Using FLASH_ATTN attention backend\n"
           "INFO [nvfp4.py:1] Using NvFp4LinearBackend.FLASHINFER_CUTLASS for NVFP4 GEMM\n")
    check("FP4 passes on a faithful log", DV.verify(fp4, "FP4_PRIMARY")["ok"], True)

    # "emulation" is in FP4's forbidden list but not in server.KERNEL_PATTERNS, so the inherited
    # scan -- which searches only KERNEL_PATTERNS-matching lines -- can never see it
    emu = fp4 + "INFO [nvfp4.py:9] falling back to emulation for NVFP4 GEMM\n"
    check("the inherited verdict misses the emulation fallback",
          E.observed_identity(emu, {}, "FP4_PRIMARY")["dispatch_verdict"]["forbidden_present"], [])
    r4 = DV.verify(emu, "FP4_PRIMARY")
    check("the additive verifier catches it", r4["ok"], False)
    check("and names the pattern", "emulation" in r4["forbidden_present"], True)
    raises("require() aborts on failing evidence",
           lambda: DV.require(emu, "FP4_PRIMARY"), SystemExit)

    fp8 = ("INFO [core.py:105] config: quantization=compressed-tensors\n"
           "INFO [cuda.py:334] Using FLASH_ATTN attention backend\n"
           "INFO [fp8.py:1] Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8\n")
    check("FP8 passes on a faithful log", DV.verify(fp8, "FP8_PRIMARY")["ok"], True)
    check("FP8 fails when the Marlin fallback appears",
          DV.verify(fp8 + "using MarlinLinearKernel\n", "FP8_PRIMARY")["ok"], False)
    check("BF16 fails if a quantization kernel appears",
          DV.verify(bf16 + "Selected CutlassFP8ScaledMMLinearKernel\n",
                    "BF16_REFERENCE")["ok"], False)


def test_p13_resolution_rule():
    """The registered resolution rule, and the arithmetic it rests on."""
    print("P13 resolution rule")
    from harness.quality import launch_variance as L
    from harness.quality import p13

    ci = L.sd_ci_chi2(1.0, 2)
    check("the sd interval at 2 df spans ~12x", round(ci["width_ratio"], 1), 12.1)
    check("an unlisted df returns None", L.sd_ci_chi2(1.0, 99), None)

    pooled = L.pooled_sigma_proj({"a": {"scaling_law": {"sigma_proj": 3e-3}},
                                  "b": {"scaling_law": {"sigma_proj": 4e-3}}})
    close("pooled sigma_proj is the rms", pooled["sigma_proj_pooled"], 3.5355339e-3, 1e-9)
    check("a comparison with no scaling law is dropped, not zeroed",
          L.pooled_sigma_proj({"a": {}}), None)

    boot = {"ci_low": 1.0e-3, "ci_high": 3.0e-3, "std_error": 5.0e-4}
    # signal far above the floor: resolved, and the inflation must not flip it
    r = p13.classify(2.0e-3, boot, 2.1e-3, 2.0e-4, 3.0e-4)
    check("a signal clear of the floor resolves", r["class"], "resolved")
    check("inflation is >= 1", r["inflation_factor"] >= 1.0, True)
    check("the inflated lower end is below the raw one",
          r["launch_inflated_ci_low_nats"] < boot["ci_low"], True)

    # same signal, floor raised above the inflated lower end: noise-limited
    r2 = p13.classify(2.0e-3, boot, 2.1e-3, 9.0e-4, 1.5e-3)
    check("a floor above the inflated lower end is noise-limited", r2["class"], "noise_limited")
    check("the margin goes negative there", r2["margin_nats"] < 0, True)

    # the rule must be MONOTONE in launch variance: more nuisance can only lose resolution
    wide = p13.classify(2.0e-3, boot, 2.0e-2, 2.0e-4, 3.0e-4)
    check("a much larger sigma_proj cannot gain resolution",
          not (wide["class"] == "resolved" and r["class"] == "noise_limited"), True)
    check("and it lowers the inflated bound",
          wide["launch_inflated_ci_low_nats"] < r["launch_inflated_ci_low_nats"], True)

    # zero launch variance reduces to the plain pre-registered bootstrap interval
    z = p13.classify(2.0e-3, boot, 0.0, 2.0e-4, 3.0e-4)
    close("sigma_proj=0 leaves the bootstrap interval untouched",
          z["launch_inflated_ci_low_nats"], boot["ci_low"], 1e-15)

    check("the registration hashes deterministically",
          p13.registration_hash() == p13.registration_hash(), True)
    check("the registered design is R=3", p13.REGISTERED["design"]["n_bf16_launches"], 3)
    check("launch 1 is the designated locked reference",
          p13.REGISTERED["design"]["designated_reference_launch_for_locked_headline"], 1)
    check("no floor is subtracted anywhere in the registration",
          p13.REGISTERED["what_is_not_changed"]["floor_subtracted_from_any_reported_kl"], False)
    check("no averaged BF16 distribution is registered",
          p13.REGISTERED["what_is_not_changed"]["averaged_or_pooled_bf16_distribution"], False)
    check("the locked headline definition is unchanged",
          p13.REGISTERED["what_is_not_changed"]["locked_headline_definition"], False)
    check("three BF16 launch roots are declared", len(p13.BF16_LAUNCH_ROOTS), 3)
    check("launch 1 is the production root itself",
          p13.BF16_LAUNCH_ROOTS[1], q.KL_DIR)
    check("the launch sources are distinct roots",
          len({s["root"] for s in p13.launch_sources()}), 3)

    # a trajectory whose BF16 reference reproduces bit-identically has floor exactly 0.0, so its
    # ratio is +inf. Only production scale surfaces this: all four smoke trajectories were non-zero.
    theta_rec = {"expected_kl_over_launches_nats": 5.0e-3,
                 "trajectory_component": {"bootstrap": boot},
                 "launch_component": {"se_launch_nats_upper_95": 2.0e-4},
                 "combined": {"bootstrap_plus_launch": {"ci_point": [1e-3, 3e-3],
                                                        "ci_upper_95": [9e-4, 3.1e-3]}}}
    idx = K.bootstrap_indices(4, 100, 1)
    qv = np.array([5e-3, 5e-3, 5e-3, 5e-3])
    phi_zero = np.array([1e-4, 0.0, 1e-4, 1e-4])
    r = L.resolvability(theta_rec, 2e-4, [1e-4, 3e-4], qv, phi_zero, idx)["per_trajectory_ratio"]
    check("a zero floor is counted, not hidden", r["trajectories_with_zero_floor"], 1)
    check("the mean of ratios is reported as undefined", r["mean"], None)
    check("and says why", bool(r["mean_undefined_because"]), True)
    check("the bootstrap interval is withheld too", r["bootstrap_ci"], None)
    check("the median stays finite over ALL trajectories", np.isfinite(r["median"]), True)
    check("the ratio of means is defined", round(r["ratio_of_means"], 4), 66.6667)

    phi_ok = np.array([1e-4, 2e-4, 1e-4, 1e-4])
    r2 = L.resolvability(theta_rec, 2e-4, [1e-4, 3e-4], qv, phi_ok, idx)["per_trajectory_ratio"]
    check("with no zero floor the mean is reported", r2["mean"] is not None, True)
    check("and the interval comes back", r2["bootstrap_ci"] is not None, True)


def main():
    import tempfile
    tests = [test_positions, test_kl_math, test_bootstrap, test_completeness,
             test_floor_reporting, test_analysis_verification, test_collection_contract,
             test_resume_provenance, test_floor_loading, test_analysis_floor_plumbing,
             test_spec_hash, test_observed_identity, test_freeze_guard,
             test_manifest_adoption, test_collect_manifest_plumbing,
             test_launch_variance_math, test_launch_variance_estimators,
             test_launch_variance_guards, test_partial_matrix_load,
             test_run_scoped_clean_tree, test_dispatch_evidence, test_p13_resolution_rule]
    if os.environ.get("QSELFTEST_ONLY"):
        want = os.environ["QSELFTEST_ONLY"]
        tests = [t for t in tests if want in t.__name__]
    needs_tmp = {"test_floor_loading", "test_analysis_floor_plumbing", "test_freeze_guard",
                 "test_manifest_adoption", "test_collect_manifest_plumbing",
                 "test_launch_variance_guards", "test_launch_variance_estimators"}
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
