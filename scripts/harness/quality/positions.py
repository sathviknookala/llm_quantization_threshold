"""Retained-position contract for teacher-forced KL. Pure; no engine, no GPU.

context(p) = prompt[0:512] + continuation[0:p-1]   length 511 + p
target(p)  = continuation[p-1]

`EVALUATION_RIG.md` A.1 owns this. The earlier `prompt_token_ids[:p]` wording was wrong under both
readings and silently so, which is why every constructed context is re-derivable and asserted.
"""

RETAINED_POSITIONS = (1, 8, 32, 64, 128, 256, 512, 1024, 1536, 2048)
PROMPT_TOKENS = 512
CONTINUATION_TOKENS = 2048


class PositionContractError(ValueError):
    pass


def context_len(p):
    _check_position(p)
    return PROMPT_TOKENS - 1 + p


def max_context_len():
    return context_len(max(RETAINED_POSITIONS))


def _check_position(p):
    if p not in RETAINED_POSITIONS:
        raise PositionContractError(f"position {p!r} is not a retained position")


def _check_inputs(prompt_ids, cont_ids):
    if len(prompt_ids) != PROMPT_TOKENS:
        raise PositionContractError(
            f"prompt has {len(prompt_ids)} tokens, contract requires {PROMPT_TOKENS}")
    if len(cont_ids) != CONTINUATION_TOKENS:
        raise PositionContractError(
            f"continuation has {len(cont_ids)} tokens, contract requires {CONTINUATION_TOKENS}")


def build_context(prompt_ids, cont_ids, p):
    """Returns (context_token_ids, target_token_id). The target is never part of the context."""
    _check_position(p)
    _check_inputs(prompt_ids, cont_ids)
    context = list(prompt_ids) + list(cont_ids[:p - 1])
    target = cont_ids[p - 1]
    if len(context) != context_len(p):
        raise PositionContractError(
            f"position {p}: built context of {len(context)}, contract says {context_len(p)}")
    return context, target


def build_all(prompt_ids, cont_ids):
    cells = []
    for p in RETAINED_POSITIONS:
        context, target = build_context(prompt_ids, cont_ids, p)
        cells.append({"position_p": p, "context_ids": context, "target_token_id": target,
                      "context_len": len(context)})
    assert_nesting(cells)
    return cells


def assert_nesting(cells):
    """Contexts within a trajectory are strictly nested; a shift bug breaks this deterministically.

    Not sufficient for validating STORED records: corruption confined past the second-longest
    context has no longer partner to expose it. `rederive_and_check` is the check for those.
    """
    ordered = sorted(cells, key=lambda c: c["position_p"])
    for shorter, longer in zip(ordered, ordered[1:]):
        a, b = shorter["context_ids"], longer["context_ids"]
        if len(a) >= len(b):
            raise PositionContractError(
                f"contexts not strictly increasing: p={shorter['position_p']} len {len(a)} "
                f"vs p={longer['position_p']} len {len(b)}")
        if b[:len(a)] != a:
            raise PositionContractError(
                f"context at p={shorter['position_p']} is not a prefix of p={longer['position_p']}")
    return True


def rederive_and_check(prompt_ids, cont_ids, position_p, context_len_seen, target_token_id_seen,
                       context_ids_seen):
    """Independent re-derivation, for checking a stored record against the contract.

    Catches a position-label scramble, which a same-cell self-consistency check cannot see.
    """
    context, target = build_context(prompt_ids, cont_ids, position_p)
    problems = []
    if context_len_seen != len(context):
        problems.append(f"context_len {context_len_seen} != {len(context)}")
    if target_token_id_seen != target:
        problems.append(f"target {target_token_id_seen} != {target}")
    if list(context_ids_seen) != context:
        problems.append("context token IDs differ from re-derivation")
    return (problems == []), problems


class GridIncompleteError(PositionContractError):
    pass


def assert_complete_grid(cells, n_trajectories):
    """Every trajectory carries all ten retained positions, exactly once.

    A partial trajectory must not be averaged: the headline is a mean of per-trajectory means, so a
    trajectory scored at 9 positions would silently weight those nine as if they were ten.
    """
    seen = {}
    for c in cells:
        t, p = c["trajectory_index"], c["position_p"]
        if p not in RETAINED_POSITIONS:
            raise GridIncompleteError(
                f"trajectory {t}: position {p!r} is not in the retained vector "
                f"{list(RETAINED_POSITIONS)}")
        key = (t, p)
        if key in seen:
            raise GridIncompleteError(f"trajectory {t}: position {p} appears more than once")
        seen[key] = c
    want_t = set(range(n_trajectories))
    got_t = {t for t, _ in seen}
    if got_t - want_t:
        raise GridIncompleteError(
            f"trajectory indices outside 0..{n_trajectories - 1}: {sorted(got_t - want_t)}")
    incomplete = {}
    for t in sorted(want_t):
        missing = [p for p in RETAINED_POSITIONS if (t, p) not in seen]
        if missing:
            incomplete[t] = missing
    if incomplete:
        raise GridIncompleteError(
            f"{len(incomplete)} trajectory/ies are not 10/10: "
            + "; ".join(f"traj {t} missing {m}" for t, m in sorted(incomplete.items())[:8]))
    expected = n_trajectories * len(RETAINED_POSITIONS)
    if len(seen) != expected:
        raise GridIncompleteError(f"grid holds {len(seen)} cells, contract requires {expected}")
    return True


def grid_order(n_trajectories):
    """Canonical (trajectory, ascending position) enumeration; the storage row order."""
    return [(t, p) for t in range(n_trajectories) for p in RETAINED_POSITIONS]
