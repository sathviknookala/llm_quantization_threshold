"""The frozen BF16 continuation set: generate once (P7), then treat as immutable input.

Every KL context in the study is a prefix of one of these trajectories, so a regenerated
trajectory set is a different experiment even when the seed and prompts are unchanged. `freeze()`
refuses to overwrite; `load()` re-verifies rather than trusting the file it reads.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import positions as P, qcommon as q  # noqa: E402
from harness.quality import qengine as E  # noqa: E402

PATH = os.path.join(q.QUALITY_DIR, "trajectories.json")
SCHEMA_VERSION = 1


def trajectory_set_hash(trajectories):
    return common.sha256_of_json(
        [[t["prompt_token_ids"], t["continuation_token_ids"]] for t in trajectories])


def _validate(rec, expect_n):
    traj = rec["trajectories"]
    if len(traj) != expect_n:
        raise SystemExit(f"ABORT: trajectory artifact holds {len(traj)}, contract says {expect_n}")
    for i, t in enumerate(traj):
        if t["trajectory_index"] != i:
            raise SystemExit(f"ABORT: trajectory {i} carries index {t['trajectory_index']}; "
                             "trajectory order is part of the identity")
        if t["prompt_index"] != i:
            raise SystemExit(f"ABORT: trajectory {i} came from prompt {t['prompt_index']}; "
                             "the first-N subset is only well defined in stored order")
        if len(t["prompt_token_ids"]) != P.PROMPT_TOKENS:
            raise SystemExit(f"ABORT: trajectory {i} prompt is {len(t['prompt_token_ids'])} tokens")
        if len(t["continuation_token_ids"]) != P.CONTINUATION_TOKENS:
            raise SystemExit(
                f"ABORT: trajectory {i} continuation is {len(t['continuation_token_ids'])} tokens, "
                f"contract requires exactly {P.CONTINUATION_TOKENS}")
        if q.prompt_hash(t["prompt_token_ids"]) != t["prompt_sha256"]:
            raise SystemExit(f"ABORT: trajectory {i} prompt does not match its recorded hash")
        if q.prompt_hash(t["continuation_token_ids"]) != t["continuation_sha256"]:
            raise SystemExit(f"ABORT: trajectory {i} continuation does not match its recorded hash")
    got = trajectory_set_hash(traj)
    if got != rec["trajectory_set_hash"]:
        raise SystemExit(f"ABORT: trajectory_set_hash is {rec['trajectory_set_hash'][:16]} but the "
                         f"content hashes to {got[:16]}")
    return rec


def load(path=PATH, expect_n=None):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: no frozen trajectory artifact at {path}; P7 has not been run")
    rec = json.load(open(path))
    if rec.get("PROVISIONAL"):
        raise SystemExit(f"ABORT: {path} is a provisional gate artifact, not production input")
    return _validate(rec, expect_n or rec["n_trajectories"])


def generate(config_id, prompts, out_dir, tag, allow_dirty=False, overrides=None):
    """One launch, groups sized under the KV wall. Returns the engine job metadata."""
    job = {
        "config_id": config_id, "task": "generate",
        "prompts": [list(p) for p in prompts],
        "group_size": q.GENERATION_GROUP_SIZE,
        "out_json": os.path.join(out_dir, f"_gen_{tag}.json"),
        "out_npy": os.path.join(out_dir, f"_unused_{tag}.npy"),
        "engine_overrides": dict(overrides or {}),
    }
    meta = E.run_job(job, os.path.join(out_dir, "logs", f"generate_{tag}.log"),
                     allow_dirty=allow_dirty)
    bad = [i for i, n in enumerate(meta["continuation_lengths"]) if n != q.GENERATION_TOKENS]
    if bad:
        raise SystemExit(f"ABORT: continuations {bad[:10]} are not exactly {q.GENERATION_TOKENS} "
                         "tokens; a short trajectory cannot be padded or truncated into one")
    return meta


def freeze(path=PATH, n=None, allow_dirty=False):
    n = n or q.N_TRAJECTORIES
    if os.path.exists(path):
        raise SystemExit(
            f"ABORT: {path} already exists. Production trajectories are immutable experimental "
            "input; regenerating them is a new experiment identity, not a rerun. Move the file "
            "aside deliberately if that is what you mean.")
    q.require_clean_tree(allow_dirty, stage="freeze_trajectories")
    prompts, manifest = q.load_prompts(n=n)
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    ident = q.config_identity(q.REFERENCE_CONFIG)
    meta = generate(q.REFERENCE_CONFIG, [p["token_ids"] for p in prompts], out_dir, "production",
                    allow_dirty=allow_dirty)

    traj = [{"trajectory_index": i,
             "prompt_index": p["index"],
             "prompt_sha256": q.prompt_hash(p["token_ids"]),
             # inlined so the artifact stands alone: the corpus body is gitignored
             "prompt_token_ids": list(p["token_ids"]),
             "continuation_sha256": q.prompt_hash(c),
             "continuation_token_ids": list(c)}
            for i, (p, c) in enumerate(zip(prompts, meta["continuations"]))]

    rec = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "production BF16 trajectory set (P7)",
        "immutable": True,
        "n_trajectories": n,
        "generated_by_config": q.REFERENCE_CONFIG,
        "engine_profile_name": q.PROFILE_NAME,
        "engine_controls": dict(q.QUALITY_ENGINE_CONTROLS),
        "generation": dict(q.GENERATION_SAMPLING),
        "generation_group_size": q.GENERATION_GROUP_SIZE,
        "generation_tokens": q.GENERATION_TOKENS,
        "corpus_version": manifest.get("corpus_version"),
        "prompt_set_hash": manifest["prompt_set_hash"],
        "prompt_subset_hash": manifest["subset_hash"],
        "kl_spec_hash": q.spec_hash(),
        "config_identity": ident,
        "observed": meta.get("observed"),
        "resolved_config": meta.get("resolved_config"),
        "engine_metrics": meta.get("engine_metrics"),
        "generate_seconds": meta.get("generate_seconds"),
        "trajectory_set_hash": trajectory_set_hash(traj),
        "git": q.git_state(),
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "created_at": common.now_iso(),
        "trajectories": traj,
    }
    _validate(rec, n)
    common.write_json(path, rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        rec = freeze(n=a.n or None, allow_dirty=a.allow_dirty)
        print(json.dumps({k: v for k, v in rec.items()
                          if k not in ("trajectories", "software", "config_identity")}, indent=2))
        print("WROTE", PATH)
    elif a.verify:
        rec = load()
        print("OK", rec["n_trajectories"], "trajectories,",
              "set_hash", rec["trajectory_set_hash"][:16],
              "spec", rec["kl_spec_hash"])
    else:
        raise SystemExit("pass --freeze or --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
