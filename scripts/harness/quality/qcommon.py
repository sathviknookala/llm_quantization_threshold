"""Identity, provenance and pre-registered constants for the quality arm.

Imports the serving harness; edits none of it. The serving sweep is a finished artifact and its
`CONFIGS`/`SERVER_CONTROLS` feed `run_sweep.SWEEP_SPEC`'s hash, so the quality registry is seeded
from them rather than extending them.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality.positions import RETAINED_POSITIONS  # noqa: E402

QUALITY_DIR = os.path.join(common.REPO, "results", "quality")
KL_DIR = os.path.join(QUALITY_DIR, "kl")
CACHE_PATH = os.path.join(QUALITY_DIR, "checkpoint_hashes.json")

QUALITY_CONFIGS = copy.deepcopy(common.CONFIGS)
REFERENCE_CONFIG = "BF16_REFERENCE"
LADDER = ("BF16_REFERENCE", "FP8_PRIMARY", "FP4_PRIMARY")

# Direct pairs. FP8->FP4 is measured with FP8 as reference; subtracting BF16-anchored KLs is barred
# because KL is not additive (D13 amendment 2026-08-25).
KL_PAIRS = (
    ("BF16_REFERENCE", "FP8_PRIMARY"),
    ("BF16_REFERENCE", "FP4_PRIMARY"),
    ("FP8_PRIMARY", "FP4_PRIMARY"),
)

VOCAB_SIZE = 128256
N_TRAJECTORIES = 64
CORPUS_WORKLOAD = "DECODE_PRIMARY"
STORAGE_DTYPE = "float32"
KL_WORKING_DTYPE = "float64"

QUALITY_ENGINE_CONTROLS = {
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 256,
    "kv_cache_dtype": "auto",
    "max_logprobs": VOCAB_SIZE,
    # the deliberate inversion of the serving contract (H7); nested prefixes make it sound
    "enable_prefix_caching": True,
    # LLM_CLASS would default to 8192 while the measured serving axis ran the server default of
    # 2048, so the longest KL context would chunk differently across the two axes
    "max_num_batched_tokens": 2048,
    # LOCKED by G9 2026-08-25. Not a neutral knob: flipping it moves FP4 by 3.74e-02 nats and
    # flips top-1 in 4/40 cells (results/quality/gates/engine_profile.json).
    "enforce_eager": False,
    "seed": 0,
    "dtype": "auto",
}
PROFILE_NAME = "graph_2048"

# Generation runs below BF16's measured KV wall (D11: [17,18]) so no trajectory is produced under
# preemption; 16 x 2560 = 40,960 tokens against 44,688.
GENERATION_GROUP_SIZE = 16

# Knobs that differ from the measured serving axis, enumerated rather than glossed.
ENGINE_DELTAS_VS_SERVING = {
    "enable_prefix_caching": "True here, False for serving -- the deliberate H7 exception",
    "max_logprobs": "128256 here, server default 20; a request-validation bound only",
    "disable_log_stats": "False here; `vllm serve` already runs with stats on, so this matches",
    "process_model": "in-process LLM here, `vllm serve` for the serving axis",
}

SCORING_SAMPLING = {
    "temperature": 0.0,
    "max_tokens": 1,
    "logprobs": VOCAB_SIZE,
    # returned logprobs are log_softmax over the ORIGINAL logits, so this costs ~128k per-token
    # decode calls that nothing reads
    "detokenize": False,
}

GENERATION_SAMPLING = {
    "max_tokens": 2048,
    "min_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": -1,
    "min_p": 0.0,
    "seed": 20260823,
    "ignore_eos": True,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "detokenize": False,
}
GENERATION_TOKENS = 2048

BOOTSTRAP = {
    "draws": 10000,
    "seed": 20260825,
    "ci": [0.025, 0.975],
    "interpolation": "linear",
    "resampling_unit": "trajectory",
}

GATES = {
    "prob_mass_tolerance": 1e-3,
    "kl_negative_tolerance": 1e-12,
    "alignment_pooled_min": 0.99,
    "replication_floor_max_frac_of_fp8": 0.01,
    "cache_equivalence_max_frac_of_fp8": 0.01,
    # AND-exclude, not OR-pass: the relative bounds are evaluated only on cells whose TRUE KL is
    # at or above the floor. Read as OR-pass the floor would auto-pass nearly every cell, since
    # storage-induced absolute deltas sit well below it.
    "precision_rel_p99_max": 1e-3,
    "precision_rel_max": 1e-2,
    "precision_abs_floor_nats": 1e-4,
    "numerics_mean_abs_max_nats": 1e-6,
    "numerics_rel_max": 5e-3,
}

KL_SPEC = {
    "schema_version": 1,
    "contract": "EVALUATION_RIG.md A.1, locked 2026-08-25 (D13 amendment)",
    "ladder": list(LADDER),
    "pairs": [list(p) for p in KL_PAIRS],
    "reference_config": REFERENCE_CONFIG,
    "n_trajectories": N_TRAJECTORIES,
    "corpus_workload": CORPUS_WORKLOAD,
    "vocab_size": VOCAB_SIZE,
    "generation": dict(GENERATION_SAMPLING),
    "generation_tokens": GENERATION_TOKENS,
    "generation_group_size": GENERATION_GROUP_SIZE,
    "engine_profile_name": PROFILE_NAME,
    "retained_positions": list(RETAINED_POSITIONS),
    "scoring": dict(SCORING_SAMPLING),
    "engine_controls": dict(QUALITY_ENGINE_CONTROLS),
    "storage_dtype": STORAGE_DTYPE,
    "kl_working_dtype": KL_WORKING_DTYPE,
    "bootstrap": dict(BOOTSTRAP),
    "gates": dict(GATES),
    "configs": {k: {"model": v["model"],
                    "expected_kernel_pattern": v["expected_kernel_pattern"],
                    "forbidden_kernel_patterns": v["forbidden_kernel_patterns"]}
                for k, v in QUALITY_CONFIGS.items()},
}


def spec_hash(spec=None):
    return common.sha256_of_json(spec if spec is not None else KL_SPEC)[:16]


def run_id(prefix="kl"):
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S')}-{spec_hash()}"


CHECKPOINT_FILES = (".safetensors", ".json", ".yaml", ".jinja", ".model")


def _content_probe(path, size, n=1 << 20):
    """First and last MiB. A recalibrated checkpoint deployed with preserved timestamps keeps its
    size and mtime, so those alone cannot key a content hash -- verified reproducible."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(n))
        if size > n:
            fh.seek(max(0, size - n))
            h.update(fh.read(n))
    return h.hexdigest()


def _checkpoint_files(path):
    out = []
    for root, _, names in os.walk(path):
        for n in sorted(names):
            if n.endswith(CHECKPOINT_FILES):
                full = os.path.join(root, n)
                st = os.stat(full)
                out.append((os.path.relpath(full, path), full, st.st_size, st.st_mtime_ns))
    return sorted(out)


def _sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def checkpoint_content_hash(path, cache_path=CACHE_PATH):
    """Identity of the weights actually loaded.

    `CONFIGS[...]["model"]` is a mutable directory path and `checkpoints/` is gitignored, so a
    recalibrated checkpoint would otherwise reuse a config ID with different weights.
    """
    files = _checkpoint_files(path)
    if not files:
        raise SystemExit(f"ABORT: no checkpoint files under {path}")
    stat_key = common.sha256_of_json(
        [[rel, size, mtime, _content_probe(full, size)] for rel, full, size, mtime in files])
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except (ValueError, OSError):
            cache = {}
    hit = cache.get(stat_key)
    if hit:
        return hit
    digest = common.sha256_of_json(
        {rel: _sha256_file(full) for rel, full, _, _ in files})
    cache[stat_key] = digest
    common.write_json(cache_path, cache)
    return digest


def tokenizer_identity(path):
    out = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        full = os.path.join(path, name)
        out[name] = _sha256_file(full) if os.path.exists(full) else None
    return out


def config_identity(config_id):
    cfg = QUALITY_CONFIGS[config_id]
    return {
        "configuration_id": config_id,
        "quantization": cfg["short"],
        "model_path": cfg["model"],
        "checkpoint_content_hash": checkpoint_content_hash(cfg["model"]),
        "tokenizer_identity": tokenizer_identity(cfg["model"]),
        "expected_kernel_pattern": cfg["expected_kernel_pattern"],
        "forbidden_kernel_patterns": cfg["forbidden_kernel_patterns"],
    }


def git_state():
    dirty = bool(subprocess.run(
        "git -C %s status --porcelain --untracked-files=no" % common.REPO, shell=True,
        capture_output=True, text=True).stdout.strip())
    head = subprocess.run("git -C %s rev-parse HEAD" % common.REPO, shell=True,
                          capture_output=True, text=True).stdout.strip()
    return {"git_head": head, "git_dirty": dirty}


def require_clean_tree(allow_dirty, stage):
    """Re-checked per launch: a run spans hours and the tree can go dirty mid-run."""
    st = git_state()
    if st["git_dirty"] and not allow_dirty:
        raise SystemExit(
            f"ABORT: working tree is dirty at {stage} and --allow-dirty was not given; "
            "a quality result must name the commit that produced it")
    st["allow_dirty"] = bool(allow_dirty)
    st["stage"] = stage
    return st


def env_identity():
    return {"gpu": common.gpu_identity(), "software": common.software_identity()}


def provenance(rid, allow_dirty=False, stage="run", extra=None):
    rec = {
        "run_id": rid,
        "kl_spec_hash": spec_hash(),
        "kl_spec": KL_SPEC,
        "git": require_clean_tree(allow_dirty, stage),
        "created_at": common.now_iso(),
    }
    rec.update(env_identity())
    if extra:
        rec.update(extra)
    return rec


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def guard_manifest(out_dir, artifact, extra=None, adopt_if_absent=frozenset()):
    """Resume must not mix runs built under different contracts.

    A key named in `adopt_if_absent` that the manifest does not carry is adopted into the returned
    record in memory. The file is never rewritten: the four root manifests are git-tracked, and a
    write here lands between `collect()`'s clean-tree check and `run_job`'s, aborting resume.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    h = spec_hash()
    if os.path.exists(path):
        rec = json.load(open(path))
        prior = rec.get("kl_spec_hash")
        if prior and prior != h:
            raise SystemExit(
                f"ABORT: KL_SPEC changed ({prior} -> {h}) but {out_dir} holds artifacts from the "
                "old spec. Move it aside or restore the spec; resume must not mix.")
        adopted = []
        for k, want in (extra or {}).items():
            if k not in rec and k in adopt_if_absent:
                rec[k] = want
                adopted.append(k)
            elif rec.get(k) != want:
                raise SystemExit(
                    f"ABORT: {path} records {k}={rec.get(k)!r} but this run has {want!r}; "
                    "resume must not mix. Move the directory aside.")
        rec["manifest_keys_adopted"] = sorted(adopted)
        return path, rec
    rec = {"artifact": artifact, "kl_spec_hash": h, "spec": KL_SPEC,
           "started_at": common.now_iso()}
    if extra:
        rec.update(extra)
    common.write_json(path, rec)
    # set after the write: the stored manifest schema stays as it was
    rec["manifest_keys_adopted"] = []
    return path, rec


PRODUCTION_FLOOR_ARTIFACT = "production-scale BF16 replication floor"


def _floor_provenance(rec, path, trajectory_set_hash):
    got = rec.get("kl_spec_hash")
    if got != spec_hash():
        raise SystemExit(
            f"ABORT: floor {path} was written under KL_SPEC {got}, the current spec is "
            f"{spec_hash()}; a floor measured under another contract is not a floor for this one")
    checked = {"kl_spec_hash": got}
    # present-key-only: the n=4 G2 artifact carries neither, and absence is not a mismatch
    prof = rec.get("engine_profile_name")
    if prof is not None:
        if prof != PROFILE_NAME:
            raise SystemExit(f"ABORT: floor {path} was measured under engine profile {prof!r}, "
                             f"not {PROFILE_NAME!r}")
        checked["engine_profile_name"] = prof
    tsh = rec.get("trajectory_set_hash")
    if tsh is not None and trajectory_set_hash is not None:
        if tsh != trajectory_set_hash:
            raise SystemExit(f"ABORT: floor {path} was measured against trajectory set "
                             f"{tsh[:16]}, not {trajectory_set_hash[:16]}")
        checked["trajectory_set_hash"] = tsh
    return {
        "floor_path": os.path.relpath(path, common.REPO),
        "floor_trajectory_set_hash": tsh,
        "provenance_checked": checked,
        "trajectory_set_hash_verified": "trajectory_set_hash" in checked,
    }


def load_floor(path, trajectory_set_hash=None):
    """Both replication-floor schemas as one `per_config` map, or a refusal.

    An unreadable or unrecognised floor used to read back as `None`, which dropped every floor
    comparison without failing anything -- how the production floor came to be passed and ignored.
    """
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: floor file {path} does not exist; a floor that cannot be read "
                         "must not be silently skipped")
    rec = json.load(open(path))
    desc = _floor_provenance(rec, path, trajectory_set_hash)
    if "per_config" in rec:
        per = rec["per_config"] or {}
        if not per:
            raise SystemExit(f"ABORT: floor {path} carries an empty per_config map")
        desc.update({
            "floor_schema": "per_config",
            "floor_artifact": rec.get("artifact") or rec.get("gate"),
            "floor_configs": sorted(per),
            "floor_n_trajectories": rec.get("n_trajectories"),
            "aggregation": "as written by the gate that produced it",
        })
        return per, desc
    if (rec.get("artifact") == PRODUCTION_FLOOR_ARTIFACT
            or all(k in rec for k in ("headline", "worst_cell", "config"))):
        short = QUALITY_CONFIGS[rec["config"]]["short"]
        per = {short: {
            "headline_nats": rec["headline"]["mean_nats"],
            # the MEAN over the six ordered pairs, the convention EVALUATION_RIG.md already
            # fixes; `worst_cell.max_nats` is a max over six dependent pair-maxima and would
            # inflate the floor 1.17x against a headline that is itself a mean over those pairs
            "max_nats": rec["worst_cell"]["mean_nats"],
            "cells": rec["worst_cell"]["cells_per_pair"],
            "n_trajectories": rec["n_trajectories"],
        }}
        desc.update({
            "floor_schema": PRODUCTION_FLOOR_ARTIFACT,
            "floor_artifact": rec.get("artifact"),
            "floor_configs": [short],
            "floor_n_trajectories": rec["n_trajectories"],
            "floor_pairs": rec.get("ordered_pairs"),
            "floor_launches": rec.get("n_launches"),
            "aggregation": "mean over ordered pairs, for both the headline and the worst cell",
            "headline_range_nats": [rec["headline"]["min_nats"], rec["headline"]["max_nats"]],
            "worst_cell_range_nats": [rec["worst_cell"]["min_nats"],
                                      rec["worst_cell"]["max_nats"]],
        })
        return per, desc
    raise SystemExit(
        f"ABORT: {path} is neither replication-floor shape: no `per_config` map (the G2 gate "
        f"shape) and no `headline`/`worst_cell`/`config` block (the {PRODUCTION_FLOOR_ARTIFACT})")


def load_prompts(n=N_TRAJECTORIES, workload=CORPUS_WORKLOAD):
    tokens = common.WORKLOADS[workload]["input_tokens"]
    stem = os.path.join(common.CORPUS_DIR, f"{workload.lower()}_{tokens}tok")
    body = json.load(open(stem + ".json"))
    manifest = json.load(open(stem + "_manifest.json"))
    # the body is gitignored as regenerable, so it is pinned to the tracked manifest rather than
    # trusted; a body that regenerated non-identically would otherwise pass the per-prompt checks
    full_hash = common.sha256_of_json([p["token_ids"] for p in body["prompts"]])
    if full_hash != manifest["prompt_set_hash"]:
        raise SystemExit(
            f"ABORT: corpus body hashes to {full_hash[:16]} but the tracked manifest says "
            f"{manifest['prompt_set_hash'][:16]}; this is not the frozen D16 corpus")
    prompts = body["prompts"][:n]
    if len(prompts) != n:
        raise SystemExit(f"ABORT: corpus holds {len(body['prompts'])} prompts, need {n}")
    for i, p in enumerate(prompts):
        if p["index"] != i:
            raise SystemExit(f"ABORT: corpus prompt {i} carries index {p['index']}; "
                             "the first-N subset is only well defined in stored order")
        if p["n_tokens"] != tokens or len(p["token_ids"]) != tokens:
            raise SystemExit(f"ABORT: corpus prompt {i} has {p['n_tokens']} tokens, want {tokens}")
    manifest = dict(manifest)
    manifest["subset_n"] = n
    manifest["subset_hash"] = common.sha256_of_json([p["token_ids"] for p in prompts])
    return prompts, manifest


def prompt_hash(token_ids):
    return hashlib.sha256(json.dumps(list(token_ids)).encode()).hexdigest()
