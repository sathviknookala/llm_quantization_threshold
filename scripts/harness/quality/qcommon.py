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
    "enforce_eager": None,
    "seed": 0,
    "dtype": "auto",
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
    stat_key = common.sha256_of_json([[rel, size, mtime] for rel, _, size, mtime in files])
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


def guard_manifest(out_dir, artifact, extra=None):
    """Resume must not mix runs built under different contracts."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "manifest.json")
    h = spec_hash()
    if os.path.exists(path):
        prior = json.load(open(path)).get("kl_spec_hash")
        if prior and prior != h:
            raise SystemExit(
                f"ABORT: KL_SPEC changed ({prior} -> {h}) but {out_dir} holds artifacts from the "
                "old spec. Move it aside or restore the spec; resume must not mix.")
        return path, json.load(open(path))
    rec = {"artifact": artifact, "kl_spec_hash": h, "spec": KL_SPEC,
           "started_at": common.now_iso()}
    if extra:
        rec.update(extra)
    common.write_json(path, rec)
    return path, rec


def load_prompts(n=N_TRAJECTORIES, workload=CORPUS_WORKLOAD):
    tokens = common.WORKLOADS[workload]["input_tokens"]
    stem = os.path.join(common.CORPUS_DIR, f"{workload.lower()}_{tokens}tok")
    body = json.load(open(stem + ".json"))
    manifest = json.load(open(stem + "_manifest.json"))
    prompts = body["prompts"][:n]
    if len(prompts) != n:
        raise SystemExit(f"ABORT: corpus holds {len(body['prompts'])} prompts, need {n}")
    for i, p in enumerate(prompts):
        if p["index"] != i:
            raise SystemExit(f"ABORT: corpus prompt {i} carries index {p['index']}; "
                             "the first-N subset is only well defined in stored order")
        if p["n_tokens"] != tokens or len(p["token_ids"]) != tokens:
            raise SystemExit(f"ABORT: corpus prompt {i} has {p['n_tokens']} tokens, want {tokens}")
    return prompts, manifest


def prompt_hash(token_ids):
    return hashlib.sha256(json.dumps(list(token_ids)).encode()).hexdigest()
