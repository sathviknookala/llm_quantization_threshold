"""Build the frozen, prefix-disjoint prompt corpus (D16: C4 en validation).

Prompts are emitted as token IDs so tokenization cannot drift between configurations,
and the manifest hash is what later runs assert against.
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilot import common  # noqa: E402

C4_REPO = "allenai/c4"
C4_FILE = "en/c4-validation.00000-of-00008.json.gz"
PREFIX_HASH_TOKENS = 64


def load_c4_docs(limit_lines):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=C4_REPO, filename=C4_FILE, repo_type="dataset")
    docs = []
    with gzip.open(path, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= limit_lines:
                break
            docs.append(json.loads(line)["text"])
    return path, docs


def build(tokenizer, docs, n_prompts, target_tokens, seed, bos_id, concat_docs):
    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)

    content_target = target_tokens - 1  # one slot reserved for BOS
    prompts, used, prefix_hashes, cursor = [], [], set(), 0

    while len(prompts) < n_prompts and cursor < len(order):
        ids, members = [], []
        while len(ids) < content_target and cursor < len(order):
            di = order[cursor]
            cursor += 1
            enc = tokenizer(docs[di], add_special_tokens=False)["input_ids"]
            if not enc:
                continue
            if not concat_docs and len(enc) < content_target:
                continue
            ids.extend(enc)
            members.append(di)
            if not concat_docs:
                break
        if len(ids) < content_target:
            continue
        ids = ids[:content_target]
        token_ids = [bos_id] + ids
        ph = hashlib.sha256(
            json.dumps(ids[:PREFIX_HASH_TOKENS]).encode()).hexdigest()
        if ph in prefix_hashes:
            continue
        prefix_hashes.add(ph)
        prompts.append({
            "index": len(prompts),
            "token_ids": token_ids,
            "n_tokens": len(token_ids),
            "prefix_hash": ph,
            "source_docs": members,
        })
        used.extend(members)
    return prompts, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="DECODE_PRIMARY", choices=list(common.WORKLOADS))
    ap.add_argument("--n-prompts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--max-lines", type=int, default=60000)
    ap.add_argument("--out-dir", default=common.CORPUS_DIR)
    a = ap.parse_args()

    target = common.WORKLOADS[a.workload]["input_tokens"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.BF16_SNAPSHOT)
    bos_id = tok.bos_token_id
    if bos_id is None:
        raise SystemExit("ABORT: tokenizer has no bos_token_id; exact length accounting undefined")

    src_path, docs = load_c4_docs(a.max_lines)
    prompts, used = build(tok, docs, a.n_prompts, target, a.seed, bos_id,
                          concat_docs=target > 1024)

    if len(prompts) < a.n_prompts:
        raise SystemExit(
            f"ABORT: only {len(prompts)}/{a.n_prompts} prompts of exactly {target} tokens; "
            "raise --max-lines")
    bad = [p["index"] for p in prompts if p["n_tokens"] != target]
    if bad:
        raise SystemExit(f"ABORT: prompts with wrong length: {bad[:5]}")
    if len({p["prefix_hash"] for p in prompts}) != len(prompts):
        raise SystemExit("ABORT: prefix hashes not unique")
    if len(set(used)) != len(used):
        raise SystemExit("ABORT: a source document was reused across prompts")

    body = {"prompts": prompts}
    manifest = {
        "corpus_version": "c4-en-validation-shard0-v1",
        "workload": a.workload,
        "decision": "D16 resolved to option (a), C4 en validation",
        "dataset_repo": C4_REPO,
        "dataset_file": C4_FILE,
        "local_source_file": src_path,
        "lines_scanned": a.max_lines,
        "seed": a.seed,
        "n_prompts": len(prompts),
        "tokens_per_prompt": target,
        "bos_token_id": bos_id,
        "tokenizer_source": common.BF16_SNAPSHOT,
        "tokenizer_name_or_path": tok.name_or_path,
        "vocab_size": len(tok),
        "prefix_hash_tokens": PREFIX_HASH_TOKENS,
        "prefix_hash_unique": True,
        "source_docs_disjoint": True,
        "excluded_datasets": ["HuggingFaceH4/ultrachat_200k (FP4 calibration draw, D7)"],
        "prompt_set_hash": common.sha256_of_json(
            [p["token_ids"] for p in prompts]),
        "prefix_hashes_hash": common.sha256_of_json([p["prefix_hash"] for p in prompts]),
        "built_at": common.now_iso(),
    }
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"{a.workload.lower()}_{target}tok")
    common.write_json(stem + ".json", body)
    common.write_json(stem + "_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
