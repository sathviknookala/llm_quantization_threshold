import argparse, json, numpy as np, os

ap = argparse.ArgumentParser()
ap.add_argument("--logits-dir", default="results/qualification/logits")
ap.add_argument("--reference", default="BF16")
ap.add_argument("--configs", nargs="+", default=["FP8", "NVFP4"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

L = a.logits_dir.rstrip("/")
labels = [a.reference] + a.configs
meta = {k: json.load(open(f"{L}/{k}_logits_meta.json")) for k in labels}

ref_ids = meta[a.reference]["token_ids"]
identical = all(meta[k]["token_ids"] == ref_ids for k in labels)
if not identical:
    raise SystemExit("ABORT: token contexts differ across configurations; KL would be invalid")

def probs(k):
    p = np.exp(np.load(f"{L}/{k}_logprobs.npy").astype(np.float64))
    return p / p.sum(axis=1, keepdims=True)

P = {k: probs(k) for k in labels}
EPS = 1e-12

def kl(a_, b_):
    pa, pb = P[a_], P[b_]
    return (pa * (np.log(pa + EPS) - np.log(pb + EPS))).sum(axis=1)

pairs = [(a.reference, c) for c in a.configs]
if len(a.configs) >= 2:
    pairs += list(zip(a.configs, a.configs[1:]))

rec = {"reference": a.reference, "configs": a.configs,
       "contexts": len(ref_ids), "ctx_tokens": len(ref_ids[0]),
       "vocab_size": meta[a.reference]["vocab_size"],
       "context_token_identity": identical,
       "bytes_per_context_fp16": meta[a.reference]["vocab_size"] * 2,
       "pairs": {}}
ref_top1 = P[a.reference].argmax(1)
for x, y in pairs:
    d = kl(x, y)
    rec["pairs"][f"{x}||{y}"] = {
        "mean_nats": float(d.mean()), "median_nats": float(np.median(d)),
        "min_nats": float(d.min()), "max_nats": float(d.max()),
        "p90_nats": float(np.quantile(d, 0.90)), "std_nats": float(d.std(ddof=1)),
        "per_context_nats": [float(v) for v in d],
    }
for c in a.configs:
    rec["pairs"].setdefault(f"{a.reference}||{c}", {})["top1_agreement_vs_reference"] = \
        float((P[c].argmax(1) == ref_top1).mean())
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(rec, open(a.out, "w"), indent=2)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_context_nats"}
                  for k, v in rec["pairs"].items()}, indent=2))
