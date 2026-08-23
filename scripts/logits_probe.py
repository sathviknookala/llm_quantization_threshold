import argparse, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--n-contexts", type=int, default=8)
ap.add_argument("--ctx-tokens", type=int, default=256)
ap.add_argument("--out-npy", required=True)
ap.add_argument("--out-json", required=True)
a = ap.parse_args()

from vllm import LLM, SamplingParams
llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.85,
          max_logprobs=128256, enforce_eager=True)
tok = llm.get_tokenizer()
V = len(tok)

base = "The history of computing hardware spans several distinct eras of technology. "
ctxs, ids_list = [], []
for i in range(a.n_contexts):
    ids = tok(f"Document {i}: chapter {i*7+3}. " + base * 60, add_special_tokens=False)["input_ids"][:a.ctx_tokens]
    ids_list.append(ids)

t0 = time.time()
sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=128256)
outs = llm.generate([{"prompt_token_ids": ids} for ids in ids_list], sp)
elapsed = time.time() - t0

mat = np.full((len(outs), V), -np.inf, dtype=np.float32)
covered = []
for i, o in enumerate(outs):
    lp = o.outputs[0].logprobs[0]
    for tid, obj in lp.items():
        mat[i, tid] = obj.logprob
    covered.append(len(lp))

p = np.exp(mat)
os.makedirs(os.path.dirname(a.out_npy) or ".", exist_ok=True)
np.save(a.out_npy, mat.astype(np.float16))
meta = {"label": a.label, "model": a.model, "vocab_size": V,
        "n_contexts": len(outs), "ctx_tokens": a.ctx_tokens,
        "logprob_entries_per_context": covered,
        "full_vocab_returned": all(c >= V for c in covered),
        "prob_mass_sum_per_context": [round(float(x), 6) for x in p.sum(axis=1)],
        "elapsed_seconds": round(elapsed, 2),
        "npy_path": a.out_npy,
        "npy_bytes": os.path.getsize(a.out_npy),
        "token_ids_sha": [int(sum(ids) % 10**9) for ids in ids_list],
        "token_ids": ids_list}
json.dump(meta, open(a.out_json, "w"), indent=2)
print(json.dumps(meta, indent=2))
