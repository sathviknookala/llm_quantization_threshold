import argparse, json, os, subprocess, time

def gpu_mem_mib():
    q = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits"
    v = subprocess.run(q, shell=True, capture_output=True, text=True).stdout.strip().split(", ")
    return dict(zip(["mem_used_mib","mem_total_mib","util_pct","power_w","sm_clock_mhz","temp_c"], v))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--max-model-len", type=int, default=32768)
ap.add_argument("--gpu-util", type=float, default=0.90)
ap.add_argument("--kv-dtype", default="auto")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rec = {"label": a.label, "model": a.model, "max_model_len": a.max_model_len,
       "gpu_memory_utilization": a.gpu_util, "kv_cache_dtype": a.kv_dtype,
       "mem_before": gpu_mem_mib()}

from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
          kv_cache_dtype=a.kv_dtype, enforce_eager=False, disable_log_stats=False)
rec["load_seconds"] = round(time.time() - t0, 2)
rec["mem_after_load"] = gpu_mem_mib()

tok = llm.get_tokenizer()
# fixed synthetic contexts so token counts are identical across configurations
base = "The history of computing hardware spans several distinct eras of technology. "
def ctx(n):
    ids = tok(base * (n // 10 + 40), add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids), len(ids)

results = []
for name, ntok, nout in [("short", 128, 64), ("medium", 2048, 128), ("long", 16384, 128)]:
    if ntok + nout > a.max_model_len:
        results.append({"workload": name, "skipped": "exceeds max_model_len"}); continue
    prompt, actual = ctx(ntok)
    sp = SamplingParams(temperature=0.0, max_tokens=nout, ignore_eos=True)
    t = time.time()
    out = llm.generate([prompt], sp)
    dt = time.time() - t
    gen = out[0].outputs[0]
    results.append({"workload": name, "prompt_tokens": actual, "output_tokens": len(gen.token_ids),
                    "wall_seconds": round(dt, 3),
                    "decode_tok_per_s": round(len(gen.token_ids) / dt, 2),
                    "text_head": gen.text[:80].replace("\n", " "),
                    "mem": gpu_mem_mib()})
rec["workloads"] = results
rec["mem_peak_after_all"] = gpu_mem_mib()
try:
    import torch
    rec["torch_max_mem_alloc_GiB"] = round(torch.cuda.max_memory_allocated()/2**30, 3)
except Exception:
    pass
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rec, open(a.out, "w"), indent=2)
print("WROTE", a.out)
print(json.dumps(rec, indent=2))
