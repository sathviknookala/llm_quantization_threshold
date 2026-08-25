"""Shared identity, telemetry and statistics helpers for the serving pilot."""

import hashlib
import json
import os
import platform
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT_DIR = os.path.join(REPO, "results", "pilot")
CORPUS_DIR = os.path.join(PILOT_DIR, "corpus")

BF16_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
)

CONFIGS = {
    "BF16_REFERENCE": {
        "model": BF16_SNAPSHOT,
        "short": "BF16",
        "weight_bytes_gb": 16.10,
        "expected_kernel_pattern": None,
        "forbidden_kernel_patterns": [
            "CutlassFP8ScaledMMLinearKernel",
            "NvFp4LinearBackend",
            "Marlin",
        ],
    },
    "FP8_PRIMARY": {
        "model": os.path.join(REPO, "checkpoints", "llama31-8b-instruct-FP8_DYNAMIC"),
        "short": "FP8",
        "weight_bytes_gb": 9.12,
        "expected_kernel_pattern": "CutlassFP8ScaledMMLinearKernel",
        "forbidden_kernel_patterns": ["Marlin"],
    },
    "FP4_PRIMARY": {
        "model": os.path.join(REPO, "checkpoints", "llama31-8b-instruct-NVFP4"),
        "short": "FP4",
        "weight_bytes_gb": 6.07,
        "expected_kernel_pattern": "FLASHINFER_CUTLASS",
        "forbidden_kernel_patterns": ["CT_EMULATIONS", "emulation"],
    },
}

# D10 weight-residency ratios, not datatype ratios.
D11_EXPECTED_RATIO = {
    "FP8_PRIMARY": CONFIGS["BF16_REFERENCE"]["weight_bytes_gb"] / CONFIGS["FP8_PRIMARY"]["weight_bytes_gb"],
    "FP4_PRIMARY": CONFIGS["BF16_REFERENCE"]["weight_bytes_gb"] / CONFIGS["FP4_PRIMARY"]["weight_bytes_gb"],
}

# Held constant across every configuration; these reproduce the qualification memory profile.
SERVER_CONTROLS = {
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 256,
    "kv_cache_dtype": "auto",
    "enable_prefix_caching": False,
    "seed": 0,
}

WORKLOADS = {
    "DECODE_PRIMARY": {"input_tokens": 512, "output_tokens": 2048},
    "PREFILL_PROBE": {"input_tokens": 8192, "output_tokens": 32},
}

SLO_TPOT_MS = 50.0
SLO_ABORT_MULTIPLE = 10.0
CELL_WALL_CAP_S = 900.0
TELEMETRY_PERIOD_S = 10.0

SMI_FIELDS = [
    "memory.used", "memory.total", "utilization.gpu", "utilization.memory",
    "power.draw", "clocks.sm", "clocks.mem", "temperature.gpu",
    "pcie.link.gen.current", "pcie.link.width.current",
]


def gpu_telemetry():
    q = ("nvidia-smi --query-gpu=" + ",".join(SMI_FIELDS)
         + " --format=csv,noheader,nounits")
    try:
        out = subprocess.run(q, shell=True, capture_output=True, text=True, timeout=10)
        vals = [v.strip() for v in out.stdout.strip().split(",")]
        rec = {}
        for k, v in zip(SMI_FIELDS, vals):
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = None
        return rec
    except Exception as exc:
        return {"error": str(exc)}


def gpu_identity():
    q = ("nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit"
         " --format=csv,noheader,nounits")
    out = subprocess.run(q, shell=True, capture_output=True, text=True, timeout=15)
    keys = ["name", "uuid", "driver_version", "memory_total_mib", "power_limit_w"]
    return dict(zip(keys, [v.strip() for v in out.stdout.strip().split(",")]))


def gpu_is_idle(max_mem_mib=512, max_util_pct=5):
    t = gpu_telemetry()
    return (t.get("memory.used") or 0) <= max_mem_mib and (t.get("utilization.gpu") or 0) <= max_util_pct


def software_identity():
    rec = {"python": platform.python_version(), "kernel": platform.release()}
    try:
        import torch
        rec["torch"] = torch.__version__
        rec["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import vllm
        rec["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        import transformers
        rec["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import compressed_tensors
        rec["compressed_tensors"] = compressed_tensors.__version__
    except Exception:
        pass
    rec["nvcc"] = subprocess.run("nvcc --version | tail -2 | head -1", shell=True,
                                 capture_output=True, text=True).stdout.strip()
    rec["git_head"] = subprocess.run("git -C %s rev-parse HEAD" % REPO, shell=True,
                                     capture_output=True, text=True).stdout.strip()
    rec["git_dirty"] = bool(subprocess.run("git -C %s status --porcelain" % REPO, shell=True,
                                           capture_output=True, text=True).stdout.strip())
    return rec


def sha256_of_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path


def quantiles(xs, qs=(0.5, 0.95, 0.99)):
    if not xs:
        return {f"p{int(q * 100)}": None for q in qs}
    s = sorted(xs)
    out = {}
    for q in qs:
        if len(s) == 1:
            out[f"p{int(q * 100)}"] = s[0]
            continue
        pos = q * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        out[f"p{int(q * 100)}"] = s[lo] + (pos - lo) * (s[hi] - s[lo])
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sample_std(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# scipy is absent in env `qnt`; two-sided 95% critical values for the df actually reachable here.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t_crit_95(df):
    if df <= 0:
        return None
    if df in _T95:
        return _T95[df]
    return 1.96 if df > 20 else _T95[max(_T95)]


PROM_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)$")


def parse_prometheus(text):
    """Sum each metric family across engine labels; vLLM emits per-engine series."""
    acc = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = PROM_RE.match(line)
        if not m:
            continue
        try:
            v = float(m.group("value"))
        except ValueError:
            continue
        name = m.group("name")
        acc[name] = acc.get(name, 0.0) + v
    return acc


ENGINE_COUNTERS = {
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
    "num_preemptions": "vllm:num_preemptions",
    "recomputed_tokens": "vllm:prompt_tokens_recomputed",
    "num_waiting_reqs": "vllm:num_requests_waiting",
    "num_running_reqs": "vllm:num_requests_running",
    "prefix_cache_queries": "vllm:prefix_cache_queries",
    "prefix_cache_hits": "vllm:prefix_cache_hits",
    "generation_tokens": "vllm:generation_tokens",
    "prompt_tokens": "vllm:prompt_tokens",
    "prompt_tokens_cached": "vllm:prompt_tokens_cached",
}


def engine_counters(prom):
    """prometheus_client appends _total to Counter names, so the source-level name misses."""
    out = {}
    for k, v in ENGINE_COUNTERS.items():
        out[k] = prom.get(v)
        if out[k] is None:
            out[k] = prom.get(v + "_total")
    return out
