"""vLLM server lifecycle for the pilot: launch, verify dispatch, scrape counters."""

import os
import re
import signal
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import common  # noqa: E402

KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
MAXCONC_RE = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
PREFIX_RE = re.compile(r"enable_prefix_caching[=:]\s*'?(True|False|None)'?")
KERNEL_PATTERNS = [
    "CutlassFP8ScaledMMLinearKernel",
    "NvFp4LinearBackend",
    "FLASHINFER_CUTLASS",
    "Marlin",
    "CT_EMULATIONS",
    "gemm_sm120",
]

# The FP4 GEMM is JIT-built by flashinfer and needs ninja/nvcc on PATH (D9).
CHILD_ENV = dict(os.environ)
CHILD_ENV["PATH"] = ("/home/sathvik/miniconda3/envs/qnt/bin:/home/sathvik/cuda-12.9/bin:"
                     + CHILD_ENV.get("PATH", ""))
CHILD_ENV["CUDA_HOME"] = "/home/sathvik/cuda-12.9"
CHILD_ENV["VLLM_LOGGING_LEVEL"] = "INFO"


def gpu_holder_pids():
    out = subprocess.run(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
        shell=True, capture_output=True, text=True, timeout=15).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def kill_gpu_holders():
    """Precise teardown: only the processes the driver reports as holding device memory."""
    pids = gpu_holder_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"  killed stray GPU holder pid={pid}", flush=True)
        except OSError:
            pass
    return pids


def wait_for_gpu_release(timeout=300, threshold_mib=512, poll=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (common.gpu_telemetry().get("memory.used") or 0) < threshold_mib:
            return True
        time.sleep(poll)
    return (common.gpu_telemetry().get("memory.used") or 0) < threshold_mib


class VllmServer:
    def __init__(self, config_id, port, log_path):
        self.config_id = config_id
        self.cfg = common.CONFIGS[config_id]
        self.port = port
        self.log_path = log_path
        self.proc = None
        self.base = f"http://127.0.0.1:{port}"
        self.startup = {}

    def command(self):
        c = common.SERVER_CONTROLS
        return [
            "vllm", "serve", self.cfg["model"],
            "--served-model-name", "pilot",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--max-model-len", str(c["max_model_len"]),
            "--gpu-memory-utilization", str(c["gpu_memory_utilization"]),
            "--max-num-seqs", str(c["max_num_seqs"]),
            "--kv-cache-dtype", c["kv_cache_dtype"],
            "--no-enable-prefix-caching",
            "--no-enable-log-requests",
            "--seed", str(c["seed"]),
        ]

    def start(self, timeout=900):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self.log_fh = open(self.log_path, "a")
        t0 = time.time()
        self.proc = subprocess.Popen(self.command(), stdout=self.log_fh,
                                     stderr=subprocess.STDOUT, env=CHILD_ENV,
                                     start_new_session=True)
        ready = False
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}; see {self.log_path}")
            try:
                if requests.get(self.base + "/health", timeout=3).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(2.0)
        if not ready:
            self.stop()
            raise RuntimeError(f"server not ready in {timeout}s; see {self.log_path}")
        self.startup = {"engine_start_seconds": round(time.time() - t0, 2)}
        self.startup.update(self.parse_log())
        self.startup["dispatch_verdict"] = self.verify_dispatch(self.startup["kernel_lines"])
        return self.startup

    def parse_log(self):
        text = open(self.log_path, errors="replace").read()
        kv = KV_RE.search(text)
        mc = MAXCONC_RE.search(text)
        pfx = PREFIX_RE.findall(text)
        lines = [ln.strip() for ln in text.splitlines()
                 if any(p in ln for p in KERNEL_PATTERNS)]
        return {
            "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
            "max_concurrency_line": (int(mc.group(1).replace(",", "")), float(mc.group(2))) if mc else None,
            "enable_prefix_caching_logged": (pfx[-1] if pfx else None),
            "kernel_lines": sorted(set(lines))[:20],
            "log_path": self.log_path,
        }

    def verify_dispatch(self, kernel_lines):
        blob = " | ".join(kernel_lines)
        want = self.cfg["expected_kernel_pattern"]
        bad = [p for p in self.cfg["forbidden_kernel_patterns"] if p in blob]
        return {
            "expected_pattern": want,
            "expected_present": (want in blob) if want else None,
            "forbidden_present": bad,
            "ok": (bad == []) and (want is None or want in blob),
        }

    def metrics(self, timeout=10):
        r = requests.get(self.base + "/metrics", timeout=timeout)
        r.raise_for_status()
        return common.engine_counters(common.parse_prometheus(r.text))

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def wait_drained(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                m = self.metrics()
                if (m.get("num_running_reqs") or 0) == 0 and (m.get("num_waiting_reqs") or 0) == 0:
                    return True
            except Exception:
                pass
            time.sleep(2.0)
        return False

    def stop(self, timeout=90, release_timeout=300):
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            self.proc.wait(timeout=timeout)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=60)
            except Exception:
                pass
        try:
            self.log_fh.close()
        except Exception:
            pass
        self.proc = None
        # vLLM v1 runs EngineCore in its own process; the parent can exit before VRAM is released,
        # and the next configuration profiles memory at startup, so wait for an actually-free GPU
        if not wait_for_gpu_release(release_timeout):
            # `pkill -f 'vllm serve'` never matched: v1 renames the child to VLLM::EngineCore, so
            # the old fallback was dead code. Ask the driver which PIDs actually hold the GPU.
            kill_gpu_holders()
            wait_for_gpu_release(120)
