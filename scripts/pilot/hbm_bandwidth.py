"""P4 - standalone device-memory streaming bandwidth.

Deliberately independent of any LLM decode measurement: deriving bandwidth from
decode throughput and then using it to validate a bandwidth-derived throughput
prediction (P1) would be circular.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pilot import common  # noqa: E402

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void copy_k(const float4* __restrict__ src, float4* __restrict__ dst, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

__global__ void triad_k(float4* __restrict__ a, const float4* __restrict__ b,
                        const float4* __restrict__ c, float s, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float4 bv = b[i], cv = c[i], av;
        av.x = bv.x + s * cv.x; av.y = bv.y + s * cv.y;
        av.z = bv.z + s * cv.z; av.w = bv.w + s * cv.w;
        a[i] = av;
    }
}

__global__ void read_k(const float4* __restrict__ src, float* __restrict__ sink, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) { float4 v = src[i]; acc += v.x + v.y + v.z + v.w; }
    if (acc == 1234.5678f) sink[0] = acc;  // never true: keeps the loads live without a store
}

void stream_copy(at::Tensor src, at::Tensor dst, long blocks, long threads) {
    long n = src.numel() / 4;
    copy_k<<<blocks, threads>>>((const float4*)src.data_ptr<float>(),
                                (float4*)dst.data_ptr<float>(), n);
}

void stream_triad(at::Tensor a, at::Tensor b, at::Tensor c, double s, long blocks, long threads) {
    long n = a.numel() / 4;
    triad_k<<<blocks, threads>>>((float4*)a.data_ptr<float>(),
                                 (const float4*)b.data_ptr<float>(),
                                 (const float4*)c.data_ptr<float>(), (float)s, n);
}

void stream_read(at::Tensor src, at::Tensor sink, long blocks, long threads) {
    long n = src.numel() / 4;
    read_k<<<blocks, threads>>>((const float4*)src.data_ptr<float>(),
                                sink.data_ptr<float>(), n);
}

"""

CPP_SRC = r"""
#include <torch/extension.h>
void stream_copy(at::Tensor src, at::Tensor dst, long blocks, long threads);
void stream_triad(at::Tensor a, at::Tensor b, at::Tensor c, double s, long blocks, long threads);
void stream_read(at::Tensor src, at::Tensor sink, long blocks, long threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stream_copy", &stream_copy);
    m.def("stream_triad", &stream_triad);
    m.def("stream_read", &stream_read);
}
"""


def cuda_memory_attrs():
    """Bus width and memory clock straight from the driver, so the spec figure is not a guess."""
    import ctypes
    for so in ("libcudart.so", "/home/sathvik/cuda-12.9/lib64/libcudart.so"):
        try:
            lib = ctypes.CDLL(so)
            break
        except OSError:
            lib = None
    if lib is None:
        return None, None
    out = []
    for attr in (37, 36):  # GlobalMemoryBusWidth, MemoryClockRate
        v = ctypes.c_int()
        if lib.cudaDeviceGetAttribute(ctypes.byref(v), attr, 0) != 0:
            return None, None
        out.append(v.value)
    return out[0], out[1]


def l2_bytes(props):
    return getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", 0)


def time_kernel(fn, iters, warmup):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1e3)
    return times


def bw_stats(times, traffic_bytes):
    gbs = sorted(traffic_bytes / t / 1e9 for t in times)
    q = common.quantiles(gbs, (0.5, 0.95))
    return {
        "iters": len(gbs),
        "median_GBs": round(q["p50"], 2),
        "p95_GBs": round(q["p95"], 2),
        "max_GBs": round(gbs[-1], 2),
        "min_GBs": round(gbs[0], 2),
        "cv_pct": round(100 * common.sample_std(gbs) / common.mean(gbs), 3) if len(gbs) > 1 else None,
        "traffic_bytes": traffic_bytes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-gib", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(common.PILOT_DIR, "p4_hbm_bandwidth.json"))
    a = ap.parse_args()

    import torch
    from torch.utils.cpp_extension import load_inline

    props = torch.cuda.get_device_properties(0)
    n_elem = int(a.array_gib * (2 ** 30) // 4)
    n_elem -= n_elem % 4
    array_bytes = n_elem * 4
    l2 = l2_bytes(props)

    ext = load_inline(
        name="pilot_stream", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )

    dev = torch.device("cuda:0")
    A = torch.ones(n_elem, dtype=torch.float32, device=dev)
    B = torch.full((n_elem,), 2.0, dtype=torch.float32, device=dev)
    C = torch.full((n_elem,), 3.0, dtype=torch.float32, device=dev)
    sink = torch.zeros(1, dtype=torch.float32, device=dev)
    torch.cuda.synchronize()

    threads = 256
    blocks = min(65535, props.multi_processor_count * 32)

    kernels = {
        "copy_read_plus_write": (lambda: ext.stream_copy(B, A, blocks, threads), 2 * array_bytes),
        "triad_2read_1write": (lambda: ext.stream_triad(A, B, C, 1.5, blocks, threads), 3 * array_bytes),
        "read_only": (lambda: ext.stream_read(B, sink, blocks, threads), array_bytes),
    }

    results = {}
    for name, (fn, traffic) in kernels.items():
        results[name] = bw_stats(time_kernel(fn, a.iters, a.warmup), traffic)

    # correctness of the streaming kernels, so a broken launch cannot masquerade as bandwidth
    ext.stream_copy(B, A, blocks, threads)
    torch.cuda.synchronize()
    copy_ok = bool(torch.all(A == 2.0).item())
    ext.stream_triad(A, B, C, 1.5, blocks, threads)
    torch.cuda.synchronize()
    triad_ok = bool(torch.allclose(A, torch.full_like(A, 2.0 + 1.5 * 3.0)))

    best = max(r["median_GBs"] for r in results.values())
    bus_bits, mem_clock_khz = cuda_memory_attrs()
    # DDR double-pumped, per the CUDA memoryClockRate convention; not a datasheet quote.
    spec = (mem_clock_khz * 1e3 * (bus_bits / 8) * 2 / 1e9) if (bus_bits and mem_clock_khz) else None

    rec = {
        "job": "P4",
        "purpose": "independent achievable HBM bandwidth; NOT derived from decode throughput",
        "measurement_method": "hand-written CUDA float4 streaming kernels, CUDA-event timed",
        "gpu": common.gpu_identity(),
        "software": common.software_identity(),
        "device_properties": {
            "name": props.name,
            "sm_count": props.multi_processor_count,
            "l2_cache_bytes": l2,
            "total_memory_bytes": props.total_memory,
            "compute_capability": f"{props.major}.{props.minor}",
        },
        "working_set": {
            "array_bytes": array_bytes,
            "arrays_allocated": 3,
            "array_over_l2_ratio": round(array_bytes / l2, 1) if l2 else None,
            "exceeds_l2": bool(l2 and array_bytes > 8 * l2),
        },
        "launch": {"blocks": blocks, "threads": threads, "warmup_iters": a.warmup},
        "kernel_correctness": {"copy": copy_ok, "triad": triad_ok},
        "results": results,
        "achieved_best_median_GBs": round(best, 2),
        "spec_bandwidth_GBs": round(spec, 1) if spec else None,
        "spec_bandwidth_source": ("derived from CUDA device attributes: bus width x memory clock x 2 "
                                  "(DDR). Not a vendor datasheet quote."),
        "memory_clock_khz": mem_clock_khz,
        "bus_width_bits": bus_bits,
        "achieved_over_spec": {
            name: round(r["median_GBs"] / spec, 3) for name, r in results.items()
        } if spec else None,
        "is_pcie_measurement": False,
        "telemetry_after": common.gpu_telemetry(),
        "timestamp": common.now_iso(),
        "not_a_citable_result": True,
    }
    common.write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in
                      ("results", "achieved_best_median_GBs", "spec_bandwidth_GBs",
                       "achieved_over_spec", "kernel_correctness", "working_set")},
                     indent=2))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
