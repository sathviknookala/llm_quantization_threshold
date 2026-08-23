import json, subprocess, sys, torch

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

out = {
    "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    "nvidia_smi_cuda": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
    "nvcc": sh("nvcc --version | tail -2 | head -1"),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    out |= {
        "gpu_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": p.total_memory,
        "total_memory_GiB": round(p.total_memory / 2**30, 3),
        "multi_processor_count": p.multi_processor_count,
    }
try:
    import vllm
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    out |= {
        "vllm": vllm.__version__,
        "cutlass_fp4_sm120": ops.cutlass_scaled_mm_supports_fp4(120),
        "cutlass_fp8_sm120": ops.cutlass_scaled_mm_supports_fp8(120),
        "cutlass_fp4_sm100": ops.cutlass_scaled_mm_supports_fp4(100),
        "fp8_dtype": str(current_platform.fp8_dtype()),
    }
except ImportError as e:
    out["vllm"] = f"absent ({e})"
try:
    import compressed_tensors
    out["compressed_tensors"] = compressed_tensors.__version__
except ImportError:
    out["compressed_tensors"] = "absent"
try:
    import llmcompressor
    from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES
    out["llmcompressor"] = llmcompressor.__version__
    out["preset_schemes"] = sorted(PRESET_SCHEMES.keys())
except ImportError:
    out["llmcompressor"] = "absent"
import transformers
out["transformers"] = transformers.__version__
print(json.dumps(out, indent=2))
