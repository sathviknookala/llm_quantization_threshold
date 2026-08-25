# Hardware Profile

## Purpose

This document records the measured deployment machine and the constraints it places on the project. Hardware is fixed context/control for the primary study, not an experimental variable.

Profile captured: **2026-08-22**.

## GPU

```text
GPU:                  NVIDIA RTX PRO 4000 Blackwell
Architecture:         Blackwell
GPU count:            1
Compute capability:   12.0 / SM120
VRAM:                 24,467 MiB reported by nvidia-smi
PyTorch total memory: ~25.15 GB decimal
Power limit:          145 W current / requested / default / maximum
Driver:               575.64.03
Display attached:     no
MIG:                  N/A
```

### PCIe

```text
Device maximum generation: 5
Host maximum generation:   4
Reported link width:        x16
Idle current generation:    1
```

The captured profile was taken while the GPU was essentially idle:

```text
Performance state: P8
GPU utilization:   0%
Power:             ~5–6 W
SM clock:          23 MHz
Memory clock:      405 MHz
```

Therefore the Gen1 reading should be treated as an **idle power-management state**, not the final benchmark link state. Verify negotiated PCIe generation again under sustained load before final measurements.

### Memory bandwidth — MEASURED 2026-08-23 (pilot P4)

Phase 1 is a memory-focused study (D10), which makes memory bandwidth the governing hardware
constant for the entire below-the-wall region of the concurrency sweep. It is now recorded.

```text
Memory bus width:          192 bit          (CUDA device attribute)
Memory clock:              14.001 GHz       (CUDA device attribute)
Spec bandwidth:            672.0 GB/s       derived: 192/8 x 14.001e9 x 2 (DDR)
L2 cache:                  48 MiB           (50,331,648 B)

Achieved, measured on this card:
  read-only                620.1 GB/s       92.3% of spec
  triad  (2 read, 1 write) 564.7 GB/s       84.0% of spec
  copy   (1 read, 1 write) 545.8 GB/s       81.2% of spec
```

Artifact: `results/pilot/p4_hbm_bandwidth.json`. Reproduce with
`python scripts/pilot/hbm_bandwidth.py`.

**Method, and why it is independent.** Hand-written CUDA `float4` streaming kernels, CUDA-event
timed, 20 warmup plus 100 timed iterations, 1 GiB per array against a 48 MiB L2 — a 21.3x
over-L2 working set, so this is HBM traffic and not a cache measurement. Device memory only; no
host-device transfer is involved, so it is not a PCIe figure. Both kernels are validated for
numerical correctness before timing, so a mis-launched kernel cannot masquerade as bandwidth.
Run-to-run CV is about 0.1%, and two independent invocations agreed to 0.03%.

The spec figure is **derived from driver-reported device attributes**, not quoted from a vendor
datasheet.

**Deliberately not derived from decode throughput.** The pilot's P1 job tests whether decode
throughput is explained by weight bytes divided by bandwidth. Obtaining the bandwidth from decode
throughput and then using it to validate a bandwidth-derived throughput prediction would be
circular, so P4 is a standalone microbenchmark.

**What this does to the qualification estimate.** D10 derived roughly 620-660 GB/s for all three
configurations from a two-point prefill subtraction over the coarse smoke artifacts. The
read-dominated achievable figure is 620.1 GB/s, so the FP4 derived value (618 GB/s) sits essentially
at the ceiling while the BF16 (649) and FP8 (659) derived values sit slightly *above* it. Those two
are therefore mildly optimistic, which is consistent with the stated caveat that the derivation
assumes decode cost is identical at 2k and 16k context. The useful conclusion is that batch-1 decode
on this card runs close to the achievable read bandwidth. That remains a derived statement about the
qualification artifacts, not a measured serving result.

## GPU clocks / power state

```text
Applications graphics clock:          2055 MHz
Default applications graphics clock:  2055 MHz
Applications memory clock:            14001 MHz
Default applications memory clock:    14001 MHz
Maximum graphics/SM clock reported:   3090 MHz
```

No evidence from the captured profile indicates a user-reduced power limit. The card is already configured at its reported 145 W maximum power limit.

Final benchmark runs should record power, clocks, temperature, utilization, and memory use so that thermal or power-state differences can be detected rather than silently folded into latency variance.

## Software

Project environments measured **2026-08-22**. Artifacts: `results/system/env_qnt_2026-08-22.json`,
`results/system/env_qnt-quant_2026-08-22.json`. Reproduce with `scripts/probe_env.py`.

```text
NVIDIA driver:        575.64.03
nvidia-smi CUDA:      12.9   (driver-exposed compatibility ceiling)
Local CUDA toolkit:   12.9   (nvcc V12.9.86)
```

Two conda environments, deliberately split (see "Environment split" below):

```text
conda env `qnt`         — serving / measurement
  Python                3.12.13
  PyTorch               2.10.0+cu128   (torch.version.cuda = 12.8)
  vLLM                  0.19.1
  compressed-tensors    0.15.0.1
  transformers          5.15.1

conda env `qnt-quant`   — offline checkpoint production
  Python                3.12.13
  PyTorch               2.10.0+cu128
  llmcompressor         0.10.0.3
  compressed-tensors    0.14.0.1
  transformers          4.57.6
```

These CUDA fields describe different layers:

- `nvidia-smi` reports the CUDA compatibility level exposed by the installed driver;
- `nvcc --version` reports the locally installed CUDA toolkit;
- `torch.version.cuda` reports the CUDA version against which the PyTorch build was compiled.

They are not expected to be numerically identical.

### Driver ceiling constrains the serving stack — measured 2026-08-22

Driver 575.64.03 exposes CUDA 12.9. PyTorch wheels built against **CUDA 13 do not run on this
driver**; `torch.cuda.is_available()` returns `False` with
`"The NVIDIA driver on your system is too old (found version 12090)"`.

This propagates into a hard version ceiling on the serving stack:

```text
torch 2.9.x / 2.10.0   default PyPI wheel = cu128   -> runs
torch 2.11.0+          default PyPI wheel = cu13x   -> does NOT run
vLLM <= 0.19.1         pins torch <= 2.10.0         -> installable
vLLM >= 0.20.0         pins torch >= 2.11.0         -> not installable without a driver upgrade
```

vLLM 0.19.1 is therefore the newest release usable on this machine as configured. Verified failure
mode: installing vLLM 0.27.1 pulled `torch 2.13.0+cu130` and CUDA was unavailable.

Raising this ceiling requires upgrading the NVIDIA driver to r580+ (apt offers 580/590/595/610).
That is a system-level change which would also invalidate the driver identity recorded in this
profile, so it must be a tracked decision, not an incidental step.

### Environment split

`vllm==0.19.1` pins `compressed-tensors==0.17.0`-era exact versions that conflict irreconcilably
with every `llmcompressor` release (exact `==` pins on both sides, plus opposing `transformers`
bounds). Quantization is an offline checkpoint-production step and does not need to share a process
with serving, so the two live in separate environments.

`qnt-quant` writes with compressed-tensors 0.14.0.1 and `qnt` reads with 0.15.0.1 — writer older
than reader, which is the supported compatibility direction. **This remains an assumption until a
checkpoint produced by `qnt-quant` is actually loaded by `qnt`.**

## Host

```text
CPU:                 AMD Ryzen 9 7950X 16-Core Processor
Physical cores:      16
Logical threads:     32
Sockets:             1
NUMA nodes:          1
System RAM:          124 GiB
Swap:                2 GiB
OS:                  Ubuntu 22.04 family
Kernel:              6.8.0-65-generic
Architecture:        x86_64
```

This host is expected to be sufficient for model loading, tokenization, request generation, calibration/evaluation preprocessing, and single-GPU serving. That remains an assumption until CPU utilization and request-generation behavior are observed under saturation tests.

## Experimental implications

### 1. BF16 fit constrains the largest comparable model

BF16 is the common high-precision experimental reference. Every primary model must therefore fit and serve correctly in BF16 before its lower-precision versions are useful for same-model comparisons.

Approximate weights-only memory:

```text
1B parameters @ BF16  ~2 GB
3B parameters @ BF16  ~6 GB
7B parameters @ BF16 ~14 GB
8B parameters @ BF16 ~16 GB
```

Actual serving memory is larger than weight storage because the process also needs KV cache, runtime workspaces, allocator overhead, scheduler state, and other backend allocations.

This makes roughly the 7–8B region a plausible upper end for a comfortable BF16 single-GPU study, but **the exact model must be validated empirically** rather than selected from the weights-only estimate.

### 2. Single-GPU execution simplifies attribution

The primary study has no tensor-parallel communication, NVLink/NVSwitch behavior, or multi-GPU scheduler effects. This makes serving differences easier to attribute to deployment configuration and workload.

### 3. Quantization can create capacity value, not only speed value

Lower model-resident memory can leave more VRAM for KV cache and therefore increase sustainable concurrency or context capacity even when token/sec gains are modest.

The serving side of the project should therefore treat at least these as separate outcomes:

```text
capacity effect     fewer weight bytes resident -> more KV  -> more concurrent sequences
throughput effect   fewer weight bytes to move per decode step -> potential faster per-token
```

**Corrected 2026-08-24 after pilot P1.** The capacity effect follows from checkpoint size and was
confirmed directly (BF16 KV wall at [17, 18]). The throughput effect is only a potential: decode also
moves KV and other traffic and is subject to kernel efficiency and scheduling, so the compression
ratio does not predict the speedup — the pilot measured 2.44x for FP4 at concurrency 1 against a
predicted 2.65x, falling to 2.00x by concurrency 12. Its magnitude is an output of the serving sweep,
and the study does not attribute it to a single mechanism. The arithmetic path is observed only by
`PREFILL_PROBE`.

### 3b. The power cap is continuously active — measured 2026-08-25

The sweep records `clocks_throttle_reasons.sw_power_cap` per cell. It reads **1.0 for entire timed
windows** on every configuration: the card is power-limited for the whole measurement, not
intermittently. `hw_slowdown` and `sw_thermal_slowdown` stayed at 0.0. So SM clock is set by the
operating point under a 145 W ceiling rather than by thermal headroom, which is why matched cells run
hours apart agree to within 0.76% on clock (H6 drift test, `results/sweep/`). Every serving number in
this project is measured under that ceiling.

### 4. Hardware capability is not equivalent to backend capability

A low-precision format enters the locked study only after the chosen inference stack is verified to expose an appropriate path on this GPU. Do not infer practical support solely from the GPU architecture.

### 5. The backend reports CUTLASS FP8 and FP4 support for SM120 — measured 2026-08-22

vLLM 0.19.1 compiled kernel-capability queries, run on this GPU
(artifact: `results/system/env_qnt_2026-08-22.json`):

```text
cutlass_scaled_mm_supports_fp8(120) = True
cutlass_scaled_mm_supports_fp4(120) = True
cutlass_group_gemm_supported(120)   = False
platform fp8 dtype                  = torch.float8_e4m3fn
```

This means the FP8 and FP4 CUTLASS GEMM kernels were **built for SM120** in this wheel, so the
ladder does not require a driver upgrade to reach a native low-precision path.

Scope of the claim — this is a **compiled-capability** query, not proof of dispatch. It does not
show that a given checkpoint actually routes through those kernels. vLLM also ships an NVFP4
emulation path (`VLLM_USE_NVFP4_CT_EMULATIONS`) which dequantizes and computes in higher precision;
a configuration running under emulation would still "work" while producing meaningless serving
numbers. Per-configuration dispatch evidence is still required before any FP8/FP4 row is treated as
natively accelerated.


## Still to verify

Resolved 2026-08-22: local CUDA toolkit (`nvcc` V12.9.86); inference backend and version
(vLLM 0.19.1); SM120 CUTLASS FP8/FP4 capability (see implication 5 below).

Outstanding:

- exact FP8 deployment recipe and the kernel actually dispatched at runtime;
- exact FP4 deployment recipe and the kernel actually dispatched at runtime;
- low-precision KV-cache policy;
- that a `qnt-quant` checkpoint loads in `qnt` (cross-environment compressed-tensors compatibility);
- negotiated PCIe generation under sustained load;
- whether host-side request generation becomes visible at high concurrency;
- sustained thermals / clocks / power behavior during long benchmark runs.

## Source artifact

The raw machine-profile output should be preserved under `results/system/` in the repository, for example:

```text
results/system/profile_2026-08-22.txt
```

Quantitative claims in this document should be checked against that artifact if the machine or software stack changes.
