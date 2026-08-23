# Quantization Deployment Configurations

## Purpose

This document is the reproducibility contract for every precision configuration in the study.

A row is not complete until another session can reconstruct or load the same deployment and identify the actual serving path used on the target GPU.

Do not use a shorthand label such as `FP8` or `FP4` in final results without mapping it to a configuration ID defined here.

## Required fields

Every configuration must record:

```text
Configuration ID
Base checkpoint
Tokenizer / revision
Weight precision
Activation precision
KV-cache precision
Quantization format / algorithm
Scale representation
Group / block size
Calibration required? yes/no
Calibration dataset
Calibration preprocessing
Calibration sample count
Quantization tool + version
Inference backend + version
Backend flags / scheduler config
Kernel / execution path
Native/efficient on target GPU? evidence
Checkpoint / artifact path
Creation command
Serving command
Validation artifact
Notes / known caveats
```

---

# Base model — shared by every configuration

**Status:** QUALIFIED 2026-08-22 (see `results/qualification/`)

```text
Model:                  Llama 3.1 8B Instruct
Parameters:             8,030,261,248  (measured from safetensors)
  non-embedding:        6.98 B
  embedding + lm_head:  1.05 B
Source dtype:           BF16 (all tensors; measured)
Weights on disk:        14.958 GiB / 16,060,522,496 bytes
Architecture:           LlamaForCausalLM, 32 layers, hidden 4096, intermediate 14336
Attention:              32 heads / 8 KV heads (GQA), head_dim 128
Vocab:                  128,256
max_position_embeddings:131,072
RoPE:                   llama3, theta 500000.0, factor 8.0
```

## Checkpoint provenance — read this before citing the model ID

The official repo `meta-llama/Llama-3.1-8B-Instruct` is **gated** and no accepted license was
available on this machine at qualification time (HTTP 401). The license request is pending.

Weights were therefore obtained from an ungated mirror and **verified byte-identical**:

```text
Serving path used:  NousResearch/Meta-Llama-3.1-8B-Instruct
Revision:           d10aef7999a2b5ba950ab3974312feeedbfe0b77
Official reference: meta-llama/Llama-3.1-8B-Instruct
Official revision:  0e9e39f249a16976918f6564b8830bc894c89659
```

All four weight shards match the official SHA256 checksums, verified against hub metadata **and**
recomputed locally on the downloaded files:

```text
model-00001-of-00004.safetensors  2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668
model-00002-of-00004.safetensors  09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15
model-00003-of-00004.safetensors  fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa
model-00004-of-00004.safetensors  92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b
```

`config.json`, `generation_config.json`, `model.safetensors.index.json`, `tokenizer.json` and
`special_tokens_map.json` also match the official byte sizes.

**Known deviation — `tokenizer_config.json`.** Mirror 50,870 B vs official 55,351 B. The difference
is the `chat_template` (mirror template is 348 chars; the official Llama 3.1 template is several KB
with tool-calling and date handling). `tokenizer.json` is identical, so **BPE tokenization is
unaffected**.

Consequences, stated precisely:

- within-model BF16 vs FP8 vs FP4 comparability is **unaffected** — all three rungs share one
  tokenizer, and the quality rig feeds raw token IDs;
- any *instruct/chat-formatted* downstream task would use a non-official template, so absolute
  scores are not comparable to published Llama 3.1 numbers until the official
  `tokenizer_config.json` is in place.

**Action when the license is approved:** re-pin to `meta-llama/Llama-3.1-8B-Instruct` at revision
`0e9e39f2...`. Because the weights are provably identical, no serving or KL measurement taken from
the mirror is invalidated; only the chat-template provenance changes.

---

# BF16_REFERENCE

**Status:** QUALIFIED — serving validated 2026-08-22

```text
Configuration ID:       BF16_REFERENCE
Base checkpoint:        NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef7999a2
Tokenizer / revision:   same repo/revision as weights
Weight precision:       BF16
Activation precision:   BF16
KV-cache precision:     BF16  (vLLM kv_cache_dtype=auto -> model dtype)
Quantization algorithm: none
Calibration:            none
Inference backend:      vLLM 0.19.1 (conda env `qnt`)
Kernel / execution path: default BF16 GEMM path
Measured weights resident: 14.99 GiB
Measured KV cache:      4.84 GiB = 39,664 tokens
Peak process VRAM:      22,845 MiB / 24,467 MiB
Artifact:               results/qualification/bf16_smoke.json
```

Serving command:

```bash
python scripts/qualify_serving.py \
  --model ~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77 \
  --label BF16_REFERENCE --max-model-len 32768 --gpu-util 0.90 \
  --out results/qualification/bf16_smoke.json
```

**Reference-integrity check.** The engine log reports no low-precision kernel selection for this
configuration, so the "BF16 reference" label is not silently masking a quantized path.

---

# FP8_PRIMARY

**Status:** QUALIFIED — native execution confirmed 2026-08-22

```text
Configuration ID:       FP8_PRIMARY
Base checkpoint:        same as BF16_REFERENCE (identical source)
Weight precision:       FP8 E4M3, 8-bit float, static, per-channel symmetric
Activation precision:   FP8 8-bit float, dynamic, per-token symmetric
KV-cache precision:     BF16  (kv_cache_scheme: null — deliberately unchanged)
Quantization format:    compressed-tensors "float-quantized"
Group / block size:     none (channel strategy)
Calibration required:   NO — weights use memoryless_minmax, activations dynamic
Quantization tool:      llmcompressor 0.10.0.3 (conda env `qnt-quant`)
compressed-tensors:     written 0.14.0.1 / read 0.15.0.1
Ignored modules:        lm_head
Inference backend:      vLLM 0.19.1
Kernel / execution path: CutlassFP8ScaledMMLinearKernel  (CompressedTensorsW8A8Fp8)
Checkpoint size:        8.5 GB
Measured weights resident: 8.49 GiB
Measured KV cache:      11.31 GiB = 92,608 tokens
Peak process VRAM:      22,739 MiB / 24,467 MiB
Quantization time:      22.8 s
Artifact:               results/qualification/fp8_smoke.json
```

Creation command:

```bash
python scripts/quantize.py \
  --model ~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77 \
  --scheme FP8_DYNAMIC --out checkpoints/llama31-8b-instruct-FP8_DYNAMIC
```

**Native-execution evidence.** Engine log line:

```text
Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
```

This is an explicit CUTLASS FP8 scaled-MM kernel selection, not a Marlin/dequantize-to-BF16
fallback. Corroborated by `cutlass_scaled_mm_supports_fp8(120) = True`.

---

# FP4_PRIMARY

**Status:** QUALIFIED — native SM120 execution confirmed 2026-08-22

```text
Configuration ID:       FP4_PRIMARY
Base checkpoint:        same as BF16_REFERENCE (identical source)
Weight precision:       NVFP4, 4-bit float, static, tensor_group, group size 16, symmetric
Activation precision:   NVFP4, 4-bit float, dynamic ("local"), tensor_group, group size 16
KV-cache precision:     BF16  (kv_cache_scheme: null — deliberately unchanged)
Quantization format:    compressed-tensors "nvfp4-pack-quantized"
Group / block size:     16
Calibration required:   YES (activation scales; W4A4)
Calibration dataset:    HuggingFaceH4/ultrachat_200k, split train_sft
Calibration preprocessing: chat template applied, truncated to 2048 tokens, no special tokens
Calibration samples:    128
Calibration seed:       0 (shuffle seed)
Quantization tool:      llmcompressor 0.10.0.3 (conda env `qnt-quant`)
Ignored modules:        lm_head
Inference backend:      vLLM 0.19.1
Kernel / execution path: NvFp4LinearBackend.FLASHINFER_CUTLASS
                         -> flashinfer gen_gemm_sm120_module_cutlass_fp4 (JIT built)
Checkpoint size:        5.7 GB
Measured weights resident: 5.65 GiB
Measured KV cache:      14.57 GiB = 119,360 tokens
Peak process VRAM:      23,353 MiB / 24,467 MiB
Quantization time:      390.1 s
Artifact:               results/qualification/nvfp4_smoke.json
```

Creation command:

```bash
python scripts/quantize.py \
  --model ~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77 \
  --scheme NVFP4 --out checkpoints/llama31-8b-instruct-NVFP4 \
  --calib-samples 128 --calib-seqlen 2048 --seed 0
```

**Native-execution evidence.** Engine log line:

```text
Using NvFp4LinearBackend.FLASHINFER_CUTLASS for NVFP4 GEMM
```

The dispatch resolves to `flashinfer.gemm.get_gemm_sm120_module_cutlass_fp4()`, i.e. an
**SM120-specific CUTLASS FP4 GEMM**, JIT-compiled on this machine. This is W4A4: activations are
also FP4, so the GEMM runs on FP4 inputs rather than dequantizing weights to BF16.

`VLLM_USE_NVFP4_CT_EMULATIONS` was confirmed **unset**, so the compressed-tensors NVFP4 emulation
path (which dequantizes and computes in higher precision) was not used.

**Build requirement / gotcha.** The SM120 FP4 kernel is JIT-compiled by flashinfer and needs
`ninja` and `nvcc` **on `PATH` of the serving process**. Launching by absolute interpreter path
without the env `bin` on `PATH` fails with `FileNotFoundError: 'ninja'` at engine init. Export:

```bash
export PATH=/home/sathvik/miniconda3/envs/qnt/bin:/home/sathvik/cuda-12.9/bin:$PATH
export CUDA_HOME=/home/sathvik/cuda-12.9
```

First load pays the JIT cost (~85 s vs ~25 s warm).

---

# KV-cache precision — held constant, deliberately

All three configurations serve with a **BF16 KV cache** (`kv_cache_scheme: null`, vLLM
`kv_cache_dtype=auto`). KV precision is therefore *not* varying with weight precision.

This matters because the measured KV-capacity gains below come purely from weights freeing VRAM,
not from a smaller per-token KV footprint. Per-token KV cost is identical across the ladder:

```text
32 layers x 8 KV heads x 128 head_dim x 2 (K+V) x 2 bytes = 131,072 bytes = 128 KiB / token
```

Confirmed by measurement: 4.84 GiB / 39,664 tokens = 128 KiB/token in every configuration.

A quantized KV cache is a **separate future configuration** (`FP8_KV_VARIANT`), not a silent
variation of these rows.

# Optional secondary configurations

Do not add these until the primary ladder is locked.

Potential examples:

```text
FP4_SECONDARY
GPTQ_INT4
AWQ_INT4
FP8_KV_VARIANT
```

Each secondary configuration must answer a specific question stated in `DECISIONS.md`.

## Validation rule

Before any configuration is benchmarked for serving performance:

1. load it successfully on the target machine;
2. verify output correctness against the intended reference contract;
3. record the exact software/backend versions and launch command;
4. confirm the expected precision path as far as the backend permits;
5. save the validation artifact;
6. only then perform timed runs.
