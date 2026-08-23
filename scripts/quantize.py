import argparse, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--scheme", required=True, choices=["FP8_DYNAMIC", "NVFP4", "NVFP4A16"])
ap.add_argument("--out", required=True)
ap.add_argument("--calib-samples", type=int, default=128)
ap.add_argument("--calib-seqlen", type=int, default=2048)
ap.add_argument("--calib-dataset", default="HuggingFaceH4/ultrachat_200k")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype="auto", device_map=None)

# FP8_DYNAMIC needs no calibration: weights are static per-channel, activations dynamic per-token
needs_calib = a.scheme in ("NVFP4", "NVFP4A16")
ds = None
if needs_calib:
    from datasets import load_dataset
    raw = load_dataset(a.calib_dataset, split="train_sft").shuffle(seed=a.seed).select(range(a.calib_samples))
    def prep(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, padding=False, max_length=a.calib_seqlen, truncation=True, add_special_tokens=False)
    ds = raw.map(prep, remove_columns=raw.column_names)

recipe = QuantizationModifier(targets="Linear", scheme=a.scheme, ignore=["lm_head"])
oneshot(
    model=model, dataset=ds, recipe=recipe,
    max_seq_length=a.calib_seqlen,
    num_calibration_samples=a.calib_samples if needs_calib else None,
)
model.save_pretrained(a.out, save_compressed=True)
tok.save_pretrained(a.out)
meta = {"scheme": a.scheme, "source_model": a.model, "out": a.out,
        "calibration": {"required": needs_calib,
                        "dataset": a.calib_dataset if needs_calib else None,
                        "samples": a.calib_samples if needs_calib else 0,
                        "seqlen": a.calib_seqlen if needs_calib else None,
                        "seed": a.seed},
        "ignore": ["lm_head"], "targets": "Linear",
        "elapsed_seconds": round(time.time() - t0, 1)}
json.dump(meta, open(os.path.join(a.out, "quantization_meta.json"), "w"), indent=2)
print("DONE", json.dumps(meta, indent=2))
