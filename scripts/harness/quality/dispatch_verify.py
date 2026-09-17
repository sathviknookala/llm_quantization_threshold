"""Positive dispatch evidence for the quality arm. Additive; mutates nothing.

`qengine.observed_identity` and `server.parse_log` both decide dispatch from lines pre-filtered by
`server.KERNEL_PATTERNS`. Two consequences, both found by the harness audit and both live for P13:

  * BF16 names no expected pattern and logs no line matching any of them, so its verdict is
    `ok` by SILENCE -- there is no positive evidence of what the reference configuration
    dispatched, and every KL number is anchored on it.
  * FP4's forbidden list contains `"emulation"`, which is not in `KERNEL_PATTERNS`, so the string
    can never reach the blob the forbidden scan searches. That pattern is unenforceable.

Neither is fixable where it sits. `common.CONFIGS`'s kernel patterns are hashed into `KL_SPEC`, so
editing them changes `spec_hash()` and strands every committed artifact behind the manifest guard;
`server.py` is the frozen serving path. This verifier is therefore a separate, additive check that
scans the WHOLE engine log, requires positive evidence per configuration, and writes its verdict
into the collection artifact beside the inherited one. It never relaxes the inherited verdict: both
must pass.

The evidence lines are copied into the tracked artifact because `results/quality/**/logs/` is
gitignored -- without that the proof would not survive the run that produced it.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness import common  # noqa: E402
from harness.quality import qcommon as q, qengine as E  # noqa: E402

# Verified present in every tracked engine log of the relevant configuration, on both the serving
# path (results/sweep/server_logs/) and the quality path (results/quality/*/logs/).
# FLASHINFER_CUTLASS is anchored to its NvFp4LinearBackend prefix: every engine, BF16 included,
# logs the attention-backend enumeration ['FLASH_ATTN', 'FLASHINFER', ...], and a future backend
# enum named FLASHINFER_CUTLASS in that list would otherwise fail every BF16 collection.
QUANT_KERNELS = (r"CutlassFP8ScaledMMLinearKernel", r"NvFp4LinearBackend",
                 r"NvFp4LinearBackend\.FLASHINFER_CUTLASS", r"Marlin", r"CT_EMULATIONS")

EVIDENCE = {
    "BF16_REFERENCE": {
        "what_positive_evidence_means_here": (
            "an unquantized configuration dispatches its GEMMs through torch.compile/inductor and "
            "logs no vLLM quantization-kernel line at all, so a kernel-class match is not "
            "available as evidence and its absence is not evidence either. What IS positive and "
            "checkable: the engine resolved NO quantization method, loaded bf16 weights, chose an "
            "attention backend, and compiled the graph."),
        "required": {
            "quantization_method_is_none": r"quantization=None",
            "weights_are_bfloat16": r"dtype=torch\.bfloat16",
            "attention_backend_chosen": r"Using [A-Z_]+ attention backend",
            "torch_compile_ran": r"(Dynamo bytecode transform time|"
                                 r"Directly load the compiled graph)",
        },
        "forbidden": QUANT_KERNELS,
    },
    "FP8_PRIMARY": {
        "required": {
            "quantization_method_resolved": r"quantization=compressed-tensors",
            "fp8_gemm_kernel": r"CutlassFP8ScaledMMLinearKernel",
            "attention_backend_chosen": r"Using [A-Z_]+ attention backend",
        },
        "forbidden": (r"Marlin", r"CT_EMULATIONS", r"emulation"),
    },
    "FP4_PRIMARY": {
        "required": {
            "quantization_method_resolved": r"quantization=compressed-tensors",
            "fp4_backend": r"NvFp4LinearBackend",
            "fp4_gemm_kernel": r"FLASHINFER_CUTLASS",
            "attention_backend_chosen": r"Using [A-Z_]+ attention backend",
        },
        # "emulation" is the pattern the inherited verdict cannot enforce; it is enforced here by
        # scanning the whole log instead of the KERNEL_PATTERNS-filtered subset
        "forbidden": (r"Marlin", r"CT_EMULATIONS", r"emulation"),
    },
}

# Forbidden patterns are matched case-insensitively so a differently-cased fallback banner cannot
# slip past; required patterns are not, because they name exact vLLM symbols.
FORBIDDEN_FLAGS = re.IGNORECASE


EVIDENCE_WINDOW = 160


def _lines_matching(log_text, pattern, flags=0, limit=4, window=EVIDENCE_WINDOW):
    """Matched lines, quoted as a window CENTRED on the match.

    vLLM's config dump is one ~3.5 kB line and `quantization=None` sits about 1.1 kB into it, so
    truncating from the start stored a "proof" that did not contain the thing it proved. The logs
    are gitignored, which makes the stored quote the only durable record.
    """
    rx = re.compile(pattern, flags)
    out = []
    for line in log_text.splitlines():
        m = rx.search(line)
        if not m:
            continue
        stripped = E._strip_log_prefix(line.strip())
        m2 = rx.search(stripped) or m
        lo, hi = max(0, m2.start() - window), min(len(stripped), m2.end() + window)
        out.append(("..." if lo else "") + stripped[lo:hi] + ("..." if hi < len(stripped) else ""))
        if len(out) >= limit:
            break
    return out


def verify(log_text, config_id):
    """Positive evidence verdict for one engine log. Pure; reads nothing from disk."""
    if config_id not in EVIDENCE:
        raise SystemExit(f"ABORT: no positive-evidence specification for {config_id!r}; "
                         f"known: {sorted(EVIDENCE)}")
    spec = EVIDENCE[config_id]
    required, missing = {}, []
    for name, pattern in spec["required"].items():
        hits = _lines_matching(log_text, pattern)
        required[name] = {"pattern": pattern, "present": bool(hits), "evidence": hits}
        if not hits:
            missing.append(name)
    forbidden = {}
    for pattern in spec["forbidden"]:
        hits = _lines_matching(log_text, pattern, flags=FORBIDDEN_FLAGS)
        if hits:
            forbidden[pattern] = hits
    return {
        "verifier": "quality.dispatch_verify",
        "config_id": config_id,
        "scope": "the whole engine log, not the KERNEL_PATTERNS-filtered subset",
        "supplements_not_replaces": "qengine.observed_identity.dispatch_verdict; both must pass",
        "required_evidence": required,
        "required_missing": missing,
        "forbidden_present": forbidden,
        "evidence_window_chars": EVIDENCE_WINDOW,
        "evidence_contains_its_match": all(
            bool(re.search(v["pattern"], (v["evidence"] or [""])[0])) for v in required.values()),
        "ok": not missing and not forbidden,
        "note": spec.get("what_positive_evidence_means_here"),
    }


def verify_log_file(path, config_id):
    return verify(open(path, errors="replace").read(), config_id)


def collection_log_path(root, config_id):
    short = q.QUALITY_CONFIGS[config_id]["short"]
    return os.path.join(root, "logs", f"collect_{short}.log")


def require(log_text, config_id):
    rec = verify(log_text, config_id)
    if not rec["ok"]:
        raise SystemExit(
            f"ABORT: positive dispatch evidence failed for {config_id}: "
            f"missing={rec['required_missing']} forbidden={sorted(rec['forbidden_present'])}. "
            "The inherited verdict can be satisfied by silence; this one cannot.")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="", help="engine log to verify")
    ap.add_argument("--config", default="", help="configuration id the log belongs to")
    ap.add_argument("--root", default="", help="collection root; verifies every ladder config")
    a = ap.parse_args()
    out, bad = {}, []
    if a.root:
        for cfg in q.LADDER:
            path = collection_log_path(os.path.join(common.REPO, a.root), cfg)
            out[cfg] = (verify_log_file(path, cfg) if os.path.exists(path)
                        else {"ok": False, "absent": path})
    elif a.log and a.config:
        out[a.config] = verify_log_file(a.log, a.config)
    else:
        raise SystemExit("give --root, or --log with --config")
    for cfg, rec in out.items():
        bad += [] if rec.get("ok") else [cfg]
        print(f"  {'ok  ' if rec.get('ok') else 'FAIL'} {cfg}")
        for name, ev in (rec.get("required_evidence") or {}).items():
            mark = "+" if ev["present"] else "!"
            print(f"      {mark} {name}: {(ev['evidence'] or ['MISSING'])[0][:120]}")
        for pat, hits in (rec.get("forbidden_present") or {}).items():
            print(f"      ! FORBIDDEN {pat}: {hits[0][:120]}")
    print(json.dumps({c: r.get("ok") for c, r in out.items()}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
