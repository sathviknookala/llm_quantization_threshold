"""KL numerics and trajectory-unit aggregation. Pure; no engine, no GPU.

Everything here is float64 and log-domain. The qualification prototypes added EPS=1e-12 inside both
logarithms, which floors the tail instead of failing on it; here a comparison distribution with no
mass where the reference has mass is a domain error, not a number.
"""

import numpy as np

WORKING_DTYPE = np.float64


class KLDomainError(ValueError):
    pass


def logsumexp(lp):
    lp = np.asarray(lp, dtype=WORKING_DTYPE)
    finite = lp[np.isfinite(lp)]
    if finite.size == 0:
        raise KLDomainError("distribution has no finite logprob entries")
    m = finite.max()
    return m + np.log(np.exp(lp - m).sum())


def log_normalize(lp):
    lp = np.asarray(lp, dtype=WORKING_DTYPE)
    return lp - logsumexp(lp)


def mass_presum(lp):
    """Diagnostic only: stored logprobs do not sum to exactly 1 and are renormalised before use."""
    return float(np.exp(np.asarray(lp, dtype=WORKING_DTYPE)).sum())


def _reject_non_logprob(lp, role):
    a = np.asarray(lp, dtype=WORKING_DTYPE)
    if np.isnan(a).any():
        raise KLDomainError(f"{role} distribution holds NaN; it is not a log-distribution")
    if np.isposinf(a).any():
        raise KLDomainError(f"{role} distribution holds +inf; it is not a log-distribution")


def kl_nats(lp_ref, lp_cmp, negative_tolerance=1e-12):
    """D_KL(ref || cmp) in nats, from unnormalised logprob vectors."""
    _reject_non_logprob(lp_ref, "reference")
    _reject_non_logprob(lp_cmp, "comparison")
    lr = log_normalize(lp_ref)
    lc = log_normalize(lp_cmp)
    if lr.shape != lc.shape:
        raise KLDomainError(f"shape mismatch {lr.shape} vs {lc.shape}")
    p = np.exp(lr)
    support = p > 0.0
    if not support.any():
        # nan > 0.0 is False everywhere, so a corrupted reference would otherwise sum an empty
        # support and return a clean-looking 0.0
        raise KLDomainError("reference distribution has no probability mass anywhere")
    if not np.isfinite(lc[support]).all():
        n = int((~np.isfinite(lc[support])).sum())
        raise KLDomainError(
            f"comparison assigns no mass at {n} token(s) where the reference does; "
            "KL is infinite and must not be floored")
    val = float((p[support] * (lr[support] - lc[support])).sum())
    if val < -negative_tolerance:
        raise KLDomainError(f"KL is negative ({val!r}); normalisation or ordering is wrong")
    return max(val, 0.0)


def entropy_nats(lp):
    lr = log_normalize(lp)
    p = np.exp(lr)
    s = p > 0.0
    return float(-(p[s] * lr[s]).sum())


def top1(lp):
    return int(np.argmax(np.asarray(lp)))


def underflow_margin(lp):
    """Distance from the smallest finite logprob to float32's exp underflow boundary."""
    lp = np.asarray(lp, dtype=WORKING_DTYPE)
    finite = lp[np.isfinite(lp)]
    return float(finite.min()) if finite.size else float("-inf")


def validate_distribution(lp, vocab_size, mass_tolerance):
    lp = np.asarray(lp, dtype=WORKING_DTYPE)
    finite = np.isfinite(lp)
    presum = mass_presum(lp)
    return {
        "entries": int(lp.size),
        "entries_finite": int(finite.sum()),
        "full_vocab": bool(lp.size == vocab_size),
        "all_finite": bool(finite.all()),
        "mass_presum": presum,
        "normalized": bool(abs(presum - 1.0) <= mass_tolerance),
        "min_logprob": underflow_margin(lp),
    }


def trajectory_means(cells):
    """cells: (n_trajectories, n_positions) -> mean within each trajectory."""
    a = np.asarray(cells, dtype=WORKING_DTYPE)
    if a.ndim != 2:
        raise KLDomainError(f"expected a 2-D (trajectory, position) grid, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise KLDomainError("grid holds non-finite KL values; the run is invalid")
    return a.mean(axis=1)


def headline(cells):
    return float(trajectory_means(cells).mean())


def bootstrap_indices(n, draws, seed):
    """One index matrix, shared across positions and pairs so curves and paired differences stay
    jointly consistent."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(draws, n))


def bootstrap_headline(cells, idx):
    k = trajectory_means(cells)
    return k[idx].mean(axis=1)


def bootstrap_positions(cells, idx):
    """Whole position structure travels with a resampled trajectory."""
    a = np.asarray(cells, dtype=WORKING_DTYPE)
    return a[idx].mean(axis=1)


def percentile_ci(draws, ci=(0.025, 0.975)):
    d = np.asarray(draws, dtype=WORKING_DTYPE)
    lo, hi = np.quantile(d, list(ci), method="linear")
    return float(lo), float(hi)


def bootstrap_summary(point, draws, ci=(0.025, 0.975)):
    d = np.asarray(draws, dtype=WORKING_DTYPE)
    lo, hi = percentile_ci(d, ci)
    return {
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "ci_half_width": (hi - lo) / 2.0,
        "bootstrap_mean": float(d.mean()),
        "bias": float(d.mean() - point),
        "std_error": float(d.std(ddof=1)),
        "draws": int(d.size),
    }
