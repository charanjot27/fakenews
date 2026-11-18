"""Phase 5 metrics: every number in the project is computed here from (true labels, P(fake))."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except ValueError:  # e.g. only one class present in a subset
        return float("nan")


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0))


def ece(y, p, n_bins=10) -> float:
    """Expected calibration error on P(fake): weighted gap between confidence and accuracy per bin."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def calibration_curve(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "mean_pred": float(p[m].mean()), "frac_fake": float(y[m].mean()), "n": int(m.sum())})
    return rows


def bootstrap_ci(y, pred, metric=macro_f1, n=1000, ci=0.95, seed=42):
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals[i] = metric(y[idx], pred[idx])
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def compute_metrics(y, p, threshold=0.5, n_boot=1000, ci=0.95, ece_bins=10, seed=42) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    lo, hi = bootstrap_ci(y, pred, n=n_boot, ci=ci, seed=seed) if n_boot else (float("nan"), float("nan"))
    return {
        "n": int(len(y)), "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_fake": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_real": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "macro_f1": macro_f1(y, pred),
        "macro_f1_ci_low": lo, "macro_f1_ci_high": hi,
        "roc_auc": _safe(roc_auc_score, y, p), "pr_auc": _safe(average_precision_score, y, p),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float(np.mean((p - y) ** 2)),
        "ece": ece(y, p, ece_bins),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def mcnemar(y, pred_a, pred_b) -> dict:
    """Do two models make significantly different numbers of errors on the same items?
    b = A right & B wrong, c = A wrong & B right. Exact binomial test when b + c < 25, else chi-square with
    continuity correction."""
    y, a, b_ = map(np.asarray, (y, pred_a, pred_b))
    ra, rb = a == y, b_ == y
    b = int((ra & ~rb).sum())
    c = int((~ra & rb).sum())
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "test": "none (identical errors)"}
    if b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(b), "p_value": float(p), "test": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(stats.chi2.sf(stat, 1)),
            "test": "chi-square (continuity corrected)"}


def paired_bootstrap_diff(y, pred_a, pred_b, metric=macro_f1, n=1000, seed=42, ci=0.95):
    """metric(A) - metric(B) on the same resampled items. Returns diff, CI and a two-sided p-value."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    obs = metric(y, a) - metric(y, b)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    # two-sided p: how often the bootstrap distribution crosses zero
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(1.0, max(p, 1 / n)))}


def subset_metrics(y, p, threshold=0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    return {"n": int(len(y)), "macro_f1": macro_f1(y, pred) if len(y) else float("nan"),
            "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
            "recall_fake": float(recall_score(y, pred, pos_label=1, zero_division=0)) if (y == 1).any() else float("nan"),
            "recall_real": float(recall_score(y, pred, pos_label=0, zero_division=0)) if (y == 0).any() else float("nan")}
