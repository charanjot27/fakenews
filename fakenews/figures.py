"""All saved charts (matplotlib, PNG). Every chart has a title, axis labels and a legend where needed.
Colours: fake = red, real = green; statistical models = blues, transformers = oranges/purples."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

FAKE, REAL = "#d62728", "#2ca02c"
STAT_COLORS = ["#08519c", "#3182bd", "#6baed6", "#2171b5", "#4292c6", "#9ecae1"]
TRANS_COLORS = ["#e6550d", "#fd8d3c", "#756bb1", "#9e9ac8", "#a63603", "#d94801", "#54278f", "#fdae6b", "#bcbddc",
                "#cc4c02", "#8c6bb1"]
ENS_COLOR = "#636363"

import warnings  # noqa: E402

from matplotlib import font_manager  # noqa: E402

warnings.filterwarnings("ignore", message="Glyph .* missing from font")
# Devanagari-capable fonts, used as fallbacks (Linux: Noto; macOS: Kohinoor/Sangam; Windows: Nirmala/Mangal)
_DEVA = ["Noto Sans Devanagari", "Kohinoor Devanagari", "Devanagari Sangam MN", "Devanagari MT", "Nirmala UI",
         "Mangal", "Lohit Devanagari"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = ["DejaVu Sans"] + [f for f in _DEVA if f in _installed]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})


def model_colors(lb: pd.DataFrame) -> dict:
    out, si, ti = {}, 0, 0
    for _, r in lb.iterrows():
        if r["family"] == "Statistical":
            out[r["model"]] = STAT_COLORS[si % len(STAT_COLORS)]
            si += 1
        elif r["family"] == "Transformer":
            out[r["model"]] = TRANS_COLORS[ti % len(TRANS_COLORS)]
            ti += 1
        else:
            out[r["model"]] = ENS_COLOR
    return out


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def macro_f1_bar(lb, colors, path, title_suffix=""):
    d = lb.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.5))
    err = np.array([d["macro_f1"] - d["ci_low"], d["ci_high"] - d["macro_f1"]])
    ax.barh(d["model"], d["macro_f1"], xerr=err, color=[colors[m] for m in d["model"]], capsize=3)
    for i, v in enumerate(d["macro_f1"]):
        ax.text(max(v - 0.03, 0.02), i, f"{v:.3f}", va="center", ha="right", fontsize=8, color="white",
                fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, ls="--", c="grey", lw=1)
    ax.text(0.5, len(d) - 0.4, "chance", color="grey", fontsize=8, ha="center")
    ax.set_xlabel("Macro-F1 on the test split (error bar = 95% bootstrap CI)")
    ax.set_title(f"Macro-F1 by model{title_suffix}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=STAT_COLORS[0], label="Statistical (BoW/TF-IDF)"),
                       Patch(color=TRANS_COLORS[0], label="Transformer"), Patch(color=ENS_COLOR, label="Ensemble")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    return _save(fig, path)


def roc_pr(y, probs: dict, colors, roc_path, pr_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, color=colors[m], lw=1.6, label=f"{m} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="chance")
    ax.set_xlabel("False positive rate (real items called fake)")
    ax.set_ylabel("True positive rate (fake items caught)")
    ax.set_title("ROC curves (test split)")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, roc_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m, p in probs.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=colors[m], lw=1.6, label=m)
    ax.axhline(np.mean(y), ls="--", c="grey", lw=1, label=f"base rate ({np.mean(y):.2f})")
    ax.set_xlabel("Recall (fake)")
    ax.set_ylabel("Precision (fake)")
    ax.set_title("Precision-recall curves (test split)")
    ax.legend(fontsize=7, loc="lower left")
    return _save(fig, pr_path)


def confusion_single(cm: dict, title, path, ax=None):
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
    tot = mat.sum()
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            correct = i == j
            ax.text(j, i, f"{mat[i, j]}\n({mat[i, j] / max(1, tot):.1%})", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > mat.max() * 0.6 else "black", fontweight="bold" if correct else None)
    ax.set_xticks([0, 1], ["pred real", "pred fake"])
    ax.set_yticks([0, 1], ["true real", "true fake"])
    ax.set_title(title, fontsize=9)
    if own:
        return _save(fig, path)


def confusion_grid(lb, results, path):
    n = len(lb)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.1 * rows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), lb.iterrows()):
        confusion_single(results[r["model"]]["test_metrics"]["confusion"], f"{r['model']}\nmacro-F1 {r['macro_f1']:.3f}", None, ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (test split; rows = truth, columns = prediction)", fontweight="bold")
    return _save(fig, path)


def calibration(y, probs, colors, path, n_bins=10):
    from .evaluate import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="perfectly calibrated")
    for m, p in probs.items():
        c = calibration_curve(y, p, n_bins)
        ax.plot([r["mean_pred"] for r in c], [r["frac_fake"] for r in c], "o-", ms=3, color=colors[m], lw=1.2, label=m)
    ax.set_xlabel("Predicted probability of fake")
    ax.set_ylabel("Actual share of fake items")
    ax.set_title("Calibration (reliability) chart")
    ax.legend(fontsize=7)
    return _save(fig, path)


def efficiency(lb, colors, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = np.clip(np.log10(lb["params"].astype(float).clip(lower=10)) ** 2.4 * 4, 20, 900)
    ax.scatter(lb["ms_per_item"].clip(lower=1e-3), lb["macro_f1"], s=sizes, c=[colors[m] for m in lb["model"]],
               alpha=0.75, edgecolors="black", linewidths=0.5)
    for _, r in lb.iterrows():
        ax.annotate(r["model"], (max(r["ms_per_item"], 1e-3), r["macro_f1"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per item (ms, log scale)")
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title("Accuracy vs. cost (bubble size = number of parameters)")
    return _save(fig, path)


def grouped_breakdown(bd: pd.DataFrame, dimension: str, colors, path, title, order=None):
    d = bd[bd["dimension"] == dimension]
    if d.empty:
        return None
    groups = order or sorted(d["group"].unique())
    groups = [g for g in groups if g in set(d["group"])]
    models = list(dict.fromkeys(d["model"]))
    w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(groups) + 0.25 * len(models)), 4.8))
    x = np.arange(len(groups))
    for i, m in enumerate(models):
        vals = [d[(d["model"] == m) & (d["group"] == g)]["macro_f1"].mean() for g in groups]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, color=colors.get(m, "#999"), label=m)
    ns = [int(d[d["group"] == g]["n"].iloc[0]) for g in groups]
    ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(groups, ns)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    return _save(fig, path)


def hard_case_recall(bd, colors, path):
    """Accuracy on each case type (only synthetic data has this column)."""
    d = bd[bd["dimension"] == "case_type"]
    if d.empty:
        return None
    groups = sorted(d["group"].unique())
    models = list(dict.fromkeys(d["model"]))
    w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(groups)), 4.8))
    x = np.arange(len(groups))
    for i, m in enumerate(models):
        vals = [d[(d["model"] == m) & (d["group"] == g)]["accuracy"].mean() for g in groups]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, color=colors.get(m, "#999"), label=m)
    ns = [int(d[d["group"] == g]["n"].iloc[0]) for g in groups]
    ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(groups, ns)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (test)")
    ax.set_title("Hard cases: accuracy by case type")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    return _save(fig, path)


def training_curves(results: dict, colors, path):
    trs = {m: r for m, r in results.items() if r.get("curve")}
    if not trs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, r in trs.items():
        ep = [c["epoch"] for c in r["curve"]]
        axes[0].plot(ep, [c["val_macro_f1"] for c in r["curve"]], "o-", color=colors.get(m), label=m)
        axes[1].plot(ep, [c["train_loss"] for c in r["curve"]], "o-", color=colors.get(m), label=f"{m} train")
        axes[1].plot(ep, [c["val_loss"] for c in r["curve"]], "x--", color=colors.get(m), label=f"{m} val")
    axes[0].set_title("Validation macro-F1 per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_title("Loss per epoch (train solid, validation dashed)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    for a in axes:
        a.legend(fontsize=7)
        a.xaxis.get_major_locator().set_params(integer=True)
    return _save(fig, path)


def top_features_chart(res: dict, path, k=15):
    tf = res.get("top_features")
    if not tf:
        return None
    fake, real = tf["fake"][:k], tf["real"][:k]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * (len(fake) + len(real)) + 1.2))
    names = [f for f, _ in real[::-1]] + [f for f, _ in fake[::-1]]
    vals = [w for _, w in real[::-1]] + [w for _, w in fake[::-1]]
    ax.barh(range(len(vals)), vals, color=[REAL if v < 0 else FAKE for v in vals])
    ax.set_yticks(range(len(vals)), names, fontsize=8)
    ax.axvline(0, c="black", lw=0.8)
    ax.set_xlabel("Weight (positive pushes towards FAKE, negative towards REAL)")
    ax.set_title(f"{res['label']}: strongest features")
    return _save(fig, path)


def shortcut_bar(df: pd.DataFrame, path):
    d = df.dropna(subset=["shortcut_index"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(d) + 1.5))
    cols = ["#d62728" if v >= 0.5 else "#fdae6b" if v >= 0.25 else "#9ecae1" for v in d["shortcut_index"]]
    ax.barh(d["probe"], d["shortcut_index"], color=cols)
    for i, v in enumerate(d["shortcut_index"]):
        ax.text(v + 0.01 if v >= 0 else 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(0.5, ls="--", c="grey", lw=1)
    ax.set_xlabel("Shortcut Index = (probe F1 - chance) / (full model F1 - chance)")
    ax.set_title("How much of the model's skill a shortcut explains")
    ax.set_xlim(min(0, d["shortcut_index"].min() - 0.05), max(1.1, d["shortcut_index"].max() + 0.1))
    return _save(fig, path)


def style_sweep_chart(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["style_correlation"], df["full_text_macro_f1"], "o-", c=STAT_COLORS[0], label="TF-IDF + LR (full text)")
    ax.plot(df["style_correlation"], df["style_only_macro_f1"], "s-", c=FAKE, label="Style-only probe")
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_xlabel("Style correlation s (P(sensational | fake))")
    ax.set_ylabel("Macro-F1 (test)")
    ax.set_title("Style shortcut experiment: style-only probe vs. full model")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8)
    return _save(fig, path)


def data_charts(card: dict, splits: dict, figdir: Path):
    out = {}
    fig, ax = plt.subplots(figsize=(6, 3.8))
    names = list(splits)
    fake = [int(splits[s]["label_id"].sum()) for s in names]
    real = [int((1 - splits[s]["label_id"]).sum()) for s in names]
    ax.bar(names, real, color=REAL, label="real")
    ax.bar(names, fake, bottom=real, color=FAKE, label="fake")
    ax.set_ylabel("Items")
    ax.set_title("Class balance per split")
    ax.legend()
    out["class_balance"] = _save(fig, figdir / "data_class_balance.png")
    all_df = pd.concat(splits.values())
    fig, ax = plt.subplots(figsize=(6, 3.8))
    lc = all_df.groupby(["language", "label_id"]).size().unstack(fill_value=0)
    lc = lc.reindex(columns=[0, 1], fill_value=0)
    ax.bar(lc.index, lc[0], color=REAL, label="real")
    ax.bar(lc.index, lc[1], bottom=lc[0], color=FAKE, label="fake")
    ax.set_ylabel("Items")
    ax.set_title("Language mix by label")
    ax.legend()
    out["language_mix"] = _save(fig, figdir / "data_language_mix.png")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(0, max(5, all_df["length_words"].quantile(0.99)), 30)
    ax.hist(all_df.loc[all_df.label_id == 0, "length_words"], bins=bins, alpha=0.6, color=REAL, label="real")
    ax.hist(all_df.loc[all_df.label_id == 1, "length_words"], bins=bins, alpha=0.6, color=FAKE, label="fake")
    ax.set_xlabel("Words per item")
    ax.set_ylabel("Items")
    ax.set_title("Text length by label")
    ax.legend()
    out["length"] = _save(fig, figdir / "data_length.png")
    return out
