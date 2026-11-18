"""Phase 5: leaderboard, breakdowns, significance tests, disagreements, figures and the results report.

Model selection rule (stated in every output): "best" statistical / transformer models used for the head-to-head
comparison, the ensemble and the hypothesis tests are chosen by VALIDATION macro-F1, never by test score, so the
test split stays an honest final exam. The leaderboard itself is sorted by test macro-F1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import figures as F
from .config import output_dir
from .data import load_card, load_splits
from .evaluate import compute_metrics, macro_f1, mcnemar, paired_bootstrap_diff, subset_metrics
from .predict import list_results
from .utils import FakeNewsError, append_run_log, ensure_dir, log, now_iso, read_json, write_json

ENSEMBLE = "ensemble_avg"


def load_predictions(cfg, name, split="test") -> tuple[pd.DataFrame, dict]:
    """Test (or val) rows joined with every model's P(fake) in columns p__<model>."""
    sp = load_splits(cfg, name)
    df = sp[split].copy()
    results = list_results(cfg, name)
    pdir = Path(output_dir(cfg, name)) / "results" / "predictions"
    suffix = "" if split == "test" else ".val"
    for m in list(results):
        f = pdir / f"{m}{suffix}.csv"
        if not f.exists():
            results.pop(m)
            continue
        p = pd.read_csv(f)
        if set(p["id"]) != set(df["id"]):
            log(f"WARNING: predictions of '{m}' were made on a different split; re-train it. Skipped.")
            results.pop(m)
            continue
        df = df.merge(p.rename(columns={"p_fake": f"p__{m}"}), on="id", how="left")
    return df, results


def pick_best(results: dict, family: str):
    cands = [m for m, r in results.items() if r["family"] == family]
    if not cands:
        return None
    return max(cands, key=lambda m: selection_key(results, m))


def _nz(v):
    return -1.0 if v is None or (isinstance(v, float) and np.isnan(v)) else v


def selection_key(results, m):
    """Choose on validation: macro-F1, then ROC-AUC, then lower Brier score, then name (fully deterministic)."""
    v = results[m]["val_metrics"]
    return (round(v["macro_f1"], 6), round(_nz(v.get("roc_auc")), 6), -round(v.get("brier", 0), 6), m)


def _length_bin(n, bins):
    return "short" if n < bins[0] else ("medium" if n < bins[1] else "long")


def run_compare(cfg: dict, name: str, quiet: bool = False) -> dict:
    ev = cfg["evaluation"]
    thr, seed = ev["threshold"], int(cfg["seed"])
    df, results = load_predictions(cfg, name, "test")
    if not results:
        raise FakeNewsError(f"No trained models for '{name}'. Run train-classical (and train-bert) first.")
    vdf, _ = load_predictions(cfg, name, "val")
    out = output_dir(cfg, name)
    rdir, fdir, repdir = ensure_dir(out / "results"), ensure_dir(out / "figures"), ensure_dir(out / "reports")
    y = df["label_id"].to_numpy()

    best_stat = pick_best(results, "Statistical")
    best_trans = pick_best(results, "Transformer")

    # ensemble of the best statistical + best transformer (chosen on validation)
    if best_stat and best_trans:
        df[f"p__{ENSEMBLE}"] = (df[f"p__{best_stat}"] + df[f"p__{best_trans}"]) / 2
        vdf[f"p__{ENSEMBLE}"] = (vdf[f"p__{best_stat}"] + vdf[f"p__{best_trans}"]) / 2
        results[ENSEMBLE] = {
            "model": ENSEMBLE, "label": f"Ensemble: mean of {best_stat} + {best_trans}", "family": "Ensemble",
            "mode": "Average", "about": "Average of P(fake) from the best statistical and best transformer model "
                                        "(both chosen on validation).",
            "test_metrics": compute_metrics(y, df[f"p__{ENSEMBLE}"], thr, ev["bootstrap"], ev["ci"], ev["ece_bins"], seed),
            "val_metrics": compute_metrics(vdf["label_id"], vdf[f"p__{ENSEMBLE}"], thr, 0),
            "train_seconds": results[best_stat]["train_seconds"] + results[best_trans]["train_seconds"],
            "infer_ms_per_item": results[best_stat]["infer_ms_per_item"] + results[best_trans]["infer_ms_per_item"],
            "params": results[best_stat]["params"] + results[best_trans]["params"],
            "members": [best_stat, best_trans],
        }

    # ------------------------------------------------------------------ leaderboard
    rows = []
    for m, r in results.items():
        t = r["test_metrics"]
        rows.append({
            "model": m, "label": r.get("label", m), "family": r["family"], "mode": r.get("mode"),
            "macro_f1": t["macro_f1"], "ci_low": t["macro_f1_ci_low"], "ci_high": t["macro_f1_ci_high"],
            "accuracy": t["accuracy"], "precision_fake": t["precision_fake"], "recall_fake": t["recall_fake"],
            "f1_fake": t["f1_fake"], "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"], "brier": t["brier"],
            "ece": t["ece"], "val_macro_f1": r["val_metrics"]["macro_f1"],
            "seed_mean": r.get("macro_f1_seed_mean"), "seed_std": r.get("macro_f1_seed_std"),
            "n_seeds": len(r.get("seeds", [])) or None,
            "train_seconds": r.get("train_seconds"), "ms_per_item": r.get("infer_ms_per_item"),
            "params": r.get("params"), "disk_mb": r.get("disk_mb"), "device": r.get("device", "cpu"),
        })
    lb = pd.DataFrame(rows).sort_values(["macro_f1", "roc_auc"], ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", range(1, len(lb) + 1))
    lb.to_csv(rdir / "summary.csv", index=False)
    colors = F.model_colors(lb)

    # ------------------------------------------------------------------ predictions table
    pcols = [f"p__{m}" for m in lb["model"]]
    keep = [c for c in ["id", "text_bert", "label_id", "language", "style", "case_type", "topic", "source", "date",
                        "length_words"] if c in df.columns]
    pred_table = df[keep + pcols].rename(columns={"text_bert": "text"})
    pred_table.to_csv(rdir / "predictions.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ significance (top model vs every other)
    top = lb.iloc[0]["model"]
    preds = {m: (df[f"p__{m}"].to_numpy() >= thr).astype(int) for m in lb["model"]}
    sig = []
    for m in lb["model"]:
        if m == top:
            continue
        mc = mcnemar(y, preds[top], preds[m])
        bd = paired_bootstrap_diff(y, preds[top], preds[m], n=ev["bootstrap"], seed=seed)
        sig.append({"model_a": top, "model_b": m, "macro_f1_a": macro_f1(y, preds[top]), "macro_f1_b": macro_f1(y, preds[m]),
                    "diff": bd["diff"], "diff_ci_low": bd["ci_low"], "diff_ci_high": bd["ci_high"],
                    "a_right_b_wrong": mc["b"], "a_wrong_b_right": mc["c"], "mcnemar_test": mc["test"],
                    "p_value": mc["p_value"], "significant": mc["p_value"] < ev["alpha"]})
    sig_df = pd.DataFrame(sig)
    sig_df.to_csv(rdir / "significance.csv", index=False)

    # ------------------------------------------------------------------ breakdowns
    df["length_bin"] = df["length_words"].map(lambda n: _length_bin(n, ev["length_bins"]))
    dims = [d for d in ["language", "length_bin", "style", "case_type", "topic"] if d in df.columns and df[d].notna().any()]
    bds = []
    for m in lb["model"]:
        p = df[f"p__{m}"].to_numpy()
        for d in dims:
            for g, idx in df.groupby(d).groups.items():
                sm = subset_metrics(y[df.index.get_indexer(idx)], p[df.index.get_indexer(idx)], thr)
                bds.append({"model": m, "dimension": d, "group": str(g), **sm})
    bd_df = pd.DataFrame(bds)
    bd_df.to_csv(rdir / "breakdowns.csv", index=False)

    # ------------------------------------------------------------------ statistical vs transformer
    svt = None
    if best_stat and best_trans:
        mc = mcnemar(y, preds[best_stat], preds[best_trans])
        bd = paired_bootstrap_diff(y, preds[best_trans], preds[best_stat], n=ev["bootstrap"], seed=seed)
        svt = {"best_statistical": best_stat, "best_transformer": best_trans,
               "f1_statistical": macro_f1(y, preds[best_stat]), "f1_transformer": macro_f1(y, preds[best_trans]),
               "diff_transformer_minus_statistical": bd["diff"], "diff_ci": [bd["ci_low"], bd["ci_high"]],
               "mcnemar": mc, "significant": mc["p_value"] < ev["alpha"]}
        dis = df[preds[best_stat] != preds[best_trans]].copy()
        dis = pd.DataFrame({
            "id": dis["id"], "text": dis["text_bert"], "true_label": dis["label_id"].map({1: "fake", 0: "real"}),
            f"p_fake_{best_stat}": dis[f"p__{best_stat}"].round(4), f"p_fake_{best_trans}": dis[f"p__{best_trans}"].round(4),
            "who_is_right": np.where((dis[f"p__{best_stat}"] >= thr).astype(int) == dis["label_id"],
                                     f"{best_stat} right, {best_trans} wrong", f"{best_trans} right, {best_stat} wrong"),
            **{c: dis[c] for c in ("language", "style", "case_type") if c in dis.columns},
        })
        dis["gap"] = (dis[f"p_fake_{best_stat}"] - dis[f"p_fake_{best_trans}"]).abs()
        dis.sort_values("gap", ascending=False).head(ev["n_disagreements"]).to_csv(
            rdir / "disagreements.csv", index=False, encoding="utf-8-sig")
        svt["n_disagreements"] = int(len(dis))
        svt["disagreement_counts"] = dis["who_is_right"].value_counts().to_dict()

    # ------------------------------------------------------------------ most confident errors
    errs = []
    for m in lb["model"]:
        p = df[f"p__{m}"].to_numpy()
        wrong = np.where(preds[m] != y)[0]
        conf = np.abs(p[wrong] - 0.5)
        for i in wrong[np.argsort(-conf)][: ev["n_errors"]]:
            errs.append({"model": m, "id": df.iloc[i]["id"], "text": df.iloc[i]["text_bert"],
                         "true_label": "fake" if y[i] else "real", "p_fake": round(float(p[i]), 4),
                         **{c: df.iloc[i][c] for c in ("language", "style", "case_type") if c in df.columns}})
    pd.DataFrame(errs).to_csv(rdir / "errors.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ figures
    probs = {m: df[f"p__{m}"].to_numpy() for m in lb["model"]}
    card = load_card(cfg, name)
    synth = "WARNING" in card
    suffix = " - SYNTHETIC data" if synth else ""
    figs = {
        "macro_f1_bar": F.macro_f1_bar(lb, colors, fdir / "macro_f1_bar.png", suffix),
        "roc": None, "pr": None,
        "confusion_grid": F.confusion_grid(lb, results, fdir / "confusion_matrices.png"),
        "calibration": F.calibration(y, probs, colors, fdir / "calibration.png", ev["ece_bins"]),
        "efficiency": F.efficiency(lb, colors, fdir / "efficiency.png"),
        "per_language": F.grouped_breakdown(bd_df, "language", colors, fdir / "per_language.png",
                                            "Macro-F1 by language", ["en", "hinglish", "hi"]),
        "per_style": F.grouped_breakdown(bd_df, "style", colors, fdir / "per_style.png", "Macro-F1 by writing style"),
        "per_length": F.grouped_breakdown(bd_df, "length_bin", colors, fdir / "per_length.png",
                                          "Macro-F1 by text length", ["short", "medium", "long"]),
        "hard_cases": F.hard_case_recall(bd_df, colors, fdir / "hard_cases.png"),
        "training_curves": F.training_curves(results, colors, fdir / "training_curves.png"),
    }
    F.roc_pr(y, probs, colors, fdir / "roc_curves.png", fdir / "pr_curves.png")
    figs["roc"], figs["pr"] = str(fdir / "roc_curves.png"), str(fdir / "pr_curves.png")
    for m, r in results.items():
        F.confusion_single(r["test_metrics"]["confusion"], f"{m} (test)", fdir / "confusion" / f"{m}.png")
        if r.get("top_features"):
            F.top_features_chart(r, fdir / "top_features" / f"{m}.png")
    sp = load_splits(cfg, name)
    figs.update(F.data_charts(card, sp, fdir))

    # ------------------------------------------------------------------ summary + report
    summary = _summary_sentences(lb, results, best_stat, best_trans, svt, ev["alpha"], synth)
    comp = {"dataset": name, "created": now_iso(), "n_test": int(len(df)), "threshold": thr,
            "selection_rule": "best statistical/transformer chosen by validation macro-F1",
            "top_model": top, "best_statistical": best_stat, "best_transformer": best_trans,
            "statistical_vs_transformer": svt, "summary": summary, "synthetic": synth, "figures": figs,
            "ensemble": results.get(ENSEMBLE, {}).get("members")}
    write_json(rdir / "comparison.json", comp)
    write_json(rdir / f"{ENSEMBLE}.meta.json", results.get(ENSEMBLE, {}))
    report = write_report(cfg, name, lb, sig_df, bd_df, comp, card, results)
    append_run_log(cfg["paths"]["outputs_dir"], name, "compare", {"top_model": top, "n_models": len(lb)})
    if not quiet:
        print_leaderboard(lb)
        print()
        print(summary)
        print()
        log(f"Saved: {rdir / 'summary.csv'}, predictions.csv, significance.csv, breakdowns.csv, disagreements.csv, errors.csv")
        log(f"Figures: {fdir}")
        log(f"Report: {report}")
    return comp


def _fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _summary_sentences(lb, results, best_stat, best_trans, svt, alpha, synth) -> str:
    t = lb.iloc[0]
    s = [f"Top of the leaderboard: {t['model']} (macro-F1 {t['macro_f1']:.3f}, 95% CI {t['ci_low']:.3f}-{t['ci_high']:.3f})."]
    if best_stat:
        r = results[best_stat]["test_metrics"]
        s.append(f"Best statistical model (chosen on validation): {best_stat} (macro-F1 {r['macro_f1']:.3f}, "
                 f"95% CI {r['macro_f1_ci_low']:.3f}-{r['macro_f1_ci_high']:.3f}).")
    if best_trans:
        r = results[best_trans]["test_metrics"]
        s.append(f"Best transformer (chosen on validation): {best_trans} (macro-F1 {r['macro_f1']:.3f}, "
                 f"95% CI {r['macro_f1_ci_low']:.3f}-{r['macro_f1_ci_high']:.3f}).")
    if svt:
        d = svt["diff_transformer_minus_statistical"]
        p = svt["mcnemar"]["p_value"]
        word = "is" if p < alpha else "is not"
        s.append(f"The transformer {'leads' if d > 0 else 'trails' if d < 0 else 'ties'} by {abs(d):.3f} macro-F1; "
                 f"this difference {word} statistically significant (McNemar p = {p:.4f}, alpha = {alpha}).")
    elif not best_trans:
        s.append("No transformer has been trained yet - run train-bert to compare.")
    if synth:
        s.append("NOTE: these are SYNTHETIC-data scores; they show the pipeline works, not how good the models are.")
    return " ".join(s)


def print_leaderboard(lb: pd.DataFrame):
    hdr = f"{'#':>2}  {'model':<16} {'family':<11} {'mode':<10} {'macro-F1 (95% CI)':<22} {'acc':>6} {'AUC':>6} " \
          f"{'ECE':>6} {'train s':>8} {'ms/item':>8} {'params':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for _, r in lb.iterrows():
        params = f"{r['params'] / 1e6:.1f}M" if r["params"] and r["params"] >= 1e5 else f"{int(r['params'] or 0):,}"
        print(f"{r['rank']:>2}  {r['model']:<16} {r['family']:<11} {str(r['mode']):<10} "
              f"{r['macro_f1']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]   {r['accuracy']:>6.3f} {_fmt(r['roc_auc']):>6} "
              f"{r['ece']:>6.3f} {_fmt(r['train_seconds'], 1):>8} {_fmt(r['ms_per_item'], 2):>8} {params:>9}")


def _md_table(df: pd.DataFrame, floatfmt=3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(cfg, name, lb, sig_df, bd_df, comp, card, results) -> Path:
    out = output_dir(cfg, name)
    rel = lambda p: Path(p).relative_to(out).as_posix() if p else None  # noqa: E731
    fig = {k: rel(v) for k, v in comp["figures"].items() if v}
    L = [f"# Results report: `{name}`", "",
         f"_Generated automatically on {comp['created']} from the result files in `outputs/{name}/results/`. "
         "No number in this report was typed by hand._", ""]
    if comp["synthetic"]:
        L += ["> **SYNTHETIC DATA.** These results prove the pipeline works. They are **not** model performance "
              "and must not be reported as such.", ""]
    L += ["## 1. Data", "",
          f"- Split strategy: **{card.get('split_strategy')}** (seed {card.get('seed')})",
          f"- Sizes: {card.get('sizes')}", f"- Share fake: {card.get('share_fake')}",
          f"- Exact duplicates removed: {card.get('exact_duplicates_removed')}; near-duplicate rows: "
          f"{card.get('near_duplicate_rows')}",
          f"- Text views: {card.get('text_views')}", f"- Fingerprint: `{str(card.get('fingerprint_sha256'))[:80]}`", ""]
    if card.get("bert_text_warning"):
        L += [f"> {card['bert_text_warning']}", ""]
    L += ["## 2. Summary", "", comp["summary"], "",
          f"_Selection rule: {comp['selection_rule']}; leaderboard sorted by test macro-F1; threshold {comp['threshold']}._", "",
          "## 3. Leaderboard (test split)", ""]
    show = lb[["rank", "model", "family", "mode", "macro_f1", "ci_low", "ci_high", "accuracy", "roc_auc", "ece",
               "train_seconds", "ms_per_item", "params"]].copy()
    show["params"] = show["params"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
    L += [_md_table(show), "", f"![macro-F1]({fig.get('macro_f1_bar')})", ""]
    L += ["## 4. Significance (top model vs. every other)", "",
          "McNemar test on the same test items; `diff` is macro-F1(A) - macro-F1(B) with a paired-bootstrap 95% CI.", ""]
    if len(sig_df):
        L += [_md_table(sig_df[["model_a", "model_b", "diff", "diff_ci_low", "diff_ci_high", "a_right_b_wrong",
                                "a_wrong_b_right", "p_value", "significant"]], 4), ""]
    svt = comp.get("statistical_vs_transformer")
    if svt:
        L += ["## 5. Statistical vs. transformer", "",
              f"- Best statistical: **{svt['best_statistical']}** (macro-F1 {svt['f1_statistical']:.3f})",
              f"- Best transformer: **{svt['best_transformer']}** (macro-F1 {svt['f1_transformer']:.3f})",
              f"- Difference (transformer - statistical): {svt['diff_transformer_minus_statistical']:+.3f} "
              f"(95% CI {svt['diff_ci'][0]:+.3f} to {svt['diff_ci'][1]:+.3f}); McNemar p = {svt['mcnemar']['p_value']:.4f}",
              f"- Items where they disagree: {svt['n_disagreements']} -> {svt['disagreement_counts']} "
              "(see `results/disagreements.csv`)", ""]
    L += ["## 6. Curves and matrices", "", f"![confusion]({fig.get('confusion_grid')})", "",
          f"![roc]({fig.get('roc')}) ![pr]({fig.get('pr')})", "", f"![calibration]({fig.get('calibration')})", "",
          f"![efficiency]({fig.get('efficiency')})", ""]
    if fig.get("training_curves"):
        L += [f"![training]({fig['training_curves']})", ""]
    L += ["## 7. Breakdowns", ""]
    for dim, key in (("language", "per_language"), ("style", "per_style"), ("length_bin", "per_length"),
                     ("case_type", "hard_cases")):
        d = bd_df[bd_df["dimension"] == dim]
        if d.empty:
            continue
        val = "accuracy" if dim == "case_type" else "macro_f1"
        piv = d.pivot_table(index="model", columns="group", values=val).reset_index()
        L += [f"### By {dim} ({val})", "", _md_table(piv), ""]
        if fig.get(key):
            L += [f"![{dim}]({fig[key]})", ""]
    hyp = Path(out) / "results" / "hypotheses.json"
    if hyp.exists():
        L += ["## 8. Hypotheses", "", "See `reports/hypotheses.md` (generated by `python -m fakenews hypotheses`).", ""]
        for h in read_json(hyp)["hypotheses"]:
            L.append(f"- **{h['id']}** {h['statement']} -> **{h['verdict']}**")
        L.append("")
    L += ["## 9. Reproducibility", "",
          f"- Library versions: {next(iter(r for r in results.values() if 'versions' in r))['versions']}",
          "- Full configuration is stored in every `results/<model>.json`.",
          f"- Run log: `outputs/{name}/logs/run_log.jsonl`", ""]
    p = Path(out) / "reports" / "results_report.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p
