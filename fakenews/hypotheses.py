"""Research hypotheses, each tested automatically against the saved results.

Verdicts
  SUPPORTED     the data show the predicted effect and it is statistically significant (p < alpha)  -> "proved"
  REJECTED      the data show the OPPOSITE effect and it is significant                             -> "disproved"
  INCONCLUSIVE  no significant difference either way (not the same as disproved!)
  NOT TESTABLE  the models or columns needed for the test are missing (the reason is stated)

Model choice is always made on VALIDATION scores, so the test split is never used to pick what gets tested.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .compare import ENSEMBLE, load_predictions, pick_best, selection_key
from .config import output_dir
from .data import load_card
from .evaluate import ece, macro_f1, mcnemar, paired_bootstrap_diff
from .transformer_models import ENGLISH_ONLY, INDIC
from .utils import append_run_log, ensure_dir, log, now_iso, read_json, write_json

SUP, REJ, INC, NT = "SUPPORTED", "REJECTED", "INCONCLUSIVE", "NOT TESTABLE"


def _verdict(effect, p, alpha, positive_is_support=True):
    if p is None or np.isnan(p):
        return INC
    if p < alpha:
        return SUP if (effect > 0) == positive_is_support else REJ
    return INC


def _H(hid, title, statement, rationale, test, verdict, numbers=None, p=None, effect=None, explanation="", used=None):
    return {"id": hid, "title": title, "statement": statement, "why_it_matters": rationale, "test": test,
            "verdict": verdict, "p_value": p, "effect": effect, "numbers": numbers or {}, "explanation": explanation,
            "models_used": used or []}


def _best_of(results, names):
    c = [m for m in names if m in results]
    return max(c, key=lambda m: selection_key(results, m)) if c else None


def _pair_test(y, pa, pb, mask, alpha, n_boot, seed, thr):
    """A vs B on the same items (optionally a subset). effect = macro-F1(A) - macro-F1(B)."""
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask)
    ya, a, b = y[m], (pa[m] >= thr).astype(int), (pb[m] >= thr).astype(int)
    mc = mcnemar(ya, a, b)
    bd = paired_bootstrap_diff(ya, a, b, n=n_boot, seed=seed)
    return {"n": int(m.sum()), "f1_a": macro_f1(ya, a), "f1_b": macro_f1(ya, b), "diff": bd["diff"],
            "diff_ci": [bd["ci_low"], bd["ci_high"]], "mcnemar_p": mc["p_value"], "bootstrap_p": bd["p_value"],
            "a_right_b_wrong": mc["b"], "a_wrong_b_right": mc["c"]}


def _pair_hyp(hid, title, statement, why, A, B, y, df, mask, alpha, n_boot, seed, thr, subset_note="test split"):
    if not A or not B:
        missing = "the first group" if not A else "the second group"
        return _H(hid, title, statement, why, "McNemar test + paired bootstrap of macro-F1", NT,
                  explanation=f"Not testable: no trained model in {missing}.")
    if mask is not None and np.asarray(mask).sum() < 10:
        return _H(hid, title, statement, why, "McNemar test + paired bootstrap of macro-F1", NT,
                  explanation=f"Not testable: only {int(np.asarray(mask).sum())} items in the subset.", used=[A, B])
    r = _pair_test(y, df[f"p__{A}"].to_numpy(), df[f"p__{B}"].to_numpy(), mask, alpha, n_boot, seed, thr)
    v = _verdict(r["diff"], r["mcnemar_p"], alpha)
    expl = (f"On the {subset_note} (n={r['n']}), {A} scored macro-F1 {r['f1_a']:.3f} vs {B} {r['f1_b']:.3f} "
            f"(difference {r['diff']:+.3f}, 95% CI {r['diff_ci'][0]:+.3f} to {r['diff_ci'][1]:+.3f}). "
            f"{A} was right where {B} was wrong on {r['a_right_b_wrong']} items, and the reverse on "
            f"{r['a_wrong_b_right']}; McNemar p = {r['mcnemar_p']:.4f}.")
    return _H(hid, title, statement, why, "McNemar test (alpha {}) + paired bootstrap 95% CI of the macro-F1 "
              "difference".format(alpha), v, r, r["mcnemar_p"], r["diff"], expl, [A, B])


def _fisher_hyp(hid, title, statement, why, a_ok, a_n, b_ok, b_n, label_a, label_b, alpha, model, expect_a_lower=True):
    if a_n < 5 or b_n < 5:
        return _H(hid, title, statement, why, "Fisher exact test on correct/wrong counts", NT,
                  explanation=f"Not testable: too few items ({label_a}: {a_n}, {label_b}: {b_n}).", used=[model])
    _, p = stats.fisher_exact([[a_ok, a_n - a_ok], [b_ok, b_n - b_ok]])
    ra, rb = a_ok / a_n, b_ok / b_n
    effect = (rb - ra) if expect_a_lower else (ra - rb)
    v = _verdict(effect, p, alpha)
    expl = (f"{model}: {label_a} {a_ok}/{a_n} correct ({ra:.1%}) vs {label_b} {b_ok}/{b_n} correct ({rb:.1%}); "
            f"Fisher exact p = {p:.4f}.")
    return _H(hid, title, statement, why, f"Fisher exact test (alpha {alpha}) on correct/wrong counts", v,
              {"rate_" + label_a: ra, "rate_" + label_b: rb, "n_" + label_a: a_n, "n_" + label_b: b_n}, float(p),
              float(effect), expl, [model])


def run_hypotheses(cfg: dict, name: str, quiet=False) -> list[dict]:
    hc, ev = cfg["hypotheses"], cfg["evaluation"]
    alpha, n_boot, seed, thr = float(hc["alpha"]), int(hc["bootstrap"]), int(cfg["seed"]), ev["threshold"]
    out = output_dir(cfg, name)
    df, results = load_predictions(cfg, name, "test")
    comp_path = out / "results" / "comparison.json"
    if comp_path.exists():  # bring the ensemble back in (compare stores it)
        meta = out / "results" / f"{ENSEMBLE}.meta.json"
        if meta.exists() and read_json(meta).get("members"):
            mem = read_json(meta)["members"]
            if all(f"p__{m}" in df for m in mem):
                df[f"p__{ENSEMBLE}"] = df[[f"p__{m}" for m in mem]].mean(axis=1)
    y = df["label_id"].to_numpy()
    H = []

    # H1 transformer vs statistical
    bs, bt = pick_best(results, "Statistical"), pick_best(results, "Transformer")
    H.append(_pair_hyp("H1", "Transformers beat statistical models",
                       "The best BERT-family model has a higher macro-F1 than the best BoW/TF-IDF model.",
                       "Transformers read context and bring pretrained knowledge; if they cannot beat TF-IDF clearly, "
                       "that is an important finding, not a failure.", bt, bs, y, df, None, alpha, n_boot, seed, thr))
    # H2 TF-IDF vs BoW
    H.append(_pair_hyp("H2", "TF-IDF beats Bag-of-Words",
                       "Re-weighting word counts by rarity (TF-IDF) gives a higher macro-F1 than raw counts (BoW).",
                       "TF-IDF down-weights words that appear everywhere and boosts distinctive cue words.",
                       _best_of(results, ["tfidf_lr", "tfidf_svm"]), _best_of(results, ["bow_nb", "bow_lr"]),
                       y, df, None, alpha, n_boot, seed, thr))
    # H3 char n-grams on Hinglish
    has_lang = "language" in df.columns
    H.append(_pair_hyp("H3", "Character n-grams help on Hinglish",
                       "On Hinglish items, TF-IDF on character 3-5-grams beats TF-IDF on words.",
                       "Romanised Hindi has many spellings of one word (nahi/nahin/nai); character pieces match them.",
                       "tfidfchar_lr" if "tfidfchar_lr" in results else None,
                       "tfidf_lr" if "tfidf_lr" in results else None, y, df,
                       (df["language"] == "hinglish").to_numpy() if has_lang else np.zeros(len(df), bool),
                       alpha, n_boot, seed, thr, "Hinglish test items"))
    # H4 Indian/multilingual vs English-only transformers on Hindi + Hinglish
    ind = _best_of(results, [f"{m}_ft" for m in INDIC]) or _best_of(results, [f"{m}_emb" for m in INDIC])
    eng = _best_of(results, [f"{m}_ft" for m in ENGLISH_ONLY]) or _best_of(results, [f"{m}_emb" for m in ENGLISH_ONLY])
    mask = df["language"].isin(["hi", "hinglish"]).to_numpy() if has_lang else np.zeros(len(df), bool)
    h4 = _pair_hyp("H4", "Indian-language models help on Hindi and Hinglish",
                   "On Hindi + Hinglish items, the best multilingual/Indian model (mBERT, XLM-R, MuRIL, IndicBERT, "
                   "HingBERT) beats the best English-only model (BERT, DistilBERT, RoBERTa, ALBERT).",
                   "English tokenizers break Hindi into meaningless pieces; models trained on Indian languages should not.",
                   ind, eng, y, df, mask, alpha, n_boot, seed, thr, "Hindi + Hinglish test items")
    if ind and eng and has_lang and (df["language"] == "en").sum() >= 10:
        h4["numbers"]["english_only_subset"] = _pair_test(y, df[f"p__{ind}"].to_numpy(), df[f"p__{eng}"].to_numpy(),
                                                          (df["language"] == "en").to_numpy(), alpha, n_boot, seed, thr)
        h4["explanation"] += (f" For contrast, on English items the difference is "
                              f"{h4['numbers']['english_only_subset']['diff']:+.3f}.")
    H.append(h4)

    # H5 style shortcut (audit)
    audit_p = out / "results" / "audit.json"
    audit = pd.DataFrame(read_json(audit_p)["probes"]) if audit_p.exists() else pd.DataFrame()
    t = float(hc["shortcut_threshold"])

    def probe_hyp(hid, probe, title, statement, why):
        if audit.empty or probe not in set(audit["probe"]):
            return _H(hid, title, statement, why, "Shortcut Index with bootstrap 95% CI", NT,
                      explanation=f"Not testable: run `audit` first{' (needs a source column)' if 'source' in probe else ''}.")
        r = audit[audit["probe"] == probe].iloc[0]
        lo, hi, idx = r["index_ci_low"], r["index_ci_high"], r["shortcut_index"]
        v = SUP if lo > t else (REJ if hi < t else INC)
        expl = (f"The {probe} probe scored macro-F1 {r['macro_f1']:.3f} vs chance {r['chance_macro_f1']:.3f} and the full "
                f"model {r['reference_macro_f1']:.3f} ({r['reference_model']}). Shortcut Index = {idx:.2f} "
                f"(95% CI {lo:.2f} to {hi:.2f}); threshold {t}.")
        return _H(hid, title, statement, why, f"Shortcut Index; SUPPORTED if its 95% CI lies above {t}, REJECTED if "
                  f"below", v, {"shortcut_index": idx, "ci": [lo, hi], "probe_macro_f1": r["macro_f1"]}, None,
                  float(idx - t), expl, [r["reference_model"]])

    H.append(probe_hyp("H5", "style_only", "Writing style is a shortcut",
                       f"Writing style alone (punctuation, capitals, emojis, length) explains more than {t:.0%} of the "
                       "model's skill (Shortcut Index > threshold).",
                       "If style is enough, a calmly written fake will slip through."))
    H.append(probe_hyp("H6", "source_only", "Who posted it is a shortcut",
                       f"The source/channel name alone explains more than {t:.0%} of the model's skill.",
                       "A model that learns 'who posted' fails on new channels and can be gamed."))

    # H7 calm-tone fakes are harder
    if bs and "style" in df.columns:
        best = bt or bs
        pr = (df[f"p__{best}"] >= thr).astype(int)
        fk = df["label_id"] == 1
        neu, sen = fk & (df["style"] == "neutral"), fk & (df["style"] == "sensational")
        H.append(_fisher_hyp("H7", "Calm-sounding fakes are harder to catch",
                             "The best model catches fewer fakes written in a neutral tone than fakes written "
                             "sensationally (lower recall on neutral fakes).",
                             "Real misinformation is often written to look official.", int((pr[neu] == 1).sum()),
                             int(neu.sum()), int((pr[sen] == 1).sum()), int(sen.sum()), "neutral_fakes",
                             "sensational_fakes", alpha, best))
    else:
        H.append(_H("H7", "Calm-sounding fakes are harder to catch", "Lower recall on neutral-tone fakes.",
                    "Real misinformation is often written to look official.", "Fisher exact test", NT,
                    explanation="Not testable: the dataset has no 'style' column (synthetic data has one)."))

    # H8 debunk reports (keyword trap)
    if bs and "case_type" in df.columns and (df["case_type"] == "debunk_report").any():
        best = bt or bs
        pr = (df[f"p__{best}"] >= thr).astype(int)
        real = df["label_id"] == 0
        deb, std = real & (df["case_type"] == "debunk_report"), real & (df["case_type"] == "standard")
        H.append(_fisher_hyp("H8", "Debunk reports fall into the keyword trap",
                             "Real reports ABOUT debunks (containing 'viral', 'false', 'fake') are classified correctly "
                             "less often than ordinary real news.",
                             "Words like 'viral' and 'false' are fake-cues to a bag-of-words model, but here the news is real.",
                             int((pr[deb] == 0).sum()), int(deb.sum()), int((pr[std] == 0).sum()), int(std.sum()),
                             "debunk_reports", "standard_real", alpha, best))
    else:
        H.append(_H("H8", "Debunk reports fall into the keyword trap", "Debunk reports are misclassified more often.",
                    "Keyword cues vs. meaning.", "Fisher exact test", NT,
                    explanation="Not testable: needs a 'case_type' column with debunk_report rows (synthetic data has it)."))

    # H9 fine-tune vs embeddings
    pairs = [b for b in {k[:-3] for k in results if k.endswith("_ft")} if f"{b}_emb" in results]
    if pairs:
        subs = [_pair_hyp("H9", "", "", "", f"{b}_ft", f"{b}_emb", y, df, None, alpha, n_boot, seed, thr) for b in sorted(pairs)]
        vs = [s["verdict"] for s in subs]
        v = SUP if all(x == SUP for x in vs) else REJ if all(x == REJ for x in vs) else INC
        H.append(_H("H9", "Fine-tuning beats frozen embeddings",
                    "For the same backbone, fine-tuning all weights beats a Logistic Regression on frozen embeddings.",
                    "The gap shows how much task-specific learning adds on top of pretrained knowledge.",
                    "McNemar per backbone; SUPPORTED only if every backbone is significant in favour of fine-tuning",
                    v, {s["models_used"][0]: s["numbers"] for s in subs}, min(s["p_value"] for s in subs),
                    float(np.mean([s["effect"] for s in subs])), " ".join(s["explanation"] for s in subs),
                    [m for s in subs for m in s["models_used"]]))
    else:
        H.append(_H("H9", "Fine-tuning beats frozen embeddings", "Fine-tuned > frozen embeddings for the same backbone.",
                    "How much task-specific learning adds.", "McNemar per backbone", NT,
                    explanation="Not testable: train the same backbone in both modes, e.g. "
                                "`train-bert --models distilbert --mode embed` and `--mode finetune`."))

    # H10 ensemble
    if f"p__{ENSEMBLE}" in df and bs and bt:
        best_member = max([bs, bt], key=lambda m: selection_key(results, m))
        h = _pair_hyp("H10", "An ensemble beats its members",
                      "Averaging the probabilities of the best statistical model and the best transformer beats the "
                      "better of the two on its own.",
                      "The two families make different kinds of mistakes, so averaging can cancel some of them.",
                      ENSEMBLE, best_member, y, df, None, alpha, n_boot, seed, thr)
        H.append(h)
    else:
        H.append(_H("H10", "An ensemble beats its members", "Ensemble > best member.", "Different error types.",
                    "McNemar", NT, explanation="Not testable: needs one statistical and one transformer model, then `compare`."))

    # H11 calibration
    if bs and bt:
        pa, pb = df[f"p__{bt}"].to_numpy(), df[f"p__{bs}"].to_numpy()
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            diffs.append(ece(y[i], pa[i]) - ece(y[i], pb[i]))
        diffs = np.array(diffs)
        obs = ece(y, pa) - ece(y, pb)
        p = float(min(1.0, max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / n_boot)))
        v = _verdict(obs, p, alpha)
        H.append(_H("H11", "Transformers are over-confident",
                    "The best transformer is worse calibrated (higher expected calibration error) than the best "
                    "statistical model.", "'90% sure' should mean right 90% of the time; over-confident models mislead users.",
                    "Paired bootstrap of the ECE difference", v,
                    {"ece_transformer": ece(y, pa), "ece_statistical": ece(y, pb), "diff": obs,
                     "ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}, p, float(obs),
                    f"ECE {bt} = {ece(y, pa):.3f} vs {bs} = {ece(y, pb):.3f} (difference {obs:+.3f}, bootstrap p = {p:.4f}).",
                    [bt, bs]))
    else:
        H.append(_H("H11", "Transformers are over-confident", "Higher ECE for the best transformer.", "Calibration.",
                    "Paired bootstrap", NT, explanation="Not testable: needs a trained transformer."))

    # H12 template memorisation / H13 time
    def heldout_hyp(hid, probe, title, statement, why):
        if audit.empty or probe not in set(audit["probe"]):
            return _H(hid, title, statement, why, "Held-out split vs random split", NT,
                      explanation=f"Not testable: run `audit` (needs a {'template_id' if 'template' in probe else 'source'} column).")
        r = audit[audit["probe"] == probe].iloc[0]
        drop = r["reference_macro_f1"] - r["macro_f1"]
        v = SUP if r["ci_high"] < r["reference_macro_f1"] else INC
        expl = (f"Random split macro-F1 {r['reference_macro_f1']:.3f} vs {probe} {r['macro_f1']:.3f} "
                f"(95% CI {r['ci_low']:.3f}-{r['ci_high']:.3f}); drop = {drop:.3f}. SUPPORTED if the held-out CI lies "
                f"entirely below the random-split score.")
        return _H(hid, title, statement, why, "Held-out split vs random split (TF-IDF + LR)", v,
                  {"random_split_f1": r["reference_macro_f1"], "held_out_f1": r["macro_f1"], "drop": drop,
                   "ci": [r["ci_low"], r["ci_high"]]}, None, float(drop), expl, ["tfidf_lr"])

    H.append(heldout_hyp("H12", "template_held_out", "Models memorise templates",
                         "Scores fall when whole templates are held out of training (the model memorised templates).",
                         "Explains why high synthetic scores mean little (BUILD_INSTRUCTIONS 16.5)."))
    H.append(heldout_hyp("H13", "source_held_out", "Unseen sources are harder",
                         "Scores fall when whole sources/channels are held out of training.",
                         "Real systems meet new channels every day."))

    # H14 style sweep experiment (global)
    sw = Path(cfg["paths"]["outputs_dir"]) / "_experiments" / "style_sweep.csv"
    title14 = "The style shortcut is made by the data"
    stmt14 = ("The more the data ties sensational style to fake items (style correlation s), the better a style-only "
              "model does: its macro-F1 rises with s, and is clearly higher at the largest s than at s = 0.5.")
    if sw.exists():
        s = pd.read_csv(sw).sort_values("style_correlation")
        rho, p = stats.spearmanr(s["style_correlation"], s["style_only_macro_f1"])
        lo_row, hi_row = s.iloc[0], s.iloc[-1]
        separated = hi_row["style_ci_low"] > lo_row["style_ci_high"]
        rises = hi_row["style_only_macro_f1"] > lo_row["style_only_macro_f1"]
        v = SUP if (rho > 0 and p < alpha and separated) else (REJ if (rho < 0 and p < alpha) else INC)
        resid = lo_row["style_only_macro_f1"] - lo_row["chance_macro_f1"]
        H.append(_H("H14", title14, stmt14,
                    "Shows shortcut learning is a property of the dataset, which we can switch on and off.",
                    "Spearman correlation between s and style-only macro-F1 over the sweep, plus non-overlapping "
                    "95% CIs at the lowest and highest s", v,
                    {"spearman_rho": float(rho), "points": s[["style_correlation", "style_only_macro_f1", "style_ci_low",
                                                              "style_ci_high", "full_text_macro_f1"]].to_dict("records"),
                     "residual_above_chance_at_lowest_s": float(resid)},
                    float(p), float(hi_row["style_only_macro_f1"] - lo_row["style_only_macro_f1"]),
                    f"Spearman rho = {rho:.2f} (p = {p:.4f}). Style-only macro-F1 goes from "
                    f"{lo_row['style_only_macro_f1']:.3f} (CI {lo_row['style_ci_low']:.3f}-{lo_row['style_ci_high']:.3f}) at "
                    f"s = {lo_row['style_correlation']} to {hi_row['style_only_macro_f1']:.3f} (CI "
                    f"{hi_row['style_ci_low']:.3f}-{hi_row['style_ci_high']:.3f}) at s = {hi_row['style_correlation']}"
                    f"{'' if rises else ' (it does not rise)'}. Even at s = {lo_row['style_correlation']} the probe stays "
                    f"{resid:.3f} above chance: the templates themselves differ in style (fake templates contain '!' "
                    "and differ in length/script), which the style knob does not control. That residual is itself a "
                    "warning about built-in style differences in a dataset.", ["style probe"]))
    else:
        H.append(_H("H14", title14, stmt14, "Dataset-made shortcuts.", "Style sweep", NT,
                    explanation="Not testable: run `style-sweep` first."))

    synth = "WARNING" in load_card(cfg, name)
    payload = {"dataset": name, "created": now_iso(), "alpha": alpha, "synthetic": synth,
               "verdict_meaning": {SUP: "predicted effect found and significant (evidence FOR = 'proved' on this data)",
                                   REJ: "opposite effect found and significant (evidence AGAINST = 'disproved')",
                                   INC: "no significant difference either way; NOT the same as disproved",
                                   NT: "models/columns needed for the test are missing"},
               "hypotheses": H}
    rdir = ensure_dir(out / "results")
    write_json(rdir / "hypotheses.json", payload)
    _write_md(out / "reports" / "hypotheses.md", payload)
    append_run_log(cfg["paths"]["outputs_dir"], name, "hypotheses",
                   {h["id"]: h["verdict"] for h in H})
    if not quiet:
        print()
        for h in H:
            print(f"{h['id']:<4} {h['verdict']:<13} {h['title']}")
        print()
        log(f"Saved {rdir / 'hypotheses.json'} and {out / 'reports' / 'hypotheses.md'}")
    return H


def _write_md(path: Path, payload):
    ensure_dir(path.parent)
    L = [f"# Hypotheses: `{payload['dataset']}`", "",
         f"_Generated automatically on {payload['created']} from the result files. alpha = {payload['alpha']}._", ""]
    if payload["synthetic"]:
        L += ["> **SYNTHETIC DATA.** Verdicts here demonstrate the method. Re-run on your real dataset before "
              "making research claims.", ""]
    L += ["| Verdict | Meaning |", "|---|---|"] + [f"| {k} | {v} |" for k, v in payload["verdict_meaning"].items()] + [""]
    L += ["| ID | Hypothesis | Verdict | p-value | Effect |", "|---|---|---|---|---|"]
    for h in payload["hypotheses"]:
        p = "-" if h["p_value"] is None else f"{h['p_value']:.4f}"
        e = "-" if h["effect"] is None else f"{h['effect']:+.3f}"
        L.append(f"| {h['id']} | {h['title']} | **{h['verdict']}** | {p} | {e} |")
    L.append("")
    for h in payload["hypotheses"]:
        L += [f"## {h['id']}: {h['title']}", "", f"**Hypothesis.** {h['statement']}", "",
              f"**Why it matters.** {h['why_it_matters']}", "", f"**Test.** {h['test']}", "",
              f"**Result.** {h['explanation']}", "", f"**Verdict: {h['verdict']}**", ""]
    path.write_text("\n".join(L), encoding="utf-8")
