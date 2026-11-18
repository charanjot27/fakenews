"""Phase 9: shortcut audit. Does the model learn CONTENT, or an easy SHORTCUT?

Each probe is a deliberately limited model. Shortcut Index = (probe F1 - chance F1) / (full F1 - chance F1).
  ~0  -> the shortcut explains little of the model's skill
  ~1+ -> the shortcut explains almost all of it
'Held-out' probes (source / time / template) re-split the data and report the DROP versus a random split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import figures as F
from .compare import load_predictions, pick_best
from .config import output_dir
from .data import group_split, load_card, load_splits
from .evaluate import bootstrap_ci, macro_f1
from .preprocess import TOKEN_PATTERN, style_features
from .utils import FakeNewsError, append_run_log, ensure_dir, log, now_iso, write_json

META_TEXT = ["speaker", "party", "state", "job", "context", "subject"]


def chance_macro_f1(y_test, share_fake_train, seed=42, n=300) -> float:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_test)
    return float(np.mean([macro_f1(y, (rng.random(len(y)) < share_fake_train).astype(int)) for _ in range(n)]))


def _tfidf_lr(seed):
    return make_pipeline(TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 2), min_df=2,
                                         max_df=0.95, sublinear_tf=True),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, solver="liblinear",
                                            random_state=seed))


def _fit_eval(model, Xtr, ytr, Xte, yte, seed, n_boot):
    model.fit(Xtr, ytr)
    pred = (model.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    f1 = macro_f1(yte, pred)
    lo, hi = bootstrap_ci(yte, pred, n=n_boot, seed=seed)
    return f1, lo, hi, pred


def _style_matrix(texts):
    return pd.DataFrame([style_features(t) for t in texts])


def run_audit(cfg: dict, name: str, quiet=False) -> pd.DataFrame:
    seed = int(cfg["seed"])
    n_boot = int(cfg["evaluation"]["bootstrap"])
    sp = load_splits(cfg, name)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    tr_full = pd.concat([tr, va])  # probes are not tuned, so they may use train + val
    yte = te["label_id"].to_numpy()
    chance = chance_macro_f1(yte, tr_full["label_id"].mean(), seed)
    df_test, results = load_predictions(cfg, name, "test")
    if not results:
        raise FakeNewsError(f"Train models for '{name}' first (train-classical).")
    ref_model = pick_best(results, "Statistical") or next(iter(results))
    ref_f1 = results[ref_model]["test_metrics"]["macro_f1"]
    rows = []

    def add(probe, sees, f1, lo, hi, reference, ref, meaning, kind="probe", extra=None):
        denom = ref - chance
        idx = (f1 - chance) / denom if abs(denom) > 1e-9 else np.nan
        idx_lo = (lo - chance) / denom if abs(denom) > 1e-9 else np.nan
        idx_hi = (hi - chance) / denom if abs(denom) > 1e-9 else np.nan
        rows.append({"probe": probe, "kind": kind, "input_seen": sees, "macro_f1": f1, "ci_low": lo, "ci_high": hi,
                     "reference_model": reference, "reference_macro_f1": ref, "chance_macro_f1": chance,
                     "shortcut_index": idx, "index_ci_low": idx_lo, "index_ci_high": idx_hi,
                     "drop_vs_reference": ref - f1, "meaning": meaning, **(extra or {})})
        if not quiet:
            log(f"probe {probe:<22} macro-F1 {f1:.3f} [{lo:.3f}, {hi:.3f}]  ref {ref:.3f}  index {idx:.2f}")

    # 1. style only
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    f1, lo, hi, _ = _fit_eval(m, _style_matrix(tr_full["text_bert"]), tr_full["label_id"],
                              _style_matrix(te["text_bert"]), yte, seed, n_boot)
    add("style_only", "length, '!' count, capitals share, emoji count, link, script", f1, lo, hi, ref_model, ref_f1,
        "Labels can be guessed from writing style alone.")

    # 2. first N words
    n = int(cfg["audit"]["first_n_words"])
    first = lambda s: s.map(lambda t: " ".join(str(t).split()[:n]))  # noqa: E731
    f1, lo, hi, _ = _fit_eval(_tfidf_lr(seed), first(tr_full["text"]), tr_full["label_id"], first(te["text"]), yte, seed, n_boot)
    add(f"first_{n}_words", f"only the first {n} words (TF-IDF + LR)", f1, lo, hi, ref_model, ref_f1,
        "The opening of a post gives the answer away.")

    # 3. shuffled words for the best transformer (if trained) and the best statistical model
    rng = np.random.default_rng(seed)
    shuffled = te["text_bert"].map(lambda t: " ".join(rng.permutation(str(t).split())))
    from .predict import get_predictor
    for fam in ("Transformer", "Statistical"):
        bm = pick_best(results, fam)
        if not bm:
            continue
        try:
            fn = get_predictor(cfg, name, bm)
            pred = (np.asarray(fn(list(shuffled))) >= 0.5).astype(int)
        except Exception as e:  # noqa: BLE001
            log(f"shuffle probe skipped for {bm}: {e}")
            continue
        f1 = macro_f1(yte, pred)
        lo, hi = bootstrap_ci(yte, pred, n=n_boot, seed=seed)
        rf = results[bm]["test_metrics"]["macro_f1"]
        add(f"shuffled_words[{bm}]", "all words in random order (same trained model)", f1, lo, hi, bm, rf,
            "High = word order and context barely matter to this model.")

    # 4. source only
    if "source" in te.columns and te["source"].notna().any():
        m = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000, class_weight="balanced"))
        f1, lo, hi, _ = _fit_eval(m, tr_full[["source"]].astype(str), tr_full["label_id"], te[["source"]].astype(str),
                                  yte, seed, n_boot)
        add("source_only", "only the source / channel name", f1, lo, hi, ref_model, ref_f1,
            "The model could learn WHO posted instead of WHAT is said.")

    # 5. metadata only (e.g. LIAR speaker/party/context)
    meta = [c for c in META_TEXT if c in te.columns and te[c].notna().any()]
    if meta:
        join = lambda d: d[meta].fillna("").astype(str).agg(" | ".join, axis=1)  # noqa: E731
        mt = make_pipeline(TfidfVectorizer(lowercase=True, min_df=1),
                           LogisticRegression(max_iter=3000, class_weight="balanced"))
        f1a, loa, hia, _ = _fit_eval(mt, join(tr_full), tr_full["label_id"], join(te), yte, seed, n_boot)
        add("metadata_only", f"only metadata: {', '.join(meta)}", f1a, loa, hia, ref_model, ref_f1,
            "Who said it (speaker, party, context) predicts the label without reading the claim.")

    # 6-8. held-out splits on pooled data vs. a random split of the same pooled data
    pooled = pd.concat([tr, va, te]).reset_index(drop=True)
    frac = [cfg["split"]["train"] + cfg["split"]["val"], 0.0, cfg["split"]["test"]]

    def random_ref():
        from sklearn.model_selection import train_test_split
        a, b = train_test_split(pooled, test_size=frac[2], stratify=pooled["label_id"], random_state=seed)
        f1, lo, hi, _ = _fit_eval(_tfidf_lr(seed), a["text"], a["label_id"], b["text"], b["label_id"].to_numpy(), seed, n_boot)
        return f1

    rand_f1 = None
    for col, probe, meaning in (("source", "source_held_out", "Big drop = the model leaned on source-specific wording."),
                                ("template_id", "template_held_out", "Big drop = the model memorised templates "
                                 "rather than learning the task (synthetic data only)."),
                                ("date", "time_split", "Big drop = the model does not handle new rumours.")):
        if col not in pooled.columns or pooled[col].isna().all() or pooled[col].nunique() < 3:
            continue
        if rand_f1 is None:
            rand_f1 = random_ref()
        if col == "date":
            d = pooled.assign(_d=pd.to_datetime(pooled["date"], errors="coerce")).dropna(subset=["_d"]).sort_values("_d")
            cut = int(len(d) * (1 - frac[2]))
            a, b = d.iloc[:cut], d.iloc[cut:]
        else:
            a1, a2, b = group_split(pooled, col, [cfg["split"]["train"], cfg["split"]["val"], frac[2]], seed)
            a = pd.concat([a1, a2])
        if b["label_id"].nunique() < 2 or a["label_id"].nunique() < 2:
            log(f"{probe}: a split has only one class; skipped")
            continue
        f1, lo, hi, _ = _fit_eval(_tfidf_lr(seed), a["text"], a["label_id"], b["text"], b["label_id"].to_numpy(), seed, n_boot)
        add(probe, f"full text (TF-IDF + LR), split by {col}", f1, lo, hi, "tfidf_lr (random split, pooled data)",
            rand_f1, meaning, kind="held_out", extra={"n_test": len(b)})

    out = pd.DataFrame(rows)
    rdir, fdir = ensure_dir(output_dir(cfg, name) / "results"), ensure_dir(output_dir(cfg, name) / "figures")
    out.to_csv(rdir / "audit.csv", index=False)
    write_json(rdir / "audit.json", {"dataset": name, "created": now_iso(), "chance_macro_f1": chance,
                                     "reference_model": ref_model, "reference_macro_f1": ref_f1,
                                     "synthetic": "WARNING" in load_card(cfg, name), "probes": out.to_dict("records")})
    F.shortcut_bar(out[out["kind"] == "probe"], fdir / "shortcut_index.png")
    append_run_log(cfg["paths"]["outputs_dir"], name, "audit", {"n_probes": len(out)})
    if not quiet:
        log(f"Chance macro-F1 = {chance:.3f}; reference = {ref_model} ({ref_f1:.3f}). Saved {rdir / 'audit.csv'}")
    return out


def run_style_sweep(cfg: dict, values=None, n_rows=None, quiet=False) -> pd.DataFrame:
    """Experiment: generate synthetic data at several style correlations and watch the style shortcut appear/vanish."""
    from sklearn.model_selection import train_test_split

    from .preprocess import bert_view, classical_view
    from .synthetic import generate

    s = cfg["synthetic"]
    seed = int(cfg["seed"])
    values = values or cfg["style_sweep"]["values"]
    n_rows = int(n_rows or cfg["style_sweep"]["n_rows"])
    rows = []
    for v in values:
        df = generate(n_rows=n_rows, seed=seed, style_correlation=float(v), label_noise=s["label_noise"],
                      language_mix=s["language_mix"])
        df["text_c"] = df["text"].map(classical_view)
        df["text_b"] = df["text"].map(bert_view)
        a, b = train_test_split(df, test_size=0.3, stratify=df["label_id"], random_state=seed)
        yb = b["label_id"].to_numpy()
        full, flo, fhi, _ = _fit_eval(_tfidf_lr(seed), a["text_c"], a["label_id"], b["text_c"], yb, seed, 500)
        st = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        sty, slo, shi, _ = _fit_eval(st, _style_matrix(a["text_b"]), a["label_id"], _style_matrix(b["text_b"]), yb, seed, 500)
        ch = chance_macro_f1(yb, a["label_id"].mean(), seed)
        rows.append({"style_correlation": float(v), "full_text_macro_f1": full, "full_ci_low": flo, "full_ci_high": fhi,
                     "style_only_macro_f1": sty, "style_ci_low": slo, "style_ci_high": shi, "chance_macro_f1": ch,
                     "shortcut_index": (sty - ch) / (full - ch) if full - ch > 1e-9 else np.nan, "n_rows": n_rows})
        if not quiet:
            log(f"s={v:.2f}: full text {full:.3f} | style-only {sty:.3f} [{slo:.3f}, {shi:.3f}] | chance {ch:.3f}")
    out = pd.DataFrame(rows)
    d = ensure_dir(Path(cfg["paths"]["outputs_dir"]) / "_experiments")
    out.to_csv(d / "style_sweep.csv", index=False)
    F.style_sweep_chart(out, d / "style_sweep.png")
    if not quiet:
        log(f"Saved {d / 'style_sweep.csv'} and style_sweep.png")
    return out
