"""Phase 3: Bag-of-Words and TF-IDF models (scikit-learn).

Transparency: for every model we save the tuning table (validation score for each setting), the vocabulary
size, the 25 strongest fake/real features, timings and every test probability.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import output_dir
from .data import load_card, load_splits
from .evaluate import compute_metrics, macro_f1
from .preprocess import TOKEN_PATTERN
from .utils import FakeNewsError, append_run_log, ensure_dir, library_versions, log, now_iso, timer, write_json

CLASSICAL_MODELS = {
    "bow_nb": {"label": "BoW + Naive Bayes", "features": "bow", "clf": "nb",
               "about": "Word and word-pair counts; Multinomial Naive Bayes models how often each word appears per class."},
    "bow_lr": {"label": "BoW + Logistic Regression", "features": "bow", "clf": "lr",
               "about": "Word and word-pair counts; Logistic Regression learns one weight per word (balanced classes)."},
    "tfidf_lr": {"label": "TF-IDF + Logistic Regression", "features": "tfidf", "clf": "lr",
                 "about": "Counts re-weighted by rarity (sublinear TF, L2-normalised); usually the strongest simple baseline."},
    "tfidf_svm": {"label": "TF-IDF + Linear SVM", "features": "tfidf", "clf": "svm",
                  "about": "Linear SVM on TF-IDF; calibrated with Platt scaling (3-fold) so it outputs probabilities."},
    "tfidfchar_lr": {"label": "TF-IDF-char + Logistic Regression", "features": "char", "clf": "lr",
                     "about": "Overlapping 3-5 character pieces; robust to Hinglish spelling variants (nahi/nahin/nai)."},
}


def build_pipeline(name: str, cfg: dict, param: float, seed: int) -> Pipeline:
    c = cfg["classical"]
    spec = CLASSICAL_MODELS[name]
    word_kw = dict(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=tuple(c["ngram_range"]),
                   min_df=c["min_df"], max_df=c["max_df"], max_features=c["max_features"])
    if spec["features"] == "bow":
        vec = CountVectorizer(**word_kw)
    elif spec["features"] == "tfidf":
        vec = TfidfVectorizer(sublinear_tf=True, norm="l2", **word_kw)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(c["char_ngram_range"]), lowercase=False,
                              min_df=c["min_df"], max_df=c["max_df"], max_features=c["max_features"],
                              sublinear_tf=True, norm="l2")
    if spec["clf"] == "nb":
        clf = MultinomialNB(alpha=param)
    elif spec["clf"] == "lr":
        clf = LogisticRegression(C=param, class_weight="balanced", max_iter=3000, solver="liblinear",
                                 random_state=seed)
    else:
        clf = CalibratedClassifierCV(LinearSVC(C=param, class_weight="balanced", random_state=seed, max_iter=5000),
                                     cv=3, method="sigmoid")
    return Pipeline([("features", vec), ("clf", clf)])


def linear_weights(pipe: Pipeline) -> np.ndarray:
    """One weight per feature; positive pushes towards fake."""
    clf = pipe.named_steps["clf"]
    if isinstance(clf, MultinomialNB):
        i1, i0 = list(clf.classes_).index(1), list(clf.classes_).index(0)
        return clf.feature_log_prob_[i1] - clf.feature_log_prob_[i0]
    if isinstance(clf, LogisticRegression):
        return clf.coef_[0] * (1 if clf.classes_[1] == 1 else -1)
    if isinstance(clf, CalibratedClassifierCV):
        return np.mean([cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_], axis=0)
    raise TypeError(type(clf))


def top_features(pipe: Pipeline, k: int = 25) -> dict:
    names = pipe.named_steps["features"].get_feature_names_out()
    w = linear_weights(pipe)
    order = np.argsort(w)
    return {"fake": [(str(names[i]), float(w[i])) for i in order[::-1][:k]],
            "real": [(str(names[i]), float(w[i])) for i in order[:k]]}


def _n_params(pipe) -> int:
    clf = pipe.named_steps["clf"]
    nf = len(pipe.named_steps["features"].vocabulary_)
    if isinstance(clf, MultinomialNB):
        return 2 * nf + 2
    if isinstance(clf, CalibratedClassifierCV):
        return len(clf.calibrated_classifiers_) * (nf + 1 + 2)
    return nf + 1


def train_classical(cfg: dict, name: str, models: list[str] | None = None) -> list[dict]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    models = models or cfg["classical"]["models"]
    unknown = [m for m in models if m not in CLASSICAL_MODELS]
    if unknown:
        raise FakeNewsError(f"Unknown classical model(s) {unknown}. Choose from {list(CLASSICAL_MODELS)}")
    seed = int(cfg["seed"])
    sp = load_splits(cfg, name)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    out = output_dir(cfg, name)
    mdir, rdir, pdir = ensure_dir(out / "models"), ensure_dir(out / "results"), ensure_dir(out / "results" / "predictions")
    ev = cfg["evaluation"]
    summaries = []
    for m in models:
        spec = CLASSICAL_MODELS[m]
        grid_key, grid = ("alpha", cfg["classical"]["alpha_grid"]) if spec["clf"] == "nb" else ("C", cfg["classical"]["C_grid"])
        tuning = []
        with timer() as t_all:
            for val in grid:
                pipe = build_pipeline(m, cfg, float(val), seed)
                pipe.fit(tr["text"], tr["label_id"])
                pv = pipe.predict_proba(va["text"])[:, 1]
                tuning.append({grid_key: float(val), "val_macro_f1": macro_f1(va["label_id"], (pv >= ev["threshold"]).astype(int))})
            best = max(tuning, key=lambda r: (r["val_macro_f1"], -abs(np.log(r[grid_key]))))  # tie -> closest to 1
            pipe = build_pipeline(m, cfg, best[grid_key], seed)
            with timer() as t_fit:
                pipe.fit(tr["text"], tr["label_id"])
        with timer() as t_inf:
            p_test = pipe.predict_proba(te["text"])[:, 1]
        p_val = pipe.predict_proba(va["text"])[:, 1]
        test_m = compute_metrics(te["label_id"], p_test, ev["threshold"], ev["bootstrap"], ev["ci"], ev["ece_bins"], seed)
        val_m = compute_metrics(va["label_id"], p_val, ev["threshold"], 0, ev["ci"], ev["ece_bins"], seed)
        model_path = mdir / f"{m}.joblib"
        joblib.dump(pipe, model_path)
        tops = top_features(pipe, cfg["classical"]["top_k"])
        res = {
            "model": m, "label": spec["label"], "family": "Statistical", "mode": "Classical",
            "about": spec["about"], "backbone": None,
            "features": spec["features"], "classifier": spec["clf"],
            "settings": {**{k: cfg["classical"][k] for k in ("ngram_range", "char_ngram_range", "min_df", "max_df",
                                                               "max_features")}, grid_key: best[grid_key],
                         "token_pattern": TOKEN_PATTERN, "input_text": "classical view (column 'text')"},
            "tuning": tuning, "tuned_param": grid_key, "best_value": best[grid_key],
            "vocab_size": len(pipe.named_steps["features"].vocabulary_),
            "top_features": tops,
            "train_seconds": t_all["seconds"], "final_fit_seconds": t_fit["seconds"],
            "infer_ms_per_item": 1000 * t_inf["seconds"] / max(1, len(te)),
            "params": _n_params(pipe), "disk_mb": model_path.stat().st_size / 1e6,
            "device": "cpu", "model_path": str(model_path),
            "val_metrics": val_m, "test_metrics": test_m,
            "dataset": name, "data_card_fingerprint": load_card(cfg, name).get("fingerprint_sha256"),
            "seed": seed, "created": now_iso(), "versions": library_versions(), "config": cfg,
        }
        write_json(rdir / f"{m}.json", res)
        pd.DataFrame({"id": te["id"], "p_fake": p_test}).to_csv(pdir / f"{m}.csv", index=False)
        pd.DataFrame({"id": va["id"], "p_fake": p_val}).to_csv(pdir / f"{m}.val.csv", index=False)
        append_run_log(cfg["paths"]["outputs_dir"], name, "train-classical",
                       {"model": m, grid_key: best[grid_key], "test_macro_f1": test_m["macro_f1"],
                        "seconds": t_all["seconds"]})
        log(f"{spec['label']:<36} {grid_key}={best[grid_key]:<5} val F1={best['val_macro_f1']:.3f}  "
            f"test macro-F1={test_m['macro_f1']:.3f} [{test_m['macro_f1_ci_low']:.3f}, {test_m['macro_f1_ci_high']:.3f}]  "
            f"({t_all['seconds']:.1f}s, vocab {res['vocab_size']})")
        summaries.append(res)
    log(f"Saved models to {mdir} and results to {rdir}")
    return summaries


def load_classical(cfg: dict, name: str, model: str):
    p = Path(output_dir(cfg, name)) / "models" / f"{model}.joblib"
    if not p.exists():
        raise FakeNewsError(f"Model '{model}' is not trained for dataset '{name}'. Run: python -m fakenews "
                            f"train-classical --name {name}")
    return joblib.load(p)
