"""Word-level explanations: exactly how a BoW/TF-IDF model reaches its probability, and leave-one-word-out
importance for transformers. Used by the Live checker and Model internals pages."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from .classical import linear_weights
from .preprocess import classical_view, tokenize


def trace_classical(pipe, raw_text: str) -> dict:
    """Step-by-step trace: raw text -> cleaned text -> tokens -> feature values -> weights -> score -> P(fake)."""
    vec, clf = pipe.named_steps["features"], pipe.named_steps["clf"]
    clean = classical_view(raw_text)
    x = vec.transform([clean])
    names = vec.get_feature_names_out()
    w = linear_weights(pipe)
    nz = x.nonzero()[1]
    feats = pd.DataFrame({"feature": names[nz], "value": np.asarray(x[0, nz].todense()).ravel(), "weight": w[nz]})
    feats["contribution"] = feats["value"] * feats["weight"]
    feats = feats.sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True)
    p = float(pipe.predict_proba([clean])[0, 1])
    if isinstance(clf, LogisticRegression):
        intercept = float(clf.intercept_[0])
        how = "score = intercept + sum(value x weight); P(fake) = 1 / (1 + e^-score)"
    elif isinstance(clf, MultinomialNB):
        i1, i0 = list(clf.classes_).index(1), list(clf.classes_).index(0)
        intercept = float(clf.class_log_prior_[i1] - clf.class_log_prior_[i0])
        how = ("score = log prior ratio + sum(count x log P(word|fake)/P(word|real)); P(fake) = 1 / (1 + e^-score)")
    else:
        intercept = float(np.mean([cc.estimator.intercept_[0] for cc in clf.calibrated_classifiers_]))
        how = "SVM margin = intercept + sum(value x weight); P(fake) from Platt (sigmoid) calibration of the margin"
    score = intercept + float(feats["contribution"].sum())
    oov = [t for t in tokenize(clean) if t not in vec.vocabulary_] if vec.analyzer == "word" else []
    return {"raw": raw_text, "clean": clean, "tokens": tokenize(clean) if vec.analyzer == "word" else None,
            "unknown_tokens": oov, "features": feats, "intercept": intercept, "score": score, "p_fake": p,
            "how": how, "analyzer": vec.analyzer}


def word_scores_from_trace(trace) -> dict:
    """Spread each feature's contribution over the words it contains (bigrams split evenly)."""
    scores: dict[str, float] = {}
    if trace["analyzer"] != "word":
        toks = tokenize(trace["clean"])
        for _, r in trace["features"].iterrows():
            piece = r["feature"]
            owners = [t for t in toks if piece.strip() and piece in f" {t} "]
            for t in owners:
                scores[t] = scores.get(t, 0.0) + r["contribution"] / len(owners)
        return scores
    for _, r in trace["features"].iterrows():
        parts = r["feature"].split(" ")
        for ptok in parts:
            scores[ptok] = scores.get(ptok, 0.0) + r["contribution"] / len(parts)
    return scores


def highlight_html(raw_text: str, word_scores: dict) -> str:
    """Red = pushes towards fake, green = towards real; opacity = strength."""
    mx = max([abs(v) for v in word_scores.values()] + [1e-9])
    out = []
    for word in raw_text.split():
        key_toks = tokenize(classical_view(word))
        s = sum(word_scores.get(t, 0.0) for t in key_toks)
        a = min(1.0, abs(s) / mx) * 0.85
        col = f"rgba(214,39,40,{a:.2f})" if s > 0 else f"rgba(44,160,44,{a:.2f})"
        out.append(f'<span title="{s:+.3f}" style="background:{col};padding:1px 3px;border-radius:3px;'
                   f'margin:1px;display:inline-block">{html.escape(word)}</span>')
    return " ".join(out)


def occlusion(predict_fn, raw_text: str, max_words: int = 40) -> pd.DataFrame:
    """Remove one word at a time and measure how P(fake) changes (positive delta = the word pushed towards fake)."""
    words = raw_text.split()[:max_words]
    base = float(predict_fn([" ".join(words)])[0])
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    ps = np.asarray(predict_fn(variants)) if variants else np.array([])
    return pd.DataFrame({"word": words, "p_without": ps, "delta": base - ps, "base_p": base})


def occlusion_html(df: pd.DataFrame) -> str:
    return highlight_html(" ".join(df["word"]), {}) if df.empty else " ".join(
        f'<span title="{d:+.3f}" style="background:{"rgba(214,39,40," if d > 0 else "rgba(44,160,44,"}'
        f'{min(1.0, abs(d) / max(1e-9, df["delta"].abs().max())) * 0.85:.2f});padding:1px 3px;border-radius:3px;'
        f'margin:1px;display:inline-block">{html.escape(w)}</span>' for w, d in zip(df["word"], df["delta"]))
