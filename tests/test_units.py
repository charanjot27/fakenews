import numpy as np
import pandas as pd
import pytest

from fakenews.data import map_labels
from fakenews.evaluate import compute_metrics, ece, macro_f1, mcnemar
from fakenews.preprocess import classical_view, detect_language, tokenize
from fakenews.synthetic import generate
from fakenews.utils import FakeNewsError


# ---------------------------------------------------------------- labels
def test_text_labels_mapped():
    s = pd.Series(["FAKE", "real", "pants-fire", "mostly-true", "Hoax", "genuine"])
    assert map_labels(s).tolist() == [1, 0, 1, 0, 1, 0]


def test_numeric_labels_never_guessed():
    with pytest.raises(FakeNewsError, match="Numeric labels are never guessed"):
        map_labels(pd.Series([0, 1, 1, 0]))


def test_numeric_labels_with_fake_value():
    # WELFake convention: 0 means fake
    assert map_labels(pd.Series([0, 1, 1, 0]), fake_value=0).tolist() == [1, 0, 0, 1]


def test_unknown_label_stops():
    with pytest.raises(FakeNewsError, match="Unknown label values"):
        map_labels(pd.Series(["fake", "satire"]))


# ---------------------------------------------------------------- text
def test_hindi_words_stay_whole():
    toks = tokenize(classical_view("मौसम विभाग के अनुसार बारिश होगी"))
    assert toks == ["मौसम", "विभाग", "के", "अनुसार", "बारिश", "होगी"]


def test_punctuation_and_emoji_tokens():
    toks = tokenize(classical_view("BREAKING 🚨 free laptop!!! visit https://x.co/abc"))
    assert "!!!" in toks and "🚨" in toks and "urltoken" in toks and "breaking" in toks


def test_language_detection():
    assert detect_language("कल से 500 के नोट बंद") == "hi"
    assert detect_language("Sarkar sabhi students ko free laptop de rahi hai") == "hinglish"
    assert detect_language("The weather department said rain is likely") == "en"


# ---------------------------------------------------------------- metrics on a hand-made example
def test_metrics_hand_example():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.7])  # preds: 1 1 0 0 0 1 -> tp2 fn1 tn2 fp1
    m = compute_metrics(y, p, n_boot=0)
    assert m["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision_fake"] == pytest.approx(2 / 3)
    assert m["recall_fake"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(7 / 9)


def test_ece_perfect_and_bad():
    y = np.array([0, 1] * 50)
    assert ece(y, y.astype(float)) == pytest.approx(0.0)
    assert ece(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_mcnemar_identical_models():
    y = np.array([0, 1, 0, 1])
    assert mcnemar(y, y, y)["p_value"] == 1.0


def test_macro_f1_perfect():
    assert macro_f1([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0


# ---------------------------------------------------------------- synthetic data
def test_synthetic_properties():
    df = generate(n_rows=600, seed=7)
    assert 0.45 <= df["label_id"].mean() <= 0.55
    assert set(df["language"]) == {"en", "hinglish", "hi"}
    assert not df["text"].duplicated().any()
    assert df["text"].str.contains("[ऀ-ॿ]").any()


def test_synthetic_deterministic():
    a, b = generate(n_rows=300, seed=3), generate(n_rows=300, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert not generate(n_rows=300, seed=4).equals(a)


def test_style_knob():
    df = generate(n_rows=1000, seed=1, style_correlation=0.9, label_noise=0.0)
    share = df.groupby("label")["style"].apply(lambda s: (s == "sensational").mean())
    assert share["fake"] > 0.8 and share["real"] < 0.2
