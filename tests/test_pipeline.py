"""End-to-end checks on a 200-row synthetic dataset (statistical models; transformers with RUN_SLOW=1)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import slow
from fakenews.cli import main
from fakenews.utils import FakeNewsError


def test_prepare_no_overlap_and_deterministic(prepared):
    from fakenews.data import load_card, load_splits, prepare

    cfg = prepared
    sp = load_splits(cfg, "tiny")
    texts = [set(sp[s]["text_bert"]) for s in ("train", "val", "test")]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])
    assert load_card(cfg, "tiny")["split_overlap_texts"] == 0
    before = {s: sp[s]["id"].tolist() for s in sp}
    prepare(cfg, "synthetic", Path(cfg["paths"]["data_dir"]) / "synthetic" / "synthetic_news.csv", "tiny")
    after = load_splits(cfg, "tiny")
    assert all(before[s] == after[s]["id"].tolist() for s in before)


def test_time_and_group_splits(cfg, tmp_path):
    from fakenews.data import load_splits, prepare
    from fakenews.synthetic import make_dataset

    path = make_dataset(cfg, n_rows=300)
    prepare(cfg, "csv", path, "t", date_col="date", source_col="source", language_col="language", split="time")
    sp = load_splits(cfg, "t")
    assert sp["train"]["date"].max() <= sp["test"]["date"].min()
    prepare(cfg, "csv", path, "g", date_col="date", source_col="source", split="group")
    sp = load_splits(cfg, "g")
    assert not (set(sp["train"]["source"]) & set(sp["test"]["source"]))


def test_csv_numeric_label_without_fake_value_stops(cfg, tmp_path):
    from fakenews.data import prepare

    f = tmp_path / "num.csv"
    pd.DataFrame({"text": [f"item number {i}" for i in range(30)], "label": [0, 1] * 15}).to_csv(f, index=False)
    with pytest.raises(FakeNewsError, match="--fake-value"):
        prepare(cfg, "csv", str(f), "num")
    prepare(cfg, "csv", str(f), "num", fake_value="1")


def test_missing_column_message(cfg, tmp_path):
    from fakenews.data import prepare

    f = tmp_path / "x.csv"
    pd.DataFrame({"content": ["a b c"] * 10, "label": ["fake", "real"] * 5}).to_csv(f, index=False)
    with pytest.raises(FakeNewsError, match="Columns in the file"):
        prepare(cfg, "csv", str(f), "x")


def test_classical_compare_audit_hypotheses(prepared):
    from fakenews.audit import run_audit, run_style_sweep
    from fakenews.classical import train_classical
    from fakenews.compare import run_compare
    from fakenews.config import output_dir
    from fakenews.hypotheses import run_hypotheses

    cfg = prepared
    res = train_classical(cfg, "tiny")
    assert len(res) == 5
    again = train_classical(cfg, "tiny", ["tfidf_lr"])  # reproducibility
    assert again[0]["test_metrics"] == next(r for r in res if r["model"] == "tfidf_lr")["test_metrics"]
    comp = run_compare(cfg, "tiny", quiet=True)
    out = output_dir(cfg, "tiny")
    for f in ["summary.csv", "predictions.csv", "significance.csv", "breakdowns.csv", "errors.csv", "comparison.json"]:
        assert (out / "results" / f).exists(), f
    for f in ["macro_f1_bar.png", "confusion_matrices.png", "roc_curves.png", "calibration.png", "per_language.png"]:
        assert (out / "figures" / f).exists(), f
    assert "SYNTHETIC" in comp["summary"]
    # every number in the leaderboard equals the per-model result file
    lb = pd.read_csv(out / "results" / "summary.csv")
    for _, r in lb.iterrows():
        j = json.loads((out / "results" / f"{r['model']}.json").read_text())
        assert r["macro_f1"] == pytest.approx(j["test_metrics"]["macro_f1"])
    aud = run_audit(cfg, "tiny", quiet=True)
    assert "style_only" in set(aud["probe"])
    run_style_sweep(cfg, quiet=True)
    H = run_hypotheses(cfg, "tiny", quiet=True)
    assert [h["id"] for h in H] == [f"H{i}" for i in range(1, 15)]
    assert all(h["verdict"] in {"SUPPORTED", "REJECTED", "INCONCLUSIVE", "NOT TESTABLE"} for h in H)
    assert (out / "reports" / "hypotheses.md").exists() and (out / "reports" / "results_report.md").exists()


def test_explain_trace_matches_model(prepared):
    import numpy as np

    from fakenews.classical import load_classical, train_classical
    from fakenews.explain import trace_classical

    cfg = prepared
    train_classical(cfg, "tiny", ["tfidf_lr", "bow_nb"])
    for m in ("tfidf_lr", "bow_nb"):
        tr = trace_classical(load_classical(cfg, "tiny", m), "Garlic water cures dengue, forward to all!")
        assert 1 / (1 + np.exp(-tr["score"])) == pytest.approx(tr["p_fake"], abs=1e-6)


def test_cli_errors(prepared, capsys, monkeypatch):
    import fakenews.cli as cli

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: prepared)
    assert main(["train-bert", "--name", "tiny", "--models", "not_a_model"]) == 2
    assert "Unknown transformer model" in capsys.readouterr().err
    assert main(["compare", "--name", "nope"]) == 2
    assert main(["train-classical", "--name", "tiny", "--models", "bow_nb"]) == 0


@slow
def test_transformer_tiny(prepared):
    from fakenews.compare import run_compare
    from fakenews.transformer_models import train_transformers

    cfg = prepared
    cfg["transformers"]["epochs"] = 1
    r = train_transformers(cfg, "tiny", ["distilbert"], "finetune")[0]
    assert r["curve"] and 0 <= r["test_metrics"]["macro_f1"] <= 1
    train_transformers(cfg, "tiny", ["distilbert"], "embed")
    from fakenews.classical import train_classical
    train_classical(cfg, "tiny", ["tfidf_lr"])
    run_compare(cfg, "tiny", quiet=True)
