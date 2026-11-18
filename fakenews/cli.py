"""Phase 6: terminal commands.  Run `python -m fakenews --help` or `python -m fakenews <command> --help`."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .config import load_config
from .utils import PROJECT_ROOT, FakeNewsError, log


def _common(p):
    p.add_argument("--config", help="path to a config.yaml (default: configs/config.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config value, e.g. --set transformers.epochs=2 (repeatable)")


def _cfg(a):
    return load_config(a.config, a.overrides)


def cmd_make_data(a):
    from .synthetic import make_dataset

    cfg = _cfg(a)
    mix = None
    if a.language_mix:
        try:
            en, hg, hi = (float(x) for x in a.language_mix.split(","))
        except ValueError:
            raise FakeNewsError("--language-mix must look like 0.6,0.25,0.15 (en,hinglish,hi)")
        mix = {"en": en, "hinglish": hg, "hi": hi}
    make_dataset(cfg, a.rows, a.seed, a.style_correlation, a.label_noise, mix, a.out)


def cmd_prepare(a):
    from .data import prepare

    cfg = _cfg(a)
    if a.seed is not None:
        cfg["seed"] = a.seed
    prepare(cfg, a.source, a.path, a.name, a.text_col, a.bert_text_col, a.label_col, a.fake_value, a.split,
            a.date_col, a.source_col, a.language_col, a.already_clean, a.max_rows)


def cmd_train_classical(a):
    from .classical import train_classical

    train_classical(_cfg(a), a.name, a.models.split(",") if a.models else None)


def cmd_train_bert(a):
    from .transformer_models import train_transformers

    cfg = _cfg(a)
    t = cfg["transformers"]
    for k in ("epochs", "max_length", "batch_size", "train_subset", "device", "truncation"):
        v = getattr(a, k)
        if v is not None:
            t[k] = v
    if a.lr is not None:
        t["learning_rate"] = t["learning_rate_small"] = a.lr
    train_transformers(cfg, a.name, a.models.split(",") if a.models else None, a.mode, a.seeds)


def cmd_compare(a):
    from .compare import run_compare

    run_compare(_cfg(a), a.name)


def cmd_predict(a):
    import pandas as pd

    from .predict import predict_texts
    from .utils import get_device

    cfg = _cfg(a)
    if a.text:
        texts = [a.text]
    elif a.file:
        from .data import read_table
        df = read_table(a.file)
        col = a.text_col or ("text" if "text" in df.columns else df.columns[0])
        if col not in df.columns:
            raise FakeNewsError(f"Column '{col}' not in {list(df.columns)}")
        texts = df[col].astype(str).tolist()
    else:
        raise FakeNewsError("Give --text \"...\" or --file path.csv")
    res = predict_texts(cfg, a.name, texts, a.models.split(",") if a.models else None, get_device(a.device))
    wide = res.pivot_table(index=["item", "text"], columns="model", values="p_fake").reset_index()
    if a.out:
        wide.to_csv(a.out, index=False, encoding="utf-8-sig")
        log(f"Saved {a.out}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    thr = cfg["evaluation"]["threshold"]
    for i, g in res.groupby("item"):
        print(f"\nText: {g['text'].iloc[0]}")
        print(f"  {'model':<16} {'family':<12} {'P(fake)':>8}  prediction   ms/item")
        for _, r in g.sort_values("p_fake", ascending=False).iterrows():
            print(f"  {r['model']:<16} {r['family']:<12} {r['p_fake']:>8.3f}  {r['prediction']:<10} {r['ms']:>8.2f}")
        n_fake = int((g["p_fake"] >= thr).sum())
        print(f"  -> {n_fake} of {len(g)} models say fake. (A model judges wording, not facts.)")


def cmd_audit(a):
    from .audit import run_audit

    run_audit(_cfg(a), a.name)


def cmd_style_sweep(a):
    from .audit import run_style_sweep

    vals = [float(x) for x in a.values.split(",")] if a.values else None
    run_style_sweep(_cfg(a), vals, a.rows)


def cmd_hypotheses(a):
    from .hypotheses import run_hypotheses

    run_hypotheses(_cfg(a), a.name)


def cmd_run_all(a):
    from .audit import run_audit, run_style_sweep
    from .classical import train_classical
    from .compare import run_compare
    from .data import prepare
    from .hypotheses import run_hypotheses
    from .synthetic import make_dataset
    from .transformer_models import train_transformers

    cfg = _cfg(a)
    t0 = time.time()
    name = a.name
    if a.full:
        models, epochs, emb = ["distilbert", "albert", "mbert", "muril"], cfg["transformers"]["epochs"], ["distilbert", "mbert"]
    else:
        models, epochs, emb = ["distilbert"], 1, ["distilbert"]
    if a.models:
        models = a.models.split(",")
    cfg["transformers"]["epochs"] = epochs
    if a.rows:
        cfg["synthetic"]["n_rows"] = a.rows
    log("STEP 1/8 make-data")
    path = make_dataset(cfg)
    log("STEP 2/8 prepare")
    prepare(cfg, "synthetic", path, name)
    log("STEP 3/8 train-classical")
    train_classical(cfg, name)
    if not a.skip_bert:
        log(f"STEP 4/8 train-bert finetune {models} ({epochs} epoch(s))")
        train_transformers(cfg, name, models, "finetune")
        log(f"STEP 5/8 train-bert embeddings {emb}")
        train_transformers(cfg, name, emb, "embed")
    log("STEP 6/8 style-sweep (classical, fast)")
    run_style_sweep(cfg, quiet=True)
    log("STEP 7/8 compare + audit")
    run_compare(cfg, name, quiet=True)
    run_audit(cfg, name, quiet=True)
    log("STEP 8/8 hypotheses + final report")
    run_hypotheses(cfg, name, quiet=True)
    run_compare(cfg, name)  # re-run so the report includes the hypotheses
    run_hypotheses(cfg, name)
    log(f"run-all finished in {(time.time() - t0) / 60:.1f} min. Start the UI with: python -m fakenews app")


def cmd_app(a):
    app = PROJECT_ROOT / "app" / "Home.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port)]
    if a.headless:
        cmd += ["--server.headless", "true"]
    log("Starting: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=False, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        pass


def build_parser():
    p = argparse.ArgumentParser(prog="python -m fakenews",
                                description="SatyaCheck-Text: fake vs. real news with BoW, TF-IDF and BERT-family models.")
    sub = p.add_subparsers(dest="command", metavar="command")

    s = sub.add_parser("make-data", help="generate the synthetic dataset (Phase 1)")
    _common(s)
    s.add_argument("--rows", type=int, help="number of rows (default 3000)")
    s.add_argument("--seed", type=int)
    s.add_argument("--style-correlation", type=float, help="P(sensational|fake); 0.5 removes the style clue")
    s.add_argument("--label-noise", type=float, help="share of labels flipped (default 0.03)")
    s.add_argument("--language-mix", help="en,hinglish,hi shares, e.g. 0.6,0.25,0.15")
    s.add_argument("--out", help="output CSV path (default data/synthetic/synthetic_news.csv)")
    s.set_defaults(func=cmd_make_data)

    s = sub.add_parser("prepare", help="load a dataset, deduplicate, split once, write a data card (Phase 2/8)")
    _common(s)
    s.add_argument("--source", choices=["synthetic", "liar", "csv"], default="synthetic")
    s.add_argument("--path", help="file path (CSV/Excel/Parquet) for --source csv")
    s.add_argument("--name", help="dataset name used for data/processed/<name> and outputs/<name>")
    s.add_argument("--text-col", default="text", help="text column (used by BoW/TF-IDF)")
    s.add_argument("--bert-text-col", help="optional lightly cleaned text column for BERT models")
    s.add_argument("--label-col", default="label")
    s.add_argument("--fake-value", help="which label value means fake (REQUIRED for numeric labels)")
    s.add_argument("--split", choices=["random", "time", "group"], help="split strategy (default from config)")
    s.add_argument("--date-col", help="date column (enables --split time)")
    s.add_argument("--source-col", help="source/channel column (enables --split group and the source audit)")
    s.add_argument("--language-col", help="language column (otherwise detected heuristically)")
    s.add_argument("--already-clean", action="store_true", help="use the text column exactly as it is for BoW/TF-IDF")
    s.add_argument("--max-rows", type=int, help="random sample of N rows (quick tests)")
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train-classical", help="train the 5 BoW / TF-IDF models (Phase 3)")
    _common(s)
    s.add_argument("--name", default="synthetic")
    s.add_argument("--models", help="comma list from: bow_nb,bow_lr,tfidf_lr,tfidf_svm,tfidfchar_lr")
    s.set_defaults(func=cmd_train_classical)

    s = sub.add_parser("train-bert", help="fine-tune or embed BERT-family models (Phase 4)")
    _common(s)
    s.add_argument("--name", default="synthetic")
    s.add_argument("--models", help="comma list from: bert,distilbert,roberta,albert,mbert,xlmr,muril,indicbert,hingbert")
    s.add_argument("--mode", choices=["finetune", "embed"])
    s.add_argument("--epochs", type=int)
    s.add_argument("--max-length", type=int)
    s.add_argument("--batch-size", type=int)
    s.add_argument("--lr", type=float, help="learning rate for all models")
    s.add_argument("--train-subset", type=int, help="use only N training rows")
    s.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    s.add_argument("--truncation", choices=["head", "head_tail"])
    s.add_argument("--seeds", type=int, help="number of seeds (3 for paper numbers)")
    s.set_defaults(func=cmd_train_bert)

    s = sub.add_parser("compare", help="leaderboard, significance, breakdowns, figures, report (Phase 5)")
    _common(s)
    s.add_argument("--name", default="synthetic")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("predict", help="classify a text or a file with every saved model")
    _common(s)
    s.add_argument("--name", default="synthetic")
    s.add_argument("--text")
    s.add_argument("--file")
    s.add_argument("--text-col")
    s.add_argument("--models")
    s.add_argument("--device", default="cpu", choices=["auto", "cuda", "mps", "cpu"])
    s.add_argument("--out", help="save the probabilities to this CSV")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("audit", help="shortcut probes (Phase 9)")
    _common(s)
    s.add_argument("--name", default="synthetic")
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("style-sweep", help="experiment: style shortcut vs. style correlation (synthetic)")
    _common(s)
    s.add_argument("--values", help="comma list, e.g. 0.5,0.7,0.9")
    s.add_argument("--rows", type=int)
    s.set_defaults(func=cmd_style_sweep)

    s = sub.add_parser("hypotheses", help="test the research hypotheses and write reports/hypotheses.md")
    _common(s)
    s.add_argument("--name", default="synthetic")
    s.set_defaults(func=cmd_hypotheses)

    s = sub.add_parser("run-all", help="one-command demo on synthetic data")
    _common(s)
    s.add_argument("--name", default="synthetic")
    s.add_argument("--full", action="store_true", help="4 transformers x 3 epochs (default: quick = distilbert, 1 epoch)")
    s.add_argument("--models", help="override the transformer list")
    s.add_argument("--rows", type=int, help="synthetic rows (default 3000)")
    s.add_argument("--skip-bert", action="store_true", help="statistical models only (about 1 minute)")
    s.set_defaults(func=cmd_run_all)

    s = sub.add_parser("app", help="start the Streamlit research UI")
    s.add_argument("--port", type=int, default=8501)
    s.add_argument("--headless", action="store_true")
    s.set_defaults(func=cmd_app)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "func", None):
        p.print_help()
        return 0
    try:
        a.func(a)
    except FakeNewsError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0
