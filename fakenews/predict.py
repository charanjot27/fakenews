"""Find trained models for a dataset and turn any of them into texts -> P(fake), applying the right text view."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import output_dir
from .preprocess import bert_view, classical_view
from .utils import FakeNewsError, read_json, timer


def list_results(cfg, name) -> dict[str, dict]:
    rdir = Path(output_dir(cfg, name)) / "results"
    out = {}
    if rdir.exists():
        for f in sorted(rdir.glob("*.json")):
            if f.name.endswith(".meta.json"):
                continue
            try:
                r = read_json(f)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(r, dict) and "model" in r and "test_metrics" in r:
                out[r["model"]] = r
    return out


def get_predictor(cfg, name, model, device="cpu"):
    res = list_results(cfg, name).get(model)
    if res is None:
        raise FakeNewsError(f"No trained model '{model}' for dataset '{name}'. Trained: {list(list_results(cfg, name))}")
    if res["family"] == "Statistical":
        from .classical import load_classical

        pipe = load_classical(cfg, name, model)
        return lambda texts: pipe.predict_proba([classical_view(t) for t in texts])[:, 1]
    from .transformer_models import load_trained

    fn = load_trained(cfg, name, model, device)
    return lambda texts: fn([bert_view(t) for t in texts])


def predict_texts(cfg, name, texts, models=None, device="cpu") -> pd.DataFrame:
    avail = list_results(cfg, name)
    models = models or list(avail)
    missing = [m for m in models if m not in avail]
    if missing:
        raise FakeNewsError(f"Not trained for '{name}': {missing}. Trained: {list(avail)}")
    thr = cfg["evaluation"]["threshold"]
    rows = []
    for m in models:
        fn = get_predictor(cfg, name, m, device)
        with timer() as t:
            p = fn(list(texts))
        for i, (txt, pi) in enumerate(zip(texts, p)):
            rows.append({"item": i, "text": txt, "model": m, "family": avail[m]["family"], "p_fake": float(pi),
                         "prediction": "fake" if pi >= thr else "real", "ms": 1000 * t["seconds"] / len(texts)})
    return pd.DataFrame(rows)
