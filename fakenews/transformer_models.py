"""Phase 4: BERT-family models from Hugging Face.

Two modes:
  finetune : 2-class head on top of the pretrained encoder; the whole network is trained (AdamW, linear warm-up
             and decay, gradient clipping, early stopping on validation macro-F1, best epoch kept).
  embed    : the encoder is frozen; mean-pooled token vectors -> Logistic Regression (C tuned on validation).

The training loop is written out in plain PyTorch (no Trainer black box) so every step can be read and checked.
Everything is logged per epoch: train loss, validation loss, validation macro-F1, seconds.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import output_dir
from .data import load_card, load_splits
from .evaluate import compute_metrics, macro_f1
from .utils import (FakeNewsError, append_run_log, ensure_dir, get_device, library_versions, log, now_iso, set_seed,
                    timer, write_json)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except ImportError:  # pragma: no cover
    pass

REGISTRY = {
    "bert": {"hf": "bert-base-uncased", "size": "110M", "languages": "English", "cased": False, "small": False,
             "why": "The original reference model"},
    "distilbert": {"hf": "distilbert-base-uncased", "size": "66M", "languages": "English", "cased": False,
                   "small": True, "why": "~40% smaller and ~60% faster than BERT; good laptop default"},
    "roberta": {"hf": "roberta-base", "size": "125M", "languages": "English", "cased": True, "small": False,
                "why": "BERT trained longer on more data"},
    "albert": {"hf": "albert-base-v2", "size": "12M", "languages": "English", "cased": False, "small": True,
               "why": "Shares parameters across layers (one reading of 'Ambert')"},
    "mbert": {"hf": "bert-base-multilingual-cased", "size": "178M", "languages": "104 languages incl. Hindi",
              "cased": True, "small": False, "why": "Multilingual baseline (the other reading of 'Ambert')"},
    "xlmr": {"hf": "xlm-roberta-base", "size": "270M", "languages": "100 languages", "cased": True, "small": False,
             "why": "Strong multilingual model"},
    "muril": {"hf": "google/muril-base-cased", "size": "237M",
              "languages": "16 Indian languages + English, incl. Romanised forms", "cased": True, "small": False,
              "why": "Built for Indian languages including transliterated text; key model for this project"},
    "indicbert": {"hf": "ai4bharat/indic-bert", "size": "~33M (ALBERT-based)", "languages": "11 Indian languages + English",
                  "cased": True, "small": True, "why": "Lightweight Indian-language model (AI4Bharat)"},
    "hingbert": {"hf": "l3cube-pune/hing-bert", "size": "110M", "languages": "Hinglish (Roman script)", "cased": True,
                 "small": False, "why": "Trained on code-mixed Hindi-English"},
}
ENGLISH_ONLY = {"bert", "distilbert", "roberta", "albert"}
INDIC = {"mbert", "xlmr", "muril", "indicbert", "hingbert"}


def check_names(names):
    bad = [n for n in names if n not in REGISTRY]
    if bad:
        raise FakeNewsError(f"Unknown transformer model(s) {bad}. Choose from: {', '.join(REGISTRY)}")


def _download_error(short, hf, e):
    msg = str(e)
    if "gated" in msg.lower() or "401" in msg:
        return FakeNewsError(
            f"'{short}' ({hf}) is a GATED model on Hugging Face. Open https://huggingface.co/{hf}, accept its terms, "
            "then log in once on this machine with:  hf auth login   (or set HF_TOKEN). Then re-run.")
    return FakeNewsError(f"Could not download/load '{short}' ({hf}): {msg[:300]}\n"
                         "Check your internet connection (first download only) and the model ID.")


def load_tokenizer(short: str):
    from transformers import AutoTokenizer

    hf = REGISTRY[short]["hf"] if short in REGISTRY else short
    kw = {"keep_accents": True} if short == "indicbert" else {}  # keep Devanagari vowel signs
    try:
        return AutoTokenizer.from_pretrained(hf, **kw)
    except Exception as e:  # noqa: BLE001
        raise _download_error(short, hf, e)


def _load_model(short: str, num_labels=2, base=False):
    from transformers import AutoModel, AutoModelForSequenceClassification

    hf = REGISTRY[short]["hf"]
    try:
        if base:
            return AutoModel.from_pretrained(hf)
        return AutoModelForSequenceClassification.from_pretrained(hf, num_labels=num_labels)
    except Exception as e:  # noqa: BLE001
        raise _download_error(short, hf, e)


def encode(tok, texts, max_length=128, truncation="head"):
    """Token ids for each text. 'head_tail' keeps the first and last parts of long texts."""
    texts = [str(t) for t in texts]
    if truncation != "head_tail":
        return tok(texts, truncation=True, max_length=max_length)
    ids_all, specials = [], tok.num_special_tokens_to_add(pair=False)
    budget = max_length - specials
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            head = budget // 4
            ids = ids[:head] + ids[-(budget - head):]
        ids_all.append(tok.build_inputs_with_special_tokens(ids))
    return {"input_ids": ids_all, "attention_mask": [[1] * len(x) for x in ids_all]}


def _batches(enc, labels, batch_size, shuffle, rng, pad_id):
    import torch

    n = len(enc["input_ids"])
    order = rng.permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        L = max(len(enc["input_ids"][i]) for i in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            x = enc["input_ids"][i]
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        y = torch.tensor(np.asarray(labels)[idx], dtype=torch.long) if labels is not None else None
        yield idx, ids, mask, y


def predict_proba(model, tok, texts, device, max_length=128, batch_size=32, truncation="head"):
    import torch

    model.eval()
    enc = encode(tok, texts, max_length, truncation)
    out = np.zeros(len(texts))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    with torch.no_grad():
        for idx, ids, mask, _ in _batches(enc, None, batch_size, False, None, pad):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            out[idx] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return out


def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


# ------------------------------------------------------------------------------------------ fine-tuning
def _finetune_one(cfg, short, sp, seed, device, tcfg, model_dir: Path | None):
    import torch

    set_seed(seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    tok = load_tokenizer(short)
    model = _load_model(short).to(device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lr = float(tcfg["learning_rate_small"] if REGISTRY[short]["small"] else tcfg["learning_rate"])
    bs, ml, trunc = int(tcfg["batch_size"]), int(tcfg["max_length"]), tcfg["truncation"]
    enc_tr = encode(tok, tr["text_bert"], ml, trunc)
    y_tr = tr["label_id"].to_numpy()
    share = max(y_tr.mean(), 1 - y_tr.mean())
    weights = None
    if share > float(tcfg["class_weight_threshold"]):
        counts = np.bincount(y_tr, minlength=2)
        weights = torch.tensor(len(y_tr) / (2 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": float(tcfg["weight_decay"])},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    epochs = int(tcfg["epochs"])
    steps_per_epoch = math.ceil(len(y_tr) / bs)
    total = steps_per_epoch * epochs
    warm = int(float(tcfg["warmup_ratio"]) * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    rng = np.random.default_rng(seed)
    curve, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    with timer() as t_train:
        for ep in range(1, epochs + 1):
            model.train()
            losses = []
            with timer() as t_ep:
                try:
                    from tqdm import tqdm
                    it = tqdm(_batches(enc_tr, y_tr, bs, True, rng, pad), total=steps_per_epoch,
                              desc=f"{short} epoch {ep}/{epochs}", leave=False)
                except ImportError:
                    it = _batches(enc_tr, y_tr, bs, True, rng, pad)
                for _, ids, mask, y in it:
                    opt.zero_grad()
                    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                    loss = loss_fn(logits, y.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
                    opt.step()
                    sched.step()
                    losses.append(float(loss.item()))
            p_val = predict_proba(model, tok, va["text_bert"], device, ml, bs * 2, trunc)
            yv = va["label_id"].to_numpy()
            val_loss = float(-np.mean(yv * np.log(np.clip(p_val, 1e-7, 1)) + (1 - yv) * np.log(np.clip(1 - p_val, 1e-7, 1))))
            f1 = macro_f1(yv, (p_val >= 0.5).astype(int))
            curve.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                          "val_macro_f1": f1, "seconds": t_ep["seconds"], "lr_end": sched.get_last_lr()[0]})
            log(f"  {short} seed {seed} epoch {ep}: train loss {np.mean(losses):.4f} | val loss {val_loss:.4f} | "
                f"val macro-F1 {f1:.4f} | {t_ep['seconds']:.0f}s")
            if f1 > best_f1:
                best_f1, best_epoch, bad = f1, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad > int(tcfg["patience"]) - 1:
                    log(f"  early stopping: no validation improvement for {tcfg['patience']} epoch(s)")
                    break
    model.load_state_dict(best_state)
    with timer() as t_inf:
        p_test = predict_proba(model, tok, te["text_bert"], device, ml, bs * 2, trunc)
    p_val = predict_proba(model, tok, va["text_bert"], device, ml, bs * 2, trunc)
    info = {"curve": curve, "best_epoch": best_epoch, "learning_rate": lr, "class_weights": weights.tolist() if weights is not None else None,
            "train_seconds": t_train["seconds"], "infer_ms_per_item": 1000 * t_inf["seconds"] / max(1, len(te)),
            "params": _count_params(model), "steps": total, "warmup_steps": warm}
    if model_dir is not None:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        info["disk_mb"] = _dir_mb(model_dir)
    return p_val, p_test, info


# ------------------------------------------------------------------------------------------ embeddings
def embed_texts(model, tok, texts, device, max_length=128, batch_size=32, truncation="head"):
    import torch

    model.eval()
    enc = encode(tok, texts, max_length, truncation)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = None
    with torch.no_grad():
        for idx, ids, mask, _ in _batches(enc, None, batch_size, False, None, pad):
            h = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
            m = mask.to(device).unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
            if out is None:
                out = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
            out[idx] = v
    return out


def _embed_one(cfg, short, sp, seed, device, tcfg, model_dir: Path | None):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    set_seed(seed)
    tok = load_tokenizer(short)
    enc_model = _load_model(short, base=True).to(device)
    ml, bs, trunc = int(tcfg["max_length"]), int(tcfg["embed_batch_size"]), tcfg["truncation"]
    with timer() as t_emb:
        X = {s: embed_texts(enc_model, tok, sp[s]["text_bert"], device, ml, bs, trunc) for s in ("train", "val")}
    tuning = []
    for C in cfg["classical"]["C_grid"]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=float(C) / 10, max_iter=3000, class_weight="balanced"))
        clf.fit(X["train"], sp["train"]["label_id"])
        f1 = macro_f1(sp["val"]["label_id"], (clf.predict_proba(X["val"])[:, 1] >= 0.5).astype(int))
        tuning.append({"C": float(C) / 10, "val_macro_f1": f1})
    best = max(tuning, key=lambda r: r["val_macro_f1"])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best["C"], max_iter=3000, class_weight="balanced"))
    with timer() as t_fit:
        clf.fit(X["train"], sp["train"]["label_id"])
    with timer() as t_inf:
        Xte = embed_texts(enc_model, tok, sp["test"]["text_bert"], device, ml, bs, trunc)
        p_test = clf.predict_proba(Xte)[:, 1]
    p_val = clf.predict_proba(X["val"])[:, 1]
    info = {"tuning": tuning, "best_C": best["C"], "embedding_dim": int(X["train"].shape[1]),
            "train_seconds": t_emb["seconds"] + t_fit["seconds"], "embedding_seconds": t_emb["seconds"],
            "infer_ms_per_item": 1000 * t_inf["seconds"] / max(1, len(sp["test"])),
            "params": _count_params(enc_model) + X["train"].shape[1] + 1}
    if model_dir is not None:
        ensure_dir(model_dir)
        joblib.dump(clf, model_dir / "head.joblib")
        (model_dir / "backbone.txt").write_text(REGISTRY[short]["hf"])
        info["disk_mb"] = _dir_mb(model_dir)
    return p_val, p_test, info


# ------------------------------------------------------------------------------------------ driver
def train_transformers(cfg: dict, name: str, models: list[str] | None = None, mode: str | None = None,
                       seeds: int | None = None) -> list[dict]:
    tcfg = cfg["transformers"]
    models = models or tcfg["default_models"]
    check_names(models)
    mode = mode or tcfg["mode"]
    if mode not in ("finetune", "embed"):
        raise FakeNewsError("--mode must be finetune or embed.")
    n_seeds = int(seeds or tcfg["seeds"])
    device = get_device(tcfg["device"])
    sp = load_splits(cfg, name)
    sub = tcfg.get("train_subset")
    if sub and int(sub) < len(sp["train"]):
        sp["train"] = sp["train"].sample(n=int(sub), random_state=int(cfg["seed"]))
        log(f"Using a training subset of {sub} rows")
    out = output_dir(cfg, name)
    rdir, pdir = ensure_dir(out / "results"), ensure_dir(out / "results" / "predictions")
    ev = cfg["evaluation"]
    log(f"Device: {device} | mode: {mode} | models: {models} | seeds: {n_seeds} | train rows: {len(sp['train'])}")
    results = []
    for short in models:
        key = f"{short}_{'ft' if mode == 'finetune' else 'emb'}"
        per_seed = []
        p_test_first, p_val_first, info_first = None, None, None
        for k in range(n_seeds):
            seed = int(cfg["seed"]) + k
            mdir = (out / "models" / key) if k == 0 else None
            log(f"{key}: seed {seed} ...")
            fn = _finetune_one if mode == "finetune" else _embed_one
            if device == "mps":
                try:
                    p_val, p_test, info = fn(cfg, short, sp, seed, device, tcfg, mdir)
                except RuntimeError as e:
                    log(f"MPS error ({e}); retrying {short} on CPU")
                    p_val, p_test, info = fn(cfg, short, sp, seed, "cpu", tcfg, mdir)
            else:
                p_val, p_test, info = fn(cfg, short, sp, seed, device, tcfg, mdir)
            m = compute_metrics(sp["test"]["label_id"], p_test, ev["threshold"], ev["bootstrap"], ev["ci"], ev["ece_bins"], seed)
            per_seed.append({"seed": seed, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"]})
            if k == 0:
                p_test_first, p_val_first, info_first, test_m = p_test, p_val, info, m
        val_m = compute_metrics(sp["val"]["label_id"], p_val_first, ev["threshold"], 0, ev["ci"], ev["ece_bins"])
        f1s = [s["macro_f1"] for s in per_seed]
        reg = REGISTRY[short]
        res = {
            "model": key, "label": f"{short} ({'fine-tuned' if mode == 'finetune' else 'frozen embeddings + LR'})",
            "family": "Transformer", "mode": "Fine-tuned" if mode == "finetune" else "Embeddings",
            "backbone": short, "hf_id": reg["hf"], "languages": reg["languages"], "size": reg["size"],
            "about": reg["why"],
            "settings": {k: tcfg[k] for k in ("max_length", "truncation", "batch_size", "epochs", "warmup_ratio",
                                              "weight_decay", "grad_clip", "patience", "train_subset")}
                        | {"input_text": "bert view (column 'text_bert')", "cased": reg["cased"]},
            **{k: v for k, v in info_first.items()},
            "device": device, "model_path": str(out / "models" / key),
            "seeds": per_seed, "macro_f1_seed_mean": float(np.mean(f1s)), "macro_f1_seed_std": float(np.std(f1s)),
            "val_metrics": val_m, "test_metrics": test_m,
            "dataset": name, "data_card_fingerprint": load_card(cfg, name).get("fingerprint_sha256"),
            "seed": int(cfg["seed"]), "created": now_iso(), "versions": library_versions(), "config": cfg,
            "train_rows": len(sp["train"]),
        }
        write_json(rdir / f"{key}.json", res)
        pd.DataFrame({"id": sp["test"]["id"], "p_fake": p_test_first}).to_csv(pdir / f"{key}.csv", index=False)
        pd.DataFrame({"id": sp["val"]["id"], "p_fake": p_val_first}).to_csv(pdir / f"{key}.val.csv", index=False)
        append_run_log(cfg["paths"]["outputs_dir"], name, "train-bert",
                       {"model": key, "device": device, "test_macro_f1": test_m["macro_f1"],
                        "seed_mean": res["macro_f1_seed_mean"], "seconds": info_first["train_seconds"]})
        log(f"{key:<16} test macro-F1={test_m['macro_f1']:.3f} [{test_m['macro_f1_ci_low']:.3f}, "
            f"{test_m['macro_f1_ci_high']:.3f}] | {info_first['train_seconds']:.0f}s | {info_first['params']/1e6:.1f}M params")
        results.append(res)
    return results


# ------------------------------------------------------------------------------------------ loading for predict / UI
def load_trained(cfg, name, key, device="cpu"):
    """Returns a callable texts -> P(fake) for a saved transformer model."""
    import joblib

    mdir = Path(output_dir(cfg, name)) / "models" / key
    if not mdir.exists():
        raise FakeNewsError(f"Transformer '{key}' is not trained for dataset '{name}'. Run train-bert first.")
    tcfg = cfg["transformers"]
    if key.endswith("_emb"):
        short = key[:-4]
        tok = load_tokenizer(short)
        enc_model = _load_model(short, base=True).to(device)
        head = joblib.load(mdir / "head.joblib")
        return lambda texts: head.predict_proba(
            embed_texts(enc_model, tok, list(texts), device, tcfg["max_length"], 32, tcfg["truncation"]))[:, 1]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    short = key[:-3]
    kw = {"keep_accents": True} if short == "indicbert" else {}
    tok = AutoTokenizer.from_pretrained(mdir, **kw)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to(device)
    return lambda texts: predict_proba(model, tok, list(texts), device, tcfg["max_length"], 32, tcfg["truncation"])
