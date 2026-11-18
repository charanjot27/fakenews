"""Phase 2 + 8: load any dataset, map labels, deduplicate, split once, save splits + a data card.

Every processed dataset has the same standard columns:
    id, text (classical view), text_bert (bert view), label_id (1 = fake, 0 = real), language, length_words
plus any metadata found (source, date, topic, style, case_type, template_id, ...).
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import processed_dir
from .preprocess import bert_view, classical_view, detect_language, near_dup_key
from .utils import FakeNewsError, append_run_log, ensure_dir, file_sha256, log, now_iso, read_json, write_json

FAKE_WORDS = {"fake", "false", "hoax", "misleading", "pants-fire", "pants fire", "pants_fire", "fabricated", "1_fake"}
REAL_WORDS = {"real", "true", "genuine", "mostly-true", "mostly true", "mostly_true", "legit", "0_real"}
LIAR_DROP = {"half-true", "barely-true"}
LIAR_COLUMNS = ["liar_id", "label", "statement", "subject", "speaker", "job", "state", "party", "barely_true_c",
                "false_c", "half_true_c", "mostly_true_c", "pants_fire_c", "context"]
LIAR_URL = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/{split}.tsv"

META_CANDIDATES = ["topic", "style", "case_type", "template_id", "noisy_label", "true_label", "speaker", "party",
                   "state", "context", "subject", "job"]


# ------------------------------------------------------------------------------------------ labels
def map_labels(series: pd.Series, fake_value=None) -> pd.Series:
    """Text labels are mapped automatically. Numeric labels are never guessed: fake_value is required."""
    s = series.copy()
    if s.isna().any():
        raise FakeNewsError(f"The label column has {int(s.isna().sum())} empty values. Remove or fill them first.")
    if fake_value is not None:
        fv = str(fake_value).strip().lower()
        vals = s.astype(str).str.strip().str.lower()
        uniq = sorted(vals.unique())
        if fv not in uniq:
            raise FakeNewsError(f"--fake-value '{fake_value}' is not in the label column. Values found: {uniq}")
        if len(uniq) != 2:
            raise FakeNewsError(f"--fake-value works with exactly two label values; found {len(uniq)}: {uniq[:20]}. "
                                "Use text labels (fake/real) or clean the column first.")
        return (vals == fv).astype(int)
    if pd.api.types.is_numeric_dtype(s) or s.astype(str).str.strip().str.fullmatch(r"-?\d+(\.0)?").all():
        vals = sorted(pd.Series(s).astype(str).unique())
        raise FakeNewsError(
            f"The label column is numeric ({vals}). Numeric labels are never guessed, because datasets disagree "
            "(in WELFake 0 means fake). Tell the program which value means fake, e.g. --fake-value 1")
    vals = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index)
    out[vals.isin(FAKE_WORDS)] = 1
    out[vals.isin(REAL_WORDS)] = 0
    unknown = sorted(vals[out.isna()].unique())
    if unknown:
        raise FakeNewsError(f"Unknown label values: {unknown[:20]}. Known fake words: {sorted(FAKE_WORDS)}; "
                            f"known real words: {sorted(REAL_WORDS)}. Rename them or use --fake-value.")
    return out.astype(int)


# ------------------------------------------------------------------------------------------ loaders
def read_table(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FakeNewsError(f"File not found: {p}. Put your dataset in data/raw/ and pass its path with --path.")
    suf = p.suffix.lower()
    if suf in (".csv", ".txt"):
        try:
            return pd.read_csv(p, encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise FakeNewsError(f"{p} is not UTF-8. Save it as UTF-8 (needed for Hindi) and try again.")
    if suf == ".tsv":
        return pd.read_csv(p, sep="\t", encoding="utf-8-sig")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suf in (".jsonl", ".json"):
        return pd.read_json(p, lines=suf == ".jsonl")
    raise FakeNewsError(f"Unsupported file type '{suf}'. Use CSV, TSV, Excel, Parquet or JSON lines.")


def _download_liar(cache_dir: Path) -> dict[str, pd.DataFrame]:
    ensure_dir(cache_dir)
    frames = {}
    for split in ("train", "valid", "test"):
        f = cache_dir / f"{split}.tsv"
        if not f.exists():
            url = LIAR_URL.format(split=split)
            log(f"Downloading LIAR {split} from {url}")
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    f.write_bytes(r.read())
            except Exception as e:  # noqa: BLE001
                raise FakeNewsError(f"Could not download LIAR ({e}). Check your internet connection, or download "
                                    f"train.tsv/valid.tsv/test.tsv manually into {cache_dir}.")
        df = pd.read_csv(f, sep="\t", header=None, names=LIAR_COLUMNS, quoting=3, dtype=str)
        frames[split] = df
    return frames


def load_liar(cfg) -> tuple[pd.DataFrame, dict]:
    cache = Path(cfg["paths"]["data_dir"]) / "external" / "liar"
    frames = _download_liar(cache)
    parts = []
    for split, df in frames.items():
        df = df[~df["label"].isin(LIAR_DROP)].copy()
        df["label_id"] = df["label"].map({"true": 0, "mostly-true": 0, "false": 1, "pants-fire": 1})
        df = df.dropna(subset=["label_id"])
        df["official_split"] = {"train": "train", "valid": "val", "test": "test"}[split]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True).rename(columns={"statement": "text_raw"})
    info = {"note": "LIAR is US political English (PolitiFact). It tests general ability, not Indian content. "
                    "Binary mapping: true/mostly-true -> real, false/pants-fire -> fake; half-true and barely-true "
                    "dropped. Official train/valid/test splits kept. speaker/party/context kept as metadata only.",
            "licence": "LIAR (Wang, 2017) - research use", "source_files": str(cache),
            "sha256": {s: file_sha256(cache / f"{s}.tsv") for s in ("train", "valid", "test")}}
    df["label_id"] = df["label_id"].astype(int)
    df["language"] = "en"
    return df, info


# ------------------------------------------------------------------------------------------ splitting
def _split_random(df, cfg, seed):
    from sklearn.model_selection import train_test_split

    fr = cfg["split"]
    if df["label_id"].value_counts().min() < 3:
        raise FakeNewsError("Each class needs at least 3 items to make a stratified split.")
    tr, rest = train_test_split(df, train_size=fr["train"], stratify=df["label_id"], random_state=seed)
    va, te = train_test_split(rest, test_size=fr["test"] / (fr["val"] + fr["test"]), stratify=rest["label_id"],
                              random_state=seed)
    return tr, va, te


def _split_time(df, cfg):
    if "date" not in df or df["date"].isna().all():
        raise FakeNewsError("Time split needs a date column. Pass --date-col, or use --split random.")
    d = df.copy()
    d["_d"] = pd.to_datetime(d["date"], errors="coerce")
    bad = int(d["_d"].isna().sum())
    if bad:
        log(f"Time split: {bad} rows have no readable date and are dropped.")
        d = d.dropna(subset=["_d"])
    d = d.sort_values(["_d", "id"], kind="stable")
    n = len(d)
    a, b = int(n * cfg["split"]["train"]), int(n * (cfg["split"]["train"] + cfg["split"]["val"]))
    return d.iloc[:a].drop(columns="_d"), d.iloc[a:b].drop(columns="_d"), d.iloc[b:].drop(columns="_d")


def group_split(df, group_col, fractions, seed):
    """No group appears in more than one split. Groups are assigned greedily to the split furthest below target."""
    if group_col not in df or df[group_col].isna().all():
        raise FakeNewsError(f"Group split needs a '{group_col}' column. Pass --source-col, or use --split random.")
    rng = np.random.default_rng(seed)
    sizes = df.groupby(group_col).size()
    groups = list(sizes.index)
    if len(groups) < 3:
        raise FakeNewsError(f"Group split needs at least 3 different values in '{group_col}'; found {len(groups)}.")
    rng.shuffle(groups)
    groups.sort(key=lambda g: -sizes[g])  # big groups first, ties in random order
    target = np.array(fractions) * len(df)
    filled = np.zeros(3)
    assign = {}
    for g in groups:
        k = int(np.argmax(target - filled))
        assign[g] = k
        filled[k] += sizes[g]
    for k in range(3):  # make sure no split is empty
        if not any(v == k for v in assign.values()):
            donor = max(range(3), key=lambda j: sum(v == j for v in assign.values()))
            g = next(g for g in reversed(groups) if assign[g] == donor)
            assign[g] = k
    lab = df[group_col].map(assign)
    return df[lab == 0], df[lab == 1], df[lab == 2]


# ------------------------------------------------------------------------------------------ prepare
def prepare(cfg, source="synthetic", path=None, name=None, text_col="text", bert_text_col=None, label_col="label",
            fake_value=None, split=None, date_col=None, source_col=None, language_col=None, already_clean=False,
            max_rows=None) -> Path:
    seed = int(cfg["seed"])
    split = split or cfg["split"]["strategy"]
    if split not in ("random", "time", "group"):
        raise FakeNewsError(f"Unknown split strategy '{split}'. Use random, time or group.")
    extra_info: dict = {}

    if source == "synthetic":
        path = Path(path) if path else Path(cfg["paths"]["data_dir"]) / "synthetic" / "synthetic_news.csv"
        if not path.exists():
            raise FakeNewsError(f"{path} does not exist. Run: python -m fakenews make-data")
        raw = read_table(path)
        name = name or "synthetic"
        text_col, label_col = "text", "label"
        date_col, source_col, language_col = "date", "source", "language"
        extra_info["WARNING"] = ("SYNTHETIC DATA: proves the pipeline works; scores here are NOT model performance.")
        fingerprint = file_sha256(path)
        card_path = Path(path).with_suffix(".card.json")
        if card_path.exists():
            extra_info["generator"] = read_json(card_path).get("params")
    elif source == "liar":
        raw, extra_info = load_liar(cfg)
        name = name or "liar"
        text_col, label_col = "text_raw", None
        fingerprint = extra_info["sha256"]
        language_col = "language"
    elif source == "csv":
        if not path:
            raise FakeNewsError("--path is required with --source csv (put the file in data/raw/).")
        raw = read_table(path)
        name = name or Path(path).stem
        fingerprint = file_sha256(path)
    else:
        raise FakeNewsError(f"Unknown source '{source}'. Use synthetic, liar or csv.")

    cols = list(raw.columns)
    for c, what in ((text_col, "--text-col"), (bert_text_col, "--bert-text-col"), (label_col, "--label-col"),
                    (date_col, "--date-col"), (source_col, "--source-col"), (language_col, "--language-col")):
        if c and c not in cols:
            raise FakeNewsError(f"Column '{c}' ({what}) not found. Columns in the file: {cols}")

    n_raw = len(raw)
    if max_rows:
        raw = raw.sample(n=min(int(max_rows), len(raw)), random_state=seed)
    df = pd.DataFrame(index=raw.index)
    df["id"] = raw["id"].values if "id" in raw.columns and raw["id"].is_unique else np.arange(1, len(raw) + 1)
    df["text_original"] = raw[text_col].astype(str).where(raw[text_col].notna(), "")
    df["text"] = df["text_original"].map(lambda t: t.strip() if already_clean else classical_view(t))
    bert_src = raw[bert_text_col] if bert_text_col else raw[text_col]
    df["text_bert"] = bert_src.astype(str).where(bert_src.notna(), "").map(bert_view)
    df["label_id"] = raw["label_id"].values if source == "liar" else map_labels(raw[label_col], fake_value).values
    if date_col:
        df["date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.date.astype(str).replace("NaT", None)
    if source_col:
        df["source"] = raw[source_col].astype(str)
    if language_col:
        df["language"] = raw[language_col].astype(str).str.lower()
        lang_inferred = False
    else:
        df["language"] = df["text_bert"].map(detect_language)
        lang_inferred = True
    for c in META_CANDIDATES:
        if c in raw.columns and c not in df.columns:
            df[c] = raw[c].values
    if source == "liar":
        df["official_split"] = raw["official_split"].values

    # cleaning at load time
    before = len(df)
    df = df[(df["text"].str.len() > 0) & (df["text_bert"].str.len() > 0)]
    n_empty = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset="text_bert", keep="first")
    n_exact_dup = before - len(df)
    nd = df["text_bert"].map(near_dup_key)
    n_near_dup_rows = int(nd.duplicated(keep=False).sum())
    n_near_dup_groups = int(nd[nd.duplicated(keep=False)].nunique())
    conflicting = df.assign(_k=nd).groupby("_k")["label_id"].nunique()
    n_conflict = int((conflicting > 1).sum())
    df["length_words"] = df["text_bert"].str.split().str.len()
    if df["label_id"].nunique() < 2:
        raise FakeNewsError("After cleaning only one label is left; a classifier needs both fake and real items.")

    # split
    if source == "liar":
        tr, va, te = (df[df["official_split"] == s].drop(columns="official_split") for s in ("train", "val", "test"))
        split_used = "official (LIAR)"
    elif split == "random":
        tr, va, te = _split_random(df, cfg, seed)
        split_used = "random, stratified"
    elif split == "time":
        tr, va, te = _split_time(df, cfg)
        split_used = "time-based (oldest -> train, newest -> test)"
    else:
        f = cfg["split"]
        tr, va, te = group_split(df, "source", [f["train"], f["val"], f["test"]], seed)
        split_used = "group-based by source (no source in two splits)"
    for nm, part in (("train", tr), ("val", va), ("test", te)):
        if len(part) == 0:
            raise FakeNewsError(f"The {nm} split is empty. Use more data or a different --split.")
        if part["label_id"].nunique() < 2 and nm != "val":
            log(f"WARNING: the {nm} split has only one class.")

    out = ensure_dir(processed_dir(cfg, name))
    for nm, part in (("train", tr), ("val", va), ("test", te)):
        part.to_csv(out / f"{nm}.csv", index=False, encoding="utf-8")
    overlap = len(set(tr["text_bert"]) & set(te["text_bert"])) + len(set(tr["text_bert"]) & set(va["text_bert"])) \
        + len(set(va["text_bert"]) & set(te["text_bert"]))

    def per_split(f):
        return {nm: f(p) for nm, p in (("train", tr), ("val", va), ("test", te))}

    card = {
        "dataset": name, "created": now_iso(), "source": source, "input_file": str(path) if path else None,
        "fingerprint_sha256": fingerprint, **extra_info,
        "columns_used": {"text (classical)": text_col, "text (bert)": bert_text_col or text_col,
                         "label": label_col, "fake_value": fake_value, "date": date_col, "source": source_col,
                         "language": language_col},
        "text_views": {
            "classical": "used as-is (already clean)" if already_clean else "lowercased, links -> urltoken",
            "bert": "separate lightly cleaned column" if bert_text_col else
                    ("same column as classical, lightly cleaned (whitespace + links only)"),
        },
        "bert_text_warning": None if (bert_text_col or source != "csv") else
            "Only one text column was given. If it was heavily cleaned (stopwords/punctuation removed, stemmed), "
            "BERT scores are likely lower than they could be. See BUILD_INSTRUCTIONS 13.2.",
        "language_inferred": lang_inferred,
        "rows_in_file": n_raw, "rows_after_cleaning": len(df), "empty_texts_removed": n_empty,
        "exact_duplicates_removed": n_exact_dup, "near_duplicate_rows": n_near_dup_rows,
        "near_duplicate_groups": n_near_dup_groups, "near_duplicate_groups_with_conflicting_labels": n_conflict,
        "split_strategy": split_used, "seed": seed, "split_overlap_texts": overlap,
        "sizes": per_split(len),
        "labels": per_split(lambda p: {"fake": int(p["label_id"].sum()), "real": int((1 - p["label_id"]).sum())}),
        "share_fake": per_split(lambda p: round(float(p["label_id"].mean()), 4)),
        "language_mix": per_split(lambda p: p["language"].value_counts().to_dict()),
        "avg_words_by_label": {"fake": round(float(df.loc[df.label_id == 1, "length_words"].mean()), 2),
                               "real": round(float(df.loc[df.label_id == 0, "length_words"].mean()), 2)},
        "date_range": per_split(lambda p: [str(p["date"].min()), str(p["date"].max())]) if "date" in df else None,
        "n_sources": int(df["source"].nunique()) if "source" in df else None,
        "metadata_columns": [c for c in df.columns if c in META_CANDIDATES or c in ("source", "date")],
    }
    write_json(out / "data_card.json", card)
    append_run_log(cfg["paths"]["outputs_dir"], name, "prepare",
                   {"source": source, "split": split_used, "sizes": card["sizes"], "fingerprint": fingerprint})
    log(f"Prepared '{name}': train={len(tr)} val={len(va)} test={len(te)} | split={split_used} | "
        f"exact duplicates removed={n_exact_dup} | near-duplicate rows={n_near_dup_rows}")
    log(f"Saved splits + data card to {out}")
    return out


def load_splits(cfg, name) -> dict[str, pd.DataFrame]:
    d = processed_dir(cfg, name)
    if not (d / "train.csv").exists():
        avail = list_prepared(cfg)
        raise FakeNewsError(f"Dataset '{name}' is not prepared. Run: python -m fakenews prepare ... --name {name}. "
                            f"Prepared datasets: {avail or 'none'}")
    out = {}
    for s in ("train", "val", "test"):
        df = pd.read_csv(d / f"{s}.csv", encoding="utf-8", keep_default_na=False, na_values=[""])
        df["text"] = df["text"].fillna("").astype(str)
        df["text_bert"] = df["text_bert"].fillna("").astype(str)
        out[s] = df
    return out


def load_card(cfg, name) -> dict:
    p = processed_dir(cfg, name) / "data_card.json"
    return read_json(p) if p.exists() else {}


def list_prepared(cfg) -> list[str]:
    d = Path(cfg["paths"]["data_dir"]) / "processed"
    return sorted(p.name for p in d.iterdir() if (p / "train.csv").exists()) if d.exists() else []


def bytes_to_frame(data: bytes, filename: str) -> pd.DataFrame:
    """Used by the UI upload widgets."""
    suf = Path(filename).suffix.lower()
    bio = io.BytesIO(data)
    if suf in (".csv", ".txt"):
        return pd.read_csv(bio, encoding="utf-8-sig")
    if suf == ".tsv":
        return pd.read_csv(bio, sep="\t", encoding="utf-8-sig")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(bio)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(bio)
    raise FakeNewsError(f"Unsupported file type '{suf}'.")
