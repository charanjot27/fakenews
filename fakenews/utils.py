"""Small helpers shared by every module: errors, seeding, device choice, timing, file I/O, run logs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_FAKE = 1
LABEL_REAL = 0
LABEL_NAMES = {0: "real", 1: "fake"}


class FakeNewsError(Exception):
    """An error with a message meant for the user (printed without a traceback)."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preference: str = "auto") -> str:
    """CUDA, then Apple MPS, then CPU. A forced device that is missing is an error."""
    import torch

    pref = (preference or "auto").lower()
    has = {
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu": True,
    }
    if pref == "auto":
        for dev in ("cuda", "mps", "cpu"):
            if has[dev]:
                return dev
    if pref not in has:
        raise FakeNewsError(f"Unknown device '{preference}'. Use auto, cuda, mps or cpu.")
    if not has[pref]:
        raise FakeNewsError(f"Device '{pref}' is not available on this machine. Use --device auto.")
    return pref


@contextmanager
def timer():
    """Usage: with timer() as t: ...; t['seconds']"""
    out = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _clean_nan(obj):
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def write_json(path, data) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(_clean_nan(data), cls=_Encoder, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def library_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ["numpy", "pandas", "sklearn", "scipy", "torch", "transformers", "matplotlib", "streamlit"]:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = None
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[satyacheck] {msg}", flush=True)


def append_run_log(outputs_dir, dataset: str, step: str, details: dict) -> None:
    """Every command appends one line here, so evaluators can replay what happened and when."""
    p = ensure_dir(Path(outputs_dir) / dataset / "logs") / "run_log.jsonl"
    rec = {"time": now_iso(), "step": step, **details}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(_clean_nan(rec), cls=_Encoder, ensure_ascii=False) + "\n")


def safe_name(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name.strip())
    if not keep:
        raise FakeNewsError("Dataset name must contain letters or digits.")
    return keep
