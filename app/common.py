"""Shared UI helpers: config, dataset selector, cached loaders, download buttons. Every number shown in the UI is
read from files produced by the terminal commands (or computed live from saved predictions)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fakenews.config import load_config, output_dir, processed_dir  # noqa: E402
from fakenews.utils import read_json  # noqa: E402

FAKE, REAL = "#d62728", "#2ca02c"
DISCLAIMER = "Model predictions are not verified facts. Always check a fact-checker."


@st.cache_resource
def cfg():
    return load_config()


def datasets_with_outputs() -> list[str]:
    out = Path(cfg()["paths"]["outputs_dir"])
    if not out.exists():
        return []
    return sorted(p.name for p in out.iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "results").exists())


def _stamp(name) -> float:
    """Changes whenever any result file changes, so caches refresh after a new run."""
    d = output_dir(cfg(), name) / "results"
    return max((f.stat().st_mtime for f in d.glob("*")), default=0.0) if d.exists() else 0.0


def sidebar() -> str | None:
    names = datasets_with_outputs()
    with st.sidebar:
        if names:
            default = st.session_state.get("dataset", "synthetic" if "synthetic" in names else names[0])
            st.session_state["dataset"] = st.selectbox("Dataset", names, index=names.index(default) if default in names else 0,
                                                       help="Every dataset that has outputs/<name>/results")
        else:
            st.session_state["dataset"] = None
            st.warning("No results yet. In a terminal run:\n\n`python -m fakenews run-all`")
        st.caption(f"⚠️ {DISCLAIMER}")
    return st.session_state.get("dataset")


def require_dataset() -> str:
    name = st.session_state.get("dataset")
    if not name:
        st.info("No dataset with results yet. Run `python -m fakenews run-all` in a terminal, then refresh.")
        st.stop()
    return name


@st.cache_data
def _results(name, stamp):
    from fakenews.predict import list_results
    return list_results(cfg(), name)


def results(name) -> dict:
    return _results(name, _stamp(name))


@st.cache_data
def _json(path, stamp):
    p = Path(path)
    return read_json(p) if p.exists() else None


def result_file(name, fname):
    return _json(str(output_dir(cfg(), name) / "results" / fname), _stamp(name))


@st.cache_data
def _csv(path, stamp):
    p = Path(path)
    return pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False, na_values=[""]) if p.exists() else None


def result_csv(name, fname):
    return _csv(str(output_dir(cfg(), name) / "results" / fname), _stamp(name))


def card(name) -> dict:
    p = processed_dir(cfg(), name) / "data_card.json"
    return _json(str(p), p.stat().st_mtime if p.exists() else 0) or {}


@st.cache_data
def splits(name, stamp):
    from fakenews.data import load_splits
    return load_splits(cfg(), name)


def get_splits(name):
    p = processed_dir(cfg(), name) / "train.csv"
    return splits(name, p.stat().st_mtime if p.exists() else 0)


def fig_path(name, rel) -> Path:
    return output_dir(cfg(), name) / "figures" / rel


def show_fig(name, rel, caption=None, key=None):
    p = fig_path(name, rel)
    if not p.exists():
        st.info(f"Figure `{rel}` not generated yet - run `python -m fakenews compare --name {name}`.")
        return
    st.image(str(p), width="stretch")
    if caption:
        st.caption(caption)
    st.download_button("⬇ PNG", p.read_bytes(), file_name=f"{name}_{Path(rel).name}", mime="image/png",
                       key=key or f"dl_{rel}")


def csv_button(df: pd.DataFrame, fname: str, label="⬇ CSV", key=None):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=fname, mime="text/csv",
                       key=key or f"csv_{fname}")


def synthetic_banner(name):
    if "WARNING" in card(name):
        st.warning("**Synthetic dataset selected.** These numbers show that the pipeline works; they are NOT model "
                   "performance. Real claims come from your own dataset and LIAR.", icon="🧪")


@st.cache_resource
def predictor(name, model, stamp):
    from fakenews.predict import get_predictor
    return get_predictor(cfg(), name, model, "cpu")


def get_predictor(name, model):
    return predictor(name, model, _stamp(name))


VERDICT_STYLE = {"SUPPORTED": ("✅", "#e6f4ea", "#1e7e34"), "REJECTED": ("❌", "#fdecea", "#b02a37"),
                 "INCONCLUSIVE": ("➖", "#fff8e1", "#8a6d00"), "NOT TESTABLE": ("⏸", "#eeeeee", "#555555")}


def verdict_badge(v: str) -> str:
    icon, bg, fg = VERDICT_STYLE.get(v, ("", "#eee", "#333"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:10px;font-weight:600;'
            f'font-size:0.85em">{icon} {v}</span>')


def fmt(v, nd=3):
    try:
        return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"
    except TypeError:
        return str(v)
