"""Bring your own dataset: upload -> choose columns -> prepare + train the statistical models + compare + audit +
hypotheses, all from the browser. Transformers are trained in the terminal (commands shown at the end)."""
from pathlib import Path

import streamlit as st

import common as C
from fakenews.data import bytes_to_frame
from fakenews.utils import FakeNewsError, safe_name

st.title("Use your own dataset")
st.markdown("""
**Format:** CSV (UTF-8), Excel or Parquet with at least a **text** column and a **label** column.
Strongly recommended: a **date** column (time split), a **source/channel** column (group split + source audit),
a **language** column, and a second **lightly cleaned text column for BERT** (keep stopwords, punctuation, emojis
and case for BERT - see the two-text-columns rule in BUILD_INSTRUCTIONS 13.2).
Your file is saved to `data/raw/` and **never committed to GitHub**.
""")
up = st.file_uploader("Dataset file", type=["csv", "xlsx", "xls", "parquet", "tsv"])
if not up:
    st.stop()
try:
    df = bytes_to_frame(up.getvalue(), up.name)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the file: {e}")
    st.stop()
st.success(f"{len(df):,} rows · columns: {list(df.columns)}")
st.dataframe(df.head(8), hide_index=True)

cols = list(df.columns)
none = "(none)"
guess = lambda opts, c: cols.index(next((x for x in cols if x.lower() in opts), cols[0]))  # noqa: E731
a, b = st.columns(2)
text_col = a.selectbox("Text column (for BoW / TF-IDF)", cols, index=guess(("text", "clean_text", "content", "statement", "title"), cols))
bert_col = a.selectbox("Lightly cleaned text column for BERT (optional)", [none] + cols)
label_col = a.selectbox("Label column", cols, index=guess(("label", "labels", "class", "target", "is_fake"), cols))
vals = sorted(df[label_col].astype(str).unique())[:50]
st.caption(f"Label values found: {vals}")
fake_value = a.selectbox("Which label value means FAKE? (required for numeric labels)", ["(auto: text labels)"] + vals)
date_col = b.selectbox("Date column", [none] + cols)
source_col = b.selectbox("Source / channel column", [none] + cols)
lang_col = b.selectbox("Language column (else detected: en / hinglish / hi)", [none] + cols)
split = b.selectbox("Split strategy", ["random", "time", "group"],
                    help="time needs a date column; group (no source in two splits) needs a source column")
ds_name = b.text_input("Dataset name", safe_name(Path(up.name).stem.lower()))
already = b.checkbox("Text column is already fully cleaned (use as-is for BoW/TF-IDF)")
max_rows = b.number_input("Use at most N rows (0 = all)", 0, 10_000_000, 0)

if st.button("Prepare + train statistical models + compare + audit + hypotheses", type="primary"):
    cfg = C.cfg()
    raw = Path(cfg["paths"]["data_dir"]) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / up.name
    path.write_bytes(up.getvalue())
    nm = safe_name(ds_name)
    try:
        with st.status("Running pipeline...", expanded=True) as s:
            from fakenews.audit import run_audit
            from fakenews.classical import train_classical
            from fakenews.compare import run_compare
            from fakenews.data import prepare
            from fakenews.hypotheses import run_hypotheses
            st.write("1/5 prepare: clean, deduplicate, split once, data card")
            prepare(cfg, "csv", str(path), nm, text_col, None if bert_col == none else bert_col, label_col,
                    None if fake_value.startswith("(auto") else fake_value, split,
                    None if date_col == none else date_col, None if source_col == none else source_col,
                    None if lang_col == none else lang_col, already, max_rows or None)
            st.write("2/5 train the 5 BoW / TF-IDF models")
            train_classical(cfg, nm)
            st.write("3/5 compare")
            run_compare(cfg, nm, quiet=True)
            st.write("4/5 shortcut audit")
            run_audit(cfg, nm, quiet=True)
            st.write("5/5 hypotheses + report")
            run_hypotheses(cfg, nm, quiet=True)
            run_compare(cfg, nm, quiet=True)
            s.update(label="Done", state="complete")
        st.cache_data.clear()
        st.session_state["dataset"] = nm
        st.success(f"Dataset `{nm}` is ready - it is now selected in the sidebar. Open Model comparison.")
    except FakeNewsError as e:
        st.error(str(e))
        st.stop()
    st.markdown("**Next: add transformers from a terminal** (MuRIL is the key model for Indian content):")
    st.code(f"python -m fakenews train-bert --name {nm} --models distilbert,muril --mode embed --train-subset 5000\n"
            f"python -m fakenews train-bert --name {nm} --models distilbert,mbert,muril --epochs 3\n"
            f"python -m fakenews compare --name {nm}\npython -m fakenews audit --name {nm}\n"
            f"python -m fakenews hypotheses --name {nm}", language="bash")
    st.caption("Before trusting results: read the top TF-IDF words on the Data explorer page and remove leaky words "
               "(publisher names, URLs, 'fact check' stamps), then re-run.")
