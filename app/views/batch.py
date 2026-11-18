import pandas as pd
import streamlit as st

import common as C
from fakenews.data import bytes_to_frame

name = C.require_dataset()
st.title("Batch checker")
st.caption("Upload a CSV/Excel of texts, pick models and the text column, and download P(fake) for every row.")
res = C.results(name)
up = st.file_uploader("File", type=["csv", "xlsx", "tsv", "parquet"])
if not up:
    st.stop()
df = bytes_to_frame(up.getvalue(), up.name)
st.dataframe(df.head(), hide_index=True)
col = st.selectbox("Text column", list(df.columns))
models = st.multiselect("Models", list(res), default=[m for m in res if res[m]["family"] == "Statistical"][:2])
if st.button("Run", type="primary"):
    thr = C.cfg()["evaluation"]["threshold"]
    out = df.copy()
    texts = df[col].astype(str).tolist()
    bar = st.progress(0.0)
    for i, m in enumerate(models):
        p = C.get_predictor(name, m)(texts)
        out[f"p_fake__{m}"] = p
        out[f"pred__{m}"] = ["fake" if x >= thr else "real" for x in p]
        bar.progress((i + 1) / len(models))
    pc = [c for c in out.columns if c.startswith("p_fake__")]
    out["models_saying_fake"] = (out[pc] >= thr).sum(axis=1)
    st.dataframe(out, hide_index=True, width="stretch")
    C.csv_button(out, f"batch_predictions_{name}.csv", "⬇ Download results")
    if any(c in df.columns for c in ("label", "label_id")):
        st.caption("Your file has a label column - for a proper evaluation, use 'Use your own dataset' instead.")
