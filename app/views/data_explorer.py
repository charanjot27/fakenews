import pandas as pd
import streamlit as st

import common as C

name = C.require_dataset()
st.title("Data explorer")
C.synthetic_banner(name)
card = C.card(name)
if not card:
    st.warning("No data card found. Run `prepare`.")
    st.stop()

st.subheader("Data card")
c = st.columns(4)
c[0].metric("Train / Val / Test", " / ".join(str(card["sizes"][k]) for k in ("train", "val", "test")))
c[1].metric("Exact duplicates removed", card.get("exact_duplicates_removed"))
c[2].metric("Near-duplicate rows", card.get("near_duplicate_rows"),
            help="Same text after ignoring case, links, emojis, digits and punctuation. Reported, not removed.")
c[3].metric("Split strategy", card.get("split_strategy", "?").split(" (")[0])
st.markdown(f"- **Share fake per split:** {card.get('share_fake')}\n"
            f"- **Average words by label:** {card.get('avg_words_by_label')}\n"
            f"- **Text views:** {card.get('text_views')}\n"
            f"- **Language inferred heuristically:** {card.get('language_inferred')}\n"
            f"- **Date range:** {card.get('date_range')}\n"
            f"- **Fingerprint (SHA-256 of the input file):** `{str(card.get('fingerprint_sha256'))[:64]}`")
if card.get("bert_text_warning"):
    st.warning(card["bert_text_warning"])
with st.expander("Full data card (JSON)"):
    st.json(card)

st.subheader("Charts")
a, b, c3 = st.columns(3)
with a:
    C.show_fig(name, "data_class_balance.png", "Fake (red) vs real (green) in each split - should be similar.")
with b:
    C.show_fig(name, "data_language_mix.png", "How many items per language and label.")
with c3:
    C.show_fig(name, "data_length.png", "If lengths differ a lot by label, length alone becomes a clue (see audit).")

res = C.results(name)
src = next((m for m in ("bow_lr", "tfidf_lr") if m in res), None)
if src:
    st.subheader(f"Top words per class (from `{src}`)")
    tf = res[src]["top_features"]
    x, y = st.columns(2)
    x.markdown("**Push towards FAKE**")
    x.dataframe(pd.DataFrame(tf["fake"], columns=["word / phrase", "weight"]), hide_index=True, width="stretch")
    y.markdown("**Push towards REAL**")
    y.dataframe(pd.DataFrame(tf["real"], columns=["word / phrase", "weight"]), hide_index=True, width="stretch")
    st.caption("Leakage check (BUILD_INSTRUCTIONS 13.3): publisher names, URLs, channel names, 'fact check' stamps or "
               "datelines in these lists are warning signs - remove them from the text and re-run.")

st.subheader("Sample rows")
sp = C.get_splits(name)
split = st.radio("Split", ["train", "val", "test"], horizontal=True)
df = sp[split].copy()
df["label"] = df["label_id"].map({1: "fake", 0: "real"})
f = st.columns(4)
lab = f[0].multiselect("Label", ["fake", "real"], ["fake", "real"])
langs = sorted(df["language"].dropna().unique())
lang = f[1].multiselect("Language", langs, langs)
mask = df["label"].isin(lab) & df["language"].isin(lang)
if "topic" in df.columns:
    tops = sorted(df["topic"].dropna().unique())
    mask &= df["topic"].isin(f[2].multiselect("Topic", tops, tops))
if "case_type" in df.columns:
    cts = sorted(df["case_type"].dropna().unique())
    mask &= df["case_type"].isin(f[3].multiselect("Case type", cts, cts))
show_cols = [c for c in ["id", "label", "language", "topic", "style", "case_type", "source", "date", "text_bert", "text",
                         "noisy_label"] if c in df.columns]
st.caption("`text_bert` = what transformers read · `text` = what BoW/TF-IDF read · `noisy_label` = label flipped on "
           "purpose by the generator (synthetic only)")
st.dataframe(df.loc[mask, show_cols], hide_index=True, width="stretch", height=420)
C.csv_button(df.loc[mask, show_cols], f"{name}_{split}_sample.csv")
