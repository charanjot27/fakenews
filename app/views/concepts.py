import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import common as C
from fakenews.preprocess import TOKEN_PATTERN, classical_view

st.title("Concepts lab")

st.header("1. Bag-of-Words vs. TF-IDF, live")
st.caption("Type 2-4 short sentences (one per line). A word that appears in every sentence gets a LOW TF-IDF weight.")
txt = st.text_area("Sentences", "the news says garlic cures dengue\nthe news says rain closes schools\n"
                   "the ministry says the viral news is false", height=100)
docs = [classical_view(s) for s in txt.splitlines() if s.strip()][:6]
if len(docs) >= 2:
    cv = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False)
    X = cv.fit_transform(docs)
    tv = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False)
    T = tv.fit_transform(docs)
    vocab = cv.get_feature_names_out()
    idx = [f"S{i + 1}" for i in range(len(docs))]
    bow = pd.DataFrame(X.toarray(), index=idx, columns=vocab)
    tf = pd.DataFrame(T.toarray(), index=idx, columns=tv.get_feature_names_out())[vocab]
    everywhere = [w for w in vocab if (bow[w] > 0).all()]
    a, b = st.columns(2)
    a.markdown("**BoW: raw counts**")
    a.dataframe(bow.style.background_gradient(cmap="Blues", axis=None), width="stretch")
    b.markdown("**TF-IDF: counts × rarity, each row normalised**")
    b.dataframe(tf.style.format("{:.2f}").background_gradient(cmap="Oranges", axis=None)
                .set_properties(subset=everywhere, **{"border": "2px solid #d62728"}), width="stretch")
    idf = pd.DataFrame({"word": tv.get_feature_names_out(), "idf": tv.idf_}).sort_values("idf")
    st.markdown(f"Words in **every** sentence (outlined red): `{everywhere or 'none'}` → lowest IDF "
                f"({idf['idf'].min():.2f}); words in one sentence get the highest IDF ({idf['idf'].max():.2f}).")
    st.caption("IDF = ln((1 + n) / (1 + df)) + 1 (scikit-learn's smoothed formula).")

st.header("2. How tokenizers split Hinglish and Hindi")
st.caption("English-only tokenizers break Hindi into many tiny or unknown pieces - the model sees noise. "
           "This is why MuRIL and IndicBERT exist. (Tokenizers are downloaded once from Hugging Face.)")
sent = st.text_input("Sentence", "Sarkar sabhi students ko free laptop de rahi hai | कल से 500 के नोट बंद हो जाएंगे")
names = st.multiselect("Tokenizers", ["bert", "distilbert", "mbert", "xlmr", "muril", "indicbert", "hingbert"],
                       ["bert", "mbert", "muril", "indicbert"])
if st.button("Tokenize"):
    from fakenews.transformer_models import load_tokenizer
    rows = []
    for n in names:
        try:
            tok = load_tokenizer(n)
            pieces = tok.tokenize(sent)
            unk = sum(p == tok.unk_token for p in pieces)
            rows.append({"tokenizer": n, "pieces": len(pieces), "unknown": unk,
                         "words per piece": round(len(sent.split()) / max(1, len(pieces)), 2),
                         "split": " | ".join(pieces)})
        except Exception as e:  # noqa: BLE001
            rows.append({"tokenizer": n, "pieces": None, "unknown": None, "words per piece": None, "split": f"ERROR: {e}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Fewer pieces and zero unknowns = the tokenizer 'knows' the language. Many 1-2 character pieces or "
               "[UNK] = it does not.")

st.header("3. Notes")
p = C.ROOT / "docs" / "CONCEPTS.md"
st.markdown(p.read_text(encoding="utf-8") if p.exists() else "docs/CONCEPTS.md missing")
