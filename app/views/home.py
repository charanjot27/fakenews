import streamlit as st

import common as C

P = C.cfg().get("project", {})
st.title(P.get("title", "SatyaCheck-Text"))
st.markdown(f"##### {P.get('subtitle', '')}")
st.caption(f"Team: {P.get('team', '-')} · Guide: {P.get('guide', '-')} · {P.get('institution', '')}  "
           "(edit these in `configs/config.yaml` → `project`)")

st.markdown(
    "**Problem.** Misinformation spreads in English, Hindi and Hinglish (Hindi in Roman letters). We compare "
    "classical text models (Bag-of-Words, TF-IDF) with BERT-family transformers - including Indian-language models "
    "such as MuRIL and IndicBERT - **on the same test set**, test research hypotheses statistically, and audit "
    "whether the models learn *content* or *shortcuts* (style, source, templates). Every number in this app is read "
    "from files produced by the code; nothing is typed by hand.")

st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  data [label="Dataset\\n(synthetic / LIAR / yours)", fillcolor="#eeeeee"];
  prep [label="prepare\\nclean · dedup · split ONCE\\n+ data card", fillcolor="#eeeeee"];
  cl [label="classical view\\nlowercase, links→token", fillcolor="#deebf7"];
  bv [label="bert view\\nclose to original", fillcolor="#fee6ce"];
  stat [label="BoW / TF-IDF\\nNB · LR · SVM · char", fillcolor="#c6dbef"];
  tr [label="BERT family\\nfine-tune · embeddings", fillcolor="#fdd0a2"];
  ev [label="evaluate\\nmacro-F1 + CI, AUC, ECE\\nconfusion, calibration", fillcolor="#e5f5e0"];
  cmp [label="compare\\nMcNemar · breakdowns\\nensemble · report", fillcolor="#e5f5e0"];
  hy [label="hypotheses\\n+ shortcut audit", fillcolor="#f2f0f7"];
  data -> prep; prep -> cl -> stat -> ev; prep -> bv -> tr -> ev; ev -> cmp -> hy; }
""")

name = st.session_state.get("dataset")
if not name:
    st.info("No results yet. Run `python -m fakenews run-all` in a terminal and refresh this page.")
    st.stop()
C.synthetic_banner(name)
st.markdown(f"**Selected dataset:** `{name}` · split: **{C.card(name).get('split_strategy', '?')}**")
comp = C.result_file(name, "comparison.json")
res = C.results(name)
if not comp:
    st.info(f"Run `python -m fakenews compare --name {name}` to build the comparison.")
    st.stop()
lb = C.result_csv(name, "summary.csv")
cols = st.columns(3)
top = lb.iloc[0]
cards = [("Top of leaderboard", top["model"], top["macro_f1"], top["ci_low"], top["ci_high"])]
for label, key in (("Best statistical (chosen on val)", "best_statistical"), ("Best transformer (chosen on val)", "best_transformer")):
    m = comp.get(key)
    if m:
        r = lb[lb["model"] == m].iloc[0]
        cards.append((label, m, r["macro_f1"], r["ci_low"], r["ci_high"]))
    else:
        cards.append((label, None, None, None, None))
for col, (label, m, f1, lo, hi) in zip(cols, cards):
    with col:
        if m:
            st.metric(label, m, help="macro-F1 on the test split with 95% bootstrap CI")
            st.markdown(f"macro-F1 **{f1:.3f}** (95% CI {lo:.3f}-{hi:.3f})")
        else:
            st.metric(label, "Not trained yet")
            st.caption("run `train-bert`")
st.info(comp["summary"])

hyp = C.result_file(name, "hypotheses.json")
if hyp:
    st.subheader("Hypotheses at a glance")
    st.markdown(" ".join(f"{h['id']} {C.verdict_badge(h['verdict'])} &nbsp;" for h in hyp["hypotheses"]),
                unsafe_allow_html=True)
    st.caption("Open the **Hypotheses** page for the test, numbers and explanation behind each verdict.")
