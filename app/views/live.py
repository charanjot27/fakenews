import time

import streamlit as st

import common as C

name = C.require_dataset()
st.title("Live checker")
st.warning("A model only judges the **wording** of a text. It does not check facts. Always verify with a fact-checker.",
           icon=":material/warning:")
res = C.results(name)
if not res:
    st.info("No trained models.")
    st.stop()

EXAMPLES = {
    "Real": "Schools in Patna will remain closed on Monday due to heavy rain, the district administration said on its official website.",
    "Fake": "Garlic water cures dengue in just 3 days, doctors are hiding this from you!",
    "Hard real": "The Health Ministry clarified that a viral message claiming hot water kills the virus is false.",
    "Hard fake": "According to an official notice, board exams for classes 10 and 12 have been cancelled this year.",
    "Hinglish": "Sarkar sabhi students ko free laptop de rahi hai, is link par jaldi register karo!",
    "Hindi": "कल से 500 के नोट बंद हो जाएंगे, यह संदेश सभी को भेजें।",
}
cols = st.columns(len(EXAMPLES))
for col, (k, v) in zip(cols, EXAMPLES.items()):
    if col.button(k, width="stretch"):
        st.session_state["live_text"] = v
text = st.text_area("News text", key="live_text", height=110,
                    placeholder="Paste a headline, WhatsApp forward or short article...")
stat = [m for m in res if res[m]["family"] == "Statistical"]
trans = [m for m in res if res[m]["family"] == "Transformer"]
models = st.multiselect("Models", list(res), default=stat + trans[:3])
explain = st.checkbox("Transformer word importance (slow: one model run per word)", value=False)
if not trans:
    st.caption("Transformers: Not trained yet - run `train-bert`.")

if st.button("Check", type="primary") and text.strip():
    thr = C.cfg()["evaluation"]["threshold"]
    outs = []
    for m in models:
        fn = C.get_predictor(name, m)
        t0 = time.perf_counter()
        p = float(fn([text])[0])
        outs.append((m, p, (time.perf_counter() - t0) * 1000))
    n_fake = sum(p >= thr for _, p, _ in outs)
    st.subheader(f"{n_fake} of {len(outs)} models say fake")
    for i in range(0, len(outs), 3):
        cs = st.columns(3)
        for col, (m, p, ms) in zip(cs, outs[i:i + 3]):
            with col.container(border=True):
                lab = "FAKE" if p >= thr else "REAL"
                color = C.FAKE if p >= thr else C.REAL
                st.markdown(f"**{m}** <span style='float:right;color:{color};font-weight:700'>{lab}</span>",
                            unsafe_allow_html=True)
                st.progress(p, text=f"P(fake) = {p:.3f}")
                st.caption(f"{res[m]['family']} · {ms:.0f} ms (first call includes loading)")

    st.subheader("Why? Word-level explanations")
    from fakenews.classical import load_classical
    from fakenews.explain import highlight_html, occlusion, occlusion_html, trace_classical, word_scores_from_trace
    for m in models:
        if res[m]["family"] == "Statistical":
            tr = trace_classical(load_classical(C.cfg(), name, m), text)
            st.markdown(f"**{m}** - red pushes towards fake, green towards real (exact weights; see Model internals "
                        "for the full calculation)")
            st.markdown(highlight_html(text, word_scores_from_trace(tr)), unsafe_allow_html=True)
        elif explain:
            if len(text.split()) > 40:
                st.caption(f"{m}: text too long for word importance (limit 40 words).")
                continue
            occ = occlusion(C.get_predictor(name, m), text)
            st.markdown(f"**{m}** - leave-one-word-out: red = removing the word lowers P(fake)")
            st.markdown(occlusion_html(occ), unsafe_allow_html=True)
