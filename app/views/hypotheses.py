from pathlib import Path

import pandas as pd
import streamlit as st

import common as C

name = C.require_dataset()
st.title("Hypotheses: proved, disproved or inconclusive")
C.synthetic_banner(name)
hyp = C.result_file(name, "hypotheses.json")
if not hyp:
    st.info(f"Run `python -m fakenews hypotheses --name {name}` (after compare and audit).")
    st.stop()

st.markdown("Each hypothesis is stated **before** looking at the result, then tested automatically with a "
            f"statistical test on the saved predictions (alpha = {hyp['alpha']}). Models are always chosen on the "
            "validation split, so the test split is never used to decide what gets tested.")
with st.expander("What the verdicts mean", expanded=True):
    for k, v in hyp["verdict_meaning"].items():
        st.markdown(f"{C.verdict_badge(k)} &nbsp; {v}", unsafe_allow_html=True)

H = hyp["hypotheses"]
counts = pd.Series([h["verdict"] for h in H]).value_counts()
cols = st.columns(4)
for col, v in zip(cols, ["SUPPORTED", "REJECTED", "INCONCLUSIVE", "NOT TESTABLE"]):
    col.metric(v.title(), int(counts.get(v, 0)))

tbl = pd.DataFrame([{"ID": h["id"], "Hypothesis": h["title"], "Verdict": h["verdict"],
                     "p-value": h["p_value"], "Effect": h["effect"]} for h in H])
tbl[["p-value", "Effect"]] = tbl[["p-value", "Effect"]].apply(pd.to_numeric, errors="coerce")
st.dataframe(tbl.style.format({"p-value": "{:.4f}", "Effect": "{:+.3f}"}, na_rep="-")
             .map(lambda v: f"background-color:{C.VERDICT_STYLE.get(v, ('', '#fff'))[1]}", subset=["Verdict"]),
             hide_index=True, width="stretch")
C.csv_button(tbl, f"{name}_hypotheses.csv")

for h in H:
    with st.expander(f"{h['id']} · {h['title']}"):
        st.markdown(C.verdict_badge(h["verdict"]), unsafe_allow_html=True)
        st.markdown(f"**Hypothesis.** {h['statement']}")
        st.markdown(f"**Why it matters.** {h['why_it_matters']}")
        st.markdown(f"**Test.** {h['test']}")
        st.markdown(f"**Result.** {h['explanation']}")
        if h["models_used"]:
            st.caption("Models used: " + ", ".join(f"`{m}`" for m in h["models_used"]))
        if h["numbers"]:
            st.json(h["numbers"], expanded=False)
        if h["id"] == "H14":
            p = Path(C.cfg()["paths"]["outputs_dir"]) / "_experiments" / "style_sweep.png"
            if p.exists():
                st.image(str(p))
                sw = pd.read_csv(p.with_suffix(".csv"))
                st.dataframe(sw, hide_index=True)
        if h["id"] in ("H5", "H6", "H12", "H13"):
            st.caption("Details on the **Shortcut audit** page.")

rep = Path(C.cfg()["paths"]["outputs_dir"]) / name / "reports" / "hypotheses.md"
if rep.exists():
    st.download_button("⬇ Hypotheses report (Markdown)", rep.read_bytes(), f"{name}_hypotheses.md")
