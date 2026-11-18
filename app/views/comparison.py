import pandas as pd
import streamlit as st

import common as C

name = C.require_dataset()
st.title("Model comparison")
C.synthetic_banner(name)
comp = C.result_file(name, "comparison.json")
lb = C.result_csv(name, "summary.csv")
if comp is None or lb is None:
    st.info(f"Not compared yet - run `python -m fakenews compare --name {name}`.")
    st.stop()

st.subheader("Statistical vs. Transformer")
st.info(comp["summary"])
st.caption(f"Selection rule: {comp['selection_rule']}. Leaderboard is sorted by test macro-F1. "
           f"Decision threshold: P(fake) ≥ {comp['threshold']}.")
if not comp.get("best_transformer"):
    st.warning("Transformer models: **Not trained yet** - run `python -m fakenews train-bert --name " + name + "`.")

st.subheader("Leaderboard (test split)")
show = lb[["rank", "model", "family", "mode", "macro_f1", "ci_low", "ci_high", "accuracy", "precision_fake",
           "recall_fake", "roc_auc", "pr_auc", "brier", "ece", "val_macro_f1", "train_seconds", "ms_per_item", "params"]]
hi_better = ["macro_f1", "accuracy", "precision_fake", "recall_fake", "roc_auc", "pr_auc", "val_macro_f1"]
lo_better = ["brier", "ece", "train_seconds", "ms_per_item"]
sty = (show.style.highlight_max(subset=hi_better, props="color:#4ade80;font-weight:700").highlight_min(subset=lo_better, props="color:#4ade80;font-weight:700")
       .format({c: "{:.3f}" for c in hi_better + ["ci_low", "ci_high", "brier", "ece"]} |
               {"train_seconds": "{:.1f}", "ms_per_item": "{:.2f}", "params": "{:,.0f}"}, na_rep="-"))
st.dataframe(sty, hide_index=True, width="stretch")
st.caption("Green = best value in the column. macro-F1 = mean of fake-F1 and real-F1 (headline metric); "
           "ci = 95% bootstrap interval; ECE = calibration error (lower is better); Brier = probability error.")
C.csv_button(lb, f"{name}_leaderboard.csv")

C.show_fig(name, "macro_f1_bar.png", "Bars overlap within their error bars → the difference may be luck. "
           "Check the significance table below.")

tabs = st.tabs(["Confusion matrices", "ROC & PR", "Calibration", "Efficiency", "Per language", "By style & length",
                "Hard cases", "Training curves"])
with tabs[0]:
    C.show_fig(name, "confusion_matrices.png", "Rows = truth, columns = prediction. Off-diagonal cells are mistakes: "
               "top-right = real called fake (false alarm), bottom-left = fake missed.")
with tabs[1]:
    a, b = st.columns(2)
    with a:
        C.show_fig(name, "roc_curves.png", "Higher and further left = better ranking of fake above real.")
    with b:
        C.show_fig(name, "pr_curves.png", "Precision vs recall for the fake class.")
with tabs[2]:
    C.show_fig(name, "calibration.png", "On the diagonal = honest probabilities; below = over-confident about fake.")
with tabs[3]:
    C.show_fig(name, "efficiency.png", "Top-left is best: accurate and fast. Bubble size = parameters.")
with tabs[4]:
    C.show_fig(name, "per_language.png", "Where Indian-language models should differ from English-only BERT.")
with tabs[5]:
    a, b = st.columns(2)
    with a:
        C.show_fig(name, "per_style.png", "Does the model still work on calm (neutral) items?")
    with b:
        C.show_fig(name, "per_length.png", "Short < 15 words ≤ medium < 40 words ≤ long.")
with tabs[6]:
    C.show_fig(name, "hard_cases.png", "debunk_report = real news about a fake claim; calm_fake = fake in an official tone.")
with tabs[7]:
    C.show_fig(name, "training_curves.png", "Validation macro-F1 and loss per epoch for each transformer.")

st.subheader("Significance: top model vs. every other model")
sig = C.result_csv(name, "significance.csv")
if sig is not None and len(sig):
    st.dataframe(sig.style.format({c: "{:.4f}" for c in ["macro_f1_a", "macro_f1_b", "diff", "diff_ci_low",
                                                          "diff_ci_high", "p_value"]}), hide_index=True, width="stretch")
    st.caption("McNemar test on the same test items: counts where A was right and B wrong, and the reverse. "
               "p < 0.05 → the difference is unlikely to be luck. `diff` CI is from a paired bootstrap.")
    C.csv_button(sig, f"{name}_significance.csv")

st.subheader("Disagreements browser")
dis = C.result_csv(name, "disagreements.csv")
if dis is None or dis.empty:
    st.info("Needs one statistical and one transformer model.")
else:
    svt = comp["statistical_vs_transformer"]
    st.markdown(f"**{svt['n_disagreements']}** test items where `{svt['best_statistical']}` and "
                f"`{svt['best_transformer']}` disagree: {svt['disagreement_counts']}")
    who = st.selectbox("Show", ["all"] + sorted(dis["who_is_right"].unique()))
    d = dis if who == "all" else dis[dis["who_is_right"] == who]
    st.dataframe(d, hide_index=True, width="stretch", height=380)
    st.caption("This is the most useful table for seeing how the two approaches differ. Sorted by probability gap.")
    C.csv_button(dis, f"{name}_disagreements.csv")

with st.expander("Each model's most confident mistakes"):
    err = C.result_csv(name, "errors.csv")
    if err is not None and len(err):
        m = st.selectbox("Model", list(dict.fromkeys(err["model"])))
        st.dataframe(err[err["model"] == m], hide_index=True, width="stretch")
