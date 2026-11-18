"""Model internals: everything an evaluator needs to see what a model does, how it was trained and tuned,
and how a single prediction is produced step by step."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
from fakenews.config import output_dir
from fakenews.evaluate import compute_metrics

name = C.require_dataset()
st.title("Model internals (transparency)")
st.caption("Pick any trained model to see what it reads, how it was trained and tuned, its learned weights, its "
           "confusion matrix at any threshold, and a step-by-step trace of a single prediction.")
C.synthetic_banner(name)
res = C.results(name)
if not res:
    st.info("No trained models yet.")
    st.stop()
model = st.selectbox("Model", list(res), format_func=lambda m: f"{m} - {res[m].get('label', '')}")
r = res[model]

# ------------------------------------------------------------------ 1. what it is
st.header("1. What this model is")
a, b = st.columns([2, 1])
with a:
    st.markdown(f"**{r.get('label')}** · family **{r['family']}** · mode **{r.get('mode')}**")
    st.markdown(r.get("about", ""))
    if r["family"] == "Statistical":
        st.markdown(
            "**Pipeline:** `text` (classical view) → "
            f"{'CountVectorizer (word + word-pair counts)' if r['features'] == 'bow' else 'TfidfVectorizer (words, sublinear TF, L2 norm)' if r['features'] == 'tfidf' else 'TfidfVectorizer (character 3-5-grams)'}"
            f" → {'Multinomial Naive Bayes' if r['classifier'] == 'nb' else 'Logistic Regression (balanced)' if r['classifier'] == 'lr' else 'Linear SVM + Platt calibration'} → P(fake)")
    else:
        st.markdown(f"**Backbone:** [`{r['hf_id']}`](https://huggingface.co/{r['hf_id']}) · languages: {r.get('languages')} "
                    f"· size ≈ {r.get('size')}")
        st.markdown("**Pipeline:** `text_bert` (close to original) → sub-word tokenizer → transformer encoder → "
                    + ("2-class head, whole network fine-tuned → softmax → P(fake)" if r["mode"] == "Fine-tuned"
                       else "mean of token vectors (frozen) → standardise → Logistic Regression → P(fake)"))
with b:
    st.metric("Parameters", f"{r.get('params', 0):,}")
    st.metric("Training time", f"{r.get('train_seconds', 0):.1f} s")
    st.metric("Inference", f"{r.get('infer_ms_per_item', 0):.2f} ms / item")
    st.caption(f"Device: {r.get('device')} · seed {r.get('seed')} · trained {r.get('created')}")

st.subheader("Exact settings used")
st.json(r.get("settings", {}), expanded=False)

# ------------------------------------------------------------------ 2. training & tuning
st.header("2. How it was trained and tuned")
if r.get("tuning"):
    key = r.get("tuned_param") or ("C" if "C" in r["tuning"][0] else list(r["tuning"][0])[0])
    t = pd.DataFrame(r["tuning"])
    st.markdown(f"Hyper-parameter **{key}** was chosen on the **validation split** (never the test split) by macro-F1:")
    best = r.get("best_value", r.get("best_C"))
    st.dataframe(t.style.apply(lambda row: ["background-color:#c7e9c0" if row[key] == best else "" for _ in row], axis=1)
                 .format({"val_macro_f1": "{:.4f}"}), hide_index=True)
if r.get("curve"):
    cv = pd.DataFrame(r["curve"])
    st.markdown(f"Fine-tuned for up to {r['settings']['epochs']} epoch(s) with AdamW (lr {r.get('learning_rate')}), "
                f"{r.get('warmup_steps')} warm-up steps of {r.get('steps')}, gradient clipping "
                f"{r['settings']['grad_clip']}, early stopping (patience {r['settings']['patience']}). "
                f"**Best epoch kept: {r.get('best_epoch')}.** Class weights: {r.get('class_weights') or 'off (balanced data)'}.")
    fig = go.Figure()
    fig.add_scatter(x=cv["epoch"], y=cv["train_loss"], name="train loss", mode="lines+markers")
    fig.add_scatter(x=cv["epoch"], y=cv["val_loss"], name="val loss", mode="lines+markers")
    fig.add_scatter(x=cv["epoch"], y=cv["val_macro_f1"], name="val macro-F1", mode="lines+markers", yaxis="y2")
    fig.update_layout(title="Training curve", xaxis_title="Epoch", yaxis_title="Loss",
                      yaxis2=dict(title="macro-F1", overlaying="y", side="right", range=[0, 1]), height=360)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(cv, hide_index=True)
if r.get("seeds") and len(r["seeds"]) > 1:
    st.markdown(f"**Seeds:** macro-F1 mean {r['macro_f1_seed_mean']:.3f} ± {r['macro_f1_seed_std']:.3f}")
    st.dataframe(pd.DataFrame(r["seeds"]), hide_index=True)

# ------------------------------------------------------------------ 3. learned weights
if r.get("top_features"):
    st.header("3. What it learned (strongest weights)")
    st.caption(f"Vocabulary size: {r.get('vocab_size'):,}. Positive weight pushes towards FAKE, negative towards REAL. "
               "These lists are the model - nothing is hidden.")
    C.show_fig(name, f"top_features/{model}.png", key=f"tf_{model}")
    x, y = st.columns(2)
    x.dataframe(pd.DataFrame(r["top_features"]["fake"], columns=["→ FAKE", "weight"]), hide_index=True, width="stretch")
    y.dataframe(pd.DataFrame(r["top_features"]["real"], columns=["→ REAL", "weight"]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 4. evaluation + threshold explorer
st.header("4. Evaluation")
v, t = r["val_metrics"], r["test_metrics"]
keys = ["accuracy", "precision_fake", "recall_fake", "f1_fake", "f1_real", "macro_f1", "roc_auc", "pr_auc", "brier", "ece"]
st.dataframe(pd.DataFrame({"metric": keys, "validation": [v.get(k) for k in keys], "test": [t.get(k) for k in keys]})
             .style.format({"validation": "{:.4f}", "test": "{:.4f}"}, na_rep="-"), hide_index=True)
st.caption(f"Test macro-F1 95% CI: {t['macro_f1_ci_low']:.3f} - {t['macro_f1_ci_high']:.3f} (1,000 bootstrap resamples).")

st.subheader("Threshold explorer")
st.caption("All metrics above use P(fake) ≥ 0.5. Move the threshold to see the trade-off between catching fakes and "
           "false alarms. Computed live from the saved test predictions.")
pred_f = output_dir(C.cfg(), name) / "results" / "predictions" / f"{model}.csv"
if pred_f.exists():
    pr = pd.read_csv(pred_f).merge(C.get_splits(name)["test"][["id", "label_id", "text_bert"]], on="id")
    thr = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05)
    m = compute_metrics(pr["label_id"], pr["p_fake"], thr, 0)
    cm = m["confusion"]
    a, b = st.columns([1, 1])
    with a:
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["pred real", "pred fake"], y=["true real", "true fake"], colorscale="Blues",
                                   text=z, texttemplate="%{text}", showscale=False))
        fig.update_layout(title=f"Confusion matrix at threshold {thr:.2f}", height=330, yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.metric("macro-F1", f"{m['macro_f1']:.3f}")
        st.metric("Fakes caught (recall)", f"{m['recall_fake']:.1%}")
        st.metric("Of items called fake, really fake (precision)", f"{m['precision_fake']:.1%}")
        st.metric("Real items wrongly called fake", cm["fp"])
    fig = go.Figure()
    for lab, col, nm in ((1, C.FAKE, "true fake"), (0, C.REAL, "true real")):
        fig.add_histogram(x=pr.loc[pr.label_id == lab, "p_fake"], name=nm, marker_color=col, opacity=0.6, nbinsx=25)
    fig.add_vline(x=thr, line_dash="dash")
    fig.update_layout(barmode="overlay", title="Distribution of P(fake) on the test split", xaxis_title="P(fake)",
                      yaxis_title="items", height=320)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ 5. trace one prediction
st.header("5. Trace one prediction step by step")
src = st.radio("Text", ["pick a test item", "type my own"], horizontal=True)
if src == "pick a test item" and pred_f.exists():
    i = st.number_input("Row of the test split", 0, len(pr) - 1, 0)
    row = pr.iloc[int(i)]
    text = row["text_bert"]
    st.markdown(f"True label: **{'fake' if row['label_id'] else 'real'}** · saved P(fake) = {row['p_fake']:.4f}")
else:
    text = st.text_area("Text", "Garlic water cures dengue in just 3 days, doctors are hiding this from you!")

if text:
    if r["family"] == "Statistical":
        from fakenews.classical import load_classical
        from fakenews.explain import highlight_html, trace_classical, word_scores_from_trace

        pipe = load_classical(C.cfg(), name, model)
        tr = trace_classical(pipe, text)
        st.markdown("**Step 1 - raw text**")
        st.code(tr["raw"], language=None)
        st.markdown("**Step 2 - classical view** (lowercase, links → `urltoken`)")
        st.code(tr["clean"], language=None)
        if tr["tokens"] is not None:
            st.markdown(f"**Step 3 - tokens** ({len(tr['tokens'])}); not in the vocabulary (ignored): "
                        f"`{tr['unknown_tokens'] or 'none'}`")
            st.code(" | ".join(tr["tokens"]), language=None)
        else:
            st.markdown("**Step 3 - character 3-5-grams** of each word (e.g. ' gar', 'garl', 'arli' ...)")
        st.markdown(f"**Step 4 - feature values × learned weights** ({len(tr['features'])} active features)")
        st.dataframe(tr["features"].style.format({"value": "{:.4f}", "weight": "{:+.4f}", "contribution": "{:+.4f}"})
                     .background_gradient(subset=["contribution"], cmap="RdYlGn_r"), hide_index=True, width="stretch",
                     height=300)
        st.markdown(f"**Step 5 - score.** {tr['how']}")
        st.code(f"intercept {tr['intercept']:+.4f} + sum of contributions {tr['features']['contribution'].sum():+.4f} "
                f"= score {tr['score']:+.4f}\n"
                + (f"sigmoid(score) = {1 / (1 + np.exp(-tr['score'])):.4f}\n" if r["classifier"] != "svm" else "")
                + f"model's P(fake) = {tr['p_fake']:.4f}  ->  {'FAKE' if tr['p_fake'] >= 0.5 else 'REAL'}", language=None)
        st.markdown("**Word highlights** (red → fake, green → real; hover for the value)")
        st.markdown(highlight_html(text, word_scores_from_trace(tr)), unsafe_allow_html=True)
    else:
        from fakenews.transformer_models import load_tokenizer

        short = r["backbone"]
        try:
            tok = load_tokenizer(short)
            enc = tok(text, truncation=True, max_length=r["settings"]["max_length"])
            pieces = tok.convert_ids_to_tokens(enc["input_ids"])
            st.markdown(f"**Step 1 - bert view** (close to original)")
            st.code(text, language=None)
            st.markdown(f"**Step 2 - sub-word tokens** ({len(pieces)}; unknown: {pieces.count(tok.unk_token)})")
            st.code(" | ".join(pieces), language=None)
            st.markdown("**Step 3 - token ids** fed to the encoder")
            st.code(str(enc["input_ids"]), language=None)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Tokenizer not available offline: {e}")
        if st.button("Run the model + leave-one-word-out importance"):
            from fakenews.explain import occlusion, occlusion_html
            fn = C.get_predictor(name, model)
            with st.spinner("Running..."):
                occ = occlusion(fn, text, 30)
            p = occ["base_p"].iloc[0]
            st.markdown(f"**Step 4 - P(fake) = {p:.4f} → {'FAKE' if p >= 0.5 else 'REAL'}**")
            st.markdown("**Step 5 - word importance:** each word is removed in turn; red = removing it lowers P(fake).")
            st.markdown(occlusion_html(occ), unsafe_allow_html=True)
            st.dataframe(occ[["word", "p_without", "delta"]], hide_index=True)

# ------------------------------------------------------------------ 6. raw files
st.header("6. Raw files")
with st.expander("Full result JSON for this model"):
    rr = {k: v for k, v in r.items() if k != "config"}
    st.json(rr, expanded=False)
    st.download_button("⬇ JSON", json.dumps(r, indent=2, ensure_ascii=False), f"{name}_{model}.json")
logf = Path(output_dir(C.cfg(), name)) / "logs" / "run_log.jsonl"
if logf.exists():
    with st.expander("Run log - every command that produced these results"):
        rows = [json.loads(line) for line in logf.read_text(encoding="utf-8").splitlines() if line.strip()]
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, width="stretch")
