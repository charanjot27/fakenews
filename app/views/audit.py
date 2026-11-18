import streamlit as st

import common as C

name = C.require_dataset()
st.title("Shortcut audit")
C.synthetic_banner(name)
st.markdown("Does the model learn **content**, or an easy **shortcut**? Each probe is a deliberately limited model.\n\n"
            "**Shortcut Index = (probe macro-F1 − chance) ÷ (full model macro-F1 − chance).** "
            "≈0: the shortcut explains little. ≈1 or more: it explains almost all of the model's apparent skill.")
aud = C.result_csv(name, "audit.csv")
if aud is None:
    st.info(f"Run `python -m fakenews audit --name {name}`.")
    st.stop()
meta = C.result_file(name, "audit.json")
st.caption(f"Chance macro-F1 (random guesser keeping the class ratio): {meta['chance_macro_f1']:.3f} · reference model "
           f"for the index: `{meta['reference_model']}` ({meta['reference_macro_f1']:.3f})")

probes = aud[aud["kind"] == "probe"]
C.show_fig(name, "shortcut_index.png", "Red ≥ 0.5: the shortcut explains at least half the skill.")
st.dataframe(probes[["probe", "input_seen", "macro_f1", "ci_low", "ci_high", "reference_model", "shortcut_index",
                     "index_ci_low", "index_ci_high", "meaning"]]
             .style.format({c: "{:.3f}" for c in ["macro_f1", "ci_low", "ci_high", "shortcut_index", "index_ci_low",
                                                    "index_ci_high"]}), hide_index=True, width="stretch")

held = aud[aud["kind"] == "held_out"]
if len(held):
    st.subheader("Held-out splits: does the score survive new sources / templates / time?")
    st.dataframe(held[["probe", "input_seen", "reference_macro_f1", "macro_f1", "ci_low", "ci_high", "drop_vs_reference",
                       "meaning"]].rename(columns={"reference_macro_f1": "random_split_f1", "macro_f1": "held_out_f1"})
                 .style.format({c: "{:.3f}" for c in ["random_split_f1", "held_out_f1", "ci_low", "ci_high",
                                                        "drop_vs_reference"]}), hide_index=True, width="stretch")
    st.caption("A model that scores 95% on a random split and 70% on a held-out split is a 70% model.")
C.csv_button(aud, f"{name}_audit.csv")

st.subheader("How to read each probe")
st.markdown("""
- **style_only** - sees only length, '!' and '?' counts, share of capitals, emojis, links and script. High → labels can be guessed from *how* it is written.
- **first_5_words** - sees only the opening. High → the start of a post gives the answer away (e.g. "BREAKING:").
- **shuffled_words[model]** - the *same trained model* reads the words in random order. High → word order and context barely matter to it (expected ≈1 for bag-of-words).
- **source_only** - sees only the channel name. High → it could learn *who* posted rather than *what* is said.
- **metadata_only** - (LIAR) only speaker, party, state, job, context. Our LIAR pilot found this beats the statement text.
- **source_held_out / template_held_out / time_split** - full text, but no source/template/time period is shared between train and test.
""")
