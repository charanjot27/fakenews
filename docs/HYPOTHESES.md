# Research hypotheses

These 14 hypotheses were written **before** looking at any results. `python -m fakenews hypotheses --name <dataset>`
tests every one automatically on the saved predictions and writes `outputs/<dataset>/reports/hypotheses.md` and
`results/hypotheses.json`. The UI's **Hypotheses** page shows the same files.

## Verdicts

| Verdict | Meaning | In plain words |
|---|---|---|
| **SUPPORTED** | The predicted effect is present and statistically significant (p < alpha, default 0.05) | "proved" on this dataset |
| **REJECTED** | The *opposite* effect is present and significant | "disproved" |
| **INCONCLUSIVE** | No significant difference either way | *not* the same as disproved: the data cannot tell |
| **NOT TESTABLE** | A needed model or column is missing (the reason is printed) | train the missing model / add the column |

Rules that keep this honest:
- The "best" model of a family is always chosen on the **validation** split, never on the test split.
- Paired tests (McNemar, paired bootstrap) compare models **on the same test items**.
- Every number comes from result files; the explanation sentences are templates filled in from them.
- A verdict on synthetic data shows that the method works. Make research claims only from your real dataset (time- or source-based split).

## The hypotheses

| ID | Hypothesis | Test |
|---|---|---|
| H1 | The best BERT-family model beats the best BoW/TF-IDF model (macro-F1) | McNemar + paired bootstrap CI |
| H2 | TF-IDF beats Bag-of-Words | McNemar + paired bootstrap CI |
| H3 | On Hinglish items, character n-gram TF-IDF beats word TF-IDF | same, on the Hinglish subset |
| H4 | On Hindi + Hinglish items, Indian/multilingual transformers beat English-only ones (English subset reported for contrast) | same, on the Hindi + Hinglish subset |
| H5 | Writing style alone explains > 50% of the model's skill | Shortcut Index with a bootstrap CI (style-only probe) |
| H6 | The source/channel alone explains > 50% of the model's skill | Shortcut Index (source-only probe) |
| H7 | Calm (neutral-tone) fakes are caught less often than sensational fakes | Fisher exact test on recall |
| H8 | Real reports *about* debunks ("viral message is false") are misclassified more often than ordinary real news (keyword trap) | Fisher exact test |
| H9 | Fine-tuning beats frozen embeddings + LR for the same backbone | McNemar per backbone |
| H10 | Averaging the best statistical model and the best transformer beats the better member | McNemar + paired bootstrap |
| H11 | The best transformer is worse calibrated (higher ECE) than the best statistical model | paired bootstrap of the ECE difference |
| H12 | Scores fall when whole templates are held out (template memorisation; synthetic data) | held-out CI vs random-split score |
| H13 | Scores fall when whole sources are held out | held-out CI vs random-split score |
| H14 | The style shortcut grows with the style correlation built into the data | Spearman over the `style-sweep` experiment + CIs at the ends |

## Adding your own hypothesis

Add a block to `fakenews/hypotheses.py` → `run_hypotheses()`. The helpers `_pair_hyp` (two models on the same items,
optionally on a subset mask) and `_fisher_hyp` (two rates) cover most cases. Every hypothesis returns the same
fields: statement, why it matters, test, verdict, p-value, effect, numbers and explanation.
