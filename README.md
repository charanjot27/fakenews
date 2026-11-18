# SatyaCheck-Text

**Fake vs. real news detection for English, Hinglish and Hindi.** The project compares Bag-of-Words, TF-IDF and
BERT-family models (BERT, DistilBERT, RoBERTa, ALBERT, mBERT, XLM-R, MuRIL, IndicBERT, HingBERT) on the same test
set. It tests 14 research hypotheses statistically, audits the models for shortcut learning, and includes a
research UI for presentations.

> Minor project: SatyaCheck / HinCheck (text module). The full specification is in
> [`BUILD_INSTRUCTIONS.md`](BUILD_INSTRUCTIONS.md); the concepts are in [`docs/CONCEPTS.md`](docs/CONCEPTS.md)
> and the hypotheses in [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md).
>
> **Team guide (PDF):** [`docs/SatyaCheck_Text_Team_Guide.pdf`](docs/SatyaCheck_Text_Team_Guide.pdf) explains how
> everything works, with all results, charts and a tour of the UI. It is regenerated from the result files with
> `python scripts/make_guide.py` followed by `node scripts/print_pdf.js` (Playwright).

**Rule followed everywhere:** no number is typed by hand. Every table, chart, report sentence and UI value is read
from result files that the code produced.

---

## 1. Setup (once)

Use Python **3.11 or 3.12** in its own environment. Python 3.13 also works; avoid 3.14.

```bash
conda create -n satyacheck python=3.11 -y
```

```bash
conda activate satyacheck
```

```bash
pip install -r requirements.txt
```

Check the device (prints `mps` on an Apple Silicon Mac, `cuda` on Colab/Kaggle, otherwise `cpu`):

```bash
python -c "from fakenews.utils import get_device; print(get_device())"
```

## 2. First demo (one command)

```bash
python -m fakenews run-all
```

This generates 3,000 synthetic items, splits them once, trains the 5 statistical models, fine-tunes DistilBERT for
1 epoch, trains DistilBERT in embeddings mode, runs the style-sweep experiment, then the comparison, the shortcut
audit and the hypothesis tests. It takes about 5 minutes on a laptop CPU. For the bigger set (DistilBERT, ALBERT,
mBERT and MuRIL for 3 epochs each, about 40 minutes on CPU and much less on a GPU), run:

```bash
python -m fakenews run-all --full
```

Then open the research UI in your browser:

```bash
python -m fakenews app
```

## 3. All terminal commands

Every command has `--help`. Any config value can be overridden with `--set section.key=value`.

| Command | What it does |
|---|---|
| `make-data` | Synthetic dataset (`--rows`, `--seed`, `--style-correlation`, `--label-noise`, `--language-mix`) |
| `prepare` | Load → clean → remove duplicates → split **once** → data card (`--source synthetic\|liar\|csv`, `--path`, `--text-col`, `--bert-text-col`, `--label-col`, `--fake-value`, `--split random\|time\|group`, `--date-col`, `--source-col`, `--language-col`, `--name`) |
| `train-classical` | BoW+NB, BoW+LR, TF-IDF+LR, TF-IDF+SVM, TF-IDF-char+LR (`--name`, `--models`) |
| `train-bert` | Fine-tune or embed transformers (`--models distilbert,muril,...`, `--mode finetune\|embed`, `--epochs`, `--max-length`, `--batch-size`, `--train-subset`, `--device`, `--seeds`) |
| `compare` | Leaderboard, significance, breakdowns, disagreements, figures, Markdown report |
| `audit` | Shortcut probes (style, first words, shuffled words, source, metadata, held-out source/template/time) |
| `style-sweep` | Experiment: style shortcut vs. style correlation |
| `hypotheses` | Tests H1-H14 and writes `reports/hypotheses.md` |
| `predict` | `--text "..."` or `--file x.csv`; prints every model's P(fake) side by side |
| `run-all` | The one-command demo (`--full`, `--skip-bert`, `--rows`) |
| `app` | Starts the Streamlit UI (`--port`) |

### Typical sessions

**LIAR benchmark** (downloads automatically):

```bash
python -m fakenews prepare --source liar
```

```bash
python -m fakenews train-classical --name liar
```

```bash
python -m fakenews train-bert --name liar --models distilbert,bert,roberta
```

```bash
python -m fakenews compare --name liar
```

```bash
python -m fakenews audit --name liar
```

```bash
python -m fakenews hypotheses --name liar
```

## 4. Using your own dataset

Put the file (CSV in UTF-8, Excel or Parquet) in `data/raw/`. Files there are **never committed**. Then:

```bash
python -m fakenews prepare --source csv --path data/raw/my_news.csv --name mynews --text-col clean_text --bert-text-col text --label-col label --fake-value 1 --date-col date --source-col channel --split time
```

- `--fake-value` is **required for numeric labels**. Numeric labels are never guessed, because datasets disagree
  (in WELFake, 0 means fake). Text labels such as fake/false/hoax/pants-fire and real/true/genuine/mostly-true are
  mapped automatically.
- **Two text columns rule.** BoW/TF-IDF can use your fully cleaned text. BERT models need lightly cleaned text that
  keeps stopwords ("not"!), punctuation, emojis and case, so pass that column with `--bert-text-col`.
- Splits: `time` (most honest) if you have dates, `group` (no source in two splits) if you have sources, `random`
  otherwise.

Then run `train-classical`, `train-bert`, `compare`, `audit` and `hypotheses` with `--name mynews`. For big data, use
`--train-subset 5000 --mode embed` on the laptop first to pick 3-4 models, then fine-tune those on Colab or Kaggle
and copy the `outputs/mynews/` folder back.

**Or from the browser:** the UI page **Use your own dataset** uploads the file, lets you pick the columns, and runs
prepare → statistical models → compare → audit → hypotheses for you. It then shows the terminal commands for the
transformers.

## 5. Transparency: what an evaluator can inspect

| Where | What you see |
|---|---|
| UI → **Model internals** | For any model: its pipeline, exact settings, the validation tuning table (which C/alpha was picked and why), training curves per epoch, learned top weights, validation vs. test metrics, a **threshold explorer** (confusion matrix at any threshold), and a **step-by-step trace of one prediction**: raw text → cleaned text → tokens → feature values × weights → score → sigmoid → P(fake). For transformers: sub-word tokens, token ids, P(fake) and leave-one-word-out importance. |
| UI → **Live checker** | Every model's verdict on any text, plus colour-highlighted words showing what pushed towards fake or real |
| UI → **Data explorer** | Data card (duplicates removed, split, fingerprint), the text each model family actually reads, and `noisy_label` showing which synthetic labels were flipped on purpose |
| `outputs/<name>/results/<model>.json` | All metrics, settings, timings, library versions and the full config |
| `outputs/<name>/results/predictions.csv` | Every test item with every model's P(fake) |
| `outputs/<name>/results/disagreements.csv` | Items where the best statistical model and the best transformer disagree |
| `outputs/<name>/logs/run_log.jsonl` | Every command that produced the results, with times |
| `outputs/<name>/reports/results_report.md` | Auto-written report with all tables and figures, ready for the paper |

### Evaluation output

Each run produces a macro-F1 bar chart with 95% CIs, **confusion matrices** (a grid plus one per model), ROC and PR
curves, a calibration chart, an efficiency chart (macro-F1 vs. ms per item, bubble = parameters), breakdowns per
language, style, length and hard-case type, training curves, top-feature charts, a Shortcut Index chart and the
style-sweep chart. All are saved as PNG under `outputs/<name>/figures/`, and every chart in the UI has a download
button.

## 6. Hypotheses: proved, disproved or inconclusive

`python -m fakenews hypotheses` tests 14 hypotheses written in advance, for example *"transformers beat TF-IDF"*,
*"character n-grams help on Hinglish"*, *"Indian-language models help on Hindi/Hinglish"*, *"writing style is a
shortcut"*, *"calm-sounding fakes are harder"* and *"debunk reports fall into a keyword trap"*. Each one gets
SUPPORTED, REJECTED, INCONCLUSIVE or NOT TESTABLE, based on McNemar, paired bootstrap, Fisher or Spearman tests.
Models are always chosen on validation scores. See [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md).

## 7. Synthetic data, honestly

The generator ([`fakenews/synthetic.py`](fakenews/synthetic.py)) writes templated news in English (60%), Hinglish
(25%) and Hindi (15%) across 8 topics. It includes hard cases: real reports about debunks, surprising-but-true news,
and fakes written in a calm official tone. It has a style knob (`--style-correlation`; 0.5 removes the style
clue), 3% label noise and 20 synthetic channels, 4 of which post mostly fake items. Each row records its
`template_id`, `case_type` and whether its label was flipped.

**Synthetic scores show that the pipeline works; they are not model performance.** The models learn the templates
(see the template-held-out probe and H12). Real claims come from your dataset and from LIAR.

## 8. Repository layout

```text
configs/config.yaml     every setting (seed, splits, models, hyper-parameters, UI project info)
fakenews/               the package: synthetic, data, preprocess, classical, transformer_models, evaluate,
                        compare, figures, audit, hypotheses, explain, predict, cli
app/                    Streamlit UI: Home.py + views/ (one file per page)
docs/                   CONCEPTS.md, HYPOTHESES.md
tests/                  pytest checks (RUN_SLOW=1 also runs a DistilBERT test)
data/raw/               your datasets (git-ignored)
outputs/<name>/         models, results, figures, reports, logs (git-ignored)
```

## 9. Tests

```bash
python -m pytest -q
```

To also run the DistilBERT fine-tuning test:

```bash
RUN_SLOW=1 python -m pytest -q
```

## 10. Notes and troubleshooting

- **IndicBERT** (`ai4bharat/indic-bert`) is a gated model. Accept its terms on Hugging Face, then run
  `hf auth login` once.
- MPS errors on a Mac: set `PYTORCH_ENABLE_MPS_FALLBACK=1` (the code does this by default) or use `--device cpu`.
- Out of memory: use `--batch-size 8` or `--max-length 128`, or `--train-subset`.
- Hindi shows as boxes in Excel: the CSVs are written with a UTF-8 BOM. Otherwise open them in Google Sheets.
- For more, see `BUILD_INSTRUCTIONS.md` §19.

## Licences

The code is MIT licensed (see `LICENSE`). LIAR (Wang, 2017) is for research use. Pretrained models are under the
licences on their Hugging Face model cards. Never commit data, model weights or `.env` files.
