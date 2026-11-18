# SatyaCheck-Text: Build Instructions

**Fake vs. real news detection with Bag-of-Words, TF-IDF and BERT-family models, run from the terminal, with a research UI**

Minor project: SatyaCheck / HinCheck (text module)
Document type: build specification. It contains instructions only, no code.

---

## Contents

0. How to use this document
1. What we are building
2. Ground rules
3. Tech stack
4. Repository layout
5. Phase 0: Environment setup
6. Phase 1: Synthetic dataset
7. Phase 2: Loading data and making splits
8. Phase 3: Bag-of-Words and TF-IDF models
9. Phase 4: BERT-family models
10. Phase 5: Evaluation and the output section
11. Phase 6: Terminal commands
12. Phase 7: Research UI
13. Phase 8: Plugging in your big dataset
14. Phase 9: Shortcut audit (research add-on)
15. Phase 10: Testing, README and GitHub
16. Conceptual clarity: how the approaches differ
17. Presenting the results
18. Suggestions from Claude for a better fake-news model
19. Troubleshooting
20. Final checklist

---

## 0. How to use this document

- This file is a **specification**. It says what to build, in what order, and how to check that each part works. You can build it yourself, or put this file in the root of your GitHub repository and ask a coding assistant to implement it **one phase at a time**.
- Each phase ends with **acceptance checks**. Do not start the next phase until the current phase passes them.
- One rule overrides everything else: **no number is ever typed by hand into a table, slide, report or the UI.** Every number must be read from a result file that the code produced. This keeps the project honest and makes it reproducible.

---

## 1. What we are building

A small research software project that:

1. generates a **synthetic fake/real news dataset** for development and demos;
2. trains **statistical models**: Bag-of-Words (BoW) and TF-IDF, each with standard classifiers;
3. trains **several BERT-family models** from Hugging Face (BERT, DistilBERT, RoBERTa, ALBERT, multilingual BERT, XLM-R, MuRIL, IndicBERT, HingBERT);
4. compares all of them **on the same test set** and shows the differences in a clear output section;
5. runs **fully from your terminal** on your laptop;
6. has a **structured research UI** for your minor-project presentation;
7. accepts **your own big, cleaned dataset** without changing any code;
8. includes **concept notes** on how BoW, TF-IDF and BERT differ;
9. ends with **suggestions** for building a better fake-news detector.

### Requirement map

| # | Your requirement | Where it is covered |
|---|---|---|
| 1 | TF-IDF model on a synthetic dataset | Phases 1 and 3 |
| 2 | Different BERT variants | Phase 4 |
| 3 | Output section showing TF-IDF vs. BERT variants (incl. "Ambert") | Phase 5, and Phase 7 (Model comparison page) |
| 4 | Run locally in the terminal | Phases 0 and 6 |
| 5 | Structured UI for research presentation | Phase 7 |
| 6 | TF-IDF **and** BoW on synthetic data | Phase 3 |
| 7 | Plug in your big cleaned dataset | Phase 8 |
| 8 | Points on the differences and conceptual clarity | Section 16 |
| 9 | Claude's suggestions | Section 18 |

**About "Ambert":** this project interprets it as both **mBERT** (multilingual BERT) and **ALBERT** (A Lite BERT), and includes both. ("AMBERT" is also the name of a 2020 research model from ByteDance, but it is not a standard Hugging Face checkpoint, so it is left out.)

---

## 2. Ground rules

These rules apply to every phase.

1. **One split for all models.** The dataset is split into train, validation and test **once**, the split is saved to disk, and every model reads the same files. A comparison is only fair if all models see exactly the same test items.
2. **Labels are fixed:** 1 = fake, 0 = real, in every file, model and chart.
3. **Each model family gets the text it needs.**
   - BoW and TF-IDF use the **cleaned** text.
   - BERT-family models use **lightly cleaned, close-to-original** text. Removing stopwords, stemming or stripping punctuation hurts BERT, because BERT reads grammar and context.
4. **Synthetic data is for building and demonstrating the pipeline, not for research claims.** Results on synthetic data are not reported as model performance anywhere. Real claims come from your own dataset and from LIAR.
5. **Reproducibility.**
   - Use one random seed in the config.
   - Save the full configuration next to every result.
   - Record the library versions used.
6. **Nothing is hard-coded to one dataset.** Every step works on any table that has a text column and a label column.
7. **Laptop first, GPU optional.** Everything must run on a MacBook. Larger runs can move to Google Colab or Kaggle without code changes.
8. **Fail loudly.** If a column is missing, a label value is unknown or a split is empty, stop with a clear message that says how to fix it.

---

## 3. Tech stack

| Tool | Used for | Why this choice |
|---|---|---|
| Python 3.11 or 3.12 | Everything | Best wheel support for PyTorch, tokenizers and sentencepiece. Avoid 3.13 or 3.14 for this project. |
| pandas, NumPy | Loading and handling tables | Standard |
| scikit-learn | BoW, TF-IDF, Logistic Regression, Naive Bayes, Linear SVM, metrics | The standard library for statistical NLP baselines |
| PyTorch | Training BERT-family models | Supports Apple Silicon GPUs (MPS) as well as NVIDIA (CUDA) |
| Hugging Face Transformers | Downloading and fine-tuning BERT variants | One interface for every model listed in Phase 4 |
| sentencepiece, protobuf | Tokenizers for ALBERT, XLM-R and IndicBERT | Required by those models |
| matplotlib | Saved figures for the report and slides | Static, publication-quality images |
| Streamlit | Research UI | Builds a clean multi-page web app from Python; runs locally in the browser |
| joblib | Saving trained statistical models | Fast and simple |
| PyYAML | One configuration file | All settings in one place |
| SciPy | Significance test (McNemar) | Already installed with scikit-learn |
| tqdm | Progress bars in the terminal | Readable training output |

**Optional:** Plotly (interactive charts in the UI) and pytest (tests).

---

## 4. Repository layout

```text
satyacheck-text/
├── README.md                   short project overview + how to run (written in Phase 10)
├── BUILD_INSTRUCTIONS.md       this file
├── requirements.txt            pinned library versions
├── .gitignore                  ignores data/, outputs/, virtual env, model files
├── configs/
│   └── config.yaml             every setting: seed, split sizes, model list, hyperparameters
├── data/
│   ├── raw/                    YOUR big dataset goes here (never committed to GitHub)
│   ├── synthetic/              generated synthetic dataset
│   └── processed/<name>/       saved train / val / test files + data card for each dataset
├── fakenews/                   the Python package
│   ├── cli                     terminal commands (Phase 6)
│   ├── config                  reads config.yaml, allows command-line overrides
│   ├── synthetic               synthetic data generator (Phase 1)
│   ├── data                    loaders for synthetic / LIAR / your CSV, label mapping, splitting (Phase 2)
│   ├── preprocess              the two text views: "classical" and "bert" (Phases 2–4)
│   ├── classical               BoW and TF-IDF models (Phase 3)
│   ├── transformer_models      BERT-family fine-tuning and embedding mode (Phase 4)
│   ├── evaluate                metrics, confidence intervals, calibration, figures (Phase 5)
│   ├── compare                 leaderboard, significance tests, report writer (Phase 5)
│   ├── explain                 word-level explanations for the UI (Phase 7)
│   ├── audit                   shortcut probes (Phase 9)
│   └── utils                   seeding, device choice, timing, file helpers
├── app/                        Streamlit UI (Phase 7): one entry page + one file per page
├── outputs/<name>/             everything produced for a dataset
│   ├── models/                 saved statistical and transformer models
│   ├── results/                one result file per model + summary table + predictions
│   ├── figures/                all charts as PNG
│   └── reports/                auto-written results report (Markdown)
├── docs/
│   └── CONCEPTS.md             Section 16 of this file, for readers of the repo
└── tests/                      small automatic checks (Phase 10)
```

---

## 5. Phase 0: Environment setup

**Goal:** a clean, isolated Python environment on your Mac that can run everything.

1. Install a **separate Python 3.11 or 3.12 environment** for this project, either a conda environment or a virtual environment. Your current Anaconda Python is 3.14, which is too new for some ML packages, so keep it separate.
2. Create the repository folder with the layout from Section 4.
3. Write `requirements.txt` with the libraries from Section 3. Pin each version once everything works.
4. Install the requirements inside the environment.
5. Check the hardware:
   - On an Apple Silicon Mac, confirm that PyTorch reports the **MPS** device as available.
   - On Colab or Kaggle, confirm **CUDA** instead.
   - If neither is available, the code must fall back to CPU automatically.
6. Write `.gitignore` before the first commit. It must ignore:
   - `data/raw/`, `data/processed/`, `outputs/`
   - the virtual environment
   - model weight files
   - any `.env` file

**Acceptance checks**
- [ ] The environment activates and every library imports without errors.
- [ ] The device check prints mps, cuda or cpu.
- [ ] `git status` shows no data or model files.

---

## 6. Phase 1: Synthetic dataset

**Goal:** a realistic-looking but fully artificial dataset for building, testing and demonstrating the pipeline. It must be **hard enough that a model can't reach 100% without learning content**, and it must include Hindi and Hinglish.

### 6.1 Size and columns

- Default **3,000 rows**, roughly half fake and half real. The size can be changed from the terminal.
- Columns:
  - **id:** unique row number.
  - **text:** the news item.
  - **label:** "fake" or "real".
  - **label_id:** 1 or 0.
  - **language:** "en" for English, "hinglish" for Hindi written in Roman letters, "hi" for Hindi in Devanagari.
  - **topic:** one of the topics below.
  - **style:** "neutral" or "sensational".
  - **source:** a synthetic channel name, channel_01 to channel_20.
  - **date:** a random date between 2025-01-01 and 2026-10-31.

### 6.2 Topics (8)

Health, government schemes, economy and currency, technology and telecom, weather and disasters, education and exams, elections, transport.

### 6.3 How real items look

Routine, plausible news written like a report and attributed to an **institution**, never to a named person. Examples of institutions: the weather department, the central bank, the space agency, the railways, a state government, a district administration.

The real class must also include two kinds of **hard cases**:
- **"surprising but true"** items, such as a new commemorative coin;
- **real reports about debunks**, such as "The Health Ministry clarified that a viral message is false". These contain words like "viral" and "false" but are real news.

### 6.4 How fake items look

They follow common Indian misinformation patterns:
- **Miracle cures:** "X cures Y in N days, doctors are hiding this".
- **Fake giveaway schemes:** "Government is giving free laptops, register at this link".
- **Currency-ban rumours.**
- **Technology–health conspiracies:** "5G towers spread disease".
- **Fake disaster warnings** attributed to foreign agencies.
- **Chain-message threats:** "WhatsApp will charge money, forward this".
- **Fake exam cancellations.**
- **Old video presented as new.**
- **OTP and SIM scams.**

The fake class must also include **hard cases written in a calm, official tone**, so that a sensational style is not the only clue.

### 6.5 Example items (to guide the generator's templates)

| Label | Language | Example |
|---|---|---|
| Real | English | Schools in Patna will remain closed on Monday due to heavy rain, the district administration said on its official website. |
| Fake | English | Garlic water cures dengue in just 3 days, doctors are hiding this from you! |
| Real (hard) | English | The Health Ministry clarified that a viral message claiming hot water kills the virus is false. |
| Fake (hard) | English | According to an official notice, board exams for classes 10 and 12 have been cancelled this year. |
| Real | Hinglish | Railway tyohar ke season mein 40 extra trains chalayega, official release mein bataya gaya. |
| Fake | Hinglish | Sarkar sabhi students ko free laptop de rahi hai, is link par jaldi register karo! |
| Real | Hindi | मौसम विभाग के अनुसार इस साल मानसून सामान्य रहने की संभावना है। |
| Fake | Hindi | कल से 500 के नोट बंद हो जाएंगे, यह संदेश सभी को भेजें। |

### 6.6 How the generator works

1. **Templates with slots.** Write about 15 real and 15 fake templates per language. Each has slots such as city, state, institution, disease, remedy, item, group of people, amount, number and day. Fill the slots at random from lists; Hindi has its own lists in Devanagari.
2. **Language mix.** Default 60% English, 25% Hinglish, 15% Hindi. Configurable.
3. **Style knob ("style correlation").**
   - A sensational version adds things like "BREAKING:", "🚨", "!!!", "Forward to all groups", "Sabko bhejo" and a random word in capitals.
   - The probability of the sensational style is *s* for fake items and *1 − s* for real items.
   - The default *s* = 0.7 makes style a useful but imperfect clue. *s* = 0.5 removes the style clue entirely. This lets you demonstrate shortcut learning (Phase 9).
4. **Label noise.** Flip about 3% of labels at random (configurable), because real datasets are never perfectly labelled.
5. **Source metadata.** Spread items over 20 synthetic channels so that a few channels are mostly fake. This lets the audit in Phase 9 show a "source shortcut".
6. **No duplicates.** Remove exact duplicate texts, and keep generating until the requested size is reached.
7. **Determinism.** The same seed must give the same dataset.
8. **Save** the result as a CSV in `data/synthetic/`, with a note at the top of the data card saying that the data is synthetic.

**Ethics:**
- Never use names of real people.
- Mark every synthetic file clearly as synthetic.
- Never publish the synthetic items as if they were real posts.

**Acceptance checks**
- [ ] Class balance is between 45% and 55% fake.
- [ ] All three languages appear.
- [ ] Zero exact duplicates.
- [ ] Running twice with the same seed gives identical files.
- [ ] You have read 20 random rows yourself, and they look sensible in all three languages.

---

## 7. Phase 2: Loading data and making splits

**Goal:** any dataset becomes the same standard set of files, so every model works with it unchanged.

### 7.1 Three supported sources

1. **Synthetic:** the file from Phase 1.
2. **LIAR:** the public benchmark of short political statements (about 12,800 statements from PolitiFact).
   - Download the three official files (train, validation, test) from the LIAR GitHub mirror.
   - **Binary mapping:** "true" and "mostly-true" become real; "false" and "pants-fire" become fake. Drop "half-true" and "barely-true", because they are ambiguous.
   - Keep LIAR's official test split as the test set.
   - Keep the speaker, party and context columns as metadata for the audit, not as model input.
   - Note in the data card that LIAR is US political English, so it tests general ability, not Indian content.
3. **Your own dataset:** CSV, Excel or Parquet (details in Phase 8).

### 7.2 Label handling

- Text labels are mapped automatically:
  - "fake", "false", "hoax", "misleading", "pants-fire" → fake.
  - "real", "true", "genuine", "mostly-true" → real.
- **Numeric labels are never guessed.** The user must state which value means fake, because public datasets disagree (in some, 1 means fake; in WELFake, 0 means fake).
- Any label value that isn't mapped stops the program with a list of the unknown values.

### 7.3 Cleaning at load time (minimal)

- Trim spaces and drop empty texts.
- **Remove exact duplicates before splitting.** Otherwise the same item can land in both train and test, which inflates scores. Report how many were removed.
- Optional but recommended: report **near-duplicates**, such as the same message with a different emoji or link.

### 7.4 Split strategies

| Strategy | When to use | What it does |
|---|---|---|
| Random, stratified (default) | Synthetic data; datasets without dates | 70% train, 15% validation, 15% test, keeping the fake/real ratio equal in each part |
| Time-based | Your dataset has a date column | Train on the oldest items and test on the newest. This is the most honest, because real systems face new rumours. |
| Group-based | Your dataset has a source or channel column | No source appears in more than one split. This tests whether the model learned content or just "who posted it". |

### 7.5 What gets saved

For each dataset name, save to `data/processed/<name>/`:
- the three split files;
- a **data card** with:
  - number of items per split and label;
  - language mix;
  - average text length per label;
  - duplicates removed;
  - split strategy and seed;
  - date range, if dates exist;
  - a fingerprint (hash) of the original file, so you can always tell which version of your data was used.

**Acceptance checks**
- [ ] Synthetic and LIAR both produce the three split files and a data card.
- [ ] No text appears in more than one split.
- [ ] Re-running "prepare" with the same settings produces identical splits.
- [ ] A CSV with a numeric label and no stated fake value stops with a clear message.

---

## 8. Phase 3: Bag-of-Words and TF-IDF models

**Goal:** fast, interpretable statistical baselines. Every BERT model must be compared against these.

### 8.1 Text view for statistical models

- Lowercase the text.
- Replace links with a single placeholder word.
- Keep Hindi characters.
- If your own dataset already has a cleaned text column, use it as it is and don't clean it again.

**Important for Hindi:** the default word pattern in scikit-learn splits Hindi words at vowel signs (matras), so a word like "बारिश" falls apart. The tokenizer must be configured to treat the whole Devanagari Unicode block (U+0900–U+097F) as word characters. Check this with one Hindi sentence before training.

### 8.2 Feature extractors

| Name | What it counts | Settings |
|---|---|---|
| **BoW** (word counts) | How many times each word or word pair appears | Single words and word pairs (1–2 grams); ignore words seen in fewer than 2 documents; ignore words in more than 95% of documents; at most 50,000 features |
| **TF-IDF** (words) | Word counts re-weighted so that words common in every document count less | Same as BoW, plus sublinear term frequency and normalised vectors |
| **TF-IDF** (characters) | Overlapping 3–5 character pieces | Handles Hinglish spelling variation ("nahi", "nahin", "nai") and typos. Recommended extra baseline. |

### 8.3 Classifiers

| Model name | Features | Classifier | Notes |
|---|---|---|---|
| BoW + Naive Bayes | BoW | Multinomial Naive Bayes | The classic text baseline |
| BoW + Logistic Regression | BoW | Logistic Regression with balanced class weights | Interpretable |
| TF-IDF + Logistic Regression | TF-IDF words | Logistic Regression | Usually the strongest simple baseline |
| TF-IDF + Linear SVM | TF-IDF words | Linear SVM, calibrated so it outputs probabilities | Strong for short texts |
| TF-IDF-char + Logistic Regression | TF-IDF characters | Logistic Regression | Best for Hinglish |

### 8.4 Training procedure

1. Fit the feature extractor on the **training split only**. Fitting it on all data leaks test-set vocabulary.
2. Tune one setting per classifier on the **validation split**:
   - regularisation strength C ∈ {0.25, 1, 4};
   - Naive Bayes smoothing alpha ∈ {0.1, 0.5, 1}.

   Choose by validation macro-F1.
3. Evaluate once on the **test split**.
4. Save:
   - the trained pipeline;
   - vocabulary size;
   - the **25 words or phrases that push hardest towards fake and towards real**;
   - training time;
   - inference time per item;
   - test-set probabilities for every item.

**Acceptance checks**
- [ ] All five models train on synthetic data in under a minute on a laptop.
- [ ] The top fake-indicating words look like misinformation patterns ("cures", "free", "forward", "register") and the top real ones like reporting language ("said", "official", "announced").
- [ ] On synthetic data with style correlation 0.7, punctuation and emoji words appear in the fake list. That's the style shortcut, and it's expected.
- [ ] Result files exist for all five models.

---

## 9. Phase 4: BERT-family models

**Goal:** fine-tune several pretrained transformer models from Hugging Face on the same splits, and compare them with the statistical baselines.

### 9.1 Model registry

All names below are Hugging Face model IDs. Parameter counts are approximate; confirm them on each model card.

| Short name | Hugging Face ID | Size (approx.) | Languages | Why include it |
|---|---|---|---|---|
| bert | bert-base-uncased | 110M | English | The original reference model |
| distilbert | distilbert-base-uncased | 66M | English | About 40% smaller and about 60% faster than BERT; good laptop default |
| roberta | roberta-base | 125M | English | BERT trained longer on more data; often stronger on English |
| albert | albert-base-v2 | 12M | English | Shares parameters across layers, so it is very small (one reading of "Ambert") |
| mbert | bert-base-multilingual-cased | 178M | 104 languages incl. Hindi | Multilingual baseline (the other reading of "Ambert") |
| xlmr | xlm-roberta-base | 270M | 100 languages | Strong multilingual model |
| muril | google/muril-base-cased | 237M | 16 Indian languages + English, plus their Romanised forms | Built by Google for Indian languages, including transliterated text; **the key model for this project** |
| indicbert | ai4bharat/indic-bert | small (ALBERT-based) | 11 Indian languages + English | Lightweight Indian-language model from AI4Bharat |
| hingbert | l3cube-pune/hing-bert | 110M | Hinglish (Roman script) | Trained on code-mixed Hindi–English; useful when most of your data is Hinglish |

**Default set for a laptop:** distilbert, albert, mbert, muril. **Full set for the paper (on a GPU):** all nine.

### 9.2 Two ways to use each model

1. **Fine-tuning (main mode).**
   - Add a 2-class output layer on top of the pretrained model.
   - Train the whole network on your training split.
   - This gives the best accuracy.
2. **Frozen embeddings (fast mode).**
   - Do not train the network.
   - Pass each text through the pretrained model and average the token vectors into one sentence vector.
   - Train a Logistic Regression on those vectors.
   - This runs quickly on a laptop, and it shows how much the pretrained model "knows" before any fine-tuning.

   Reporting both modes is a nice research point.

### 9.3 Fine-tuning settings (defaults in config.yaml)

| Setting | Default | Notes |
|---|---|---|
| Maximum length | 128 tokens | Use 256 for long articles. If an article is longer than that, an option keeps its first and last parts ("head + tail"). |
| Batch size | 16 | Use 8 if memory runs out |
| Learning rate | 2e-5 | 3e-5 for small models (DistilBERT, ALBERT) |
| Epochs | 3 | With early stopping |
| Warm-up | 10% of steps, then linear decay | Standard |
| Weight decay | 0.01 | Standard |
| Gradient clipping | 1.0 | Stabilises training |
| Early stopping | Stop when validation macro-F1 has not improved for 1 epoch; keep the best epoch | Prevents over-fitting |
| Class weighting | Turn on automatically if one class is more than 60% of the data | Handles imbalance |
| Seeds | 1 for development, 3 for final paper numbers | Report the mean and spread over seeds |
| Device | Automatic: CUDA, then Apple MPS, then CPU | Can be forced from the terminal |
| Training subset | Optional limit on training rows | Lets you test big datasets quickly on a laptop |

### 9.4 Tokenizer notes (check each one once)

- **Uncased** models (bert, distilbert, albert) lowercase the text. **Cased** models (mbert, muril) keep capitals. Don't lowercase text yourself before cased models.
- **IndicBERT** uses an ALBERT-style tokenizer that can strip accent marks. For Devanagari, accents must be kept, otherwise vowel signs disappear.
- In the UI's Concepts lab (Phase 7), show how each tokenizer splits the same Hinglish and Hindi sentence. English-only tokenizers break Hindi into many tiny or unknown pieces. That's the visual proof of why MuRIL and IndicBERT exist.

### 9.5 What to save per model

- the best model weights, tokenizer and configuration;
- the training curve: loss and validation macro-F1 per epoch;
- the parameter count;
- training time;
- inference time per item;
- the device used;
- test-set probabilities for every item;
- the exact settings used.

**Acceptance checks**
- [ ] distilbert and muril fine-tune on synthetic data on your Mac in minutes, not hours.
- [ ] The frozen-embedding mode works for at least two models.
- [ ] Re-running with the same seed gives nearly the same scores. Small differences on MPS or GPU are normal.
- [ ] If a model download fails, the error message names the model and suggests checking the internet connection or the model ID.

---

## 10. Phase 5: Evaluation and the output section

**Goal:** one clear, fair, statistically sound comparison of all models. This is the **output section** that shows the differences between TF-IDF, BoW and the BERT variants.

### 10.1 Metrics (computed for every model on the test split)

| Metric | Meaning in plain words |
|---|---|
| Accuracy | Share of items labelled correctly. Misleading when classes are unbalanced. |
| Precision (fake) | Of the items the model called fake, how many really were fake |
| Recall (fake) | Of the truly fake items, how many the model caught |
| F1 (fake) | Balance of precision and recall for the fake class |
| **Macro-F1 (headline metric)** | Average of the fake-class F1 and the real-class F1. Treats both classes equally. |
| ROC-AUC | How well the model ranks fake above real across all thresholds |
| PR-AUC | Like ROC-AUC but focused on the fake class; better when fakes are rare |
| Brier score | How close the probabilities are to the truth (lower is better) |
| ECE (calibration error) | Whether "90% sure" really means right 90% of the time (lower is better) |
| 95% confidence interval for macro-F1 | From 1,000 bootstrap resamples of the test set; shows how much the score could vary |
| McNemar test | Whether the best model is *significantly* better than each other model, or just luckier on this test set |
| Training time, inference time per item, parameter count, disk size | The cost side of the comparison |

### 10.2 Breakdowns (where the differences become visible)

- **By language:** English vs. Hinglish vs. Hindi. This is where MuRIL, IndicBERT and HingBERT should differ from English-only BERT.
- **By text length:** short (under 15 words), medium, long.
- **By style:** neutral vs. sensational. Does the model still catch calm-sounding fakes?
- **Statistical vs. transformer:** the best BoW/TF-IDF model vs. the best BERT-family model, with the macro-F1 difference and its significance.

### 10.3 Terminal output

After a comparison run, the terminal prints:

1. A **leaderboard table** sorted by macro-F1. Columns:
   - model, family (Statistical or Transformer), mode (Fine-tuned or Embeddings);
   - macro-F1 with its 95% interval;
   - accuracy, ROC-AUC, ECE;
   - training time, milliseconds per item, parameters.
2. A **one-paragraph summary** written from the numbers, for example: "Best model: X (macro-F1 …). Best statistical model: Y (…). The difference is / is not significant (McNemar p = …)." These sentences are templates filled automatically from the result files.
3. The paths of every saved file.

**Layout of the leaderboard.** Cells are left empty here on purpose; they are filled by your runs.

| Rank | Model | Family | Mode | Macro-F1 (95% CI) | Accuracy | ROC-AUC | ECE | Train time | ms / item | Params |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | | |
| 2 | | | | | | | | | | |
| … | | | | | | | | | | |

### 10.4 Saved files (in `outputs/<name>/`)

| File | Content |
|---|---|
| results/one file per model | All metrics, settings, timings, versions |
| results/summary table | The leaderboard as a CSV (opens in Excel) |
| results/predictions table | One row per test item: text, true label, and every model's probability of fake |
| results/disagreements table | Items where the best statistical model and the best transformer disagree. **This is the most useful file for understanding the difference between the two approaches.** |
| results/errors table | Each model's most confident mistakes |
| results/significance table | McNemar p-values: best model vs. every other |
| figures/macro-F1 bar chart | All models, with confidence-interval error bars, coloured by family |
| figures/ROC curves | All models on one chart |
| figures/precision–recall curves | All models on one chart |
| figures/confusion matrices | A grid with one small matrix per model |
| figures/calibration chart | Predicted probability vs. actual fake rate |
| figures/efficiency chart | Macro-F1 vs. milliseconds per item; bubble size = number of parameters |
| figures/per-language chart | Grouped bars of macro-F1 for English / Hinglish / Hindi per model |
| figures/training curves | Validation macro-F1 per epoch for each transformer |
| reports/results report | An auto-generated Markdown report containing all tables and figures, ready to copy into your paper or slides |

**Acceptance checks**
- [ ] Every number in the report matches the result files.
- [ ] Figures have titles, axis labels and a legend, and are readable when pasted into a slide.
- [ ] The disagreements table has examples in both directions (TF-IDF right and BERT wrong, and the reverse).

---

## 11. Phase 6: Terminal commands

**Goal:** the whole project runs from your terminal with a few memorable commands. All commands are subcommands of one entry point (the package run as a module). Each prints what it is doing and where it saved the results.

| Command | What it does | Main options |
|---|---|---|
| make-data | Generates the synthetic dataset (Phase 1) | number of rows, seed, style correlation, label noise, language mix, output path |
| prepare | Loads a dataset and creates the saved splits and data card (Phase 2) | source (synthetic / liar / csv), file path, text column, optional BERT-text column, label column, which value means fake, split strategy, dataset name |
| train-classical | Trains the five BoW / TF-IDF models (Phase 3) | dataset name, which models |
| train-bert | Fine-tunes or embeds the chosen transformer models (Phase 4) | dataset name, list of short model names, mode (finetune / embed), epochs, max length, batch size, training subset, device, number of seeds |
| compare | Builds the leaderboard, breakdowns, significance tests, figures and report (Phase 5) | dataset name |
| predict | Classifies one text or a file of texts with every saved model and prints a side-by-side table | dataset name, text or file path, which models |
| audit | Runs the shortcut probes (Phase 9) | dataset name |
| run-all | One-command demo: make synthetic data, prepare, train statistical models, train a small transformer set for 1 epoch, compare | quick or full |
| app | Starts the Streamlit UI in your browser | port |

**Typical sessions:**
- **First demo:** run-all, then app.
- **LIAR:** prepare with source liar → train-classical → train-bert with distilbert, bert, roberta → compare → app.
- **Your dataset:** prepare with source csv (Phase 8) → train-classical → train-bert in embeddings mode for a quick look → train-bert fine-tune on your best 3–4 models → compare → audit → app.

**Acceptance checks**
- [ ] run-all finishes on a laptop within about 15–20 minutes and leaves a complete output folder.
- [ ] Every command has a help message listing its options.
- [ ] Wrong inputs (unknown model name, missing dataset) give a clear error, not a long traceback.

---

## 12. Phase 7: Research UI

**Goal:** a clean, structured web app for your minor-project presentation that reads everything from the output folders. It never trains heavy models itself; training happens in the terminal.

### 12.1 General design

- Streamlit, wide layout, one page per topic, navigation in the sidebar.
- The sidebar holds:
  - a **dataset selector**, listing every dataset that has outputs;
  - a short disclaimer: "Model predictions are not verified facts. Always check a fact-checker."
- Consistent colours everywhere: **fake = red, real = green**. Statistical models and transformer models each get their own colour family.
- Every chart has a title, axis labels and a one-line caption explaining how to read it.
- Each chart and table has a **download button** (PNG or CSV) so you can drop it straight into slides.
- **Graceful empty states:** if only the statistical models are trained, transformer sections say "Not trained yet — run train-bert" instead of crashing.
- Load models and results once and cache them, so pages open quickly.

### 12.2 Pages

**1. Home**
- Project title, team, guide, and a one-paragraph problem statement.
- A **pipeline diagram:** data → split → (BoW / TF-IDF models) and (BERT-family models) → evaluation → comparison.
- Three headline cards, read from the results: best model overall, best statistical model, best transformer, each with macro-F1 and its 95% interval.
- A clear note on which dataset is selected, and a warning banner when it is the synthetic one.

**2. Data explorer**
- The data card: sizes, split strategy, duplicates removed, date range.
- Charts:
  - class balance per split;
  - language mix;
  - text length distribution by label;
  - top words per class (from BoW).
- A filterable table of sample rows by label, language and topic.

**3. Model comparison (the main research page)**
- The leaderboard with the best value in each column highlighted.
- The macro-F1 bar chart with confidence intervals.
- Tabs for: ROC and PR curves, confusion matrices, calibration, efficiency, per-language results, training curves.
- The significance table (McNemar).
- An auto-written **"Statistical vs. Transformer" summary box**, using the template sentences from Phase 5.
- A **disagreements browser:** read examples where TF-IDF and BERT disagree, with both probabilities shown.

**4. Live checker**
- A text box for any news text, plus example buttons: real, fake, hard real, hard fake, Hinglish, Hindi.
- A multiselect of trained models.
- One result card per model:
  - predicted label;
  - probability of fake, shown as a bar;
  - time taken.
- An agreement line, for example "5 of 7 models say fake".
- **Explanation:**
  - For BoW and TF-IDF with Logistic Regression: highlight the words in the text that pushed towards fake (red) or real (green).
  - For transformers: an optional "word importance" view that removes one word at a time and measures how the probability changes. Limit it to short texts, because it is slow.
- A visible warning: a model only judges the wording of a text; it does not check facts.

**5. Batch checker**
- Upload a CSV, choose a model and the text column, run, then view and download the results with probabilities.

**6. Concepts lab (for teaching your audience)**
- **BoW vs. TF-IDF live demo.** The user types 2–4 short sentences. The page shows the word-count matrix and the TF-IDF matrix side by side, and highlights how a word that appears in every sentence gets a low TF-IDF weight.
- **Tokenizer comparison.** One sentence (default: a Hinglish one and a Hindi one) is split by the BERT, mBERT, MuRIL and IndicBERT tokenizers. Show the pieces, the number of pieces and any unknown tokens. This visually explains why Indian-language models matter.
- Short text explanations taken from Section 16.

**7. Shortcut audit**
- The probe results from Phase 9 as a table and a bar chart of the Shortcut Index, with a plain-language explanation of each probe.

**8. About and limitations**
- Datasets used and their licences, models used with links to their model cards.
- Ethics note, limitations, how to cite, and the team.

**Acceptance checks**
- [ ] The app opens with one terminal command and works offline after models are downloaded once.
- [ ] Every page works on the synthetic dataset; switching datasets in the sidebar updates every page.
- [ ] No page shows a number that is not in the result files.
- [ ] A first-time viewer can understand the Model comparison page without your explanation.

---

## 13. Phase 8: Plugging in your big dataset

**Goal:** run the whole pipeline on your own cleaned dataset without touching any code.

### 13.1 Format

- **File:** CSV (UTF-8 encoding, so Hindi displays correctly), Excel or Parquet. Put it in `data/raw/`.
- **Required columns:**
  - a text column;
  - a label column.
- **Strongly recommended columns:**
  - **date:** enables the honest time-based split;
  - **source or channel:** enables the group split and the source audit;
  - **language or script:** enables per-language results;
  - **a second, lightly cleaned text column** for BERT (see 13.2).

### 13.2 The "two text columns" rule

You cleaned your data with many steps. Some of them help BoW and TF-IDF but **harm BERT**:

| Step | Good for BoW / TF-IDF? | Good for BERT? |
|---|---|---|
| Removing URLs, phone numbers, HTML | Yes | Yes |
| Fixing broken encoding, extra spaces | Yes | Yes |
| Lowercasing | Yes | Only for uncased models; harms cased models (mBERT, MuRIL) |
| Removing stopwords ("not", "no", "is", "the") | Sometimes | **No.** "Not" can flip the meaning of a claim. |
| Stemming / lemmatisation | Sometimes | **No.** BERT has its own sub-word tokenizer. |
| Removing punctuation and emojis | Sometimes | **No.** They carry tone and context. |
| Translating Hindi to English | Changes the task | Changes the task; keep the original as well |

So, if you can, keep **two columns**: your fully cleaned text (for BoW and TF-IDF) and a lightly cleaned version (for BERT). If you only have the fully cleaned text, the pipeline still runs, but the data card and report must say so, because it will likely lower the BERT scores.

### 13.3 Checks before training (do all of them)

1. **Label check.** Read 20 random items per class and confirm that the labels mean what you think.
2. **Duplicates.** Look at the exact and near-duplicate counts reported by "prepare". Forwarded messages repeat a lot.
3. **Leakage words.** After training TF-IDF, read its top 25 fake and real words. Treat these as warning signs:
   - publisher names, e.g. "Reuters" appearing only in real items, which is a known problem in some public datasets;
   - URLs or channel names;
   - "fact check", "false" or "misleading" stamps;
   - datelines.

   If you find any, remove them from the text and re-run.
4. **Balance.** If one class is over 60%, class weighting turns on automatically; also report macro-F1, not just accuracy.
5. **Language mix and length.** Check that they look as expected in the data card.

### 13.4 Split and compute plan

- **Split:** time-based if dates exist; otherwise group-based if sources exist; random only as the last option.
- **Compute:**
  - **Statistical models:** fine on a laptop for any size.
  - **Transformers on up to about 20,000 rows:** a laptop works with distilbert, albert and muril (expect long runs for muril).
  - **Larger datasets:** first use the training-subset option and embeddings mode on the laptop to choose your best 3–4 models. Then fine-tune those on full data on Google Colab or Kaggle (free GPUs). Copy the output folder back so the UI can show it.

**Acceptance checks**
- [ ] "prepare" runs on your file and the data card matches your own counts.
- [ ] You have read and recorded the top TF-IDF words, and removed any leakage.
- [ ] The final report states which split strategy and which text columns were used.

---

## 14. Phase 9: Shortcut audit (research add-on)

**Goal:** show whether models learned *content* or easy *shortcuts*. This connects the text module to the research paper.

### 14.1 Probes

Each probe is a deliberately limited model trained and tested on the same splits.

| Probe | Input the model is allowed to see | High score means |
|---|---|---|
| Style only | Length, number of "!", share of capital letters, emoji count, presence of a link | Labels can be guessed from writing style alone |
| First five words | Only the first five words | The opening of a post gives the answer away |
| Shuffled words | All words in random order (transformers) | Word order and context barely matter |
| Source only | Only the source or channel name | The model can learn "who posted" instead of "what is said" |
| Source held out | Full text, but with a group split by source | A big drop in score means the model leaned on source-specific wording |
| Time split | Full text, trained on older items and tested on newer ones | A big drop means the model does not handle new rumours |

### 14.2 Shortcut Index

For each probe:

**Shortcut Index = (probe macro-F1 − chance macro-F1) ÷ (full-model macro-F1 − chance macro-F1)**

Chance is a random guesser that keeps the class ratio. A value near 0 means the shortcut explains little. A value near 1 or above means the shortcut explains almost all of the model's apparent skill.

### 14.3 What to expect

- On synthetic data with style correlation 0.7, the style-only probe should score well. Generating again with 0.5 should make it collapse. This is a clean classroom demonstration of shortcut learning.
- On LIAR, our pilot run in this project found that a model using **only information about who said it** (speaker, party, state, job title, context; macro-F1 0.655) did slightly **better** than TF-IDF on the statement text (0.634). That is a real shortcut, and worth showing in your presentation.

---

## 15. Phase 10: Testing, README and GitHub

1. **Automatic checks**:
   - label mapping, including the numeric-label error;
   - Hindi words stay whole after tokenization;
   - no overlap between splits;
   - metrics give the right answer on a tiny hand-made example;
   - each command runs on a 200-row synthetic dataset.
2. **Smoke test:** a full run-all on 200 rows with distilbert for 1 epoch, then open the UI.
3. **Reproducibility test:** run the statistical models twice and confirm identical results.
4. **README:** what the project does, setup steps, the typical sessions from Phase 6, two or three UI screenshots, the dataset licences and the team.
5. **GitHub hygiene:**
   - **Never commit data, model weights or `.env` files.**
   - Commit in small steps, one phase per commit or pull request.
   - Tag a version (for example v1.0) when you freeze results for the report, so the numbers can always be traced to the exact code.
   - Add a licence.
   - Large trained models, if you want to share them, go to the Hugging Face Hub, not GitHub.

---

## 16. Conceptual clarity: how the approaches differ

This section also goes into `docs/CONCEPTS.md` and the UI's Concepts lab.

### 16.1 Bag-of-Words vs. TF-IDF

1. **What they store.** BoW stores *how many times* each word appears. TF-IDF stores the same counts, **re-weighted by how rare the word is** across all documents.
2. **Common words.** In BoW, frequent words like "the", "hai" or "news" get large values even though they say little. TF-IDF pushes them down automatically, because they appear everywhere.
3. **Distinctive words.** TF-IDF boosts words that are frequent in *this* text but rare overall, such as "cures", "forward", "register". These are often exactly the misinformation cues.
4. **Length.** Long texts get big BoW counts just for being long. TF-IDF vectors are normalised, so long and short texts are compared fairly.
5. **What they share.**
   - Both ignore word order, so "dog bites man" = "man bites dog".
   - Both ignore meaning, so "free" and "muft" are unrelated words to them.
   - Both treat each spelling as a different word, so "nahi" ≠ "nahin". Character n-grams fix only the spelling part.
6. **Classifier pairing.** BoW pairs naturally with Naive Bayes, which models word counts. TF-IDF pairs best with Logistic Regression or a Linear SVM.
7. **Why keep both in the paper.** BoW is the simplest possible baseline. TF-IDF is the strong classical baseline. If BERT can't beat TF-IDF clearly, that is an important finding, not a failure.

### 16.2 Statistical models vs. BERT-family models

| Dimension | BoW / TF-IDF | BERT family |
|---|---|---|
| How text is represented | A sparse vector of word counts or weights | Dense vectors where each word's meaning depends on its neighbours (contextual embeddings) |
| Word order and context | Ignored (except word pairs) | Used. "Not true" and "true" get different representations. |
| Knowledge before training | None; learns only from your labelled data | Pretrained on billions of words, so it brings language knowledge |
| Synonyms and paraphrases | Not understood | Partly understood |
| Hinglish and spelling variation | Weak (character n-grams help) | Depends on the model: MuRIL and HingBERT handle it; English BERT does not |
| Data needed | Works with a few hundred examples | Benefits from thousands; can over-fit tiny datasets |
| Compute | Seconds on a CPU | Minutes to hours; a GPU helps |
| Interpretability | High: you can list the exact words that drive a decision | Lower: needs extra explanation methods |
| Typical failure | Fooled by fakes written in a calm, official style | Can be over-confident; can still learn style or source shortcuts |
| Role in this project | Baseline and sanity check | Main models; Indian-language variants are the novelty |

### 16.3 How the BERT variants differ

1. **Size vs. speed.**
   - **DistilBERT:** a compressed BERT, smaller and faster.
   - **ALBERT:** shares weights between layers to be very small.
   - **RoBERTa:** the same architecture as BERT, trained longer on more data.
2. **Languages.**
   - **bert, distilbert, roberta, albert:** English only.
   - **mBERT and XLM-R:** cover about 100 languages, but Hindi is a small share of their training data.
   - **MuRIL and IndicBERT:** made for Indian languages. MuRIL is also trained on transliterated (Romanised) text.
   - **HingBERT:** trained specifically on Hinglish.
3. **Tokenizer.** Each model splits words into sub-word pieces differently. A tokenizer that doesn't know Hindi breaks a Hindi word into many meaningless pieces, so the model sees noise. This is the single most visible difference for Indian content; show it in the Concepts lab.
4. **Cased vs. uncased.** Uncased models lowercase everything. Cased models keep capitals, which can carry signal ("BREAKING", "URGENT").
5. **Expectation to test, not assume:** on English-only data, the English models should do well. On Hinglish and Hindi data, MuRIL, IndicBERT and HingBERT should do better. The per-language chart is where you show whether this holds.

### 16.4 Fine-tuning vs. frozen embeddings

- **Fine-tuning** adjusts all of the model's weights for your task. It is the most accurate but the slowest.
- **Frozen embeddings plus Logistic Regression** only reads the pretrained knowledge. It is fast and cheap, and the gap between the two modes shows how much task-specific learning adds.

### 16.5 Why a high score on synthetic data means little

The synthetic generator uses a limited set of templates, so models can learn the templates instead of the task. Synthetic data proves the **pipeline works**. Only your real dataset and LIAR can show how good a **model** is. Say this explicitly in your presentation; examiners respect it.

### 16.6 What "fake" means to a text model

A text classifier judges **how a text is written and what it resembles**, not whether its claim is true in the world. A true but sensational post can be called fake, and a calm, well-written lie can be called real. That's why the full SatyaCheck system adds evidence-based steps (claim matching, image provenance, evidence retrieval) on top of the text model.

---

## 17. Presenting the results

**Suggested flow for the minor-project presentation:**
1. **Problem:** misinformation in Indian languages, and why text-only detection is hard.
2. **Concepts lab demo:** BoW vs. TF-IDF matrix; tokenizer comparison on a Hinglish sentence.
3. **Dataset:** the synthetic data (for the pipeline), LIAR (a public benchmark) and your dataset (the real test), using the data card numbers.
4. **Model comparison page:** leaderboard, per-language chart, efficiency chart.
5. **Live checker:** two real and two fake examples, including one hard case, showing where models disagree.
6. **Shortcut audit:** the style or source shortcut finding.
7. **Limitations and next steps:** Section 18.

**Honest-reporting rules:**
- Report macro-F1 with its confidence interval.
- Say which split was used.
- Say when a difference is not significant.
- Never present synthetic-data scores as model performance.

---

## 18. Suggestions from Claude for a better fake-news model

They're roughly in order of value for your project.

1. **Fix the evaluation before improving the model.** Time-based or source-based splits and duplicate removal change results more than any model choice. A model that scores 95% on a random split and 70% on a time split is a 70% model.
2. **Run the shortcut audit on every dataset you use**, including your own, and remove leaky words such as publisher names, URLs and fact-check stamps. Our LIAR pilot already showed a speaker-only model beating a text model.
3. **Use Indian-language models as the main candidates.** MuRIL first, then IndicBERT and HingBERT, with TF-IDF on character n-grams as the strong Hinglish baseline.
4. **Test cross-script consistency.** The same claim written in Devanagari, Romanised Hindi and English should get the same answer. Measure how often it doesn't; this is a novel, easy-to-explain metric.
5. **Calibrate and allow "unverified".**
   - Use temperature scaling so that probabilities are honest.
   - Add an abstain zone: when the model is unsure, it says "unverified" instead of guessing.
   - Report accuracy at different coverage levels.
6. **Ensemble a statistical model and a transformer.** Averaging the probabilities of TF-IDF-char and MuRIL is often more robust than either alone, because they make different kinds of mistakes. Your disagreements table will show this.
7. **Add claim matching.** Most viral fakes are *repeats* of already-debunked claims. Matching a post against fact-check claims (multilingual sentence embeddings such as LaBSE, plus a vector index such as FAISS) often beats classification. You already collected 52,307 fact-checked claims in Stage 1.
8. **Add evidence-based checking.** Retrieve trusted articles and use a multilingual natural-language-inference model to check whether the evidence supports or contradicts the claim. This moves from "sounds fake" to "is contradicted".
9. **Collect more real (true) examples from the same sources.** Only 293 of your 52,307 fact-check claims came out as SUPPORTED in the automatic label suggestion. Without true items from the same channels, a model learns "fact-checked topic = fake".
10. **Test robustness.** Check typos, Romanised spelling variants, emojis, and especially **fakes rewritten in a calm, neutral style** (for example by an LLM). A good model must survive the style rewrite.
11. **Use domain-adaptive pretraining.** Continue MuRIL's pretraining on a large amount of *unlabelled* Indian social-media text before fine-tuning. This usually gives a solid gain on noisy Hinglish.
12. **Report multiple seeds and significance.** For the paper, run 3 seeds, report mean ± spread, and use McNemar or bootstrap tests before claiming one model is better.
13. **Do error analysis by category.** Group mistakes into types: satire, partly true, old-media-new-claim, health, politics, calm-tone fakes. That tells you what to fix next, and it makes a strong paper section.
14. **Use LLMs carefully.** An LLM can suggest labels, but a person must confirm them against evidence. Never use "ask the LLM if it's fake" as ground truth.
15. **Document everything.** Write a datasheet for your dataset and a model card for your best model (intended use, limitations, languages, known biases). Reviewers and examiners value this.
16. **Plan for deployment.** For the SatyaCheck bot, export the best small model (DistilBERT or a distilled MuRIL) to a faster format and measure latency on a normal CPU, because real users won't have GPUs.

---

## 19. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| Packages fail to install | You are probably on Python 3.13 or 3.14. Create a 3.11 or 3.12 environment. |
| MPS error during training on Mac | Some operations aren't supported on Apple's GPU yet. Set PyTorch's MPS fallback environment variable (PYTORCH_ENABLE_MPS_FALLBACK=1) or force CPU for that model. |
| Out of memory | Lower the batch size (16 → 8 → 4) or max length (256 → 128), or use the training-subset option. |
| Model download fails or is slow | Check the internet connection and the exact model ID. Download each model once; after that, the local cache works offline. |
| Error mentioning sentencepiece | ALBERT, XLM-R and IndicBERT need the sentencepiece library; install it in the environment. |
| Hindi text shows as boxes or "?" in Excel | Save CSV files as UTF-8 with a byte-order mark, or open them in Google Sheets. |
| Hindi words split into pieces in BoW/TF-IDF | The Devanagari-aware word pattern from 8.1 is missing. |
| IndicBERT output looks broken for Hindi | The tokenizer is stripping accents (vowel signs); turn accent-keeping on (9.4). |
| Warning about "tokenizers parallelism" | Harmless; disable tokenizer parallelism in the environment to silence it. |
| Synthetic scores near 100% | Expected with the default style setting. Lower the style correlation to 0.5 and raise label noise to see a realistic drop. |
| BERT scores worse than TF-IDF | Check that BERT received lightly cleaned text (13.2), that training ran for more than 1 epoch, and that the learning rate is not too high. On small or very templated data, TF-IDF can simply be competitive; that's a valid finding. |
| UI shows "not trained yet" | Run the matching training command for that dataset, then refresh. |
| The UI port is already in use | Start the app on another port with the app command's port option. |

---

## 20. Final checklist

**Build**
- [ ] Phase 0: environment works; device detected; .gitignore in place.
- [ ] Phase 1: synthetic data generated, checked by eye, deterministic.
- [ ] Phase 2: synthetic and LIAR prepared; data cards saved; no split overlap.
- [ ] Phase 3: five BoW / TF-IDF models trained; top words make sense.
- [ ] Phase 4: at least distilbert, albert, mbert and muril trained (fine-tune), plus two in embeddings mode.
- [ ] Phase 5: leaderboard, figures, significance, disagreements and report generated.
- [ ] Phase 6: every command has help text; run-all works end to end.
- [ ] Phase 7: all eight UI pages work on synthetic data.
- [ ] Phase 8: your dataset prepared with the two-text-column rule; leakage checked.
- [ ] Phase 9: shortcut audit run on synthetic, LIAR and your dataset.
- [ ] Phase 10: tests pass; README with screenshots; version tagged on GitHub.

**Research**
- [ ] Main results come from your real dataset with a time or source split.
- [ ] Macro-F1 is reported with confidence intervals, and differences with significance tests.
- [ ] Per-language results show whether Indian-language models help.
- [ ] Limitations are stated, including that a text model judges wording, not truth.
- [ ] No number anywhere was typed by hand.

---

*End of build instructions.*
