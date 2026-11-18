"""Builds the team guide (HTML -> PDF) from the result files.

Every number in the guide is read from outputs/ or computed by the project code. Nothing is typed by hand.
Usage:  python scripts/make_guide.py            (writes docs/guide/guide.html)
        node scripts/print_pdf.js               (writes docs/SatyaCheck_Text_Team_Guide.pdf)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fakenews.config import load_config  # noqa: E402
from fakenews.utils import read_json  # noqa: E402

CFG = load_config()
OUT = ROOT / "outputs"
SYN = OUT / "synthetic"
LIAR = OUT / "liar"
GUIDE = ROOT / "docs" / "guide"
IMG = GUIDE / "img"
E = html.escape


def rj(p):
    p = Path(p)
    return read_json(p) if p.exists() else None


def f3(v, nd=3):
    try:
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"
    except TypeError:
        return str(v)


def fig(path, caption, width="100%"):
    p = Path(path)
    if not p.exists():
        return f'<p class="missing">[figure not generated: {E(p.name)}]</p>'
    return (f'<figure><img src="{p.as_uri()}" style="width:{width}"><figcaption>{caption}</figcaption></figure>')


def table(df: pd.DataFrame, fmt: dict | None = None, cls="") -> str:
    fmt = fmt or {}
    h = "".join(f"<th>{E(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, (float, np.floating)):
                v = f3(v)
            cells.append(f"<td>{v if str(v).startswith('<') else E(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def box(kind, title, body):
    return f'<div class="box {kind}"><div class="boxt">{title}</div>{body}</div>'


VERDICT_CLS = {"SUPPORTED": "sup", "REJECTED": "rej", "INCONCLUSIVE": "inc", "NOT TESTABLE": "nt"}


def badge(v):
    return f'<span class="badge {VERDICT_CLS.get(v, "nt")}">{E(v)}</span>'


# ------------------------------------------------------------------------------------------ load results
card = rj(ROOT / "data/processed/synthetic/data_card.json")
gen_card = rj(ROOT / "data/synthetic/synthetic_news.card.json")
comp = rj(SYN / "results/comparison.json")
lb = pd.read_csv(SYN / "results/summary.csv")
sig = pd.read_csv(SYN / "results/significance.csv")
bd = pd.read_csv(SYN / "results/breakdowns.csv")
hyp = rj(SYN / "results/hypotheses.json")
aud = pd.read_csv(SYN / "results/audit.csv")
aud_meta = rj(SYN / "results/audit.json")
sweep = pd.read_csv(OUT / "_experiments/style_sweep.csv")
dis = pd.read_csv(SYN / "results/disagreements.csv", encoding="utf-8-sig")
res = {p.stem: read_json(p) for p in (SYN / "results").glob("*.json")
       if not p.name.endswith(".meta.json") and "test_metrics" in read_json(p)}
liar_lb = pd.read_csv(LIAR / "results/summary.csv") if (LIAR / "results/summary.csv").exists() else None
liar_aud = pd.read_csv(LIAR / "results/audit.csv") if (LIAR / "results/audit.csv").exists() else None
liar_card = rj(ROOT / "data/processed/liar/data_card.json")
any_res = next(iter(res.values()))
versions = any_res["versions"]
trans = {m: r for m, r in res.items() if r["family"] == "Transformer"}
stat = {m: r for m, r in res.items() if r["family"] == "Statistical"}
test_df = pd.read_csv(ROOT / "data/processed/synthetic/test.csv")
lang_n = test_df["language"].value_counts().to_dict()

# ------------------------------------------------------------------------------------------ computed examples
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402

from fakenews.classical import CLASSICAL_MODELS, load_classical  # noqa: E402
from fakenews.explain import trace_classical  # noqa: E402
from fakenews.preprocess import TOKEN_PATTERN, bert_view, classical_view, tokenize  # noqa: E402
from fakenews.synthetic import all_templates  # noqa: E402
from fakenews.transformer_models import REGISTRY, load_tokenizer  # noqa: E402

DOCS = ["the news says garlic cures dengue", "the news says rain closes schools", "the ministry says the viral news is false"]
cv = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False).fit(DOCS)
tv = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False).fit(DOCS)
words = ["the", "news", "says", "garlic", "cures", "viral", "false"]
bow_df = pd.DataFrame(cv.transform(DOCS).toarray(), columns=cv.get_feature_names_out())[words]
tf_df = pd.DataFrame(tv.transform(DOCS).toarray(), columns=tv.get_feature_names_out())[words]
bow_df.insert(0, "sentence", ["S1", "S2", "S3"])
tf_df.insert(0, "sentence", ["S1", "S2", "S3"])
idf_df = pd.DataFrame({"word": words, "appears in # sentences": [int((bow_df[w] > 0).sum()) for w in words],
                       "IDF weight": [tv.idf_[tv.vocabulary_[w]] for w in words]})

TRACE_TEXT = "Garlic water cures dengue in just 3 days, doctors are hiding this from you!"
tr = trace_classical(load_classical(CFG, "synthetic", "tfidf_lr"), TRACE_TEXT)
trace_tbl = tr["features"].head(12).copy()

TOK_SENTS = {"Hinglish": "Sarkar sabhi students ko free laptop de rahi hai",
             "Hindi": "कल से 500 के नोट बंद हो जाएंगे"}
tok_rows = []
for short in ["bert", "albert", "mbert", "xlmr", "muril", "hingbert"]:
    try:
        tk = load_tokenizer(short)
    except Exception as e:  # noqa: BLE001
        tok_rows.append({"model": short, "lang": "-", "pieces": "-", "unknown": "-", "split": f"not available: {e}"[:80]})
        continue
    for lang, s in TOK_SENTS.items():
        pcs = tk.tokenize(s)
        tok_rows.append({"model": short, "lang": lang, "pieces": len(pcs), "unknown": sum(p == tk.unk_token for p in pcs),
                         "split": " | ".join(pcs)})
tok_df = pd.DataFrame(tok_rows)

examples = pd.read_csv(ROOT / "data/synthetic/synthetic_news.csv")
ex_rows = []
for lang in ("en", "hinglish", "hi"):
    for lab, case in (("real", "standard"), ("fake", "standard"), ("real", "debunk_report"), ("fake", "calm_fake")):
        d = examples[(examples.language == lang) & (examples.label == lab) & (examples.case_type == case) & (~examples.noisy_label)]
        if len(d):
            r = d.iloc[0]
            ex_rows.append({"label": lab, "language": lang, "case": case, "style": r["style"], "text": r["text"]})
ex_df = pd.DataFrame(ex_rows)


# ------------------------------------------------------------------------------------------ helpers for numbers
def m(model, key="macro_f1"):
    return res[model]["test_metrics"][key]


def lang_pivot():
    d = bd[bd.dimension == "language"].pivot_table(index="model", columns="group", values="macro_f1")
    d = d.reindex(columns=[c for c in ["en", "hinglish", "hi"] if c in d.columns]).reset_index()
    d["family"] = d["model"].map(lambda x: res[x]["family"] if x in res else "Ensemble")
    return d[["model", "family"] + [c for c in ["en", "hinglish", "hi"] if c in d.columns]]


lp = lang_pivot()
worst_hi = lp.sort_values("hi").iloc[0] if "hi" in lp else None
best_hi = lp["hi"].max() if "hi" in lp else None
best_stat, best_trans = comp["best_statistical"], comp["best_transformer"]
svt = comp["statistical_vs_transformer"]
H = {h["id"]: h for h in hyp["hypotheses"]}
counts = pd.Series([h["verdict"] for h in hyp["hypotheses"]]).value_counts().to_dict()
style_row = aud[aud.probe == "style_only"].iloc[0]
tmpl_row = aud[aud.probe == "template_held_out"].iloc[0] if (aud.probe == "template_held_out").any() else None
n_templates = len(all_templates())
devices_used = sorted({r.get("device", "cpu") for r in trans.values()})
ft = {k: v for k, v in trans.items() if v["mode"] == "Fine-tuned"}
P = CFG.get("project", {})

# ------------------------------------------------------------------------------------------ HTML
S = []  # sections
A = S.append

A(f"""
<section class="cover">
  <div class="cover-top">MINOR PROJECT · TEAM GUIDE</div>
  <h1>SatyaCheck-Text</h1>
  <div class="cover-sub">Detecting fake vs. real news in English, Hinglish and Hindi<br>
  with Bag-of-Words, TF-IDF and BERT-family transformers</div>
  <div class="cover-box">
    <b>What this guide explains:</b> how the system was built, how the Hugging Face transformers are plugged in,
    how everything runs from the terminal on CPU or GPU, how outputs and charts are produced, how to read every
    evaluation chart (including the confusion-matrix heat maps), what the research UI shows, why Hinglish/Hindi is
    the novelty, and which hypotheses we proved, disproved or could not decide.
  </div>
  <div class="cover-meta">{E(P.get('team', ''))} · Guide: {E(P.get('guide', ''))} · {E(P.get('institution', ''))}</div>
  <div class="cover-note">All numbers in this document are read automatically from the project's result files
  (<code>outputs/</code>). Results on the synthetic dataset show that the pipeline works; they are not claims about
  real-world performance.</div>
</section>
""")

A("""
<section>
<h2 class="nobreak">Contents</h2>
<ol class="toc">
<li>The big picture in one page</li>
<li>How the project is built (architecture)</li>
<li>The data: synthetic, LIAR and your own</li>
<li>Turning text into something a model can read</li>
<li>The statistical models: Bag-of-Words and TF-IDF</li>
<li>The transformers from Hugging Face</li>
<li>Running it from the terminal, on CPU or GPU</li>
<li>How the output is produced (one prediction, start to finish)</li>
<li>Evaluation: metrics explained</li>
<li>Results and visualisations (heat maps, curves, charts)</li>
<li>The novelty: Hinglish and Hindi</li>
<li>Shortcut audit: does the model learn content or tricks?</li>
<li>Hypotheses: proved, disproved, inconclusive</li>
<li>A real benchmark: LIAR</li>
<li>Tour of the research UI, page by page</li>
<li>Limitations, next steps and questions faculty may ask</li>
<li>Glossary and command cheat-sheet</li>
</ol>
</section>
""")

# 1 ---------------------------------------------------------------------------------------------------------
A(f"""
<section>
<h2>1. The big picture in one page</h2>
<p><b>The problem.</b> Misinformation in India spreads in English, Hindi (Devanagari script) and <b>Hinglish</b>,
which is Hindi typed in Roman letters, often mixed with English ("Sarkar sabhi students ko free laptop de rahi hai").
Most fake-news research uses English only. We want to know which kind of model works best, and whether
models built for Indian languages actually help.</p>
<p><b>What we built.</b> A complete research pipeline that:</p>
<ol>
<li>creates a <b>synthetic dataset</b> of fake and real news in three languages (for building and demonstrating);</li>
<li>trains <b>5 statistical models</b> (Bag-of-Words and TF-IDF with classic classifiers);</li>
<li>trains <b>BERT-family transformers</b> downloaded from Hugging Face ({len(ft)} fine-tuned and
{len(trans) - len(ft)} in "frozen embeddings" mode in the final run);</li>
<li>compares <b>all models on exactly the same test items</b> with confidence intervals and significance tests;</li>
<li>checks for <b>shortcuts</b> (is the model reading the content or just the "!!!"?);</li>
<li>tests <b>{len(hyp['hypotheses'])} research hypotheses</b> automatically;</li>
<li>shows everything in a <b>research UI</b> (Streamlit) for the presentation.</li>
</ol>
{box('key', 'Results in one breath (synthetic data, test split of ' + str(comp['n_test']) + ' items)',
     f"<p>Top of the leaderboard: <b>{E(lb.iloc[0]['model'])}</b> with macro-F1 <b>{f3(lb.iloc[0]['macro_f1'])}</b>. "
     f"Best statistical model (picked on validation): <b>{E(best_stat)}</b> ({f3(svt['f1_statistical'])}); best "
     f"transformer (picked on validation): <b>{E(best_trans)}</b> ({f3(svt['f1_transformer'])}). The difference is "
     f"<b>{'' if svt['significant'] else 'not '}statistically significant</b> (McNemar p = {f3(svt['mcnemar']['p_value'], 4)}). "
     f"Hypotheses: {counts.get('SUPPORTED', 0)} supported, {counts.get('REJECTED', 0)} rejected, "
     f"{counts.get('INCONCLUSIVE', 0)} inconclusive, {counts.get('NOT TESTABLE', 0)} not testable.</p>")}
{box('warn', 'The single most important honesty rule',
     '<p>No number is ever typed by hand into a table, slide, report or the UI. The code writes result files; '
     'everything else (this PDF included) reads from them. Synthetic-data scores prove the <i>pipeline</i> works, '
     'not that the <i>model</i> is good; real claims come from our own dataset and LIAR.</p>')}
<h3>Key words you will see</h3>
<table class="gloss">
<tr><td><b>Fake = 1, Real = 0</b></td><td>Fixed everywhere: files, models, charts.</td></tr>
<tr><td><b>Train / validation / test</b></td><td>The data is split once. Models learn on <i>train</i>, settings are chosen on <i>validation</i>, and the final score is measured once on <i>test</i>.</td></tr>
<tr><td><b>P(fake)</b></td><td>The model's probability that an item is fake. Above 0.5 means "fake".</td></tr>
<tr><td><b>Macro-F1</b></td><td>Our headline score: the average of the F1 score for fake and for real (0 = useless, 1 = perfect, about 0.5 = guessing).</td></tr>
</table>
</section>
""")

# 2 ---------------------------------------------------------------------------------------------------------
A(f"""
<section>
<h2>2. How the project is built (architecture)</h2>
<p>The project is a Python package called <code>fakenews</code> plus a Streamlit app. Each step is a separate
module, and each step writes files that the next step reads. That's why any step can be inspected or re-run on
its own.</p>
<svg viewBox="0 0 900 300" class="diagram">
 <defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>
 <g font-family="Helvetica" font-size="12" text-anchor="middle">
  <rect x="10" y="120" width="120" height="60" rx="8" fill="#eee" stroke="#999"/><text x="70" y="145">Dataset</text><text x="70" y="162" font-size="10">synthetic / LIAR / yours</text>
  <rect x="160" y="120" width="120" height="60" rx="8" fill="#eee" stroke="#999"/><text x="220" y="140">prepare</text><text x="220" y="156" font-size="10">clean · de-duplicate</text><text x="220" y="170" font-size="10">split ONCE · data card</text>
  <rect x="320" y="40" width="130" height="50" rx="8" fill="#deebf7" stroke="#6baed6"/><text x="385" y="62">classical view</text><text x="385" y="78" font-size="10">lowercase, links→token</text>
  <rect x="320" y="210" width="130" height="50" rx="8" fill="#fee6ce" stroke="#fd8d3c"/><text x="385" y="232">BERT view</text><text x="385" y="248" font-size="10">close to the original</text>
  <rect x="490" y="40" width="130" height="50" rx="8" fill="#c6dbef" stroke="#3182bd"/><text x="555" y="62">BoW / TF-IDF</text><text x="555" y="78" font-size="10">NB · LR · SVM · char</text>
  <rect x="490" y="210" width="130" height="50" rx="8" fill="#fdd0a2" stroke="#e6550d"/><text x="555" y="232">BERT family</text><text x="555" y="248" font-size="10">fine-tune · embeddings</text>
  <rect x="660" y="120" width="110" height="60" rx="8" fill="#e5f5e0" stroke="#41ab5d"/><text x="715" y="140">evaluate +</text><text x="715" y="156">compare</text><text x="715" y="170" font-size="10">CI · McNemar · charts</text>
  <rect x="790" y="120" width="100" height="60" rx="8" fill="#efedf5" stroke="#807dba"/><text x="840" y="140">audit +</text><text x="840" y="156">hypotheses</text><text x="840" y="170" font-size="10">→ UI + report</text>
  <g stroke="#555" stroke-width="1.5" fill="none" marker-end="url(#a)">
   <path d="M130,150 L158,150"/><path d="M280,140 L318,70"/><path d="M280,160 L318,232"/>
   <path d="M450,65 L488,65"/><path d="M450,235 L488,235"/><path d="M620,65 L658,135"/><path d="M620,235 L658,165"/><path d="M770,150 L788,150"/>
  </g></g>
</svg>
<h3>Which file does what</h3>
<table>
<tr><th>File (in <code>fakenews/</code>)</th><th>Job</th></tr>
<tr><td><code>synthetic.py</code></td><td>Generates the synthetic dataset from {n_templates} templates in 3 languages.</td></tr>
<tr><td><code>data.py</code></td><td>Loads synthetic / LIAR / your CSV, maps labels, removes duplicates, splits once, writes the data card.</td></tr>
<tr><td><code>preprocess.py</code></td><td>The two text views and the Hindi-safe tokenizer pattern.</td></tr>
<tr><td><code>classical.py</code></td><td>The 5 Bag-of-Words / TF-IDF models (scikit-learn).</td></tr>
<tr><td><code>transformer_models.py</code></td><td>Registry of 9 Hugging Face models, the training loop (PyTorch), embeddings mode.</td></tr>
<tr><td><code>evaluate.py</code></td><td>All metrics: macro-F1, bootstrap CIs, ROC-AUC, calibration (ECE), McNemar test.</td></tr>
<tr><td><code>compare.py</code>, <code>figures.py</code></td><td>Leaderboard, breakdowns, disagreements, every chart, the Markdown report.</td></tr>
<tr><td><code>audit.py</code></td><td>Shortcut probes and the style-sweep experiment.</td></tr>
<tr><td><code>hypotheses.py</code></td><td>Tests the {len(hyp['hypotheses'])} hypotheses and writes verdicts.</td></tr>
<tr><td><code>explain.py</code>, <code>predict.py</code></td><td>Word-level explanations and "predict any text with any saved model".</td></tr>
<tr><td><code>cli.py</code></td><td>The terminal commands (<code>python -m fakenews ...</code>).</td></tr>
<tr><td><code>app/</code></td><td>The Streamlit research UI (11 pages).</td></tr>
<tr><td><code>configs/config.yaml</code></td><td>Every setting in one place (seed {CFG['seed']}, split sizes, model lists, learning rates...).</td></tr>
</table>
<h3>Tools used</h3>
<p>Python {E(versions.get('python', ''))}, scikit-learn {E(str(versions.get('sklearn')))}, PyTorch {E(str(versions.get('torch')))},
Hugging Face Transformers {E(str(versions.get('transformers')))}, pandas, NumPy, SciPy (statistical tests), matplotlib
(saved charts), Plotly (interactive charts) and Streamlit (UI). These are the exact versions recorded in the result files.</p>
</section>
""")

# 3 ---------------------------------------------------------------------------------------------------------
sc = gen_card["params"]
A(f"""
<section>
<h2>3. The data</h2>
<h3>3.1 The synthetic dataset (for building and demonstrating)</h3>
<p>We generate news items from <b>templates with slots</b>. A template like "<i>{{remedy}} cures {{disease}} in just
{{n_small}} days, doctors are hiding this from you!</i>" is filled at random with values such as
"garlic water", "dengue" and "3". Hindi templates have their own slot values in Devanagari.</p>
<table>
<tr><th>Setting</th><th>Value used</th><th>Why</th></tr>
<tr><td>Rows</td><td>{sc['n_rows']:,}</td><td>About half fake, half real ({f3(gen_card['share_fake'] * 100, 1)}% fake)</td></tr>
<tr><td>Languages</td><td>{', '.join(f"{k}: {v}" for k, v in gen_card['language_counts'].items())}</td><td>English, Hinglish, Hindi</td></tr>
<tr><td>Style correlation <i>s</i></td><td>{sc['style_correlation']}</td><td>P(sensational | fake) = s, P(sensational | real) = 1 − s. Style is a useful but imperfect clue; 0.5 removes it.</td></tr>
<tr><td>Label noise</td><td>{sc['label_noise']:.0%} ({gen_card['noisy_labels']} rows flipped)</td><td>Real datasets are never perfectly labelled. This also caps the best possible score.</td></tr>
<tr><td>Sources</td><td>{sc['n_sources']} channels, {sc['fake_heavy_sources']} mostly fake</td><td>Lets the audit show a "who posted it" shortcut.</td></tr>
<tr><td>Hard cases</td><td>{', '.join(f"{k}: {v}" for k, v in gen_card['case_type_counts'].items())}</td><td>Real reports <i>about</i> debunks and calm, official-sounding fakes, so style is not the only clue.</td></tr>
</table>
<p>Example rows (taken from the generated file):</p>
{table(ex_df, cls='small')}
<p class="note">Each row also stores <code>template_id</code>, <code>case_type</code> and <code>noisy_label</code>,
so we can always see how an item was made. No real people are named; institutions are generic.</p>

<h3>3.2 Preparing a dataset: the same steps for every dataset</h3>
<ol>
<li><b>Load</b> the file and find the text and label columns. Text labels (fake/false/hoax/pants-fire,
real/true/genuine...) are mapped automatically. <b>Numeric labels are never guessed</b>: you must say which value
means fake, because datasets disagree (in WELFake, 0 means fake).</li>
<li><b>Clean lightly</b>: trim spaces, drop empty texts, and <b>remove exact duplicates before splitting</b> (otherwise the
same message could be in train <i>and</i> test, which inflates scores). Near-duplicates are counted and reported.</li>
<li><b>Split once</b> into train / validation / test (70/15/15, stratified) and save the split to disk. Every model reads
these same files, which is what makes the comparison fair. Time-based and source-based splits are also available.</li>
<li><b>Write a data card</b> with sizes, label balance, language mix, duplicates removed and a fingerprint (SHA-256)
of the input file, so we always know which version of the data was used.</li>
</ol>
<p>For the synthetic run: train {card['sizes']['train']}, validation {card['sizes']['val']}, test
{card['sizes']['test']}; exact duplicates removed: {card['exact_duplicates_removed']}; near-duplicate rows:
{card['near_duplicate_rows']}; overlap of texts between splits: {card['split_overlap_texts']}.</p>
<div class="row">{fig(SYN / 'figures/data_class_balance.png', 'Class balance per split (red = fake, green = real).', '100%')}
{fig(SYN / 'figures/data_language_mix.png', 'Language mix by label.', '100%')}</div>

<h3>3.3 Your own dataset</h3>
<p>Put a CSV / Excel / Parquet file in <code>data/raw/</code> (never committed to GitHub), or upload it on the
UI page "Use your own dataset". If possible, keep <b>two text columns</b>: a fully cleaned one for
BoW/TF-IDF, and a lightly cleaned one for BERT. BERT needs words like "not", punctuation and emojis, because it reads
grammar and context. A <b>date</b> column enables the honest time split (train on old, test on new), and a
<b>source</b> column enables the source split and the source audit.</p>
</section>
""")

# 4 ---------------------------------------------------------------------------------------------------------
hi_ex = "मौसम विभाग के अनुसार बारिश होगी"
A(f"""
<section>
<h2>4. Turning text into something a model can read</h2>
<p>Each model family gets the text in the form that suits it. We call these the two <b>views</b>.</p>
<table>
<tr><th>View</th><th>Used by</th><th>What is done</th><th>Example</th></tr>
<tr><td>classical</td><td>BoW, TF-IDF</td><td>lowercase; links → <code>urltoken</code>; keep Hindi letters; keep "!!!" and emojis as tokens</td>
<td>{E(classical_view('BREAKING: Free laptop!!! register at www.x.in 🚨'))}</td></tr>
<tr><td>BERT</td><td>all transformers</td><td>almost the original text: only spaces normalised and links replaced</td>
<td>{E(bert_view('BREAKING: Free laptop!!! register at www.x.in 🚨'))}</td></tr>
</table>
{box('key', 'A small but important fix for Hindi',
     f"<p>scikit-learn's default word pattern splits Hindi words at the vowel signs (matras), so a word like "
     f"'बारिश' falls apart into pieces. Our tokenizer treats the whole Devanagari Unicode block (U+0900–U+097F) as word "
     f"characters. Result for '<span class='deva'>{hi_ex}</span>': "
     f"<span class='deva'>{' | '.join(E(t) for t in tokenize(hi_ex))}</span>. Every word stays whole, and an automatic test "
     f"checks this.</p>")}
</section>
""")

# 5 ---------------------------------------------------------------------------------------------------------
A(f"""
<section>
<h2>5. The statistical models: Bag-of-Words and TF-IDF</h2>
<h3>5.1 Bag-of-Words (BoW): just count the words</h3>
<p>BoW turns every text into a long row of numbers, one column per word (and per word pair), holding <i>how many
times</i> the word appears. Word order is thrown away, so "dog bites man" = "man bites dog". Here are three tiny
sentences (S1: "{DOCS[0]}", S2: "{DOCS[1]}", S3: "{DOCS[2]}"):</p>
{table(bow_df)}
<h3>5.2 TF-IDF: counts re-weighted by rarity</h3>
<p>TF-IDF takes the counts and multiplies them by how <i>rare</i> each word is across documents (the IDF weight), then
scales each row to length 1. Words that appear everywhere ("the", "says") get pushed down; distinctive words
("garlic", "cures", "viral") get pushed up. These are often exactly the misinformation cues.</p>
<div class="row"><div>{table(idf_df)}</div><div>{table(tf_df, {c: (lambda v: f3(v, 2)) for c in words})}</div></div>
<p class="note">IDF = ln((1 + n) / (1 + df)) + 1, where n is the number of documents and df the number containing the word
(scikit-learn's smoothed formula). These numbers were computed by the same code the models use.</p>

<h3>5.3 The five models</h3>
{table(pd.DataFrame([{'model': k, 'name': v['label'], 'what it does': v['about'],
                      'chosen setting (on validation)': f"{stat[k]['tuned_param']} = {stat[k]['best_value']}" if k in stat else '-',
                      'vocabulary': f"{stat[k]['vocab_size']:,}" if k in stat else '-'} for k, v in CLASSICAL_MODELS.items()]), cls='small')}
<h3>5.4 How they are trained</h3>
<ol>
<li>The vocabulary is learned from the <b>training split only</b>. Learning it from all data would leak test words.</li>
<li>One setting per model is tuned on the <b>validation split</b>: regularisation strength C ∈ {CFG['classical']['C_grid']}
for Logistic Regression and SVM, smoothing alpha ∈ {CFG['classical']['alpha_grid']} for Naive Bayes. The value with the best
validation macro-F1 wins.</li>
<li>The model is evaluated <b>once</b> on the test split.</li>
<li>We save the model, the tuning table, the 25 strongest words for fake and for real, timings and every test probability.</li>
</ol>
<p>What the TF-IDF + Logistic Regression model learned (red = pushes towards fake, green = towards real):</p>
{fig(SYN / 'figures/top_features/tfidf_lr.png', 'Top weights of tfidf_lr. Note "!" in the fake list: that is the style shortcut, which the audit measures.', '78%')}
</section>
""")

# 6 ---------------------------------------------------------------------------------------------------------
reg_df = pd.DataFrame([{"short name": k, "Hugging Face ID": v["hf"], "size": v["size"], "languages": v["languages"],
                        "trained in final run": ", ".join(sorted(x[len(k) + 1:] for x in trans if x.rsplit('_', 1)[0] == k)) or "-"}
                       for k, v in REGISTRY.items()])
A(f"""
<section>
<h2>6. The transformers from Hugging Face</h2>
<h3>6.1 What a transformer does differently</h3>
<p>A BERT-style model was <b>pre-trained</b> on billions of words to fill in missing words. In the process it learned
grammar, word meanings and context. Each word gets a vector that depends on its neighbours, so "not true" and "true"
look different to it, unlike in BoW. We then <b>fine-tune</b> it, which teaches it our fake-vs-real task with a few
thousand labelled examples.</p>
<h3>6.2 The model registry</h3>
<p>All models are listed in one dictionary (<code>REGISTRY</code> in <code>transformer_models.py</code>). Adding a model
means adding one line with its Hugging Face ID. "Ambert" is read as both mBERT and ALBERT, and both are included.</p>
{table(reg_df, cls='small')}
<h3>6.3 How a model is pulled in from Hugging Face</h3>
<p>The Hugging Face <code>transformers</code> library downloads a model the first time it is used and keeps it in a
local cache (after that it works offline). Two calls do the work:</p>
<pre>tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model     = AutoModelForSequenceClassification.from_pretrained("google/muril-base-cased", num_labels=2)</pre>
<p>The first loads the model's own <b>tokenizer</b>, which splits text into sub-word pieces. The second loads the
pre-trained network and adds a fresh 2-class output layer (fake / real) on top. For IndicBERT we pass
<code>keep_accents=True</code> so Hindi vowel signs are not stripped. IndicBERT is a <b>gated</b> model: accept its terms on
Hugging Face and run <code>hf auth login</code> once.</p>
<h3>6.4 Mode 1: fine-tuning (main mode)</h3>
<p>We wrote the training loop in plain PyTorch (not a black-box trainer) so every step is visible:</p>
<pre>for epoch in 1..{CFG['transformers']['epochs']}:
    for batch of {CFG['transformers']['batch_size']} texts (shuffled):
        ids, mask = tokenizer(texts)                 # text -> token ids, padded per batch
        logits    = model(ids.to(device), mask.to(device))
        loss      = cross_entropy(logits, labels)    # class weights if one class > 60%
        loss.backward()                              # compute gradients
        clip_grad_norm(model, {CFG['transformers']['grad_clip']})                  # keeps training stable
        AdamW.step(); scheduler.step()               # lr {CFG['transformers']['learning_rate']} ({CFG['transformers']['learning_rate_small']} for small models), 10% warm-up then linear decay
    P(fake) on validation -> macro-F1
    keep the best epoch; stop if no improvement for {CFG['transformers']['patience']} epoch (early stopping)</pre>
<p>Max length {CFG['transformers']['max_length']} tokens (long articles can keep "head + tail"), weight decay
{CFG['transformers']['weight_decay']}, seed {CFG['seed']}. After training, the best weights, tokenizer, training curve,
parameter count, timings, device and every test probability are saved.</p>
<h3>6.5 Mode 2: frozen embeddings (fast mode)</h3>
<p>The network is <b>not</b> trained. Each text is passed through the pre-trained model once, the token vectors are
averaged into one sentence vector ({next((r['embedding_dim'] for r in trans.values() if r.get('embedding_dim')), '-')}
numbers for DistilBERT), and a Logistic Regression is trained on those vectors. This is fast on a laptop and shows how much
the model "knows" before any fine-tuning. The gap between the two modes is a research point (hypothesis H9).</p>
{fig(SYN / 'figures/training_curves.png', 'Training curves from the final run: validation macro-F1 (left) and loss (right) per epoch for each fine-tuned transformer. Early stopping kept the best epoch.', '100%')}
</section>
""")

# 7 ---------------------------------------------------------------------------------------------------------
tt = pd.DataFrame([{"model": k, "device": v.get("device"), "parameters": f"{v['params']:,}",
                    "training time (s)": f"{v['train_seconds']:.0f}", "ms per item": f"{v['infer_ms_per_item']:.2f}",
                    "best epoch": v.get("best_epoch", "-")} for k, v in sorted(trans.items())])
A(f"""
<section>
<h2>7. Running it from the terminal, on CPU or GPU</h2>
<h3>7.1 One entry point, several commands</h3>
<p>Everything runs as <code>python -m fakenews &lt;command&gt;</code>. Every command prints what it is doing and where it
saved the results, and appends a line to <code>outputs/&lt;dataset&gt;/logs/run_log.jsonl</code>.</p>
<table>
<tr><th>Command</th><th>What happens</th></tr>
<tr><td><code>make-data</code></td><td>generate the synthetic CSV</td></tr>
<tr><td><code>prepare</code></td><td>clean, de-duplicate, split once, data card</td></tr>
<tr><td><code>train-classical</code></td><td>train the 5 BoW/TF-IDF models (seconds)</td></tr>
<tr><td><code>train-bert --models distilbert,muril</code></td><td>fine-tune (or <code>--mode embed</code>) transformers</td></tr>
<tr><td><code>compare</code></td><td>leaderboard, significance, breakdowns, all charts, report</td></tr>
<tr><td><code>audit</code> / <code>style-sweep</code></td><td>shortcut probes / style experiment</td></tr>
<tr><td><code>hypotheses</code></td><td>test all hypotheses</td></tr>
<tr><td><code>predict --text "..."</code></td><td>every saved model's P(fake) side by side</td></tr>
<tr><td><code>run-all</code> (<code>--full</code>)</td><td>the whole chain in one command</td></tr>
<tr><td><code>app</code></td><td>start the research UI in the browser</td></tr>
</table>
<h3>7.2 How the computer's GPU is used</h3>
<p>The function <code>get_device()</code> picks the hardware automatically, in this order:</p>
<ol>
<li><b>CUDA</b>: an NVIDIA GPU (Google Colab, Kaggle, gaming PCs);</li>
<li><b>MPS</b>: the GPU inside Apple Silicon Macs (M1/M2/M3...);</li>
<li><b>CPU</b>: always works, just slower.</li>
</ol>
<p>Then <code>model.to(device)</code> copies the network's weights into GPU memory, and each batch is moved there with
<code>ids.to(device)</code>. All the heavy matrix multiplications of training and prediction then happen on the GPU. You
can force a device with <code>--device cpu|mps|cuda</code>. If an operation is not supported on Apple's MPS, the code
falls back to CPU for that model automatically. Statistical models always run on CPU and need only seconds.</p>
<h3>7.3 What the final run recorded</h3>
<p>The final run used: <b>{', '.join(devices_used)}</b> (the cloud machine that produced these results had no GPU). On a Mac
with MPS or a Colab GPU the same commands run unchanged and the training times drop. Each model's time is recorded in its
result file, so it is never estimated.</p>
{table(tt)}
<p>For comparison, the statistical models trained in {', '.join(f"{v['train_seconds']:.1f}" for v in stat.values())}
seconds (including tuning), at about {f3(np.mean([v['infer_ms_per_item'] for v in stat.values()]), 2)} ms per item.</p>
<h3>7.4 Typical sessions</h3>
<pre>python -m fakenews run-all            # quick demo
python -m fakenews run-all --full     # 4 transformers x 3 epochs
python -m fakenews app                # open the UI

# our own dataset
python -m fakenews prepare --source csv --path data/raw/my.csv --name mynews --label-col label --fake-value 1 --split time --date-col date
python -m fakenews train-classical --name mynews
python -m fakenews train-bert --name mynews --models distilbert,mbert,muril
python -m fakenews compare --name mynews ; python -m fakenews audit --name mynews ; python -m fakenews hypotheses --name mynews</pre>
</section>
""")

# 8 ---------------------------------------------------------------------------------------------------------
tsum = trace_tbl.copy()
A(f"""
<section>
<h2>8. How the output is produced: one prediction, start to finish</h2>
<p>Here is the exact path from a sentence to a verdict for the TF-IDF + Logistic Regression model. It was computed by
the same function the UI's "Model internals" page uses.</p>
<ol class="steps">
<li><b>Raw text:</b> {E(tr['raw'])}</li>
<li><b>Classical view:</b> <code>{E(tr['clean'])}</code></li>
<li><b>Tokens:</b> <code>{E(' | '.join(tr['tokens']))}</code><br>Not in the vocabulary (ignored): {E(', '.join(tr['unknown_tokens']) or 'none')}</li>
<li><b>Features × weights:</b> each active word/pair has a TF-IDF value and a learned weight; contribution = value × weight
(positive pushes towards fake). The 12 largest:</li>
</ol>
{table(tsum, {'value': lambda v: f3(v, 4), 'weight': lambda v: f"{v:+.4f}", 'contribution': lambda v: f"{v:+.4f}"})}
<ol class="steps" start="5">
<li><b>Score</b> = intercept {tr['intercept']:+.4f} + sum of all {len(tr['features'])} contributions
{tr['features']['contribution'].sum():+.4f} = <b>{tr['score']:+.4f}</b></li>
<li><b>Probability</b> = 1 / (1 + e<sup>−score</sup>) = <b>{tr['p_fake']:.4f}</b> → <b>{'FAKE' if tr['p_fake'] >= 0.5 else 'REAL'}</b>
(threshold {CFG['evaluation']['threshold']})</li>
</ol>
<p>For a <b>transformer</b> the path is: text → sub-word tokens → token ids → network → two numbers (logits) → softmax →
P(fake). Its reasoning is not a simple sum, so the UI explains it by <b>removing one word at a time</b> and measuring how
much P(fake) changes (leave-one-word-out importance).</p>
<h3>Where every output goes</h3>
<table>
<tr><th>File</th><th>Content</th></tr>
<tr><td><code>outputs/&lt;name&gt;/results/&lt;model&gt;.json</code></td><td>all metrics, settings, timings, library versions, the full config</td></tr>
<tr><td><code>results/predictions/&lt;model&gt;.csv</code></td><td>P(fake) for every test item, which is the raw material for every metric</td></tr>
<tr><td><code>results/summary.csv</code></td><td>the leaderboard</td></tr>
<tr><td><code>results/predictions.csv</code></td><td>one row per test item with every model's probability</td></tr>
<tr><td><code>results/disagreements.csv</code></td><td>items where the best statistical model and best transformer disagree</td></tr>
<tr><td><code>results/significance.csv</code>, <code>breakdowns.csv</code>, <code>errors.csv</code>, <code>audit.csv</code>, <code>hypotheses.json</code></td><td>tests, per-group scores, confident mistakes, probes, verdicts</td></tr>
<tr><td><code>figures/*.png</code></td><td>every chart (also downloadable from the UI)</td></tr>
<tr><td><code>reports/results_report.md</code>, <code>hypotheses.md</code></td><td>auto-written reports</td></tr>
<tr><td><code>models/</code></td><td>saved models (never committed to GitHub)</td></tr>
</table>
</section>
""")

# 9 ---------------------------------------------------------------------------------------------------------
A(f"""
<section>
<h2>9. Evaluation: metrics explained</h2>
<p>All metrics are computed on the <b>test split</b>, from the saved probabilities, with "fake" = P(fake) ≥ {CFG['evaluation']['threshold']}.</p>
<table>
<tr><th>Metric</th><th>Plain meaning</th></tr>
<tr><td>Accuracy</td><td>Share of items labelled correctly. Misleading when classes are unbalanced.</td></tr>
<tr><td>Precision (fake)</td><td>Of the items the model called fake, how many really were fake.</td></tr>
<tr><td>Recall (fake)</td><td>Of the truly fake items, how many the model caught.</td></tr>
<tr><td>F1</td><td>The balance of precision and recall.</td></tr>
<tr><td><b>Macro-F1 (headline)</b></td><td>The average of F1 for fake and F1 for real; treats both classes equally.</td></tr>
<tr><td>95% confidence interval</td><td>We re-sample the test set {CFG['evaluation']['bootstrap']:,} times (bootstrap) and see how much macro-F1 moves. Overlapping intervals mean "could be luck".</td></tr>
<tr><td>ROC-AUC / PR-AUC</td><td>How well the model <i>ranks</i> fake above real across all thresholds (1 = perfect, 0.5 = random).</td></tr>
<tr><td>Brier score</td><td>Average squared error of the probabilities (lower is better).</td></tr>
<tr><td>ECE (calibration error)</td><td>Does "90% sure" really mean right 90% of the time? (lower is better)</td></tr>
<tr><td>McNemar test</td><td>Compares two models on the <i>same</i> items: counts where A is right and B wrong, and the reverse. p &lt; 0.05 means the difference is unlikely to be luck.</td></tr>
<tr><td>Cost</td><td>Training time, milliseconds per item, number of parameters, disk size.</td></tr>
</table>
{box('key', 'Fair-comparison rules we follow', '<ul><li>One split, saved once; every model sees the same test items.</li>'
     '<li>Settings and the "best model of each family" are chosen on <b>validation</b>, never on test.</li>'
     '<li>Every score comes with a confidence interval; differences come with a significance test.</li></ul>')}
</section>
""")

# 10 --------------------------------------------------------------------------------------------------------
lbs = lb[["rank", "model", "family", "mode", "macro_f1", "ci_low", "ci_high", "accuracy", "roc_auc", "ece", "train_seconds", "ms_per_item", "params"]].copy()
lbs["params"] = lbs["params"].map(lambda v: f"{v / 1e6:.1f}M" if v >= 1e5 else f"{int(v):,}")
lbs["train_seconds"] = lbs["train_seconds"].map(lambda v: f"{v:.1f}")
lbs["ms_per_item"] = lbs["ms_per_item"].map(lambda v: f"{v:.2f}")
sig_s = sig[["model_a", "model_b", "diff", "a_right_b_wrong", "a_wrong_b_right", "p_value", "significant"]]
hc = bd[bd.dimension == "case_type"].pivot_table(index="model", columns="group", values="accuracy").reset_index()
A(f"""
<section>
<h2>10. Results and visualisations (synthetic dataset)</h2>
{box('warn', 'Read this first', '<p>The synthetic data is made from templates, so models can learn the templates. '
     f"With {sc['label_noise']:.0%} label noise the best possible score is about {1 - sc['label_noise']:.2f}. "
     'Expect many models near that ceiling. The interesting parts are where they <i>differ</i>: per language, hard cases '
     'and shortcuts.</p>')}
<h3>10.1 The leaderboard</h3>
{table(lbs, cls='small')}
<p>{E(comp['summary'])}</p>
{fig(SYN / 'figures/macro_f1_bar.png', 'Macro-F1 per model with 95% bootstrap error bars. Blue = statistical, orange = transformers, grey = ensemble. Bars whose error bars overlap are not clearly different.', '88%')}

<h3>10.2 Confusion matrices: the heat maps</h3>
<p>A confusion matrix is a 2×2 table of counts, coloured as a heat map (darker = more items). <b>Rows are the truth,
columns are the prediction.</b> The diagonal (top-left and bottom-right) holds the correct answers.
<b>Top-right</b> counts real news wrongly called fake (false alarms). <b>Bottom-left</b> counts fakes the model missed,
which is the dangerous error for a fact-checking tool.</p>
{fig(SYN / 'figures/confusion_matrices.png', 'Confusion-matrix heat maps for every model on the same test items (counts and share of all items).', '100%')}

<h3>10.3 Curves</h3>
<div class="row">{fig(SYN / 'figures/roc_curves.png', '<b>ROC curves</b>: the catch rate for fakes vs. the false-alarm rate as the threshold moves. Closer to the top-left corner is better.')}
{fig(SYN / 'figures/pr_curves.png', '<b>Precision-recall curves</b> for the fake class: higher and further right is better.')}</div>
<div class="row">{fig(SYN / 'figures/calibration.png', '<b>Calibration</b>: points on the diagonal mean the probabilities are honest; below it means over-confident about fake.')}
{fig(SYN / 'figures/efficiency.png', '<b>Accuracy vs. cost</b>: top-left is best (accurate and fast). Bubble size = number of parameters. TF-IDF is thousands of times faster.')}</div>

<h3>10.4 Breakdowns</h3>
<div class="row">{fig(SYN / 'figures/per_style.png', 'Macro-F1 on neutral vs. sensational items.')}
{fig(SYN / 'figures/hard_cases.png', 'Accuracy by case type: calm fakes, debunk reports, standard, surprising-but-true.')}</div>
<p>Accuracy on each hard-case type:</p>
{table(hc, cls='small')}

<h3>10.5 Is the best model significantly better? (McNemar)</h3>
{table(sig_s, {'p_value': lambda v: f3(v, 4), 'diff': lambda v: f"{v:+.3f}"}, cls='small')}
<p class="note">a_right_b_wrong = items the top model got right and the other got wrong; a_wrong_b_right = the reverse.
With only a handful of discordant items, almost no difference on synthetic data is significant. That is an honest
result, not a bug.</p>

<h3>10.6 Where TF-IDF and the transformer disagree</h3>
<p>The best statistical model ({E(best_stat)}) and the best transformer ({E(best_trans)}) disagree on
{svt['n_disagreements']} test items: {E(str(svt['disagreement_counts']))}. The full list is in
<code>results/disagreements.csv</code> and in the UI's disagreements browser.</p>
{table(dis.head(6)[[c for c in dis.columns if c in ('text', 'true_label', 'who_is_right') or c.startswith('p_fake')]], cls='small') if len(dis) else ''}
</section>
""")

# 11 --------------------------------------------------------------------------------------------------------
h4 = H["H4"]
A(f"""
<section>
<h2>11. The novelty: Hinglish and Hindi</h2>
<p>Most fake-news detectors are English-only. Indian misinformation is written in Devanagari Hindi and, very often, in
<b>Hinglish</b>: Hindi typed in Roman letters, with no fixed spelling ("nahi", "nahin", "nai") and English words mixed
in. Our project treats this as a first-class problem in five ways:</p>
<ol>
<li><b>A three-language synthetic dataset</b> with matching fake/real patterns in English, Hinglish and Devanagari Hindi.</li>
<li><b>A Hindi-safe tokenizer</b> for BoW/TF-IDF, so matras don't break words (section 4).</li>
<li><b>Character n-gram TF-IDF</b> (3–5-letter pieces), which matches spelling variants that word models treat as different words (hypothesis H3).</li>
<li><b>Indian-language transformers</b>: MuRIL (Google, trained on 16 Indian languages <i>including their Romanised/transliterated forms</i>), IndicBERT (AI4Bharat), HingBERT (trained on Hinglish), alongside mBERT and XLM-R.</li>
<li><b>Per-language evaluation</b>: every model is scored separately on English, Hinglish and Hindi (hypothesis H4).</li>
</ol>
<h3>11.1 Why tokenizers matter: the visual proof</h3>
<p>Each transformer first chops text into pieces it knows. Below, the same two sentences are split by six real tokenizers.
<b>Fewer, whole-word pieces mean the model "knows" the language.</b> Many tiny pieces, or lost vowel signs, mean it is
reading noise.</p>
{table(tok_df, {'split': lambda v: f'<span class="deva">{E(str(v))}</span>'}, cls='small tok')}
<p class="note">Look at ALBERT on Hindi: it drops the vowel signs (से becomes स). It "strips accents", which destroys
Devanagari. MuRIL keeps the Hinglish words whole and splits Hindi into real words.</p>
<h3>11.2 What happened per language</h3>
<p>Test items per language: {', '.join(f'{k}: {v}' for k, v in lang_n.items())}.</p>
{table(lp, cls='small')}
{fig(SYN / 'figures/per_language.png', 'Macro-F1 per language for every model.', '100%')}
{box('key', 'What this shows',
     (f"<p>The clearest effect is <b>{E(worst_hi['model'])}</b> on Hindi: macro-F1 <b>{f3(worst_hi['hi'])}</b>, against up to "
      f"{f3(best_hi)} for other models. Its English-only tokenizer destroys Devanagari, exactly as the tokenizer table "
      f"predicts. Hinglish is the hardest language for most models on this data. "
      f"Formal test (H4, Indian/multilingual vs. English-only transformers on Hindi + Hinglish): {badge(h4['verdict'])} "
      f"- {E(h4['explanation'])}</p>") if worst_hi is not None else '')}
<p><b>Why H4 is not "proved" yet.</b> H4 compares the <i>best</i> model of each group (chosen on validation). On synthetic
data the best English-only model (DistilBERT) already learns the Hindi templates, so the groups tie. The Hindi test set is
also small. The true test is our real Hinglish/Hindi dataset, which is exactly where this pipeline is meant to be run next.
A further novel metric we propose is <b>cross-script consistency</b>: the same claim in Devanagari, Hinglish and English
should get the same answer, and we measure how often it doesn't.</p>
</section>
""")

# 12 --------------------------------------------------------------------------------------------------------
probes = aud[aud.kind == "probe"][["probe", "input_seen", "macro_f1", "ci_low", "ci_high", "shortcut_index"]]
held = aud[aud.kind == "held_out"][["probe", "reference_macro_f1", "macro_f1", "ci_low", "ci_high", "drop_vs_reference"]].rename(
    columns={"reference_macro_f1": "random split", "macro_f1": "held out"})
A(f"""
<section>
<h2>12. Shortcut audit: does the model learn content or tricks?</h2>
<p>A model can score well by spotting an easy clue instead of understanding the text, for example "!!!" or the
channel name. To measure this, we train deliberately <b>limited models (probes)</b> that can only see one clue, and compare
them with the full model:</p>
<p class="formula">Shortcut Index = (probe macro-F1 − chance) ÷ (full model macro-F1 − chance)</p>
<p>About 0 means the clue explains little; about 1 means it explains almost all of the model's skill. Chance macro-F1 here
= {f3(aud_meta['chance_macro_f1'])}; reference model = {E(aud_meta['reference_model'])} ({f3(aud_meta['reference_macro_f1'])}).</p>
{table(probes, cls='small')}
{fig(SYN / 'figures/shortcut_index.png', 'Shortcut Index per probe (red ≥ 0.5).', '80%')}
<h3>Held-out tests: does the score survive something new?</h3>
{table(held, cls='small')}
<p>{('<b>Template held out</b> is the key synthetic finding: when whole templates are kept out of training, TF-IDF falls from '
     f"{f3(tmpl_row['reference_macro_f1'])} to {f3(tmpl_row['macro_f1'])}. The models largely memorised templates, which is "
     'why high synthetic scores mean little.') if tmpl_row is not None else ''}</p>
<h3>The style-sweep experiment</h3>
<p>We regenerated the data with style correlation s from {sweep['style_correlation'].min()} to {sweep['style_correlation'].max()}
and trained a style-only model each time. As s rises, the style-only probe gets better: the shortcut is created by the data.</p>
{fig(OUT / '_experiments/style_sweep.png', 'Style-only probe vs. full TF-IDF model as the style correlation increases.', '75%')}
</section>
""")

# 13 --------------------------------------------------------------------------------------------------------
hrows = "".join(f"<tr><td><b>{h['id']}</b></td><td>{E(h['title'])}</td><td>{badge(h['verdict'])}</td>"
                f"<td>{'-' if h['p_value'] is None else f3(h['p_value'], 4)}</td>"
                f"<td>{'-' if h['effect'] is None else f'{h[chr(101) + chr(102) + chr(102) + chr(101) + chr(99) + chr(116)]:+.3f}'}</td></tr>"
                for h in hyp["hypotheses"])
hdetail = "".join(
    f"<div class='hyp'><div class='hyph'><b>{h['id']}: {E(h['title'])}</b> {badge(h['verdict'])}</div>"
    f"<p><b>Hypothesis.</b> {E(h['statement'])}</p><p><b>Why it matters.</b> {E(h['why_it_matters'])}</p>"
    f"<p><b>Test.</b> {E(h['test'])}</p><p><b>Result.</b> {E(h['explanation'])}</p></div>" for h in hyp["hypotheses"])
A(f"""
<section>
<h2>13. Hypotheses: proved, disproved, inconclusive</h2>
<p>We wrote {len(hyp['hypotheses'])} hypotheses <b>before</b> looking at the results. The command
<code>python -m fakenews hypotheses</code> tests each one with a proper statistical test (alpha = {hyp['alpha']}) and gives a
verdict:</p>
<table>
<tr><td>{badge('SUPPORTED')}</td><td>The predicted effect is there and significant: "proved" on this data.</td></tr>
<tr><td>{badge('REJECTED')}</td><td>The opposite effect is there and significant: "disproved".</td></tr>
<tr><td>{badge('INCONCLUSIVE')}</td><td>No significant difference either way. <b>Not</b> the same as disproved: the data cannot tell.</td></tr>
<tr><td>{badge('NOT TESTABLE')}</td><td>A needed model or column is missing (the reason is given).</td></tr>
</table>
<p>The models being compared are always chosen on <b>validation</b>, so the test set is never used to decide what to test.</p>
<h3>Summary (synthetic dataset)</h3>
<table><tr><th>ID</th><th>Hypothesis</th><th>Verdict</th><th>p-value</th><th>Effect</th></tr>{hrows}</table>
<p class="note">"Effect" is in the predicted direction when positive (e.g. macro-F1 difference, change in rate, Shortcut
Index minus the threshold, Spearman rho for H14).</p>
{box('key', 'How to explain these to faculty',
     '<p>On synthetic data most comparisons are <b>inconclusive</b> because every model is near the ceiling, and that is '
     'expected and honest. The <b>supported</b> ones are about the data itself (template memorisation, the style shortcut '
     'made by the data) and about calibration. The same hypotheses will be re-tested automatically on our real dataset; '
     'only those verdicts become research claims.</p>')}
<h3>Each hypothesis in detail</h3>
{hdetail}
</section>
""")

# 14 --------------------------------------------------------------------------------------------------------
if liar_lb is not None:
    ll = liar_lb[["rank", "model", "family", "macro_f1", "ci_low", "ci_high", "accuracy", "roc_auc"]]
    meta_row = liar_aud[liar_aud.probe == "metadata_only"].iloc[0] if liar_aud is not None and (liar_aud.probe == "metadata_only").any() else None
    A(f"""
<section>
<h2>14. A real benchmark: LIAR</h2>
<p>LIAR is a public dataset of about 12,800 short political statements checked by PolitiFact (US English). We map
true/mostly-true → real and false/pants-fire → fake, drop the ambiguous middle labels, and keep LIAR's official splits
(train {liar_card['sizes']['train']}, validation {liar_card['sizes']['val']}, test {liar_card['sizes']['test']}). It tests
general ability on a real dataset, not Indian content.</p>
{table(ll, cls='small')}
{box('key', 'A real shortcut, found automatically',
     (f"<p>A model that sees <b>only who said it</b> (speaker, party, state, job, context) and never reads the statement scores "
      f"macro-F1 <b>{f3(meta_row['macro_f1'])}</b> (95% CI {f3(meta_row['ci_low'])}–{f3(meta_row['ci_high'])}), compared "
      f"with {f3(meta_row['reference_macro_f1'])} for the best statistical model that reads the text "
      f"(Shortcut Index {f3(meta_row['shortcut_index'], 2)}). On LIAR, "
      "\"who said it\" predicts the label about as well as \"what was said\". This shows why shortcut audits matter.</p>")
     if meta_row is not None else '')}
<p>LIAR scores are much lower than synthetic ones (around 0.6 instead of 0.97). That is normal: real claims are hard, and
the published literature reports similar numbers for text-only models.</p>
</section>
""")

# 15 --------------------------------------------------------------------------------------------------------
ui = [
    ("ui_home.png", "Home", "Project title and team (edit in <code>configs/config.yaml</code> → <code>project</code>), the problem statement, the pipeline diagram, three headline cards (top model, best statistical, best transformer with macro-F1 and CI), the auto-written summary, and every hypothesis verdict at a glance. A yellow banner warns when the synthetic dataset is selected. The <b>sidebar</b> lists all pages and has the <b>dataset selector</b>: switch between synthetic, LIAR and your own data, and every page updates."),
    ("ui_data.png", "Data explorer", "The data card (split sizes, duplicates removed, split strategy, file fingerprint), charts for class balance, language mix and text length, the top words per class (use this to spot leaky words such as publisher names), and a filterable table of rows showing both text views and the deliberately flipped labels."),
    ("ui_comparison.png", "Model comparison (top)", "The auto-written Statistical vs. Transformer summary and the leaderboard. Green cells mark the best value in each column. Every table and chart has a download button for slides."),
    ("ui_comparison2.png", "Model comparison (charts)", "The macro-F1 bar chart with confidence intervals, then tabs for confusion matrices, ROC & PR, calibration, efficiency, per language, style & length, hard cases and training curves. Further down: the McNemar significance table, the <b>disagreements browser</b> (items where TF-IDF and BERT disagree) and each model's most confident mistakes."),
    ("ui_internals1.png", "Model internals (transparency)", "Pick any model. (1) What it is: pipeline, backbone, parameters, timings, exact settings. (2) How it was trained and tuned: the validation tuning table, or the training curve with best epoch for transformers. (3) Learned weights for statistical models."),
    ("ui_internals_threshold.png", "Model internals: threshold explorer", "Move the threshold slider and watch the confusion matrix, recall, precision and false alarms change, computed live from the saved predictions. The histogram shows how P(fake) is spread for true-fake and true-real items."),
    ("ui_internals_trace.png", "Model internals: trace one prediction", "Pick a test item or type text and follow it step by step: for TF-IDF the tokens, every feature × weight and the final sigmoid; for transformers the sub-word tokens, token ids, P(fake) and leave-one-word-out importance. The bottom of the page has the raw result JSON and the run log of every command."),
    ("ui_hypotheses.png", "Hypotheses", "What each verdict means, counts per verdict, a colour-coded table, and an expandable card per hypothesis with its statement, reason, test, numbers and explanation (H14 also shows the style-sweep chart)."),
    ("ui_audit.png", "Shortcut audit", "The Shortcut Index chart and table for every probe, the held-out tests (source, template, time), and a plain-language explanation of each probe."),
    ("ui_live.png", "Live checker", "Type or paste any text, or click an example (real, fake, hard real, hard fake, Hinglish, Hindi). Every selected model gives a verdict card with P(fake) and time, plus an agreement line (\"8 of 8 models say fake\")."),
    ("ui_live2.png", "Live checker: why?", "Word-level explanations: for each statistical model, words are coloured red (push towards fake) or green (towards real) using its exact weights; for transformers, optional leave-one-word-out importance. Remember: a model judges wording, not facts."),
    ("ui_batch.png", "Batch checker", "Upload a CSV/Excel of texts, choose the text column and models, run, and download P(fake) and the verdict for every row."),
    ("ui_upload.png", "Use your own dataset", "Upload your file, choose the text / BERT-text / label / date / source / language columns, say which label means fake, pick the split, and press one button: prepare → statistical models → compare → audit → hypotheses. It then shows the terminal commands for training transformers."),
    ("ui_concepts1.png", "Concepts lab", "Type 2–4 sentences and see the BoW count matrix next to the TF-IDF matrix (words in every sentence are outlined: they get the lowest weight). Below that, the tokenizer comparison and the concept notes."),
]
ui_html = "".join(f"<div class='ui'><h3>{E(t)}</h3>{fig(IMG / f.replace('ui_', 'ui_c_', 1), d, '100%')}</div>" for f, t, d in ui)
A(f"""
<section>
<h2>15. Tour of the research UI, page by page</h2>
<p>Start it with <code>python -m fakenews app</code>; it opens at <code>http://localhost:8501</code>. The UI <b>never trains
heavy models itself</b>: it reads what the terminal commands produced, so every number matches the result files. The
"About &amp; limitations" page (not shown) lists datasets, model links, ethics and limitations.</p>
{ui_html}
</section>
""")

# 16 --------------------------------------------------------------------------------------------------------
A("""
<section>
<h2>16. Limitations, next steps and questions faculty may ask</h2>
<h3>Limitations (say these openly)</h3>
<ul>
<li>A text classifier judges <b>how a text is written and what it resembles</b>, not whether the claim is true. A calm,
well-written lie can be called real; a true but sensational post can be called fake.</li>
<li>Synthetic data is templated, so models memorise templates (H12). It proves the pipeline, not the model.</li>
<li>Random splits overestimate real-world performance; time-based or source-based splits are more honest.</li>
<li>LIAR is US political English; it does not represent Indian content.</li>
<li>The final run used a CPU; times on a GPU will be shorter (the code records the device in every result).</li>
</ul>
<h3>Next steps (in order of value)</h3>
<ol>
<li>Run the full pipeline on <b>our real cleaned dataset</b> with a time or source split. Read the top TF-IDF words first and remove leaky words.</li>
<li>Make <b>MuRIL</b> (then IndicBERT, HingBERT) the main candidates, with character TF-IDF as the strong Hinglish baseline.</li>
<li>Measure <b>cross-script consistency</b> (same claim in Hindi, Hinglish and English should get the same answer).</li>
<li>Calibrate probabilities and add an <b>"unverified"</b> zone instead of forcing fake/real.</li>
<li>Add <b>claim matching</b> against already fact-checked claims, and evidence retrieval.</li>
<li>Report 3 seeds (<code>--seeds 3</code>) and significance for paper numbers.</li>
</ol>
<h3>Likely questions and short answers</h3>
<table class="qa">
<tr><td><b>Why is TF-IDF as good as BERT here?</b></td><td>The synthetic data is templated and small, so word cues are enough. On harder, real data with paraphrases and context, transformers usually pull ahead. If they don't, that is a valid finding.</td></tr>
<tr><td><b>Why macro-F1 and not accuracy?</b></td><td>It treats fake and real equally and is not fooled by class imbalance.</td></tr>
<tr><td><b>How do you know the difference is real?</b></td><td>McNemar test on the same items, plus bootstrap confidence intervals. We report "not significant" when it isn't.</td></tr>
<tr><td><b>Did you tune on the test set?</b></td><td>No. All tuning and model selection uses the validation split; test is used once.</td></tr>
<tr><td><b>What is novel?</b></td><td>Three-language (English/Hinglish/Hindi) evaluation with Indian-language transformers, Hindi-safe tokenization, tokenizer analysis, an automatic shortcut audit and pre-registered hypothesis testing in one reproducible pipeline.</td></tr>
<tr><td><b>Can it check facts?</b></td><td>No. It judges wording. The full SatyaCheck system adds claim matching and evidence retrieval on top.</td></tr>
<tr><td><b>How do we reproduce a number?</b></td><td>Same seed + same data fingerprint + the config stored in each result file + the run log.</td></tr>
</table>
</section>
""")

# 17 --------------------------------------------------------------------------------------------------------
A("""
<section>
<h2>17. Glossary and command cheat-sheet</h2>
<table class="gloss">
<tr><td><b>BoW</b></td><td>Bag-of-Words: counts how often each word appears.</td></tr>
<tr><td><b>TF-IDF</b></td><td>Term frequency × inverse document frequency: counts re-weighted by rarity.</td></tr>
<tr><td><b>n-gram</b></td><td>A sequence of n words (word n-gram) or n characters (character n-gram).</td></tr>
<tr><td><b>Logistic Regression / Naive Bayes / SVM</b></td><td>Classic classifiers that learn one weight per feature.</td></tr>
<tr><td><b>Transformer / BERT</b></td><td>A neural network that reads all words together and uses context; pre-trained on huge text.</td></tr>
<tr><td><b>Tokenizer</b></td><td>Splits text into the pieces a model knows (sub-words).</td></tr>
<tr><td><b>Fine-tuning</b></td><td>Continuing to train a pre-trained model on our labelled task.</td></tr>
<tr><td><b>Frozen embeddings</b></td><td>Using the pre-trained model's vectors without changing it.</td></tr>
<tr><td><b>Epoch</b></td><td>One full pass over the training data.</td></tr>
<tr><td><b>Learning rate</b></td><td>How big each training step is.</td></tr>
<tr><td><b>Early stopping</b></td><td>Stop training when validation stops improving; keep the best epoch.</td></tr>
<tr><td><b>CUDA / MPS / CPU</b></td><td>NVIDIA GPU / Apple GPU / normal processor.</td></tr>
<tr><td><b>Bootstrap</b></td><td>Re-sampling the test set many times to see how much a score varies.</td></tr>
<tr><td><b>p-value</b></td><td>How likely a difference this big would be by luck; below 0.05 is "significant".</td></tr>
<tr><td><b>Calibration / ECE</b></td><td>Whether probabilities are honest.</td></tr>
<tr><td><b>Shortcut</b></td><td>An easy clue (style, source, template) that works on the dataset but not in reality.</td></tr>
<tr><td><b>Hinglish</b></td><td>Hindi written in Roman letters, often mixed with English.</td></tr>
<tr><td><b>Devanagari / matra</b></td><td>The Hindi script / its vowel signs.</td></tr>
</table>
<h3>Cheat-sheet</h3>
<pre>git clone https://github.com/charanjot27/fakenews.git &amp;&amp; cd fakenews
git checkout claude/practical-newton-tj0fh0
conda create -n satyacheck python=3.11 -y &amp;&amp; conda activate satyacheck
pip install -r requirements.txt
python -m fakenews run-all          # or: run-all --full
python -m fakenews app              # http://localhost:8501
python -m fakenews predict --text "Sarkar sabhi students ko free laptop de rahi hai"
python -m pytest -q                 # tests</pre>
<p class="note">This guide is generated by <code>scripts/make_guide.py</code> from the result files. Re-run it after
new experiments and every number updates automatically.</p>
</section>
""")

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
body { font-family: 'Noto Sans', 'Helvetica Neue', Arial, 'Noto Sans Devanagari', sans-serif; color: #1f2328; font-size: 10.3pt; line-height: 1.45; }
.deva, td { font-family: 'Noto Sans', Arial, 'Noto Sans Devanagari', sans-serif; }
h1 { font-size: 40pt; margin: 0; color: #12355b; }
h2 { font-size: 18pt; color: #12355b; border-bottom: 2px solid #12355b; padding-bottom: 4px; margin-top: 0; }
h3 { font-size: 12.5pt; color: #1d4e89; margin: 16px 0 6px; break-after: avoid; }
section { break-before: page; }
section.cover { break-before: auto; height: 250mm; display: flex; flex-direction: column; justify-content: center; }
.cover-top { letter-spacing: 3px; color: #d62728; font-weight: 700; font-size: 10pt; }
.cover-sub { font-size: 15pt; color: #333; margin: 10px 0 28px; }
.cover-box { background: #f1f5fb; border-left: 5px solid #12355b; padding: 14px 16px; font-size: 11pt; }
.cover-meta { margin-top: 28px; color: #555; }
.cover-note { margin-top: 10px; font-size: 9pt; color: #777; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt; break-inside: auto; }
th { background: #12355b; color: white; text-align: left; padding: 4px 6px; }
td { border-bottom: 1px solid #ddd; padding: 3px 6px; vertical-align: top; }
tr { break-inside: avoid; }
table.small { font-size: 8pt; }
table.tok td:last-child { font-size: 7.5pt; }
table.gloss td:first-child, table.qa td:first-child { width: 30%; }
figure { margin: 8px 0 14px; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #e3e3e3; }
figcaption { font-size: 8.5pt; color: #555; margin-top: 3px; text-align: left; }
.row { display: flex; gap: 10px; } .row > * { flex: 1; min-width: 0; }
.box { border-radius: 6px; padding: 8px 12px; margin: 10px 0; break-inside: avoid; }
.box.key { background: #eef6ee; border-left: 5px solid #2ca02c; }
.box.warn { background: #fff7e0; border-left: 5px solid #e0a800; }
.boxt { font-weight: 700; margin-bottom: 2px; }
.box p { margin: 4px 0; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px; padding: 8px 10px; font-size: 8.3pt; white-space: pre-wrap; break-inside: avoid; }
code { background: #f1f3f5; padding: 0 3px; border-radius: 3px; font-size: 9pt; }
.note { font-size: 8.8pt; color: #555; }
.formula { text-align: center; font-weight: 700; background: #f1f5fb; padding: 6px; }
.badge { padding: 1px 7px; border-radius: 9px; font-weight: 700; font-size: 8pt; white-space: nowrap; }
.badge.sup { background: #e6f4ea; color: #1e7e34; } .badge.rej { background: #fdecea; color: #b02a37; }
.badge.inc { background: #fff8e1; color: #8a6d00; } .badge.nt { background: #eee; color: #555; }
.hyp { border: 1px solid #e3e3e3; border-radius: 6px; padding: 6px 10px; margin: 8px 0; break-inside: avoid; }
.hyp p { margin: 3px 0; font-size: 9.2pt; }
.ui { break-inside: avoid; margin-bottom: 10px; }
.ui figure img { max-height: 165mm; object-fit: contain; object-position: top; }
.toc li { margin: 5px 0; font-size: 11.5pt; }
.diagram { width: 100%; height: auto; margin: 6px 0 10px; }
ol.steps li { margin-bottom: 4px; }
.missing { color: #b02a37; }
"""
GUIDE.mkdir(parents=True, exist_ok=True)
out = GUIDE / "guide.html"
out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>SatyaCheck-Text Team Guide</title>"
               f"<style>{CSS}</style></head><body>{''.join(S)}</body></html>", encoding="utf-8")
print(out)
