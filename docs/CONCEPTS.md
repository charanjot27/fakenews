# Concepts: how the approaches differ

_From BUILD_INSTRUCTIONS.md, Section 16. Also shown in the UI's Concepts lab._

### Bag-of-Words vs. TF-IDF

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

### Statistical models vs. BERT-family models

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

### How the BERT variants differ

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

### Fine-tuning vs. frozen embeddings

- **Fine-tuning** adjusts all of the model's weights for your task. It is the most accurate but the slowest.
- **Frozen embeddings plus Logistic Regression** only reads the pretrained knowledge. It is fast and cheap, and the gap between the two modes shows how much task-specific learning adds.

### Why a high score on synthetic data means little

The synthetic generator uses a limited set of templates, so models can learn the templates instead of the task. Synthetic data proves the **pipeline works**. Only your real dataset and LIAR can show how good a **model** is. Say this explicitly in your presentation; examiners respect it.

### What "fake" means to a text model

A text classifier judges **how a text is written and what it resembles**, not whether its claim is true in the world. A true but sensational post can be called fake, and a calm, well-written lie can be called real. That's why the full SatyaCheck system adds evidence-based steps (claim matching, image provenance, evidence retrieval) on top of the text model.
