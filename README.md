# Detecting Racism in Football Commentary

**A hybrid NLP approach comparing BERT and RoBERTa, with a rule layer for coded language**

MS Data Science capstone · Rochester Institute of Technology · 2025
Sneha Mondal · Advisor: Dr. Ashique Khudabukhsh

---

## About this repository

This is a **documentation archive** of the capstone project. It contains the final poster
and an interim project summary.

**The source code is not included.** It was developed locally and lost before it was pushed
to a remote. Everything reported below is quoted from the two documents in this repository,
not reproduced from code here.

Because the figures come from project documents written at two different stages, this README
states which document each number comes from, and flags the places where the documents do not
explain themselves. Nothing here has been reinterpreted to make the results look tidier than
the sources support.

---

## The problem

Football commentary carries a kind of racism that general-purpose toxicity classifiers miss.
Explicit slurs are the easy case. The harder case is coded language — remarks about a player's
"tactical IQ," being "surprisingly articulate for his background," or being "naturally athletic
but lacking technical skill."

Phrases like these are not racist in themselves. "Football IQ" and "naturally athletic" are
ordinary football vocabulary, and the same words can be straightforward analysis or a racial
stereotype depending on who is being described and how. That ambiguity is the actual research
problem: the system has to treat such phrases as *candidate* coded language whose meaning is
resolved by context, rather than as a blocklist.

Four challenges shaped the design:

1. **Pattern diversity** — explicit and implicit racism look nothing alike to a model.
2. **False positives** — heated but legitimate football criticism resembles abuse on the
   surface, and a detector that cannot separate them is unusable in practice.
3. **Data scarcity** — annotated examples of *implicit* racism are rare.
4. **Context dependence** — the same sentence can be analysis or abuse depending on what
   surrounds it.

---

## Approach

A hybrid system rather than a single classifier:

1. **Transformer classifier** — BERT, fine-tuned, for broad contextual coverage.
2. **SVM on BERT features** — a linear-kernel SVM over pooled `[CLS]` embeddings, for
   generalization on a small corpus.
3. **Rule layer** — lexicons and pattern detectors for explicit slurs, candidate coded-language
   cues, and football-analysis exemptions.
4. **Ensemble** — rule override plus majority vote across BERT and the SVM.

### Feature flags

| Flag | Contents |
|---|---|
| `slur_list` | Explicit slurs and animal terms |
| `implicit_indicators` | 60+ candidate coded-language phrases ("football IQ," "surprisingly articulate") |
| `football_exemption` | Football-analysis vocabulary ("tactic," "possession") |

The `implicit_indicators` list is a set of phrases that *can* function as coded stereotypes in
context. It is a retrieval layer, not a verdict — which is why the exemption rule below exists.

### Rule override

> If model confidence < 0.75 **and** only `football_exemption` tokens are present
> → force the prediction to non-racist.

This rule was introduced to reduce false positives on legitimate tactical criticism.

---

## Data

*Source: final poster.*

Manually labelled from YouTube, Reddit, Twitter and news articles, then extended with targeted
synthetic templates to cover cases the real corpus was too thin to teach.

| Component | Count | Notes |
|---|---|---|
| Base corpus | 950 | 66.1% non-racist, 33.9% racist — manually labelled |
| Implicit-racism templates | 45 | Synthetic |
| Explicit-slur templates | 8 | Synthetic |
| Benign animal references | 8 | Non-racist controls |
| Media-critique templates | 20 | Non-racist sports-media criticism |
| **Final dataset** | **1,031** | |

- **Balancing:** undersampled 1:1 to 375 / 375
- **Class weights:** 1:2 (non-racist : racist), applied during training
- **Splits:** 80/20 train/validation, plus a separate challenging edge-case test set

The benign-animal and media-critique buckets exist specifically to teach the model what *not*
to flag — a deliberate choice to protect precision rather than chase recall.

---

## Models

*Source: final poster.*

| Stage | Model | Key setting | Racist-class F₁ |
|---|---|---|---|
| 1 | BERT | Base model | 0.72 |
| 2 | RoBERTa | Same setup | 0.61 |
| 3 | Optimized BERT | Fine-tuned params | 0.76 |
| 4 | SVM | On pooled BERT `[CLS]` embeddings | 0.68 |
| 5 | **Final hybrid** | **+ rule engine** | **0.76** |

Configuration:

- **BERT** — `bert-base-uncased`, class weights `[1.0, 2.0]`, threshold sweep → 0.82
- **RoBERTa** — `roberta-base`, `lr=3e-5`, `weight_decay=0.01`, 3 epochs
- **SVM** — linear kernel on pooled BERT `[CLS]` embeddings
- **Hybrid** — rule override + majority vote (BERT and SVM)

**BERT outperformed RoBERTa** (F₁ 0.72 vs 0.61) on this task and corpus, and was used in the
final hybrid.

---

## Results

*Source: final poster, "Key Results on Held-Out Test."*

| Metric | Value |
|---|---|
| Overall accuracy | **75.2%** |
| Racist-class F₁ | **0.68** |
| Precision, racist class | 62.3% |
| Recall, racist class | 76.0% |

> **Unresolved in the source documents.** The stage table above reports F₁ 0.76 for the final
> hybrid, while the held-out results report 0.68 for the racist class. The poster does not state
> which evaluation split or averaging method each figure uses, so the two are not reconciled
> here. **The 75.2% / 0.68 held-out figures are the ones this project reports.**

### Error analysis

*Source: final poster.*

| Error type | Rate | Dominant cause |
|---|---|---|
| False positives | 16.3% | Comments *discussing* racism without containing it |
| False negatives | 8.5% | Subtle implicit racism in coded sports language |

**Calibration.** Average confidence on *incorrect* predictions was **0.82**; on *correct*
predictions, **0.69**. The model was more confident when it was wrong than when it was right,
which means its confidence scores could not be used as-is for routing or thresholding.

---

## Earlier stage of the project

*Source: interim project summary.*

The interim summary documents an earlier stage with a **different corpus and different
evaluation**, and is included here for the development history rather than as a comparison
against the final numbers.

| | Interim summary |
|---|---|
| Baseline corpus | 694 comments, 107 racist (15%), 23 explicit examples |
| Baseline precision, racist class | 26% |
| Expanded corpus | 923 examples (306 racist / 616 non-racist), undersampled 1:1.5 |
| Validation accuracy | 71.2% |
| Validation F₁, racist class | 0.633 |

> **The two documents are not directly comparable.** The corpora, class ratios and evaluation
> sets differ between the interim summary and the final poster, so the 26% baseline precision
> and the poster's 62.3% were not measured on the same data. The summary also reports a
> separate "challenging test set" at 93% accuracy and F₁ 0.95; that result belongs to the
> earlier stage and is **superseded by the poster**, which is the later document and the final
> one for this project.

### Example predictions

*Source: interim project summary — these confidences come from the earlier-stage model, not
the final hybrid.*

None of these contain an explicit slur:

| Comment | Confidence |
|---|---|
| "Lukaku missed again, these people just don't have the intelligence for tactical play." | 99% |
| "He's surprisingly articulate for a player from that background." | 97% |
| "Players like Saka are just naturally athletic, but lack technical skills." | 98% |
| "Football was better before all these foreigners came in." | 97% |

---

## Directions for further work

Drawn from the next-steps sections of both documents:

- **Evaluate temperature scaling** against the calibration finding above.
- **Active learning** to collect further implicit-racism examples.
- **Human review routing** for predictions in the 0.75–0.90 confidence band.
- **Multi-class output** separating non-racist, explicit racist and implicit racist, rather
  than collapsing the two positive cases.
- **More neutral refereeing and officiating commentary**, identified in the interim summary as
  an area for improvement on false positives.

---

## Documents

### Capstone (2025)

| File | What it is |
|---|---|
| [`Capstone_Poster.pptx`](Capstone_Poster.pptx) | Final poster — architecture, data composition, model comparison, held-out results, error analysis |
| [`Project_Summary.pdf`](Project_Summary.pdf) | Interim summary from an earlier stage — baseline figures, the 923-example corpus and 1:1.5 balancing that preceded the final design |

Where the two disagree, the poster is final.

### Earlier coursework (November 2023) — `docs/`

These three documents come from an **earlier course project**, *Football and Racism*,
**co-authored with Siva Meenakshi**, about eighteen months before the capstone. They are
planning and literature-review documents written in advance of implementation, and they
describe a **different intended design** from the one the capstone ultimately used — LLama2
rather than RoBERTa, PostgreSQL/MongoDB for storage, Instagram as an additional source, and
BoW/TF-IDF/Word2Vec feature extraction alongside sentiment analysis, NER and topic modelling.

They are included for the research trail, not as capstone results. Nothing in them should be
read as describing the final system.

| File | What it is |
|---|---|
| [`docs/2023-11_Architecture_Write_Up.pdf`](docs/2023-11_Architecture_Write_Up.pdf) | Intended architecture, data-collection plan and annotation protocol |
| [`docs/2023-11_Approach_Write_Up.pdf`](docs/2023-11_Approach_Write_Up.pdf) | Collection process by source, preprocessing and feature-extraction plan, analysis and reporting |
| [`docs/2023-11_Related_Research.pdf`](docs/2023-11_Related_Research.pdf) | Literature review — bias mitigation, race in NLP, extremism detection, veiled hate speech |

The architecture write-up documents the data-collection and annotation methodology:
YouTube Data API collection across Big Five league channels, Reddit via the API and PRAW
across r/Soccer, r/PremierLeague and r/Football, BeautifulSoup scraping of sports news
portals, and manual annotation by two annotators into three labels (racist, not racist,
implicitly racist) with reliability checks and iterative review.

The literature review's engagement with Han & Tsvetkov's *Fortifying Toxic Speech Detectors
Against Veiled Toxicity* and Field et al.'s *A Survey of Race, Racism, and Anti-Racism in NLP*
anticipates the implicit-racism problem the capstone went on to address.

---

## References

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
- Davidson, T. et al. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.*
- Kilvington, D. (2020). *The Virtual Stages of Hate: Using Goffman's Work to Conceptualise the Motivations for Online Hate.*
- Farrington, N. et al. (2020). *Race, Racism and Sports Journalism.*
