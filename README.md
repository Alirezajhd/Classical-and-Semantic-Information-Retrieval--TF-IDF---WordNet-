# Classical and Semantic Information Retrieval (TF-IDF + WordNet)

An ad hoc retrieval system for scientific-paper search. It uses an inverted index built from scratch, TF-IDF with cosine similarity as the main ranker, and a WordNet-based semantic reranking step applied only to the top 20 lexical results. It runs on an ordinary CPU. The bonus adds a small learning-to-rank (LTR) reranker written in NumPy.

Built for the Text Mining & NLP course (Assignment 2), Shiraz University, Spring 2026.

## Why

Plain word matching has two problems. It treats every matching word as equally important, and it misses relevant documents that use different words for the same idea. TF-IDF fixes the first by down-weighting common words. WordNet is used to attack the second, but only as a supporting signal on a small candidate pool, because semantic scoring on everything tends to add noise.

## How it works

```
raw query / document text
   -> preprocess   tokenize -> lowercase -> stop words -> Porter stemming
   -> inverted index (term -> postings with tf, df)
   -> TF-IDF cosine ranking via posting lists   (top 100)
   -> WordNet semantic reranking of the top 20  (Wu-Palmer, nouns/verbs only)
   -> (bonus) ridge-regression LTR reranking of the same top 20
   -> rankings + metrics + query trace
```

**Preprocessing.** Tokenization with `(?u)\b\w\w+\b`, case folding, NLTK English stop words, minimum token length 2, Porter stemming. Every step can be switched on or off in `config.json`. Vocabulary drops from 78,277 to 58,136 terms (-25.8%).

**Inverted index.** For each term it stores the posting list, the term frequency per document and the document frequency. Documents are indexed as title + text. Title weighting is applied in the ranker, not in the index, so the index always holds true counts. Only documents sharing a term with the query are scored, and there is no linear scan.

| Statistic | Value |
|---|---|
| Documents | 25,657 |
| Unique terms | 58,136 |
| Postings | 1,883,493 |
| Average document length | 114.01 tokens |

**TF-IDF ranking.** Sublinear TF with smooth IDF, ranked by cosine similarity:

```
w(t,d)   = (1 + ln tf_eff(t,d)) * idf(t)
idf(t)   = ln((N + 1) / (df(t) + 1)) + 1
tf_eff   = tf_raw + tf_title * (w_title - 1)
```

**Semantic reranking (Option B).** For each query, the top K = 20 TF-IDF results are reranked. Content words are the nouns and verbs found by POS tagging the unstemmed words. For each query word, the best Wu-Palmer similarity among the document's content words of the same POS is taken, then averaged over the query words. Exact matches score 1.0 directly, at most 2 WordNet senses are checked per word, and document content words are capped at 100. If either word list is empty, the semantic score is 0.0 and the system falls back to the lexical score.

```
Score_final = alpha * Score_lexical + (1 - alpha) * Score_semantic
```

Both scores are rescaled to [0, 1] within the top-20 pool before combining, because raw TF-IDF cosines are much smaller than WordNet scores.

**Bonus: LTR.** Ridge regression (written from scratch in NumPy) over 8 features per query-document pair: TF-IDF score, WordNet score, log document length, matched query terms, coverage ratio, summed IDF, query length and title matches. It is trained with 5-fold cross-validation over queries, so no query is scored by a model that saw it.

## Results

Dev set (BEIR / SCIDOCS-derived), final configuration `alpha = 0.6`, `title_weight = 3.0`:

| System | MAP | MRR | P@1 | P@5 | P@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| Random | 0.0012 | 0.0056 | 0.0029 | 0.0017 | 0.0010 | 0.0014 | 0.0016 |
| Overlap | 0.0530 | 0.1754 | 0.0900 | 0.0620 | 0.0459 | 0.0589 | 0.0840 |
| TF-IDF | 0.1105 | 0.2890 | 0.1757 | 0.1211 | 0.0850 | 0.1146 | 0.1552 |
| **Final (semantic)** | **0.1128** | **0.2996** | **0.1871** | 0.1229 | 0.0849 | 0.1142 | 0.1536 |
| LTR (bonus) | 0.1125 | 0.2984 | 0.1871 | 0.1209 | 0.0871 | 0.1114 | 0.1563 |

Ablations (dev):

| Condition | MAP | NDCG@10 |
|---|---|---|
| Baseline TF-IDF | 0.1105 | 0.1552 |
| No stop-word removal | 0.1097 | 0.1542 |
| No stemming | 0.1061 | 0.1425 |
| Overlap instead of TF-IDF | 0.0530 | 0.0840 |
| Semantic, raw scores | 0.1114 | 0.1486 |
| Semantic, normalized (final) | 0.1128 | 0.1536 |

Alpha sweep (dev MAP):

| alpha | 0.0 | 0.2 | 0.4 | 0.6 | 0.75 | 0.8 | 0.85 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| MAP | 0.0889 | 0.1027 | 0.1099 | **0.1128** | 0.1112 | 0.1109 | 0.1109 | 0.1104 | 0.1105 |

Notes:

- IDF weighting matters far more than anything else tested (MAP more than doubles over plain overlap).
- Stemming is the most valuable preprocessing step (no stemming: -4.0% MAP, -8.2% NDCG@10).
- Semantic reranking helps most at the very top of the list (P@1 +6.5%, MAP +2.1%), and barely moves NDCG@10.
- A moderate title boost works best: MAP is 0.1080 at w = 1, 0.1105 at w = 3 and 0.1084 at w = 15.
- LTR roughly matches the semantic system, trading a little MAP for better P@10 and NDCG@10.

## Installation

```bash
pip install numpy nltk
# NLTK resources used (stop words, WordNet, POS tagger)
python -m nltk.downloader stopwords wordnet averaged_perceptron_tagger
```

Python 3.x. Verify the exact NLTK resource names against what your code loads.

## Data

The dataset is course-supplied and is not included in this repository:

```
HW2_data/
├── documents.jsonl      # doc_id, title, text
├── queries_dev.tsv
├── qrels_dev.tsv        # binary relevance
├── queries_test.tsv
└── README_dataset.txt
```

## Usage

```bash
python system.py \
  --docs HW2_data/documents.jsonl \
  --dev-queries HW2_data/queries_dev.tsv \
  --dev-qrels HW2_data/qrels_dev.tsv \
  --test-queries HW2_data/queries_test.tsv \
  --outdir results
```

`system.py` has four modes, selected with `--mode`:

| Mode | What it does |
|---|---|
| `full` (default) | Runs everything and writes all required files |
| `alpha-sweep` | Sweeps the semantic weight alpha |
| `title-weight-sweep` | Sweeps the title weight |
| `ablation` | Runs the ablation conditions |

The inverted index is cached to disk and rebuilt only when the documents or preprocessing settings change. All randomness uses seed 42.

## Output files

| File | Contents |
|---|---|
| `dev_rankings.tsv`, `test_rankings.tsv` | `query_id, doc_id, rank, score, method` |
| `dev_metrics.json` | MAP, MRR, P@1/5/10, NDCG@5/10 for random, overlap, tfidf, final (and ltr) |
| `summary.json` | Preprocessing, methods, top-k, seed, score weights |
| `query_trace.jsonl` | Per-query log: preprocessing, lexical top 10, semantic step, rank changes with reasons |
| `report.pdf` | Written report |
| `extra/` | Sweeps, ablation logs, cached similarities |

## Repository layout

```
system.py        # CLI and end-to-end pipeline
preprocess.py    # tokenization, stop words, stemming
index.py         # inverted index construction, caching and lookup
ranker.py        # TF-IDF cosine ranking, overlap and random baselines
semantic.py      # WordNet reranking
ltr.py           # bonus: ridge-regression learning to rank
evaluator.py     # metrics and output writing
config.py        # loads and checks config.json
config.json
```

## Known limitations

- **Short queries.** With only one or two content words, one lucky wrong WordNet match swings the semantic score (e.g. "Outlier Analysis" lost two relevant documents from the top 10).
- **Technical jargon.** WordNet is a general-purpose dictionary and judges specialized vocabulary poorly (e.g. an LTCC waveguide filter query).
- **Figurative titles.** Short, metaphorical queries such as "The Off-Switch Game" are missed by both lexical and semantic stages.
- **Uniform trust.** The semantic score is trusted equally on every query, so the average gain hides per-query harm.

## Acknowledgements

Course: Text Mining & NLP, Dr. Mostafa Fakhrahmad, Shiraz University. Dataset: BEIR / SCIDOCS (Thakur et al., SIGIR 2021).

References: Manning, Raghavan & Schütze (2008); Croft, Metzler & Strohman (2010); Nogueira & Cho (2019); Jurafsky & Martin (2023).
