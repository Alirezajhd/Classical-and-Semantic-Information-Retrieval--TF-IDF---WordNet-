"""
Learning to Rank bonus module (Section 5).

Implements pointwise ridge regression LTR with k-fold cross-validation over queries.
"""
import math
import random

try:
    import numpy as _np
except ImportError:
    _np = None


class RidgeLTR:
    """Minimal pointwise LTR regressor: standardized features + closed-form
    ridge regression (bias term left unregularized). Pure NumPy, no sklearn."""

    def __init__(self, lam=1.0):
        self.lam = lam
        self.mean_ = None
        self.std_ = None
        self.theta_ = None

    def fit(self, X, y):
        X = _np.asarray(X, dtype=float)
        y = _np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        Xb = _np.hstack([Xs, _np.ones((Xs.shape[0], 1))])
        n_feat = Xb.shape[1]
        reg = self.lam * _np.eye(n_feat)
        reg[-1, -1] = 0.0  # don't regularize the bias term
        self.theta_ = _np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        return self

    def predict(self, X):
        X = _np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.std_
        Xb = _np.hstack([Xs, _np.ones((Xs.shape[0], 1))])
        return Xb @ self.theta_


def ltr_feature_vector(tfidf_score, semantic_score, doc_length, overlap_count, coverage, idf_sum, query_len, title_overlap):
    return [tfidf_score, semantic_score, math.log1p(max(doc_length, 0)),
            overlap_count, coverage, idf_sum, query_len, title_overlap]


def ltr_title_overlap(title_term_counts, query_tokens_final, doc_id):
    title_terms = title_term_counts.get(doc_id)
    if not title_terms:
        return 0
    return sum(1 for t in set(query_tokens_final) if t in title_terms)


def ltr_term_overlap(index, query_tokens_final, doc_id):
    count = 0
    for t in set(query_tokens_final):
        if doc_id in index.get_postings(t):
            count += 1
    return count


def ltr_build_rows(queries, tfidf_run, raw_tokens_map, preproc_cfg, semantic_cfg,
                    index, query_tokens_cache, doc_words_lookup, rerank_k, idf_lookup=None,
                    title_term_counts=None):
    from preprocess import preprocess
    from semantic import get_content_words, semantic_score_detailed

    rows = []
    similarity_metric = semantic_cfg.get("similarity_metric", "wup")
    semantic_preproc = {
        "lowercase": preproc_cfg.get("lowercase", True),
        "remove_stopwords": False,
        "stemming": False,
        "min_token_length": 1,
    }
    for qid, qtext in queries.items():
        pool = tfidf_run.get(qid, [])[:rerank_k]
        if not pool:
            continue
        raw_q_tokens = preprocess(qtext, semantic_preproc)
        q_words = get_content_words(raw_q_tokens)
        q_tokens_final = query_tokens_cache.get(qid, [])
        query_len = len(q_tokens_final)
        idf_sum = sum(idf_lookup(t) for t in set(q_tokens_final)) if idf_lookup else 0.0
        for doc_id, tfidf_score in pool:
            d_words = doc_words_lookup(doc_id)
            sem_score, _ = semantic_score_detailed(q_words, d_words, similarity_metric)
            doc_len = index.doc_lengths.get(doc_id, 0)
            overlap = ltr_term_overlap(index, q_tokens_final, doc_id)
            coverage = overlap / max(1, query_len)
            title_overlap = ltr_title_overlap(title_term_counts, q_tokens_final, doc_id) if title_term_counts else 0
            feat = ltr_feature_vector(tfidf_score, sem_score, doc_len, overlap, coverage, idf_sum, query_len, title_overlap)
            rows.append((qid, doc_id, feat))
    return rows


def run_ltr(dev_queries, dev_qrels, test_queries, dev_tfidf_run, test_tfidf_run,
            raw_tokens_map, preproc_cfg, semantic_cfg, index,
            dev_query_tokens, test_query_tokens, rerank_k=20, lam=1.0,
            n_folds=5, seed=42, idf_lookup=None, title_term_counts=None):
    """Returns (dev_ltr_run, test_ltr_run, trained_final_model)."""
    if _np is None:
        print("       [LTR] numpy not available -- skipping LTR bonus.")
        return {qid: ranked for qid, ranked in dev_tfidf_run.items()}, \
               {qid: ranked for qid, ranked in test_tfidf_run.items()}, None

    from semantic import get_content_words

    doc_words_cache = {}
    def doc_words_lookup(doc_id):
        if doc_id not in doc_words_cache:
            doc_words_cache[doc_id] = get_content_words(raw_tokens_map.get(doc_id, []))
        return doc_words_cache[doc_id]

    dev_rows = ltr_build_rows(dev_queries, dev_tfidf_run, raw_tokens_map, preproc_cfg,
                               semantic_cfg, index, dev_query_tokens, doc_words_lookup, rerank_k,
                               idf_lookup=idf_lookup, title_term_counts=title_term_counts)

    def label_of(qid, doc_id):
        return 1.0 if doc_id in dev_qrels.get(qid, set()) else 0.0

    qids = list(dev_queries.keys())
    rnd = random.Random(seed)
    rnd.shuffle(qids)
    n_folds = max(2, min(n_folds, len(qids))) if qids else 1
    folds = [qids[i::n_folds] for i in range(n_folds)]
    fold_of_qid = {qid: i for i, fold in enumerate(folds) for qid in fold}

    oof_scores = {}
    for i in range(n_folds):
        train_rows = [r for r in dev_rows if fold_of_qid.get(r[0]) != i]
        heldout_rows = [r for r in dev_rows if fold_of_qid.get(r[0]) == i]
        if not train_rows or not heldout_rows:
            continue
        X_train = [r[2] for r in train_rows]
        y_train = [label_of(r[0], r[1]) for r in train_rows]
        model = RidgeLTR(lam=lam).fit(X_train, y_train)
        X_ho = [r[2] for r in heldout_rows]
        preds = model.predict(X_ho)
        for (qid, doc_id, _), pred in zip(heldout_rows, preds):
            oof_scores[(qid, doc_id)] = float(pred)

    dev_ltr_run = {}
    for qid in dev_queries:
        ranked = dev_tfidf_run.get(qid, [])
        pool, tail = ranked[:rerank_k], ranked[rerank_k:]
        scored_pool = [(doc_id, oof_scores.get((qid, doc_id), tfidf_score))
                       for doc_id, tfidf_score in pool]
        scored_pool.sort(key=lambda x: -x[1])
        dev_ltr_run[qid] = scored_pool + tail

    final_model = None
    test_ltr_run = {qid: ranked for qid, ranked in test_tfidf_run.items()}
    if dev_rows:
        X_all = [r[2] for r in dev_rows]
        y_all = [label_of(r[0], r[1]) for r in dev_rows]
        final_model = RidgeLTR(lam=lam).fit(X_all, y_all)

        test_rows = ltr_build_rows(test_queries, test_tfidf_run, raw_tokens_map, preproc_cfg,
                                     semantic_cfg, index, test_query_tokens, doc_words_lookup, rerank_k,
                                     idf_lookup=idf_lookup, title_term_counts=title_term_counts)
        test_preds = {}
        if test_rows:
            X_test = [r[2] for r in test_rows]
            preds = final_model.predict(X_test)
            for (qid, doc_id, _), pred in zip(test_rows, preds):
                test_preds[(qid, doc_id)] = float(pred)

        test_ltr_run = {}
        for qid in test_queries:
            ranked = test_tfidf_run.get(qid, [])
            pool, tail = ranked[:rerank_k], ranked[rerank_k:]
            scored_pool = [(doc_id, test_preds.get((qid, doc_id), tfidf_score))
                           for doc_id, tfidf_score in pool]
            scored_pool.sort(key=lambda x: -x[1])
            test_ltr_run[qid] = scored_pool + tail

    return dev_ltr_run, test_ltr_run, final_model