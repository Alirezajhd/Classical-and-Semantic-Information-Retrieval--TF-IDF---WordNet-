"""
Lexical retrieval ranking (Section 4.3).

Implements TF-IDF ranking with cosine similarity from scratch,
plus overlap (boolean) and random baselines.
"""
import math
import random
from collections import defaultdict, Counter


class TfidfRanker:
    def __init__(self, index, tf_weight="sublinear", idf_weight="smooth",
                 title_term_counts=None, title_weight=1.0):
        self.index = index
        self.tf_weight = tf_weight
        self.idf_weight = idf_weight
        self.idf = {}
        self.doc_norms = {}
        self.weighted_postings = defaultdict(dict)
        self.title_term_counts = title_term_counts or {}
        self.title_weight = title_weight
        self._build()

    def _effective_tf(self, term, doc_id, raw_tf):
        """Raw TF plus an additive title boost, used only for scoring."""
        if self.title_weight == 1.0 or not self.title_term_counts:
            return raw_tf
        title_tf = self.title_term_counts.get(doc_id, {}).get(term, 0)
        if not title_tf:
            return raw_tf
        return raw_tf + title_tf * (self.title_weight - 1)

    def _build(self):
        N = max(self.index.N, 1)
        for term in self.index.vocab():
            df = self.index.get_df(term)
            if self.idf_weight == "smooth":
                self.idf[term] = math.log((N + 1) / (df + 1)) + 1.0 if df > 0 else 0.0
            elif self.idf_weight == "probabilistic":
                self.idf[term] = math.log((N - df) / df) if df > 0 and df < N else 0.0
            else:  # raw
                self.idf[term] = math.log(N / df) if df > 0 else 0.0

        norms = defaultdict(float)
        for term in self.index.vocab():
            idf = self.idf[term]
            if idf == 0: continue
            
            postings = self.index.get_postings(term)
            for doc_id, raw_tf in postings.items():
                tf = self._effective_tf(term, doc_id, raw_tf)
                if self.tf_weight == "sublinear":
                    weighted_val = (1 + math.log(tf)) * idf
                elif self.tf_weight == "binary":
                    weighted_val = (1 if tf > 0 else 0) * idf
                elif self.tf_weight == "raw":
                    weighted_val = tf * idf
                else:  # sublinear default
                    weighted_val = (1 + math.log(tf)) * idf
                    
                self.weighted_postings[term][doc_id] = weighted_val
                norms[doc_id] += weighted_val * weighted_val
        
        self.doc_norms = {doc_id: math.sqrt(v) for doc_id, v in norms.items()}

    def _query_weights(self, query_tokens):
        tf_counts = Counter(query_tokens)
        weights = {}
        for term, tf in tf_counts.items():
            if term in self.idf:
                if self.tf_weight == "sublinear":
                    weights[term] = (1 + math.log(tf)) * self.idf[term]
                elif self.tf_weight == "binary":
                    weights[term] = (1 if tf > 0 else 0) * self.idf[term]
                elif self.tf_weight == "raw":
                    weights[term] = tf * self.idf[term]
                else:
                    weights[term] = (1 + math.log(tf)) * self.idf[term]
        return weights

    def score(self, query_tokens, topk=100):
        q_weights = self._query_weights(query_tokens)
        if not q_weights:
            return []
            
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0:
            return []

        dot = defaultdict(float)
        for term, qw in q_weights.items():
            term_postings = self.weighted_postings.get(term, {})
            for doc_id, dw in term_postings.items():
                dot[doc_id] += qw * dw

        results = []
        for doc_id, dot_product in dot.items():
            dnorm = self.doc_norms.get(doc_id, 0.0)
            if dnorm == 0:
                continue
            
            cos = dot_product / (q_norm * dnorm)
            results.append((doc_id, cos))

        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:topk]

    def candidate_doc_ids(self, query_tokens):
        cands = set()
        for term in set(query_tokens):
            cands.update(self.index.get_postings(term).keys())
        return cands


class OverlapRanker:
    def __init__(self, index):
        self.index = index

    def score(self, query_tokens, topk=100):
        q_terms = set(query_tokens)
        overlap = defaultdict(int)
        for term in q_terms:
            for doc_id in self.index.get_postings(term).keys():
                overlap[doc_id] += 1
        
        results = sorted(overlap.items(), key=lambda x: (-x[1], x[0]))
        return results[:topk]


class RandomRanker:
    def __init__(self, index, seed=42):
        self.index = index
        self.seed = seed

    def score(self, query_tokens, topk=100, query_id=""):
        cands = list(self.index.get_candidate_documents(query_tokens))
        rnd = random.Random(f"{self.seed}-{query_id}")
        rnd.shuffle(cands)
        
        n = len(cands)
        return [(doc_id, float(n - i)) for i, doc_id in enumerate(cands[:topk])]