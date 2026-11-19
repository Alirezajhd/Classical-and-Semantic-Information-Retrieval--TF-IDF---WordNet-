"""
Inverted index construction and lookup (Section 4.2).

Builds an inverted index from scratch storing term-document postings,
term frequencies, and document frequencies. Also provides utilities
for loading documents/queries/qrels and caching the index.
"""
import csv
import hashlib
import json
import os
import pickle
from collections import defaultdict, Counter


class InvertedIndex:
    def __init__(self):
        self.postings = defaultdict(dict)
        self.doc_freq = defaultdict(int)
        self.doc_lengths = {}
        self.doc_ids = []
        self.N = 0
        self._total_postings = 0

    def add_document(self, doc_id, tokens_or_counts):
        """
        Supports both raw token lists and pre-weighted frequency Counters.
        Performance optimization: Avoids redundant list concatenation.
        """
        if isinstance(tokens_or_counts, (dict, Counter)):
            tf_counts = tokens_or_counts
            doc_len = sum(tf_counts.values())
        else:
            tf_counts = Counter(tokens_or_counts)
            doc_len = len(tokens_or_counts)

        for term, tf in tf_counts.items():
            if doc_id not in self.postings[term]:
                self.doc_freq[term] += 1
                self._total_postings += 1
            self.postings[term][doc_id] = tf
            
        self.doc_lengths[doc_id] = doc_len
        self.doc_ids.append(doc_id)
        self.N += 1

    def get_postings(self, term):
        return self.postings.get(term, {})

    def get_df(self, term):
        return self.doc_freq.get(term, 0)

    def get_candidate_documents(self, query_tokens):
        candidates = set()
        for term in query_tokens:
            if term in self.postings:
                candidates.update(self.postings[term].keys())
        return candidates

    def vocab(self):
        return self.postings.keys()

    def stats(self):
        return {
            "num_unique_terms": len(self.postings),
            "num_postings": self._total_postings,
            "num_docs": self.N,
            "avg_doc_length": sum(self.doc_lengths.values()) / self.N if self.N > 0 else 0
        }

    def get_sample_posting_lists(self, n=3):
        samples = {}
        sample_terms = list(self.postings.keys())[:n]
        for term in sample_terms:
            samples[term] = dict(list(self.postings[term].items())[:10])
        return samples

    def top_df_terms(self, n=10):
        return sorted(self.doc_freq.items(), key=lambda x: x[1], reverse=True)[:n]

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def load_documents(path):
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs[str(obj["doc_id"])] = obj
    return docs


def load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            queries[str(row["query_id"])] = row["query_text"]
    return queries


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["relevance"]) > 0:
                qrels[str(row["query_id"])].add(str(row["doc_id"]))
    return qrels


def build_index(docs, preproc_cfg, verbose=True, title_weight=6.0, outdir="results"):
    """
    Builds the required inverted index (doc-id postings + raw term frequency per
    section 4.2) plus a raw-token map used by the WordNet semantic module.
    """
    from preprocess import preprocess

    cache_dir = os.path.join(outdir, "extra")
    index_path = os.path.join(cache_dir, "index.pkl")
    cfg_fingerprint_path = os.path.join(cache_dir, "index_cfg.json")
    title_map_path = os.path.join(cache_dir, "title_term_counts.pkl")
    raw_map_path = os.path.join(cache_dir, "raw_tokens_map.pkl")

    doc_id_hash = hashlib.md5("|".join(sorted(docs.keys())).encode("utf-8")).hexdigest()
    fingerprint_payload = {
        "preproc": preproc_cfg,
        "num_docs": len(docs),
        "doc_id_hash": doc_id_hash,
    }
    cfg_fingerprint = json.dumps(fingerprint_payload, sort_keys=True)

    cache_valid = (
        os.path.exists(index_path)
        and os.path.exists(cfg_fingerprint_path)
        and os.path.exists(title_map_path)
        and os.path.exists(raw_map_path)
        and open(cfg_fingerprint_path, encoding="utf-8").read() == cfg_fingerprint
    )

    if cache_valid:
        if verbose:
            print(f"       Loading cached index (config + corpus unchanged)...")
        with open(title_map_path, "rb") as f:
            title_term_counts = pickle.load(f)
        with open(raw_map_path, "rb") as f:
            raw_tokens_map = pickle.load(f)
        return InvertedIndex.load(index_path), raw_tokens_map, title_term_counts

    index = InvertedIndex()
    raw_tokens_map = {}
    title_term_counts = {}

    raw_cfg = {
        "lowercase": preproc_cfg.get("lowercase", True), 
        "remove_stopwords": False, 
        "stemming": False, 
        "min_token_length": 1
    }
    raw_vocab = set()

    for doc_id, doc in docs.items():
        title = doc.get("title") or ""
        text = doc.get("text") or ""
        
        raw_t = preprocess(f"{title} {text}", raw_cfg)
        raw_tokens_map[doc_id] = raw_t
        raw_vocab.update(raw_t)

        title_tokens = preprocess(title, preproc_cfg) if title else []
        body_tokens = preprocess(text, preproc_cfg) if text else []

        doc_counts = Counter(body_tokens)
        doc_counts.update(title_tokens)
        index.add_document(doc_id, doc_counts)

        if title_tokens:
            title_term_counts[doc_id] = Counter(title_tokens)

    os.makedirs(cache_dir, exist_ok=True)
    index.save(index_path)
    with open(cfg_fingerprint_path, "w", encoding="utf-8") as f:
        f.write(cfg_fingerprint)
    with open(title_map_path, "wb") as f:
        pickle.dump(title_term_counts, f)
    with open(raw_map_path, "wb") as f:
        pickle.dump(raw_tokens_map, f)

    if verbose:
        print(f"       Vocab Size - Before: {len(raw_vocab)} | After: {len(index.vocab())}")

    return index, raw_tokens_map, title_term_counts