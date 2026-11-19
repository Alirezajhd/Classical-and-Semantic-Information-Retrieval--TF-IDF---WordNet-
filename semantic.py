"""
Semantic retrieval module (Section 4.4).

Implements WordNet-based semantic reranking using Wu-Palmer similarity
(Equation 1, 2, 3) as described in the assignment.
"""
import math
import os
import pickle
import time
from collections import defaultdict
from preprocess import preprocess

_DEFAULT_CONFIG = {
    "semantic": {
        "enabled": True,
        "mode": "rerank",
        "alpha": 0.8,
        "rerank_k": 20,
        "max_synsets_per_word": 3,
        "max_doc_content_words": 60,
        "pos_tags": ["NOUN", "VERB"],
        "similarity_metric": "wup",
        "cache_enabled": True,
        "cache_path": "results/extra/wordnet_sim_cache.pkl",
        "alpha_mode": "fixed",
        "alpha_min": 0.5,
        "alpha_max": 0.95
    }
}

_CONFIG = _DEFAULT_CONFIG

def set_config(cfg):
    global _CONFIG
    _CONFIG = cfg

def get_config():
    """Get semantic configuration with proper fallback for all parameters."""
    semantic_cfg = _CONFIG.get("semantic", {})
    
    if isinstance(_CONFIG, dict) and "semantic" not in _CONFIG:
        semantic_cfg = {
            "enabled": _CONFIG.get("semantic_enabled", True),
            "mode": _CONFIG.get("semantic_mode", "rerank"),
            "alpha": _CONFIG.get("alpha", 0.8),
            "rerank_k": _CONFIG.get("rerank_k", 20),
            "max_synsets_per_word": _CONFIG.get("max_synsets_per_word", 3),
            "max_doc_content_words": _CONFIG.get("max_doc_content_words", 60),
            "pos_tags": _CONFIG.get("pos_tags", ["NOUN", "VERB"]),
            "similarity_metric": _CONFIG.get("similarity_metric", "wup"),
            "cache_enabled": _CONFIG.get("cache_enabled", True),
            "cache_path": _CONFIG.get("cache_path", "results/extra/wordnet_sim_cache.pkl"),
            "alpha_mode": _CONFIG.get("alpha_mode", "fixed"),
            "alpha_min": _CONFIG.get("alpha_min", 0.5),
            "alpha_max": _CONFIG.get("alpha_max", 0.95)
        }
    else:
        semantic_cfg.setdefault("alpha_mode", "fixed")
        semantic_cfg.setdefault("alpha_min", 0.5)
        semantic_cfg.setdefault("alpha_max", 0.95)
        semantic_cfg.setdefault("similarity_metric", "wup")
        semantic_cfg.setdefault("pos_tags", ["NOUN", "VERB"])
    
    return semantic_cfg

def get_max_synsets(): return get_config().get("max_synsets_per_word", 3)
def get_max_doc_words(): return get_config().get("max_doc_content_words", 60)
def get_rerank_k(): return get_config().get("rerank_k", 20)
def get_alpha(): return get_config().get("alpha", 0.8)
def get_pos_tags(): return set(get_config().get("pos_tags", ["NOUN", "VERB"]))
def get_cache_path():
    path = get_config().get("cache_path", "results/extra/wordnet_sim_cache.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
def is_cache_enabled(): return get_config().get("cache_enabled", True)

_wn = None
_pos_tag = None
_nltk_ready = False
_synset_cache = {}
_pair_sim_cache = {}
_cache_dirty = False

def _ensure_nltk():
    global _wn, _pos_tag, _nltk_ready
    if _nltk_ready: return
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        from nltk import pos_tag
        _wn, _pos_tag = wn, pos_tag
        _nltk_ready = True
    except ImportError:
        _nltk_ready = False
        return
    if is_cache_enabled(): 
        _load_disk_cache()

def _load_disk_cache():
    global _pair_sim_cache
    path = get_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            _pair_sim_cache.update(data.get("pair_sim", {}))
        except:
            pass

def save_disk_cache():
    if not _cache_dirty or not is_cache_enabled(): return
    try:
        with open(get_cache_path(), "wb") as f:
            pickle.dump({"pair_sim": _pair_sim_cache}, f)
    except:
        pass

def get_content_words(raw_tokens):
    _ensure_nltk()
    if not raw_tokens or not _nltk_ready: 
        return []
    
    allowed = get_pos_tags()
    tagged = _pos_tag(raw_tokens)
    seen, result = set(), []
    
    for word, tag in tagged:
        pos = None
        if "NOUN" in allowed and tag.startswith("NN"): 
            pos = _wn.NOUN
        elif "VERB" in allowed and tag.startswith("VB"): 
            pos = _wn.VERB
            
        if pos:
            key = (word, pos)
            if key not in seen:
                seen.add(key)
                result.append(key)
                
    return result[:get_max_doc_words()]

def _get_synsets_with_morphy(word, pos):
    key = (word, pos)
    if key in _synset_cache: 
        return _synset_cache[key]
    
    syns = _wn.synsets(word, pos=pos)
    if not syns:
        base = _wn.morphy(word, pos)
        if base: 
            syns = _wn.synsets(base, pos=pos)

    res = tuple(syns[:get_max_synsets()])
    _synset_cache[key] = res
    return res

def _pair_similarity(w1, pos1, w2, pos2, similarity_metric="wup"):
    if w1 == w2: 
        return 1.0
    
    a, b = (w1, pos1), (w2, pos2)
    if b < a: 
        a, b = b, a
    key = (a[0], a[1], b[0], b[1], similarity_metric)
    
    if is_cache_enabled() and key in _pair_sim_cache:
        return _pair_sim_cache[key]

    syns1 = _get_synsets_with_morphy(*a)
    syns2 = _get_synsets_with_morphy(*b)
    
    best = 0.0
    for s1 in syns1:
        for s2 in syns2:
            if similarity_metric == "wup":
                sim = s1.wup_similarity(s2)
            else:
                sim = s1.path_similarity(s2)
            
            if sim and sim > best: 
                best = sim
                if best >= 0.9: break
        if best >= 0.9: break
    
    if is_cache_enabled():
        global _cache_dirty
        _pair_sim_cache[key] = best
        _cache_dirty = True
    return best

def semantic_score_detailed(q_words, d_words, similarity_metric="wup"):
    """
    Strict implementation of Equation (2).
    Calculates simple average of maximum similarities.
    """
    if not q_words or not d_words:
        return 0.0, "No overlapping content words."

    d_word_set = {w for w, _ in d_words}

    total_max_sim = 0.0
    significant_pairs = []

    for qw, qpos in q_words:
        best_sim = 0.0
        best_match = None
        if qw in d_word_set:
            best_sim = 1.0
            best_match = qw
        else:
            for dw, dpos in d_words:
                if qpos == dpos:
                    sim = _pair_similarity(qw, qpos, dw, dpos, similarity_metric)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = dw
                    if best_sim >= 1.0: break

        total_max_sim += best_sim
        if best_sim > 0.6:
            significant_pairs.append(f"'{qw}'~'{best_match}'")

    avg_score = total_max_sim / len(q_words)
    
    reason = f"Key matches: {', '.join(significant_pairs[:2])}" if significant_pairs else "Low semantic overlap."
    return avg_score, reason

def compute_dynamic_alpha(pool_scores, alpha_min=0.5, alpha_max=0.95):
    if not pool_scores or len(pool_scores) < 2:
        return alpha_max
    top = pool_scores[0]
    rest = pool_scores[1:]
    if top <= 0:
        return alpha_min
    confidence = max(0.0, min(1.0, (top - sum(rest) / len(rest)) / top))
    return alpha_min + (alpha_max - alpha_min) * confidence

def rerank(lexical_results, query_words, doc_words_lookup, alpha=None, top_k=None,
           similarity_metric="wup", normalize=True):
    """
    Implementation of Equation (3): Final = alpha * Lexical + (1 - alpha) * Semantic
    """
    top_k = top_k if top_k is not None else get_rerank_k()
    alpha = alpha if alpha is not None else get_alpha()
    pool = lexical_results[:top_k]
    if not pool: return []

    raw_entries = []
    for rank_before, (doc_id, lex_score) in enumerate(pool, start=1):
        d_words = doc_words_lookup(doc_id)
        sem_score, reason = semantic_score_detailed(query_words, d_words, similarity_metric)
        raw_entries.append({
            "doc_id": doc_id,
            "lexical_score": lex_score,
            "semantic_score": sem_score,
            "rank_before": rank_before,
            "reason": reason,
        })

    if normalize:
        lex_vals = [e["lexical_score"] for e in raw_entries]
        sem_vals = [e["semantic_score"] for e in raw_entries]
        lo_l, hi_l = min(lex_vals), max(lex_vals)
        lo_s, hi_s = min(sem_vals), max(sem_vals)
        span_l = (hi_l - lo_l) or 1.0
        span_s = (hi_s - lo_s) or 1.0
    else:
        lo_l = lo_s = 0.0
        span_l = span_s = 1.0

    entries = []
    for e in raw_entries:
        norm_lex = (e["lexical_score"] - lo_l) / span_l
        norm_sem = (e["semantic_score"] - lo_s) / span_s
        combined = (alpha * norm_lex) + ((1 - alpha) * norm_sem)
        entries.append({
            "doc_id": e["doc_id"],
            "lexical_score": e["lexical_score"],
            "semantic_score": e["semantic_score"],
            "combined_score": combined,
            "rank_before": e["rank_before"],
            "reason": e["reason"],
        })

    entries.sort(key=lambda e: (-e["combined_score"], e["rank_before"]))
    for rank_after, e in enumerate(entries, start=1):
        e["rank_after"] = rank_after
    return entries


def run_semantic_final(queries, tfidf_run, raw_tokens_map, preproc_cfg, semantic_cfg, config, doc_words_cache=None):
    from semantic import get_content_words, rerank, compute_dynamic_alpha

    if doc_words_cache is None:
        doc_words_cache = {}

    def doc_words_lookup(doc_id):
        if doc_id not in doc_words_cache:
            doc_words_cache[doc_id] = get_content_words(raw_tokens_map.get(doc_id, []))
        return doc_words_cache[doc_id]
    
    final_run = {}
    trace_extra = {}
    alpha = semantic_cfg.get("alpha", 0.8)
    rerank_k = semantic_cfg.get("rerank_k", 20)
    similarity_metric = semantic_cfg.get("similarity_metric", "wup")
    normalize_scores = semantic_cfg.get("normalize_scores", True)
    alpha_mode = semantic_cfg.get("alpha_mode", "fixed")
    alpha_min = semantic_cfg.get("alpha_min", 0.5)
    alpha_max = semantic_cfg.get("alpha_max", 0.95)
    
    total = len(queries)
    t_start = time.time()
    n_done = 0
    
    for qid, qtext in queries.items():
        lexical_results = tfidf_run.get(qid, [])
        semantic_preproc = {
            "lowercase": preproc_cfg.get("lowercase", True),
            "remove_stopwords": False, 
            "stemming": False, 
            "min_token_length": 1
        }
        raw_q_tokens = preprocess(qtext, semantic_preproc)
        q_words = get_content_words(raw_q_tokens)

        if alpha_mode == "dynamic":
            pool_scores = [s for _, s in lexical_results[:rerank_k]]
            query_alpha = compute_dynamic_alpha(pool_scores, alpha_min, alpha_max)
        else:
            query_alpha = alpha

        reranked_entries = rerank(
            lexical_results, q_words, doc_words_lookup, 
            alpha=query_alpha, 
            top_k=rerank_k,
            similarity_metric=similarity_metric,
            normalize=normalize_scores,
        )
        
        reranked_ids = {e["doc_id"] for e in reranked_entries}
        tail = [(d, s) for d, s in lexical_results if d not in reranked_ids]
        new_order = [(e["doc_id"], e["combined_score"]) for e in reranked_entries] + tail
        final_run[qid] = new_order
        
        top10_before = [d for d, _ in lexical_results[:10]]
        top10_after = [d for d, _ in new_order[:10]]
        trace_extra[qid] = {
            "applied": True, 
            "type": "rerank", 
            "entries": reranked_entries,
            "candidate_pool_size": len(lexical_results[:rerank_k]),
            "top10_changed": top10_before != top10_after,
            "alpha_used": round(query_alpha, 4),
        }
        
        n_done += 1
        if n_done % 25 == 0 or n_done == total:
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            remaining = (total - n_done) / rate if rate > 0 else 0
            print(f"       ... {n_done}/{total} queries ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, {len(doc_words_cache)} unique docs POS-tagged so far)")
            
    return final_run, trace_extra