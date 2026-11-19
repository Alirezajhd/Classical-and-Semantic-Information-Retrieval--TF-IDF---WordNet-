"""
Evaluation module (Section 4.5).

Implements MAP, MRR, Precision@k, and NDCG@k from scratch.
Also provides utilities for writing ranking files, metrics, and traces.
"""
import csv
import json
import time
import math
import os
from collections import defaultdict

_LOG_CACHE = [0.0, 0.0] 

def _get_log_discount(rank):
    global _LOG_CACHE
    while len(_LOG_CACHE) <= rank:
        _LOG_CACHE.append(1.0 / math.log2(len(_LOG_CACHE) + 1))
    return _LOG_CACHE[rank]

def precision_at_k(ranked_doc_ids, relevant_set, k):
    if k <= 0 or not ranked_doc_ids:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_set)
    return hits / k

def average_precision(ranked_doc_ids, relevant_set):
    if not relevant_set:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, d in enumerate(ranked_doc_ids, start=1):
        if d in relevant_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant_set)

def reciprocal_rank(ranked_doc_ids, relevant_set):
    for i, d in enumerate(ranked_doc_ids, start=1):
        if d in relevant_set:
            return 1.0 / i
    return 0.0

def dcg_at_k(ranked_doc_ids, relevant_set, k):
    dcg = 0.0
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if d in relevant_set:
            dcg += _get_log_discount(i)
    return dcg

def ndcg_at_k(ranked_doc_ids, relevant_set, k):
    dcg = dcg_at_k(ranked_doc_ids, relevant_set, k)
    if dcg == 0: return 0.0
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(_get_log_discount(i) for i in range(1, ideal_hits + 1))
    return dcg / idcg

METRIC_FUNCS = {
    "MAP": lambda r, rel: average_precision(r, rel),
    "MRR": lambda r, rel: reciprocal_rank(r, rel),
    "P@1": lambda r, rel: precision_at_k(r, rel, 1),
    "P@5": lambda r, rel: precision_at_k(r, rel, 5),
    "P@10": lambda r, rel: precision_at_k(r, rel, 10),
    "NDCG@5": lambda r, rel: ndcg_at_k(r, rel, 5),
    "NDCG@10": lambda r, rel: ndcg_at_k(r, rel, 10),
}

def evaluate_run(run, qrels, metrics=None):
    """Evaluate with configurable metrics."""
    if metrics is None:
        metrics = ["MAP", "MRR", "P@1", "P@5", "P@10", "NDCG@5", "NDCG@10"]
    
    metric_funcs = {name: METRIC_FUNCS[name] for name in metrics if name in METRIC_FUNCS}
    
    per_query = {}
    for qid, ranked in run.items():
        relevant = qrels.get(qid, set())
        per_query[qid] = {name: fn(ranked, relevant) for name, fn in metric_funcs.items()}
    
    if not per_query:
        return {name: 0.0 for name in metric_funcs}, {}
    
    avg = {name: sum(m[name] for m in per_query.values()) / len(per_query)
           for name in metric_funcs}
    return avg, per_query

def build_rankings_rows(run, method_name):
    rows = []
    for qid, ranked in run.items():
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rows.append({
                "query_id": qid, 
                "doc_id": doc_id, 
                "rank": rank,
                "score": score, 
                "method": method_name
            })
    return rows

def write_rankings_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "rank", "score", "method"])
        for r in rows:
            writer.writerow([r["query_id"], r["doc_id"], r["rank"],
                              f"{r['score']:.6f}", r["method"]])

def write_dev_metrics_json(path, dev_metrics):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2)

def write_summary_json(path, summary):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def write_query_trace_with_queries(path, queries, query_tokens_cache, tfidf_run, final_run, trace_extra, preproc_cfg):
    from preprocess import preprocess

    raw_cfg = {
        "lowercase": preproc_cfg.get("lowercase", True), 
        "remove_stopwords": False, 
        "stemming": False, 
        "min_token_length": 1
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, qtext in queries.items():
            raw_tokens = preprocess(qtext, raw_cfg)
            final_tokens = query_tokens_cache.get(qid, [])
            lex_results = tfidf_run.get(qid, [])
            final_results = final_run.get(qid, [])
            extra = trace_extra.get(qid, {"applied": False})
            
            record = {
                "query_id": qid,
                "query_text": qtext,
                "query_tokens_raw": raw_tokens,
                "query_tokens_final": final_tokens,
                "method": "wordnet_rerank" if extra.get("applied") else "tfidf_only",
                "lexical_top10": [{"doc_id": d, "score": round(s, 4), "rank": i + 1} 
                                 for i, (d, s) in enumerate(lex_results[:10])],
                "semantic_step": {
                    "applied": extra.get("applied", False),
                    "type": extra.get("type", "none"),
                    "details": extra.get("details", ""),
                    "alpha_used": extra.get("alpha_used"),
                },
                "final_top10": [{"doc_id": d, "score": round(s, 4), "rank": i + 1} 
                               for i, (d, s) in enumerate(final_results[:10])],
                "summary": {
                    "num_docs_scored_lexically": len(lex_results),
                    "num_docs_semantically_reranked": extra.get("candidate_pool_size", 0),
                    "top10_changed": extra.get("top10_changed", False),
                },
            }

            if extra.get("applied") and "entries" in extra:
                candidates = []
                for e in extra["entries"][:5]:
                    reason = []
                    if e["semantic_score"] > 0: reason.append("semantic match")
                    if e["rank_after"] < e["rank_before"]:
                        reason.append(f"promoted +{e['rank_before'] - e['rank_after']}")
                    
                    candidates.append({
                        "doc_id": e["doc_id"],
                        "lexical_score": round(e["lexical_score"], 4),
                        "semantic_score": round(e["semantic_score"], 4),
                        "combined_score": round(e["combined_score"], 4),
                        "rank_before": e["rank_before"],
                        "rank_after": e["rank_after"],
                        "reason": ", ".join(reason) if reason else "rank stable"
                    })
                record["candidate_documents"] = candidates
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_outputs(outdir, config, dev_runs, test_final_run, dev_metrics, 
                    dev_queries, dev_query_tokens, dev_trace_extra, stats, t0, verbose,
                    test_ltr_run=None):
    
    from preprocess import get_preprocessing_config
    preproc_cfg = get_preprocessing_config(config)
    
    dev_rows = []
    for method in ["random", "overlap", "tfidf", "final", "ltr"]:
        if method in dev_runs:
            dev_rows.extend(build_rankings_rows(dev_runs[method], method))
    write_rankings_tsv(os.path.join(outdir, "dev_rankings.tsv"), dev_rows)

    test_rows = build_rankings_rows(test_final_run, "final")
    if test_ltr_run:
        test_rows.extend(build_rankings_rows(test_ltr_run, "ltr"))
    write_rankings_tsv(os.path.join(outdir, "test_rankings.tsv"), test_rows)

    write_dev_metrics_json(os.path.join(outdir, "dev_metrics.json"), dev_metrics)

    summary = {
        "student_name": config.get("student_name", ""),
        "student_id": config.get("student_id", ""),
        "token_pattern": r"(?u)\b\w\w+\b",
        "preprocessing": config.get("preprocessing", {}),
        "indexing": config.get("indexing", {}),
        "retrieval": config.get("retrieval", {}),
        "semantic": config.get("semantic", {}),
        "ltr": config.get("ltr", {}),
        "evaluation": config.get("evaluation", {}),
        "random_seed": config.get("random_seed", 42),
        "index_stats": stats,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    write_summary_json(os.path.join(outdir, "summary.json"), summary)

    write_query_trace_with_queries(
        os.path.join(outdir, "query_trace.jsonl"),
        dev_queries,
        dev_query_tokens, 
        dev_runs.get("tfidf", {}), 
        dev_runs.get("final", {}), 
        dev_trace_extra,
        preproc_cfg
    )

    os.makedirs(os.path.join(outdir, "extra"), exist_ok=True)
    with open(os.path.join(outdir, "extra", "index_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "sample_posting_lists": stats.get("sample_posting_lists", {}),
            "top10_df_terms": stats.get("top10_df_terms", [])
        }, f, indent=2)

    if verbose:
        print(f"       Wrote all outputs to {outdir}/")