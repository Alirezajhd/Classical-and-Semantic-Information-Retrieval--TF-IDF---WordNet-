#!/usr/bin/env python3
"""
Shiraz University - Text Mining & NLP Assignment 2
Main entry point. Implements CLI and orchestrates the full pipeline.
"""
import argparse
import os
import random
import time
import json

from config import load_config, set_config
from preprocess import get_preprocessing_config
from index import build_index, load_documents
from ranker import TfidfRanker, OverlapRanker, RandomRanker
from semantic import run_semantic_final
from evaluator import evaluate_run, write_outputs
from ltr import run_ltr


def run_full(args, config, docs, dev_queries, dev_qrels, test_queries, index, raw_tokens_map, title_term_counts=None):
    preproc_cfg = get_preprocessing_config(config)
    retrieval_cfg = config.get("retrieval", {})
    semantic_cfg = config.get("semantic", {})
    output_cfg = config.get("output", {})
    eval_cfg = config.get("evaluation", {})
    outdir = output_cfg.get("outdir", "results")
    eval_metrics = eval_cfg.get("metrics", ["MAP", "MRR", "P@1", "P@5", "P@10", "NDCG@5", "NDCG@10"])
    
    t0 = time.time()
    
    # Pass TF/IDF weighting options to ranker
    tf_weight = retrieval_cfg.get("tf_weight", "sublinear")
    idf_weight = retrieval_cfg.get("idf_weight", "smooth")
    title_weight = config.get("indexing", {}).get("title_weight", 1.0)
    ranker = TfidfRanker(index, tf_weight=tf_weight, idf_weight=idf_weight,
                          title_term_counts=title_term_counts, title_weight=title_weight)
    overlap_ranker = OverlapRanker(index)
    random_ranker = RandomRanker(index, seed=config["random_seed"])
    
    print("[4/6] Running baselines (random, overlap) and TF-IDF on dev set ...")
    topk = retrieval_cfg.get("topk", 100)
    
    # Build runs
    dev_runs = {"random": {}, "overlap": {}, "tfidf": {}}
    dev_query_tokens = {}
    for qid, qtext in dev_queries.items():
        from preprocess import preprocess
        q_tokens = preprocess(qtext, preproc_cfg)
        dev_query_tokens[qid] = q_tokens
        dev_runs["random"][qid] = random_ranker.score(q_tokens, topk=topk, query_id=qid)
        dev_runs["overlap"][qid] = overlap_ranker.score(q_tokens, topk=topk)
        dev_runs["tfidf"][qid] = ranker.score(q_tokens, topk=topk)
    
    if semantic_cfg.get("enabled", True):
        current_alpha = semantic_cfg.get('alpha', 0.8)
        print(f"[5/6] Running semantic step (alpha={current_alpha}) on dev set ...")
        
        doc_words_cache = {}
        from semantic import run_semantic_final
        dev_final_run, dev_trace_extra = run_semantic_final(
            dev_queries, dev_runs["tfidf"], raw_tokens_map, preproc_cfg, 
            semantic_cfg, config, doc_words_cache=doc_words_cache)
        dev_runs["final"] = dev_final_run
    else:
        dev_runs["final"] = dev_runs["tfidf"]
        dev_trace_extra = {}
    
    print("[6/6] Running final system on test set & writing outputs ...")
    test_runs = {"random": {}, "overlap": {}, "tfidf": {}}
    test_query_tokens = {}
    for qid, qtext in test_queries.items():
        from preprocess import preprocess
        q_tokens = preprocess(qtext, preproc_cfg)
        test_query_tokens[qid] = q_tokens
        test_runs["random"][qid] = random_ranker.score(q_tokens, topk=topk, query_id=qid)
        test_runs["overlap"][qid] = overlap_ranker.score(q_tokens, topk=topk)
        test_runs["tfidf"][qid] = ranker.score(q_tokens, topk=topk)
    
    if semantic_cfg.get("enabled", True):
        test_doc_words_cache = {}
        test_final_run, _ = run_semantic_final(
            test_queries, test_runs["tfidf"], raw_tokens_map, preproc_cfg, 
            semantic_cfg, config, doc_words_cache=test_doc_words_cache)
    else:
        test_final_run = test_runs["tfidf"]

    # --- Bonus Section 5: Learning to Rank ---
    test_ltr_run = None
    ltr_cfg = config.get("ltr", {})
    if ltr_cfg.get("enabled", False):
        print("[bonus] Training pointwise LTR (ridge) with k-fold CV over dev queries ...")
        dev_ltr_run, test_ltr_run, _ = run_ltr(
            dev_queries, dev_qrels, test_queries,
            dev_runs["tfidf"], test_runs["tfidf"],
            raw_tokens_map, preproc_cfg, semantic_cfg, index,
            dev_query_tokens, test_query_tokens,
            rerank_k=ltr_cfg.get("rerank_k", 20),
            lam=ltr_cfg.get("lambda", 1.0),
            n_folds=ltr_cfg.get("cv_folds", 5),
            seed=config.get("random_seed", 42),
            idf_lookup=lambda t: ranker.idf.get(t, 0.0),
            title_term_counts=title_term_counts,
        )
        dev_runs["ltr"] = dev_ltr_run

    print("       Evaluating dev runs ...")
    dev_metrics = {}
    for method in ["random", "overlap", "tfidf", "final", "ltr"]:
        if method in dev_runs:
            doc_ids_only = {qid: [d for d, _ in ranked] for qid, ranked in dev_runs[method].items()}
            avg, _ = evaluate_run(doc_ids_only, dev_qrels, metrics=eval_metrics)
            dev_metrics[method] = {k: round(v, 4) for k, v in avg.items()}
            print(f"         {method:8s} -> {dev_metrics[method]}")

    stats = index.stats()
    stats["sample_posting_lists"] = index.get_sample_posting_lists()
    stats["top10_df_terms"] = index.top_df_terms()
    
    from evaluator import write_outputs
    write_outputs(outdir, config, dev_runs, test_final_run, dev_metrics, 
              dev_queries, dev_query_tokens, dev_trace_extra, stats, t0, True,
              test_ltr_run=test_ltr_run)


def run_alpha_sweep(args, config, docs, dev_queries, dev_qrels, raw_tokens_map, index, title_term_counts=None):
    set_config(config)
    
    print("[Sweep] Starting Alpha Grid Sweep...")
    results = []
    retrieval_cfg = config.get("retrieval", {})
    tf_weight = retrieval_cfg.get("tf_weight", "sublinear")
    idf_weight = retrieval_cfg.get("idf_weight", "smooth")
    title_weight = config.get("indexing", {}).get("title_weight", 1.0)
    from ranker import TfidfRanker
    ranker = TfidfRanker(index, tf_weight=tf_weight, idf_weight=idf_weight,
                          title_term_counts=title_term_counts, title_weight=title_weight)
    from preprocess import get_preprocessing_config, preprocess
    preproc_cfg = get_preprocessing_config(config)
    
    print("       Computing TF-IDF run once...")
    tfidf_run = {}
    for qid, qtext in dev_queries.items():
        tfidf_run[qid] = ranker.score(preprocess(qtext, preproc_cfg))

    alpha_values = config["analysis"].get("alpha_values", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    from evaluator import evaluate_run
    from semantic import run_semantic_final
    eval_metrics = config.get("evaluation", {}).get("metrics", ["MAP", "NDCG@10"])
    
    for alpha in alpha_values:
        print(f"       Testing Alpha={alpha}...")
        config["semantic"]["alpha"] = alpha
        
        final_run, _ = run_semantic_final(
            dev_queries, tfidf_run, raw_tokens_map, 
            preproc_cfg, config["semantic"], config,
            doc_words_cache={}
        )
        
        doc_ids_only = {qid: [d for d, _ in ranked] for qid, ranked in final_run.items()}
        metrics, _ = evaluate_run(doc_ids_only, dev_qrels, metrics=eval_metrics)
        
        result = {"alpha": alpha}
        for metric_name in eval_metrics:
            if metric_name in metrics:
                result[metric_name] = metrics[metric_name]
        results.append(result)

    import json
    import os
    sweep_path = os.path.join(config["output"]["outdir"], "extra", "alpha_sweep.json")
    os.makedirs(os.path.dirname(sweep_path), exist_ok=True)
    with open(sweep_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    best_alpha = max(results, key=lambda x: x.get("MAP", 0))
    print(f"[Sweep] Best Alpha found: {best_alpha['alpha']} (MAP: {best_alpha.get('MAP', 0):.4f})")


def run_title_weight_sweep(args, config, docs, dev_queries, dev_qrels, index, title_term_counts=None):
    """
    IMPROVEMENT (optional, not required by spec): title_weight is a
    ranking-quality hyperparameter (how much to trust title matches over body
    matches) that was previously just guessed (15.0, then retuned to 3.0 by
    hand). This sweep tries a small grid on the DEV set and reports MAP/NDCG@10
    for each, the same way the required alpha sweep already does for alpha --
    so the choice going into the final config.json is backed by dev-set
    evidence instead of a guess, which is exactly the kind of design-choice
    justification section 6 (Experimental Settings) and section 9 (ablation)
    ask for. Reuses the already-built index (title_weight no longer affects
    the raw index -- see build_index's docstring), so this is cheap: it only
    rebuilds the small TfidfRanker weighting for each candidate value.
    """
    print("[TitleWeightSweep] Starting Title Weight Sweep...")
    from preprocess import get_preprocessing_config, preprocess
    preproc_cfg = get_preprocessing_config(config)
    retrieval_cfg = config.get("retrieval", {})
    tf_weight = retrieval_cfg.get("tf_weight", "sublinear")
    idf_weight = retrieval_cfg.get("idf_weight", "smooth")
    from evaluator import evaluate_run
    eval_metrics = config.get("evaluation", {}).get("metrics", ["MAP", "NDCG@10"])

    query_tokens_cache = {qid: preprocess(qtext, preproc_cfg) for qid, qtext in dev_queries.items()}
    values = config.get("analysis", {}).get("title_weight_values", [1.0, 2.0, 3.0, 5.0, 8.0, 15.0])

    results = []
    for w in values:
        from ranker import TfidfRanker
        ranker = TfidfRanker(index, tf_weight=tf_weight, idf_weight=idf_weight,
                              title_term_counts=title_term_counts, title_weight=w)
        run = {qid: [d for d, _ in ranker.score(toks)] for qid, toks in query_tokens_cache.items()}
        metrics, _ = evaluate_run(run, dev_qrels, metrics=eval_metrics)
        result = {"title_weight": w}
        for metric_name in eval_metrics:
            if metric_name in metrics:
                result[metric_name] = round(metrics[metric_name], 4)
        results.append(result)
        print(f"       title_weight={w}: MAP={metrics.get('MAP', 0):.4f}, NDCG@10={metrics.get('NDCG@10', 0):.4f}")

    import json
    import os
    outdir = config["output"]["outdir"]
    sweep_path = os.path.join(outdir, "extra", "title_weight_sweep.json")
    os.makedirs(os.path.dirname(sweep_path), exist_ok=True)
    with open(sweep_path, "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x.get("MAP", 0))
    print(f"[TitleWeightSweep] Best title_weight found: {best['title_weight']} (MAP: {best.get('MAP', 0):.4f})")


def run_ablation(args, config, docs, dev_queries, dev_qrels):
    print("[Ablation] Starting Ablation Study...")
    outdir = config["output"]["outdir"]
    
    base_preproc = config["preprocessing"].copy()
    ablation_tests = config["analysis"].get("ablation_tests", 
        ["stopwords", "stemming", "lemmatization", "overlap_vs_tfidf", "semantic"])
    
    tests = []
    for test in ablation_tests:
        if test == "stopwords":
            cfg = base_preproc.copy()
            cfg["remove_stopwords"] = False
            cfg["stopword_removal"] = False
            tests.append(("No Stopwords", cfg, "tfidf"))
        elif test == "stemming":
            tests.append(("No Stemming", {**base_preproc, "stemming": False}, "tfidf"))
        elif test == "lemmatization":
            tests.append(("With Lemmatization", {**base_preproc, "lemmatize": True, "stemming": False}, "tfidf"))
        elif test == "overlap_vs_tfidf":
            tests.append(("Overlap (vs TF-IDF)", None, "overlap"))
        elif test == "semantic":
            tests.append(("Semantic (vs TF-IDF)", None, "semantic"))
        elif test == "score_normalization":
            tests.append(("Semantic - raw (unnormalized) Eq.3", None, "semantic_no_norm"))
    
    # Add baseline
    tests = [("Baseline", base_preproc, "tfidf")] + tests
    
    ablation_results = {}
    eval_metrics = config.get("evaluation", {}).get("metrics", ["MAP"])
    retrieval_cfg = config.get("retrieval", {})
    tf_weight = retrieval_cfg.get("tf_weight", "sublinear")
    idf_weight = retrieval_cfg.get("idf_weight", "smooth")
    
    from preprocess import get_preprocessing_config, preprocess
    from index import build_index
    from ranker import TfidfRanker, OverlapRanker
    from semantic import run_semantic_final
    from evaluator import evaluate_run
    
    for name, preproc_cfg, ranker_type in tests:
        print(f"       Running: {name}...")
        
        if ranker_type == "overlap":
            idx, _, _ = build_index(docs, base_preproc, verbose=False, outdir=outdir)
            ranker = OverlapRanker(idx)
            run = {}
            for qid, qtext in dev_queries.items():
                run[qid] = [d for d, _ in ranker.score(preprocess(qtext, base_preproc))]
        elif ranker_type in ("semantic", "semantic_no_norm"):
            idx, raw_tokens_map, title_term_counts = build_index(docs, base_preproc, verbose=False, outdir=outdir)
            ranker = TfidfRanker(idx, tf_weight=tf_weight, idf_weight=idf_weight,
                                  title_term_counts=title_term_counts,
                                  title_weight=config.get("indexing", {}).get("title_weight", 1.0))
            
            tfidf_run = {}
            for qid, qtext in dev_queries.items():
                tfidf_run[qid] = ranker.score(preprocess(qtext, base_preproc))
            
            semantic_cfg = dict(config.get("semantic", {}))
            semantic_cfg["enabled"] = True
            semantic_cfg["normalize_scores"] = (ranker_type == "semantic")
            final_run, _ = run_semantic_final(
                dev_queries, tfidf_run, raw_tokens_map, 
                base_preproc, semantic_cfg, config,
                doc_words_cache={}
            )
            run = {qid: [d for d, _ in ranked] for qid, ranked in final_run.items()}
        else:
            idx, _, title_term_counts = build_index(docs, preproc_cfg, verbose=False, outdir=outdir)
            ranker = TfidfRanker(idx, tf_weight=tf_weight, idf_weight=idf_weight,
                                  title_term_counts=title_term_counts,
                                  title_weight=config.get("indexing", {}).get("title_weight", 1.0))
            run = {}
            for qid, qtext in dev_queries.items():
                run[qid] = [d for d, _ in ranker.score(preprocess(qtext, preproc_cfg))]
        
        metrics, _ = evaluate_run(run, dev_qrels, metrics=eval_metrics)
        ablation_results[name] = {k: round(v, 4) for k, v in metrics.items()}

    import json
    import os
    os.makedirs(os.path.join(outdir, "extra"), exist_ok=True)
    with open(os.path.join(outdir, "extra", "ablation_study.json"), 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print("[Ablation] Results saved to extra/ablation_study.json")


def main():
    parser = argparse.ArgumentParser(description="Shiraz University - Text Mining & NLP Assignment 2")
    parser.add_argument("--docs", required=True)
    parser.add_argument("--dev-queries", required=True)
    parser.add_argument("--dev-qrels", required=True)
    parser.add_argument("--test-queries", required=False)
    parser.add_argument("--config", required=False, default="config.json")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="full", choices=["full", "alpha-sweep", "ablation", "title-weight-sweep"])
    args = parser.parse_args()

    from config import load_config
    config = load_config(args.config)
    config["output"]["outdir"] = args.outdir
    config["random_seed"] = args.seed
    import random
    random.seed(config["random_seed"])
    
    print(f"[*] Starting system in '{args.mode}' mode...")

    from index import load_documents, build_index
    from preprocess import get_preprocessing_config

    print(f"[1/6] Loading documents from {args.docs} ...")
    docs = load_documents(args.docs)
    print(f"       {len(docs)} documents loaded.")

    print("[2/6] Building inverted index ...")
    preproc_cfg = get_preprocessing_config(config)
    title_weight = config.get("indexing", {}).get("title_weight", 2)
    index, raw_tokens_map, title_term_counts = build_index(
        docs, preproc_cfg, title_weight=title_weight, outdir=args.outdir)
    stats = index.stats()
    print(f"       {stats}")

    from index import load_queries, load_qrels

    print("[3/6] Loading queries & qrels ...")
    dev_queries = load_queries(args.dev_queries)
    dev_qrels = load_qrels(args.dev_qrels)
    
    if args.mode == "full":
        if not args.test_queries:
            print("Error: --test-queries is required for 'full' mode.")
            return
        test_queries = load_queries(args.test_queries)
        print(f"       dev queries: {len(dev_queries)}, test queries: {len(test_queries)}")
        run_full(args, config, docs, dev_queries, dev_qrels, test_queries, index, raw_tokens_map, title_term_counts)
    elif args.mode == "alpha-sweep":
        run_alpha_sweep(args, config, docs, dev_queries, dev_qrels, raw_tokens_map, index, title_term_counts)
    elif args.mode == "ablation":
        run_ablation(args, config, docs, dev_queries, dev_qrels)
    elif args.mode == "title-weight-sweep":
        run_title_weight_sweep(args, config, docs, dev_queries, dev_qrels, index, title_term_counts)

    print(f"[*] System execution complete. Check the '{args.outdir}' folder for results.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()