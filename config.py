"""
Configuration management for the assignment.
"""
import json


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config.setdefault("random_seed", 42)
    config.setdefault("preprocessing", {})
    config.setdefault("indexing", {})
    config.setdefault("retrieval", {})
    config.setdefault("semantic", {})
    config.setdefault("evaluation", {})
    config.setdefault("output", {})
    config.setdefault("analysis", {})
    config.setdefault("ltr", {})
    
    preproc = config["preprocessing"]
    preproc.setdefault("lowercase", True)
    preproc.setdefault("remove_stopwords", True)
    preproc.setdefault("stemming", True)
    preproc.setdefault("lemmatize", False)
    preproc.setdefault("min_token_length", 2)
    preproc.setdefault("stopword_source", "nltk")
    
    indexing = config["indexing"]
    indexing.setdefault("use_stemming_in_index", False)
    indexing.setdefault("use_lemmatization_in_index", True)
    indexing.setdefault("store_term_frequencies", True)
    indexing.setdefault("store_document_frequencies", True)

    if preproc.get("stemming", True) != indexing["use_stemming_in_index"]:
        print(f"[config warning] indexing.use_stemming_in_index="
              f"{indexing['use_stemming_in_index']} disagrees with "
              f"preprocessing.stemming={preproc.get('stemming', True)}. "
              f"Per spec 4.1, documents and queries must use the same "
              f"pipeline; preprocessing.stemming is the value actually used.")
    if preproc.get("lemmatize", False) != indexing["use_lemmatization_in_index"]:
        print(f"[config warning] indexing.use_lemmatization_in_index="
              f"{indexing['use_lemmatization_in_index']} disagrees with "
              f"preprocessing.lemmatize={preproc.get('lemmatize', False)}. "
              f"Per spec 4.1, documents and queries must use the same "
              f"pipeline; preprocessing.lemmatize is the value actually used.")
    
    retrieval = config["retrieval"]
    retrieval.setdefault("topk", 100)
    retrieval.setdefault("tf_weight", "sublinear")
    retrieval.setdefault("idf_weight", "smooth")
    retrieval.setdefault("normalization", "cosine")
    
    semantic = config["semantic"]
    semantic.setdefault("enabled", True)
    semantic.setdefault("mode", "rerank")
    semantic.setdefault("alpha", 0.8)
    semantic.setdefault("rerank_k", 20)
    semantic.setdefault("max_synsets_per_word", 3)
    semantic.setdefault("max_doc_content_words", 60)
    semantic.setdefault("pos_tags", ["NOUN", "VERB"])
    semantic.setdefault("similarity_metric", "wup")
    semantic.setdefault("cache_enabled", True)
    semantic.setdefault("cache_path", "results/extra/wordnet_sim_cache.pkl")
    semantic.setdefault("alpha_mode", "fixed")
    semantic.setdefault("alpha_min", 0.5)
    semantic.setdefault("alpha_max", 0.95)
    semantic.setdefault("normalize_scores", True)
    
    evaluation = config["evaluation"]
    evaluation.setdefault("metrics", ["MAP", "MRR", "P@1", "P@5", "P@10", "NDCG@5", "NDCG@10"])
    evaluation.setdefault("relevance_binary", True)
    
    output = config["output"]
    output.setdefault("outdir", "results")
    output.setdefault("overwrite", True)
    output.setdefault("verbose", True)
    output.setdefault("save_intermediate", True)
    
    analysis = config["analysis"]
    analysis.setdefault("run_alpha_sweep", False)
    analysis.setdefault("alpha_values", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    analysis.setdefault("run_ablation", False)
    analysis.setdefault("ablation_tests", ["stopwords", "stemming", "lemmatization", "overlap_vs_tfidf", "semantic"])
    analysis.setdefault("title_weight_values", [1.0, 2.0, 3.0, 5.0, 8.0, 15.0])

    ltr = config["ltr"]
    ltr.setdefault("enabled", False)
    ltr.setdefault("model", "ridge")
    ltr.setdefault("lambda", 1.0)
    ltr.setdefault("rerank_k", semantic.get("rerank_k", 20))
    ltr.setdefault("cv_folds", 5)
    
    return config


# For semantic module compatibility
_CONFIG = {}

def set_config(cfg):
    global _CONFIG
    _CONFIG = cfg

def get_config():
    return _CONFIG