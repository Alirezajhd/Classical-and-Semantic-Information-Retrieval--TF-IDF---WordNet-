#!/usr/bin/env python
"""
Clear all caches for the IR system.
"""
import os
import shutil

def clear_cache():
    """Remove all cached data."""
    paths_to_remove = [
        "results/extra/wordnet_sim_cache.pkl",
        "results/extra/",
        "results/query_trace.jsonl",
        "results/dev_rankings.tsv",
        "results/test_rankings.tsv",
        "results/dev_metrics.json",
        "results/summary.json",
    ]
    
    for path in paths_to_remove:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            else:
                os.remove(path)
                print(f"Removed file: {path}")
    
    print("Cache cleared successfully!")

if __name__ == "__main__":
    clear_cache()