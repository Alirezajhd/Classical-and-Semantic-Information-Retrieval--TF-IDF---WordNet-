"""
Text preprocessing module.

Implements configurable tokenization, case folding, stopword removal,
and Porter stemming as required by Section 4.1.
"""
import re
from collections import Counter


class TextPreprocessor:
    def __init__(self, cfg=None):
        self.cfg = cfg or {
            "lowercase": True,
            "remove_stopwords": True,
            "stemming": True,
            "lemmatize": False,
            "min_token_length": 1
        }
        self._ensure_resources()
        self.porter = None
        self.lemmatizer = None
        self.stop_words = None
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def _ensure_resources(self):
        try:
            import nltk
            for res in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
                try:
                    nltk.data.find(res)
                except LookupError:
                    nltk.download(res, quiet=True)
        except ImportError:
            pass

    def _init_nltk(self):
        if self.porter is None:
            try:
                from nltk.stem.porter import PorterStemmer
                from nltk.stem import WordNetLemmatizer
                from nltk.corpus import stopwords
                self.porter = PorterStemmer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except ImportError:
                self.porter = None
                self.lemmatizer = None
                self.stop_words = set()

    def preprocess(self, text):
        self._init_nltk()
        if not text or not isinstance(text, str):
            return []

        tokens = self.token_pattern.findall(text)

        if self.cfg.get("lowercase", True):
            tokens = [t.lower() for t in tokens]

        remove_sw = self.cfg.get("remove_stopwords", self.cfg.get("stopword_removal", True))
        if remove_sw and self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]

        min_len = self.cfg.get("min_token_length", 1)
        if min_len and min_len > 1:
            tokens = [t for t in tokens if len(t) >= min_len]

        if self.cfg.get("lemmatize", False):
            if self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        if self.cfg.get("stemming", True):
            if self.porter:
                tokens = [self.porter.stem(t) for t in tokens]

        return tokens


_preprocessor_cache = {}

def preprocess(text, cfg=None):
    cfg = cfg or {}
    key = tuple(sorted(cfg.items()))
    tp = _preprocessor_cache.get(key)
    if tp is None:
        tp = TextPreprocessor(cfg)
        _preprocessor_cache[key] = tp
    return tp.preprocess(text)

def get_preprocessing_config(cfg):
    if cfg and "preprocessing" in cfg:
        return cfg["preprocessing"]
    return cfg