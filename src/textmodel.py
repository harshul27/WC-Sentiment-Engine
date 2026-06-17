"""Dependency-free TF-IDF + linear-model scorer (numpy only).

This is the *runtime* counterpart to the offline-trained emotion and sentiment
models. The training script (scripts/train_emotion.py) fits scikit-learn's
TfidfVectorizer using the exact analyzer callables defined here, then exports
the learned vocabulary, idf weights, and logistic-regression coefficients to a
compact npz + json pair. Because train time and run time share these analyzers
and the same TF-IDF maths, the exported model reproduces scikit-learn's
predictions without ever shipping scikit-learn into the Streamlit app.

Nothing here imports anything heavier than numpy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"

_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def word_analyzer(text: str, ngram_max: int = 2) -> list[str]:
    """Lowercased word unigrams + n-grams (used by the emotion model)."""
    tokens = _TOKEN.findall(str(text).lower())
    grams = list(tokens)
    for n in range(2, ngram_max + 1):
        grams.extend(
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        )
    return grams


def char_analyzer(text: str, ngram_min: int = 2, ngram_max: int = 5) -> list[str]:
    """Lowercased character n-grams (language-agnostic; emoji-safe).

    Used by the multilingual sentiment model so Spanish/Portuguese/Arabic/…
    posts and emoji are scored without a per-language word vocabulary.
    """
    s = str(text).lower()
    grams: list[str] = []
    length = len(s)
    for n in range(ngram_min, ngram_max + 1):
        if length < n:
            break
        grams.extend(s[i : i + n] for i in range(length - n + 1))
    return grams


ANALYZERS = {"word": word_analyzer, "char": char_analyzer}


class TfidfLinearModel:
    """Exported TF-IDF + multinomial-logistic model scored in pure numpy."""

    def __init__(
        self,
        vocab: dict[str, int],
        idf: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        classes: list[str],
        analyzer: str,
    ) -> None:
        self.vocab = vocab
        self.idf = np.asarray(idf, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.classes = list(classes)
        self.analyzer = analyzer
        self._fn = ANALYZERS[analyzer]

    def _features(self, messages: list[str]) -> np.ndarray:
        rows = len(messages)
        width = len(self.idf)
        matrix = np.zeros((rows, width), dtype=np.float64)
        for r, message in enumerate(messages):
            counts: dict[int, float] = {}
            for token in self._fn(message):
                col = self.vocab.get(token)
                if col is not None:
                    counts[col] = counts.get(col, 0.0) + 1.0
            if not counts:
                continue
            idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
            vals = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
            vals = vals * self.idf[idx]
            norm = float(np.sqrt((vals * vals).sum()))
            if norm > 0.0:
                vals = vals / norm
            matrix[r, idx] = vals
        return matrix

    def decision(self, messages: list[str]) -> np.ndarray:
        return self._features(list(messages)) @ self.coef.T + self.intercept

    def predict_proba(self, messages: list[str]) -> np.ndarray:
        scores = self.decision(messages)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, messages: list[str]) -> list[str]:
        proba = self.predict_proba(messages)
        return [self.classes[i] for i in proba.argmax(axis=1)]

    def save(self, npz_path: Path, json_path: Path | None = None) -> None:
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = Path(json_path) if json_path else npz_path.with_suffix(".json")
        np.savez_compressed(
            npz_path,
            idf=self.idf.astype(np.float32),
            coef=self.coef.astype(np.float32),
            intercept=self.intercept.astype(np.float32),
        )
        json_path.write_text(
            json.dumps(
                {"vocab": self.vocab, "classes": self.classes, "analyzer": self.analyzer}
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, npz_path: Path, json_path: Path | None = None) -> "TfidfLinearModel":
        npz_path = Path(npz_path)
        json_path = Path(json_path) if json_path else npz_path.with_suffix(".json")
        arrays = np.load(npz_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        return cls(
            vocab={k: int(v) for k, v in meta["vocab"].items()},
            idf=arrays["idf"],
            coef=arrays["coef"],
            intercept=arrays["intercept"],
            classes=meta["classes"],
            analyzer=meta["analyzer"],
        )


def try_load(name: str, model_dir: Path = MODEL_DIR) -> "TfidfLinearModel | None":
    """Load a committed model by name, or None when its artifacts are absent."""
    npz_path = Path(model_dir) / f"{name}.npz"
    json_path = Path(model_dir) / f"{name}.json"
    if not npz_path.exists() or not json_path.exists():
        return None
    try:
        return TfidfLinearModel.load(npz_path, json_path)
    except (OSError, ValueError, KeyError):
        return None
