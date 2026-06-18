"""Offline training, testing & cross-validation for the crowd-emotion models.

Run locally (not in CI - it downloads public datasets and needs scikit-learn):

    python scripts/train_emotion.py

What it does:
  1. Pulls two real, labelled, public datasets:
       - dair-ai/emotion (English, 6-class) -> emotion categorisation benchmark
       - cardiffnlp/tweet_sentiment_multilingual (8 languages, neg/neu/pos)
         -> multilingual panic-direction benchmark (addresses the coverage gap)
  2. Measures the CURRENT lexicon classifier on both held-out test sets.
  3. Trains lightweight TF-IDF + logistic-regression models (sharing the exact
     numpy analyzers in src/textmodel.py), with stratified 5-fold CV + a
     held-out test split, and verifies the exported numpy model reproduces
     scikit-learn's predictions.
  4. If the current model is below the 0.80 accuracy bar, exports the trained
     models to data/models/*.npz|json (pure-numpy runtime, no new deps) and
     writes the full benchmark to data/benchmarks/emotion_benchmark.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import emotion  # noqa: E402
import textmodel  # noqa: E402
from model import score_messages  # noqa: E402

MODEL_DIR = ROOT / "data" / "models"
BENCH_DIR = ROOT / "data" / "benchmarks"
ACCURACY_BAR = 0.80

EMOTION_MAP = {0: "despair", 1: "joy", 2: "joy", 3: "anger", 4: "panic", 5: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_LANGS = (
    "english", "spanish", "portuguese", "french", "italian", "german", "hindi", "arabic"
)


def _parquet_urls(dataset: str, config: str, split: str) -> list[str]:
    url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset}"
    files = requests.get(url, timeout=60).json()["parquet_files"]
    return [f["url"] for f in files if f["config"] == config and f["split"] == split]


def _load(dataset: str, config: str, split: str) -> pd.DataFrame:
    frames = [pd.read_parquet(u) for u in _parquet_urls(dataset, config, split)]
    return pd.concat(frames, ignore_index=True)


def load_emotion_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.concat(
        [_load("dair-ai/emotion", "split", s) for s in ("train", "validation")],
        ignore_index=True,
    )
    test = _load("dair-ai/emotion", "split", "test")
    for frame in (train, test):
        frame["y"] = frame["label"].map(EMOTION_MAP)
    return train, test


def load_sentiment_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for lang in SENTIMENT_LANGS:
        for split in ("train", "validation"):
            df = _load("cardiffnlp/tweet_sentiment_multilingual", lang, split)
            df["lang"] = lang
            train_parts.append(df)
        test = _load("cardiffnlp/tweet_sentiment_multilingual", lang, "test")
        test["lang"] = lang
        test_parts.append(test)
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    for frame in (train, test):
        frame["y"] = frame["label"].map(SENTIMENT_MAP)
    return train, test


def lexicon_emotion_predictions(texts: pd.Series) -> list[str]:
    scores = emotion.classify_comments(texts)
    labels = scores.idxmax(axis=1).str.removeprefix("emo_")
    labels[scores.max(axis=1) <= 0.0] = "neutral"
    return labels.tolist()


def lexicon_sentiment_predictions(texts: pd.Series, band: float = 0.15) -> list[str]:
    panic = score_messages(texts).to_numpy()
    out = np.where(panic > band, "negative", np.where(panic < -band, "positive", "neutral"))
    return out.tolist()


def _train_model(
    train: pd.DataFrame, analyzer_name: str, analyzer, max_features: int
) -> tuple[textmodel.TfidfLinearModel, object, object]:
    vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=2, max_features=max_features)
    matrix = vectorizer.fit_transform(train["text"].astype(str))
    clf = LogisticRegression(max_iter=2000, C=8.0, class_weight="balanced")
    clf.fit(matrix, train["y"])
    model = textmodel.TfidfLinearModel(
        vocab={str(k): int(v) for k, v in vectorizer.vocabulary_.items()},
        idf=vectorizer.idf_,
        coef=clf.coef_,
        intercept=clf.intercept_,
        classes=list(clf.classes_),
        analyzer=analyzer_name,
    )
    return model, vectorizer, clf


def _cv_scores(vectorizer, clf, train: pd.DataFrame) -> dict[str, float]:
    matrix = vectorizer.transform(train["text"].astype(str))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, matrix, train["y"], cv=skf, scoring="accuracy")
    f1 = cross_val_score(clf, matrix, train["y"], cv=skf, scoring="f1_macro")
    return {
        "cv_accuracy_mean": round(float(acc.mean()), 4),
        "cv_accuracy_std": round(float(acc.std()), 4),
        "cv_macro_f1_mean": round(float(f1.mean()), 4),
    }


def main() -> None:
    print("Downloading datasets...")
    emo_train, emo_test = load_emotion_data()
    sent_train, sent_test = load_sentiment_data()
    print(f"  emotion: {len(emo_train)} train / {len(emo_test)} test")
    print(f"  sentiment: {len(sent_train)} train / {len(sent_test)} test "
          f"across {sent_test['lang'].nunique()} languages")

    # --- current lexicon baselines -------------------------------------------
    emo_classes = sorted(set(EMOTION_MAP.values()))
    lex_emo_pred = lexicon_emotion_predictions(emo_test["text"])
    lex_emo_acc = accuracy_score(emo_test["y"], lex_emo_pred)
    lex_emo_f1 = f1_score(emo_test["y"], lex_emo_pred, labels=emo_classes, average="macro")
    lex_sent_pred = lexicon_sentiment_predictions(sent_test["text"])
    lex_sent_acc = accuracy_score(sent_test["y"], lex_sent_pred)
    print(f"\nCURRENT lexicon  emotion acc={lex_emo_acc:.4f} macroF1={lex_emo_f1:.4f}"
          f" | multilingual sentiment acc={lex_sent_acc:.4f}")

    # --- trained models -------------------------------------------------------
    emo_model, emo_vec, emo_clf = _train_model(
        emo_train, "word", lambda t: textmodel.word_analyzer(t, 2), 12000
    )
    sent_model, sent_vec, sent_clf = _train_model(
        sent_train, "char", lambda t: textmodel.char_analyzer(t, 2, 5), 20000
    )

    emo_pred = emo_model.predict(emo_test["text"].tolist())
    emo_acc = accuracy_score(emo_test["y"], emo_pred)
    emo_f1 = f1_score(emo_test["y"], emo_pred, average="macro")
    emo_cv = _cv_scores(emo_vec, emo_clf, emo_train)

    sent_pred = sent_model.predict(sent_test["text"].tolist())
    sent_acc = accuracy_score(sent_test["y"], sent_pred)
    sent_cv = _cv_scores(sent_vec, sent_clf, sent_train)
    per_lang = {
        lang: round(
            accuracy_score(
                grp["y"], sent_model.predict(grp["text"].tolist())
            ),
            4,
        )
        for lang, grp in sent_test.groupby("lang")
    }

    # --- numpy/sklearn parity check ------------------------------------------
    sample = emo_test["text"].head(200).tolist()
    sk_proba = emo_clf.predict_proba(emo_vec.transform(sample))
    np_proba = emo_model.predict_proba(sample)
    parity = float(np.abs(sk_proba - np_proba).max())
    print(f"\nTRAINED  emotion acc={emo_acc:.4f} macroF1={emo_f1:.4f} CV={emo_cv}")
    print(f"TRAINED  multilingual sentiment acc={sent_acc:.4f} CV={sent_cv}")
    print(f"  per-language: {per_lang}")
    print(f"numpy<->sklearn max prob diff: {parity:.2e}")
    assert parity < 1e-5, "numpy scorer does not reproduce sklearn predictions"

    switched = lex_emo_acc < ACCURACY_BAR
    benchmark = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accuracy_bar": ACCURACY_BAR,
        "datasets": {
            "emotion": "dair-ai/emotion",
            "sentiment": "cardiffnlp/tweet_sentiment_multilingual",
        },
        "lexicon": {
            "emotion_accuracy": round(float(lex_emo_acc), 4),
            "emotion_macro_f1": round(float(lex_emo_f1), 4),
            "multilingual_sentiment_accuracy": round(float(lex_sent_acc), 4),
        },
        "trained_emotion": {
            "test_accuracy": round(float(emo_acc), 4),
            "test_macro_f1": round(float(emo_f1), 4),
            **emo_cv,
            "classes": emo_model.classes,
        },
        "trained_sentiment": {
            "test_accuracy": round(float(sent_acc), 4),
            **sent_cv,
            "per_language_accuracy": per_lang,
            "classes": sent_model.classes,
        },
        "numpy_parity_max_diff": parity,
        "decision": "switched_to_trained" if switched else "kept_lexicon",
    }
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / "emotion_benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
    )
    if switched:
        emo_model.save(MODEL_DIR / "emotion_model.npz")
        sent_model.save(MODEL_DIR / "sentiment_model.npz")
        print(f"\nLexicon {lex_emo_acc:.2%} < {ACCURACY_BAR:.0%} bar -> exported trained "
              f"models to {MODEL_DIR}")
    else:
        print(f"\nLexicon {lex_emo_acc:.2%} >= bar -> kept lexicon (no model exported)")
    print("Benchmark written to data/benchmarks/emotion_benchmark.json")


if __name__ == "__main__":
    main()
