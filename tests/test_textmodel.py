"""Tests for the dependency-free TF-IDF scorer and the shipped models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import textmodel

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"
BENCHMARK = ROOT / "data" / "benchmarks" / "emotion_benchmark.json"


def test_word_analyzer_unigrams_and_bigrams() -> None:
    grams = textmodel.word_analyzer("We are choking", ngram_max=2)
    assert "we" in grams and "are" in grams and "choking" in grams
    assert "we are" in grams and "are choking" in grams


def test_char_analyzer_covers_emoji_and_substrings() -> None:
    grams = textmodel.char_analyzer("go😭", 2, 3)
    assert "go" in grams and "o😭" in grams and "go😭" in grams


def test_save_load_round_trip_and_softmax(tmp_path: Path) -> None:
    model = textmodel.TfidfLinearModel(
        vocab={"panic": 0, "calm": 1},
        idf=np.array([1.0, 1.0]),
        coef=np.array([[3.0, -3.0], [-3.0, 3.0]]),
        intercept=np.array([0.0, 0.0]),
        classes=["panic", "calm"],
        analyzer="word",
    )
    model.save(tmp_path / "m.npz")
    loaded = textmodel.TfidfLinearModel.load(tmp_path / "m.npz")
    proba = loaded.predict_proba(["panic panic", "calm calm"])
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert loaded.predict(["panic panic"]) == ["panic"]
    assert loaded.predict(["calm calm"]) == ["calm"]


def test_try_load_missing_returns_none(tmp_path: Path) -> None:
    assert textmodel.try_load("does_not_exist", model_dir=tmp_path) is None


def test_shipped_emotion_model_loads_and_scores() -> None:
    model = textmodel.try_load("emotion_model", model_dir=MODEL_DIR)
    assert model is not None
    proba = model.predict_proba(
        ["i feel so scared and terrified right now", "i feel so happy and joyful"]
    )
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = model.predict(
        ["i feel so scared and terrified right now", "i feel so happy and joyful"]
    )
    assert preds[0] == "panic"
    assert preds[1] == "joy"


def test_benchmark_clears_accuracy_bar() -> None:
    record = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert record["decision"] == "switched_to_trained"
    assert record["lexicon"]["emotion_accuracy"] < record["accuracy_bar"]
    assert record["trained_emotion"]["test_accuracy"] >= 0.80
    assert record["trained_emotion"]["cv_accuracy_mean"] >= 0.80
