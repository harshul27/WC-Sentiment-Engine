"""Real-time match-situation classifier.

A lightweight, dependency-free nearest-centroid model that classifies every
scored minute into one of seven match situations from the live feature
vector (crowd panic, xG stability, mood volatility, arbitrage index, match
phase). Each situation carries a playbook entry: the metrics that matter
most in that situation (surfaced dynamically in the dashboard) and a
one-line read of what the situation means for the product's aim.

The model is deliberately interpretable: prototypes are the model weights,
distances are auditable, and confidence is a softmax over the negative
weighted squared distances.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: tuple[str, ...] = (
    "crowd_panic_score",
    "delta_xg_10min",
    "emotional_volatility",
    "arbitrage_index",
    "minute_norm",
)

FEATURE_WEIGHTS = np.array([1.2, 1.0, 1.0, 1.4, 0.6], dtype=np.float64)

# Prototype feature vectors - the interpretable "weights" of the model.
PROTOTYPES: dict[str, np.ndarray] = {
    "cruise_control": np.array([-0.60, 0.60, 0.15, 0.08, 0.50]),
    "balanced_contest": np.array([0.00, 0.35, 0.25, 0.15, 0.50]),
    "panic_divergence": np.array([0.65, 0.55, 0.35, 0.65, 0.60]),
    "genuine_crisis": np.array([0.70, 0.10, 0.45, 0.20, 0.65]),
    "late_drama": np.array([0.35, 0.45, 0.65, 0.35, 0.95]),
    "emotional_chaos": np.array([0.15, 0.40, 0.80, 0.30, 0.50]),
    "dead_rubber": np.array([-0.05, 0.05, 0.08, 0.05, 0.55]),
}

SOFTMAX_TEMPERATURE = 0.25

# Per-situation playbook: which metrics matter right now and why.
SITUATION_PLAYBOOK: dict[str, dict[str, object]] = {
    "panic_divergence": {
        "label": "Panic Divergence",
        "metrics": ["arbitrage_index", "crowd_panic_score", "delta_xg_10min"],
        "read": (
            "Crowd panic has decoupled from a stable match - the engine's "
            "core arbitrage spot. Watch the index against the flag threshold."
        ),
    },
    "genuine_crisis": {
        "label": "Genuine Crisis",
        "metrics": ["delta_xg_10min", "rolling_xg", "crowd_panic_score"],
        "read": (
            "The panic is justified: attacking threat has collapsed. "
            "Sentiment and pitch agree - no divergence edge here."
        ),
    },
    "cruise_control": {
        "label": "Cruise Control",
        "metrics": ["delta_xg_10min", "emo_confidence", "comment_volume"],
        "read": (
            "Comfortable, stable match with a calm crowd. Low signal value; "
            "watch for complacency if threat fades."
        ),
    },
    "balanced_contest": {
        "label": "Balanced Contest",
        "metrics": ["rolling_xg", "delta_xg_10min", "crowd_panic_score"],
        "read": (
            "Even match, even mood. The next clear chance likely moves both "
            "the crowd and the market - watch threat accumulation."
        ),
    },
    "late_drama": {
        "label": "Late Drama",
        "metrics": ["emotional_volatility", "arbitrage_index", "minute"],
        "read": (
            "Closing stages with an unstable crowd. Sentiment overshoots are "
            "common here but expire fast - short decision windows."
        ),
    },
    "emotional_chaos": {
        "label": "Emotional Chaos",
        "metrics": ["emotional_volatility", "comment_volume", "crowd_panic_score"],
        "read": (
            "The crowd is reacting to everything; mood flips minute to "
            "minute. Distrust single-minute readings, use rolling values."
        ),
    },
    "dead_rubber": {
        "label": "Dead Rubber",
        "metrics": ["comment_volume", "rolling_xg", "emotional_volatility"],
        "read": (
            "Low threat, low engagement. Thin crowd signal - treat all "
            "sentiment-derived metrics as low-confidence."
        ),
    },
}

METRIC_LABELS: dict[str, str] = {
    "arbitrage_index": "Arbitrage Index",
    "crowd_panic_score": "Crowd Panic",
    "delta_xg_10min": "xG Stability",
    "rolling_xg": "Rolling xG",
    "emo_confidence": "Confidence Share",
    "emotional_volatility": "Mood Volatility",
    "comment_volume": "Reactions/min",
    "minute": "Minute",
}


def _feature_matrix(state: pd.DataFrame) -> np.ndarray:
    minutes = pd.to_numeric(state.get("minute"), errors="coerce").fillna(0.0)
    columns = {
        "crowd_panic_score": (-1.0, 1.0),
        "delta_xg_10min": (0.0, 1.0),
        "emotional_volatility": (0.0, 1.0),
        "arbitrage_index": (0.0, 1.0),
    }
    parts: list[np.ndarray] = []
    for column, (low, high) in columns.items():
        values = pd.to_numeric(
            state.get(column, pd.Series(0.0, index=state.index)), errors="coerce"
        )
        parts.append(values.fillna(0.0).clip(low, high).to_numpy(dtype=np.float64))
    parts.append((minutes / 95.0).clip(0.0, 1.4).to_numpy(dtype=np.float64))
    return np.column_stack(parts)


def classify(state: pd.DataFrame) -> pd.DataFrame:
    """Classify every minute of a scored state frame.

    Returns the frame with two added columns: situation (label) and
    situation_confidence (softmax probability of the chosen class, in
    [0, 1]). Empty input gets the columns with no rows.
    """
    result = state.copy()
    if result.empty:
        result["situation"] = pd.Series(dtype="str")
        result["situation_confidence"] = pd.Series(dtype="float64")
        return result
    features = _feature_matrix(result)
    labels = list(PROTOTYPES)
    centroids = np.stack([PROTOTYPES[label] for label in labels])
    deltas = features[:, None, :] - centroids[None, :, :]
    distances = np.sqrt(((deltas * FEATURE_WEIGHTS) ** 2).sum(axis=2))
    scores = -(distances**2) / SOFTMAX_TEMPERATURE
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    best = probabilities.argmax(axis=1)
    result["situation"] = [labels[i] for i in best]
    result["situation_confidence"] = probabilities[
        np.arange(len(best)), best
    ].round(4)
    return result


def situation_brief(situation: str) -> dict[str, object]:
    """Playbook entry (display label, key metrics, one-line read)."""
    return SITUATION_PLAYBOOK.get(
        situation,
        {
            "label": situation.replace("_", " ").title() or "Unknown",
            "metrics": ["arbitrage_index", "crowd_panic_score", "delta_xg_10min"],
            "read": "Unrecognised situation - showing core engine metrics.",
        },
    )


def metrics_that_matter(
    state: pd.DataFrame,
) -> list[tuple[str, str, float]]:
    """The classified latest minute's key metrics as (column, label, value)."""
    if state.empty or "situation" not in state.columns:
        return []
    latest = state.iloc[-1]
    brief = situation_brief(str(latest["situation"]))
    rows: list[tuple[str, str, float]] = []
    for column in brief["metrics"]:
        if column in state.columns:
            value = pd.to_numeric(pd.Series([latest[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                rows.append((column, METRIC_LABELS.get(column, column), float(value)))
    return rows
