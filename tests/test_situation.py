"""Tests for the real-time match-situation classifier in src/situation.py."""
from __future__ import annotations

import pandas as pd

import situation


def _state(**overrides: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "minute": 50,
        "crowd_panic_score": 0.0,
        "delta_xg_10min": 0.35,
        "emotional_volatility": 0.25,
        "arbitrage_index": 0.15,
        "rolling_xg": 0.4,
        "comment_volume": 5,
        "emo_confidence": 0.2,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_classify_recovers_panic_divergence_prototype() -> None:
    state = _state(
        minute=57,
        crowd_panic_score=0.65,
        delta_xg_10min=0.55,
        emotional_volatility=0.35,
        arbitrage_index=0.65,
    )
    result = situation.classify(state)
    assert result.iloc[0]["situation"] == "panic_divergence"
    assert 0.0 <= result.iloc[0]["situation_confidence"] <= 1.0


def test_classify_separates_crisis_from_divergence() -> None:
    crisis = _state(
        minute=62,
        crowd_panic_score=0.7,
        delta_xg_10min=0.1,
        emotional_volatility=0.45,
        arbitrage_index=0.2,
    )
    assert situation.classify(crisis).iloc[0]["situation"] == "genuine_crisis"


def test_classify_cruise_and_dead_rubber() -> None:
    cruise = _state(crowd_panic_score=-0.6, delta_xg_10min=0.6, emotional_volatility=0.15)
    assert situation.classify(cruise).iloc[0]["situation"] == "cruise_control"
    rubber = _state(
        crowd_panic_score=-0.05,
        delta_xg_10min=0.05,
        emotional_volatility=0.08,
        arbitrage_index=0.05,
    )
    assert situation.classify(rubber).iloc[0]["situation"] == "dead_rubber"


def test_classify_late_drama_uses_match_phase() -> None:
    drama = _state(
        minute=90,
        crowd_panic_score=0.35,
        delta_xg_10min=0.45,
        emotional_volatility=0.65,
        arbitrage_index=0.35,
    )
    assert situation.classify(drama).iloc[0]["situation"] == "late_drama"


def test_classify_is_vectorized_over_frames() -> None:
    frame = pd.concat(
        [
            _state(minute=10, crowd_panic_score=-0.6, delta_xg_10min=0.6, emotional_volatility=0.15),
            _state(minute=60, crowd_panic_score=0.65, delta_xg_10min=0.55, arbitrage_index=0.65, emotional_volatility=0.35),
        ],
        ignore_index=True,
    )
    result = situation.classify(frame)
    assert len(result) == 2
    assert result["situation"].tolist() == ["cruise_control", "panic_divergence"]
    assert result["situation_confidence"].between(0, 1).all()


def test_classify_empty_frame() -> None:
    result = situation.classify(pd.DataFrame())
    assert "situation" in result.columns
    assert result.empty


def test_classify_tolerates_missing_columns() -> None:
    result = situation.classify(pd.DataFrame([{"minute": 30}]))
    assert result.iloc[0]["situation"] in situation.PROTOTYPES


def test_playbook_covers_every_prototype() -> None:
    for label in situation.PROTOTYPES:
        brief = situation.situation_brief(label)
        assert brief["metrics"], label
        assert brief["read"], label
        for metric in brief["metrics"]:
            assert metric in situation.METRIC_LABELS, metric


def test_metrics_that_matter_returns_values() -> None:
    state = situation.classify(
        _state(crowd_panic_score=0.65, delta_xg_10min=0.55, arbitrage_index=0.65, emotional_volatility=0.35)
    )
    rows = situation.metrics_that_matter(state)
    assert rows
    columns = [c for c, _, _ in rows]
    assert "arbitrage_index" in columns
    assert all(isinstance(v, float) for _, _, v in rows)
