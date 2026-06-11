"""Unit tests for the custom emotion model in src/emotion.py."""
from __future__ import annotations

import pandas as pd
import pytest

from emotion import (
    EMOTION_COLUMNS,
    EmotionAgent,
    classify_comments,
    dominant_emotion,
    emotion_shares,
    emotional_volatility,
    generate_takeaways,
    minute_profile,
    panic_from_profile,
)


def test_classify_comments_fires_each_emotion() -> None:
    samples = pd.Series(
        [
            "total panic, we're done, can't watch this",
            "the ref is corrupt, var robbery, disgrace",
            "what a goal!! golazo, let's go",
            "we look comfortable and in control, cruising",
            "it's over, hopeless, absolutely gutted",
            "no way, can't believe what just happened",
        ]
    )
    scores = classify_comments(samples)
    assert scores.loc[0, "emo_panic"] > 0
    assert scores.loc[1, "emo_anger"] > 0
    assert scores.loc[2, "emo_joy"] > 0
    assert scores.loc[3, "emo_confidence"] > 0
    assert scores.loc[4, "emo_despair"] > 0
    assert scores.loc[5, "emo_surprise"] > 0


def test_emotion_shares_normalize_and_keep_neutral_zero() -> None:
    scores = classify_comments(pd.Series(["panic and golazo", "nice weather today"]))
    shares = emotion_shares(scores)
    assert shares.iloc[0].sum() == pytest.approx(1.0)
    assert shares.iloc[1].sum() == 0.0


def test_minute_profile_is_continuous_and_counts_volume() -> None:
    chat = pd.DataFrame(
        {
            "minute": [0, 0, 3],
            "message": ["panic everywhere", "we're done", "cruising now"],
        }
    )
    profile = minute_profile(chat)
    assert list(profile["minute"]) == [0, 1, 2, 3]
    assert profile.loc[1, "emo_panic"] == profile.loc[0, "emo_panic"]
    assert profile.loc[0, "comment_volume"] == 2
    assert profile.loc[3, "emo_confidence"] > 0


def test_minute_profile_empty_chat() -> None:
    profile = minute_profile(pd.DataFrame(columns=["minute", "message"]))
    assert profile.empty


def test_dominant_emotion_with_neutral_fallback() -> None:
    chat = pd.DataFrame({"minute": [0, 1], "message": ["sheer panic", "hello there"]})
    profile = minute_profile(chat)
    labels = dominant_emotion(profile)
    assert labels.iloc[0] == "panic"


def test_emotional_volatility_bounds() -> None:
    chat = pd.DataFrame(
        {
            "minute": [0, 1, 2, 3],
            "message": ["panic!!", "golazo what a goal", "panic again", "cruising easy"],
        }
    )
    vol = emotional_volatility(minute_profile(chat))
    assert float(vol.min()) >= 0.0
    assert float(vol.max()) <= 1.0
    assert float(vol.iloc[-1]) > 0.0


def test_panic_from_profile_signs() -> None:
    panicked = minute_profile(
        pd.DataFrame({"minute": [0], "message": ["panic, hopeless, we're done"]})
    )
    confident = minute_profile(
        pd.DataFrame({"minute": [0], "message": ["comfortable, in control, cruising"]})
    )
    assert float(panic_from_profile(panicked).iloc[0]) > 0.5
    assert float(panic_from_profile(confident).iloc[0]) < -0.5


def test_emotion_agent_output_schema() -> None:
    chat = pd.DataFrame(
        {"minute": [0, 1, 5], "message": ["panic", "golazo", "we're done, can't watch"]}
    )
    result = EmotionAgent(window_minutes=3).run(chat)
    assert list(result["minute"]) == list(range(6))
    assert float(result["crowd_panic_score"].abs().max()) <= 1.0
    for column in EMOTION_COLUMNS:
        assert column in result.columns
    assert "dominant_emotion" in result.columns
    assert "emotional_volatility" in result.columns


def test_emotion_agent_empty_chat() -> None:
    result = EmotionAgent().run(pd.DataFrame(columns=["minute", "message"]))
    assert result.empty


def _state_row(**overrides: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "minute": 60,
        "crowd_panic_score": 0.0,
        "delta_xg_10min": 0.5,
        "arbitrage_index": 0.1,
        "dominant_emotion": "neutral",
        "emotional_volatility": 0.1,
        "emo_anger": 0.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_takeaway_market_overreaction_rule() -> None:
    state = _state_row(crowd_panic_score=0.8, arbitrage_index=0.7, dominant_emotion="panic")
    takeaways = generate_takeaways(state, threshold=0.35)
    assert any(t["headline"] == "Market overreaction signal" for t in takeaways)
    assert all(t["tone"] in {"warning", "positive", "info"} for t in takeaways)


def test_takeaway_complacency_rule() -> None:
    state = _state_row(crowd_panic_score=-0.6, delta_xg_10min=0.1)
    takeaways = generate_takeaways(state, threshold=0.9)
    assert any(t["headline"] == "Complacency risk" for t in takeaways)


def test_takeaway_anger_rule() -> None:
    state = _state_row(emo_anger=0.5)
    takeaways = generate_takeaways(state, threshold=0.9)
    assert any(t["headline"] == "Anger-driven sentiment" for t in takeaways)


def test_takeaway_aligned_default_includes_stats() -> None:
    stats = {"Mexico": {"possessionPct": "61", "shotsOnTarget": "5"}}
    takeaways = generate_takeaways(_state_row(), threshold=0.9, match_stats=stats)
    assert len(takeaways) == 1
    assert takeaways[0]["tone"] == "positive"
    assert "61" in takeaways[0]["detail"]


def test_takeaway_empty_state() -> None:
    takeaways = generate_takeaways(pd.DataFrame(), threshold=0.5)
    assert takeaways[0]["headline"] == "No crowd signal yet"
