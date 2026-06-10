"""Unit tests for the vectorized math and agent components in src/model.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import (
    ArbitrageSelector,
    MatchProgressionAgent,
    SocialListeningAgent,
    compute_arbitrage_index,
    grid_search_threshold,
    llm_panic_score,
    log_loss,
    parse_commentary,
    rolling_xg_stability,
    score_messages,
)


def test_score_messages_panic_is_positive() -> None:
    scores = score_messages(pd.Series(["this is a disaster we are choking"]))
    assert scores.iloc[0] > 0.5


def test_score_messages_calm_is_negative() -> None:
    scores = score_messages(pd.Series(["we look comfortable and in control"]))
    assert scores.iloc[0] < -0.5


def test_score_messages_bounded_and_handles_missing() -> None:
    scores = score_messages(pd.Series(["panic " * 50, None, ""]))
    assert float(scores.abs().max()) <= 1.0
    assert scores.iloc[1] == 0.0
    assert scores.iloc[2] == 0.0


def test_llm_panic_score_returns_none_without_keys() -> None:
    assert llm_panic_score("we are doomed") is None


def test_parse_commentary_extracts_events() -> None:
    lines = pd.Series(
        [
            "23' Brazil: shot on target, forces a save",
            "45+2' Argentina: goal! clinical finish",
            "not a commentary line at all",
            "60' Brazil: keeps possession in midfield",
        ]
    )
    events = parse_commentary(lines)
    assert len(events) == 3
    assert events.loc[0, "minute"] == 23
    assert events.loc[0, "team"] == "Brazil"
    assert events.loc[0, "event_type"] == "shot_on_target"
    assert events.loc[1, "event_type"] == "goal"
    assert events.loc[2, "event_type"] == "play"
    assert events.loc[2, "xg_value"] == 0.0


def test_rolling_xg_stability_bounded_and_minute_indexed() -> None:
    events = parse_commentary(
        pd.Series([f"{m}' Brazil: shot on target, forces a save" for m in range(0, 30, 3)])
    )
    stability = rolling_xg_stability(events, window_minutes=10)
    assert list(stability["minute"]) == list(range(28))
    assert float(stability["delta_xg_10min"].min()) >= 0.0
    assert float(stability["delta_xg_10min"].max()) <= 1.0


def test_rolling_xg_stability_empty_input() -> None:
    stability = rolling_xg_stability(parse_commentary(pd.Series([], dtype="str")))
    assert stability.empty
    assert list(stability.columns) == ["minute", "rolling_xg", "delta_xg_10min"]


def test_compute_arbitrage_index_core_equation() -> None:
    index = compute_arbitrage_index(np.array([0.8, -0.8, 0.0]), np.array([0.25, 0.0, 0.9]))
    assert index == pytest.approx([0.6, 0.8, 0.0])


def test_compute_arbitrage_index_clips_out_of_range_inputs() -> None:
    index = compute_arbitrage_index(np.array([5.0]), np.array([-3.0]))
    assert index == pytest.approx([1.0])


def test_log_loss_known_value() -> None:
    value = log_loss(np.array([1.0, 0.0]), np.array([0.8, 0.2]))
    assert value == pytest.approx(-np.log(0.8), rel=1e-6)


def test_grid_search_threshold_separates_classes() -> None:
    index = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    truth = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    result = grid_search_threshold(index, truth)
    assert 0.2 < result["arbitrage_flag_threshold"] < 0.8
    assert result["log_loss"] < log_loss(truth, np.full(6, 0.5))


def test_social_listening_agent_minute_indexed_and_bounded() -> None:
    chat = pd.DataFrame(
        {
            "minute": [0, 0, 1, 3, 3, 3],
            "message": [
                "total panic we are done",
                "disaster, choking again",
                "we look comfortable",
                "panic panic panic",
                "awful, terrible defending",
                "no chance we hold on",
            ],
        }
    )
    result = SocialListeningAgent(window_minutes=2, use_llm=False).run(chat)
    assert list(result["minute"]) == [0, 1, 2, 3]
    assert float(result["crowd_panic_score"].abs().max()) <= 1.0
    assert result["crowd_panic_score"].iloc[3] > 0.0


def test_social_listening_agent_empty_chat() -> None:
    result = SocialListeningAgent(use_llm=False).run(pd.DataFrame(columns=["minute", "message"]))
    assert result.empty


def test_arbitrage_selector_flags_panic_during_stable_match() -> None:
    social = pd.DataFrame({"minute": [0, 1, 2], "crowd_panic_score": [0.0, 0.9, 0.1]})
    match = pd.DataFrame(
        {
            "minute": [0, 1, 2],
            "rolling_xg": [0.0, 0.05, 0.05],
            "delta_xg_10min": [0.0, 0.05, 0.05],
        }
    )
    state = ArbitrageSelector(threshold=0.5).run(social, match)
    assert bool(state.loc[1, "flagged"])
    assert not bool(state.loc[0, "flagged"])
    assert not bool(state.loc[2, "flagged"])
    assert state["arbitrage_index"].iloc[1] == pytest.approx(0.9 * 0.95)


def test_arbitrage_selector_output_schema() -> None:
    agent_a = SocialListeningAgent(use_llm=False)
    agent_b = MatchProgressionAgent()
    chat = pd.DataFrame({"minute": [0, 1], "message": ["panic", "calm"]})
    commentary = pd.Series(["0' Brazil: kicks off", "1' Brazil: wins a corner"])
    state = ArbitrageSelector().run(agent_a.run(chat), agent_b.run(commentary))
    assert dict(state.dtypes.astype(str)) == {
        "minute": "int64",
        "crowd_panic_score": "float64",
        "rolling_xg": "float64",
        "delta_xg_10min": "float64",
        "arbitrage_index": "float64",
        "flagged": "bool",
    }
    assert not state.isna().any().any()
