"""Tests for the key-gated bookmaker odds connector in src/odds.py."""
from __future__ import annotations

import pytest

import odds


def test_odds_disabled_without_key() -> None:
    assert odds.odds_enabled() is False
    assert odds.fetch_odds() == []
    assert odds.fixture_market("Mexico", "South Africa") is None


def test_implied_probabilities_devigs_to_one() -> None:
    probs = odds.implied_probabilities({"Mexico": 2.0, "Draw": 4.0, "South Africa": 4.0})
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert probs["Mexico"] > probs["South Africa"]


def test_consensus_averages_bookmakers() -> None:
    event = {
        "home_team": "Mexico",
        "away_team": "South Africa",
        "bookmakers": [
            {"markets": [{"key": "h2h", "outcomes": [
                {"name": "Mexico", "price": 2.0}, {"name": "South Africa", "price": 4.0}
            ]}]},
            {"markets": [{"key": "h2h", "outcomes": [
                {"name": "Mexico", "price": 2.4}, {"name": "South Africa", "price": 3.6}
            ]}]},
        ],
    }
    avg = odds._consensus(event)
    assert avg["Mexico"] == pytest.approx(2.2)
    assert avg["South Africa"] == pytest.approx(3.8)


def test_fixture_market_parses_sample_events() -> None:
    events = [
        {
            "home_team": "Mexico",
            "away_team": "South Africa",
            "bookmakers": [
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "Mexico", "price": 1.5},
                    {"name": "Draw", "price": 4.0},
                    {"name": "South Africa", "price": 7.0},
                ]}]}
            ],
        }
    ]
    market = odds.fixture_market("Mexico", "South Africa", events=events)
    assert market is not None
    assert market["favorite"] == "Mexico"
    assert market["home_prob"] > market["away_prob"]
    assert 0.0 < market["certainty"] <= 1.0
    assert sum(market["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_fixture_market_no_match_returns_none() -> None:
    events = [{"home_team": "Brazil", "away_team": "Argentina", "bookmakers": []}]
    assert odds.fixture_market("Mexico", "South Africa", events=events) is None


def test_sentiment_market_divergence() -> None:
    assert odds.sentiment_market_divergence(0.8, 0.75) == pytest.approx(0.6)
    assert odds.sentiment_market_divergence(-0.5, 0.5) == pytest.approx(0.25)
    assert odds.sentiment_market_divergence(0.0, 0.9) == 0.0
