"""Tests for mood-vs-game consistency in src/consistency.py."""
from __future__ import annotations

import consistency


def _summary(dominant: str, volume: int = 40, coverage: float = 0.9) -> dict:
    shares = {f"emo_{name}": 0.05 for name in ("panic", "anger", "joy", "confidence", "despair", "surprise")}
    shares[f"emo_{dominant}"] = 0.6
    return {
        "team": "TeamX",
        "dominant": dominant,
        "shares": shares,
        "volume": volume,
        "coverage": coverage,
    }


def test_parse_score_variants() -> None:
    assert consistency.parse_score("2-1") == (2, 1)
    assert consistency.parse_score("0–0") == (0, 0)  # en dash
    assert consistency.parse_score("junk") is None
    assert consistency.parse_score("") is None


def test_game_context_statuses() -> None:
    ctx = consistency.game_context("2-0", "Mexico", "South Africa", 0.4)
    assert ctx["home"]["status"] == "leading"
    assert ctx["away"]["status"] == "trailing"
    assert ctx["away"]["margin"] == 2
    level = consistency.game_context("1-1", "A", "B")
    assert level["home"]["status"] == "level" == level["away"]["status"]


def test_trailing_team_joy_is_conflict() -> None:
    """The user's core case: a losing side cannot read as joyful without a flag."""
    ctx = consistency.game_context("0-2", "Mexico", "South Africa")
    verdicts = consistency.mood_consistency(
        {"home": _summary("joy")}, {"home": ctx["home"]}
    )
    assert verdicts["home"]["verdict"] == "conflict"
    assert "overreaction" in verdicts["home"]["explanation"].lower()


def test_trailing_team_panic_is_consistent() -> None:
    ctx = consistency.game_context("0-2", "Mexico", "South Africa")
    verdicts = consistency.mood_consistency(
        {"home": _summary("panic")}, {"home": ctx["home"]}
    )
    assert verdicts["home"]["verdict"] == "consistent"


def test_leading_team_panic_without_pressure_is_conflict() -> None:
    ctx = consistency.game_context("2-0", "Mexico", "South Africa")
    verdicts = consistency.mood_consistency(
        {"home": _summary("panic")}, {"home": ctx["home"]}
    )
    assert verdicts["home"]["verdict"] == "conflict"


def test_leading_team_panic_with_busy_keeper_is_consistent() -> None:
    keeper = {"Mexico": {"keeper": "Ochoa", "saves": 6.0, "shots_faced": 8.0, "conceded": 0.0}}
    ctx = consistency.game_context("1-0", "Mexico", "South Africa", keeper=keeper)
    verdicts = consistency.mood_consistency(
        {"home": _summary("panic")}, {"home": ctx["home"]}
    )
    assert verdicts["home"]["verdict"] == "consistent"
    assert "keeper" in verdicts["home"]["explanation"].lower()


def test_clarity_score_behaviour() -> None:
    decisive = consistency.clarity_score(_summary("panic", volume=60, coverage=1.0))
    thin = consistency.clarity_score(
        {"shares": {"emo_panic": 0.2, "emo_joy": 0.19}, "volume": 2, "coverage": 0.2}
    )
    assert 0.0 <= thin < decisive <= 1.0


def test_conflict_moments_shape() -> None:
    ctx = consistency.game_context("0-1", "A", "B")
    verdicts = consistency.mood_consistency(
        {"home": _summary("confidence"), "away": _summary("joy")},
        ctx,
    )
    moments = consistency.conflict_moments(verdicts)
    reasons = {m["reason"] for m in moments}
    assert "positive-while-losing" in reasons  # trailing home side is confident
    assert all({"reason", "team", "detail"} <= set(m) for m in moments)
