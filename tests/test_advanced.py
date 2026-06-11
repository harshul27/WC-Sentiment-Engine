"""Offline tests for the soccerdata priors layer in src/advanced.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import advanced

SCHEDULE = pd.DataFrame(
    {
        "home_team": ["Mexico", "South Korea", "Mexico"],
        "away_team": ["South Africa", "Czechia", "Czechia"],
        "score": ["2-0", "1–1", None],
    }
)


def test_build_priors_from_schedule_scores() -> None:
    priors = advanced.build_priors(SCHEDULE)
    assert set(priors["team"]) == {"Mexico", "South Africa", "South Korea", "Czechia"}
    mexico = priors.loc[priors["team"] == "Mexico"].iloc[0]
    assert mexico["matches_played"] == 1
    assert mexico["goals_for_per_match"] == 2.0
    assert mexico["goals_against_per_match"] == 0.0
    korea = priors.loc[priors["team"] == "South Korea"].iloc[0]
    assert korea["goals_for_per_match"] == 1.0
    assert np.isnan(mexico["xg_for_per_match"])


def test_build_priors_skips_unplayed_and_bad_scores() -> None:
    schedule = pd.DataFrame(
        {
            "home_team": ["A", "B"],
            "away_team": ["C", "D"],
            "score": [None, "abandoned"],
        }
    )
    assert advanced.build_priors(schedule).empty


def test_fixture_prior_edge_and_lookup(tmp_path: Path) -> None:
    priors = advanced.build_priors(SCHEDULE)
    path = tmp_path / "team_priors.parquet"
    priors.to_parquet(path)
    loaded = advanced.load_priors(path)
    prior = advanced.fixture_prior("mexico", "South Africa", priors=loaded)
    assert prior is not None
    assert prior["edge"] == 4.0
    assert prior["home"]["team"] == "Mexico"


def test_fixture_prior_unknown_teams() -> None:
    priors = advanced.build_priors(SCHEDULE)
    assert advanced.fixture_prior("Atlantis", "Mordor", priors=priors) is None


def test_load_priors_missing_file() -> None:
    frame = advanced.load_priors(Path("does/not/exist.parquet"))
    assert frame.empty
    assert list(frame.columns) == advanced.PRIOR_COLUMNS
