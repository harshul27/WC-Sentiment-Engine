"""Tests for per-team reaction attribution in src/teams.py."""
from __future__ import annotations

import pandas as pd

import teams


def test_tag_reactions_assigns_sides() -> None:
    messages = pd.Series(
        [
            "Mexico are choking again",
            "what a goal from South Africa",
            "Mexico vs South Africa is end to end",
            "the referee is awful",
        ]
    )
    tags = teams.tag_reactions(messages, "Mexico", "South Africa")
    assert tags.tolist() == [teams.HOME, teams.AWAY, teams.BOTH, teams.NEITHER]


def test_tag_reactions_uses_aliases() -> None:
    messages = pd.Series(["come on the three lions", "vamos el tri"])
    tags = teams.tag_reactions(messages, "England", "Mexico")
    assert tags.tolist() == [teams.HOME, teams.AWAY]


def test_attribution_coverage() -> None:
    labels = pd.Series([teams.HOME, teams.AWAY, teams.BOTH, teams.NEITHER])
    assert teams.attribution_coverage(labels) == 0.75
    assert teams.attribution_coverage(pd.Series([], dtype="str")) == 0.0


def test_reactions_for_team_includes_both() -> None:
    chat = pd.DataFrame(
        {
            "minute": [1, 2, 3, 4],
            "message": ["a", "b", "c", "d"],
            "team": [teams.HOME, teams.AWAY, teams.BOTH, teams.NEITHER],
        }
    )
    home = teams.reactions_for_team(chat, teams.HOME)
    assert set(home["message"]) == {"a", "c"}
    assert teams.reactions_for_team(pd.DataFrame(), teams.HOME).empty
