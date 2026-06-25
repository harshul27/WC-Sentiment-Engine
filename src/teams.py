"""Attribute each fan reaction to a team.

The crowd stream is pooled across both sides, so an emotion like "anger" is
ambiguous - angry at which team? This module tags every reaction as referring
to the home side, the away side, both, or neither, by matching team-name tokens
and a small alias map (common nicknames and the alternate names broadcasters
use). It is deliberately simple and transparent; the dashboard also reports the
share of reactions that could be attributed, so the coverage gap stays visible.
"""
from __future__ import annotations

import re

import pandas as pd

HOME = "home"
AWAY = "away"
BOTH = "both"
NEITHER = "neither"

# Alternate names / nicknames -> extra match tokens. Keys are lowercased
# substrings of the official name; values are additional whole-word tokens.
_ALIASES: dict[str, tuple[str, ...]] = {
    "korea republic": ("south korea", "korea", "taeguk"),
    "south korea": ("korea republic", "korea"),
    "united states": ("usa", "usmnt", "usneverything"),
    "ivory coast": ("cote d'ivoire", "cote divoire"),
    "netherlands": ("holland", "oranje"),
    "england": ("three lions", "engl"),
    "argentina": ("albiceleste", "argentinos"),
    "brazil": ("brasil", "selecao", "seleção"),
    "germany": ("deutschland", "die mannschaft"),
    "spain": ("espana", "españa", "la roja"),
    "mexico": ("el tri", "mexicanos"),
    "portugal": ("seleccao", "seleção das quinas"),
    "france": ("les bleus", "francia"),
    "croatia": ("hrvatska", "vatreni"),
}


def _tokens_for(name: str) -> list[str]:
    """Lowercased match tokens for a team: significant name words + aliases."""
    clean = str(name or "").lower().strip()
    if not clean:
        return []
    tokens: set[str] = set()
    # whole name and its significant words (drop short connective words)
    tokens.add(clean)
    for word in re.findall(r"[a-z']+", clean):
        if len(word) >= 4 and word not in {"team", "republic", "and"}:
            tokens.add(word)
    for key, extra in _ALIASES.items():
        if key in clean:
            tokens.update(extra)
    return [t for t in tokens if t]


def _mentions(text: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    low = str(text).lower()
    return any(tok in low for tok in tokens)


def tag_reactions(messages: pd.Series, home_team: str, away_team: str) -> pd.Series:
    """Label each message home | away | both | neither for the two teams."""
    home_tokens = _tokens_for(home_team)
    away_tokens = _tokens_for(away_team)
    labels: list[str] = []
    for message in messages.fillna("").astype(str):
        h = _mentions(message, home_tokens)
        a = _mentions(message, away_tokens)
        if h and a:
            labels.append(BOTH)
        elif h:
            labels.append(HOME)
        elif a:
            labels.append(AWAY)
        else:
            labels.append(NEITHER)
    return pd.Series(labels, index=messages.index, name="team")


def attribution_coverage(team_labels: pd.Series) -> float:
    """Share of reactions tied to at least one team (home/away/both)."""
    if team_labels is None or len(team_labels) == 0:
        return 0.0
    attributed = team_labels.isin([HOME, AWAY, BOTH]).mean()
    return float(round(attributed, 4))


def reactions_for_team(chat: pd.DataFrame, side: str) -> pd.DataFrame:
    """Rows referring to one side (`home`/`away`), including `both`-tagged posts."""
    if chat.empty or "team" not in chat.columns:
        return chat.iloc[0:0]
    return chat.loc[chat["team"].isin([side, BOTH])].reset_index(drop=True)
