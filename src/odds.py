"""Live bookmaker odds via The Odds API (key-gated, read-only).

Closes the "no market to compare against" gap: when ODDS_API_KEY is set, this
fetches the bookmaker consensus for a fixture, de-vigs it into implied
probabilities, and exposes a market-certainty value plus a sentiment-vs-market
divergence so a flagged crowd overreaction can be read against what the market
actually prices.

Strictly read-only: the engine never places, suggests, or sizes a wager, and
none of this is betting or financial advice. The Odds API free tier is ~500
requests/month, so callers poll it sparingly (the live panel, not every tick).
"""
from __future__ import annotations

import os

import requests

ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = os.environ.get("ODDS_SPORT_KEY", "soccer_fifa_world_cup")
USER_AGENT = {"User-Agent": "wc-sentiment-engine/0.1"}


def odds_enabled() -> bool:
    """True when an Odds API key is configured."""
    return bool(os.environ.get("ODDS_API_KEY"))


def implied_probabilities(decimal_odds: dict[str, float]) -> dict[str, float]:
    """De-vig decimal odds into probabilities that sum to 1.0.

    The bookmaker margin ("vig") makes raw inverse-odds sum to >1; normalising
    removes it to recover the market's implied outcome probabilities.
    """
    raw = {
        name: 1.0 / price
        for name, price in decimal_odds.items()
        if isinstance(price, (int, float)) and price > 0
    }
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {name: round(value / total, 4) for name, value in raw.items()}


def _consensus(event: dict) -> dict[str, float]:
    """Average decimal odds for the head-to-head market across bookmakers."""
    collected: dict[str, list[float]] = {}
    for bookmaker in event.get("bookmakers", []) or []:
        for market in bookmaker.get("markets", []) or []:
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []) or []:
                name = str(outcome.get("name", ""))
                price = outcome.get("price")
                if name and isinstance(price, (int, float)) and price > 0:
                    collected.setdefault(name, []).append(float(price))
    return {name: sum(vals) / len(vals) for name, vals in collected.items() if vals}


def fetch_odds(sport_key: str = SPORT_KEY, timeout: float = 20.0) -> list[dict]:
    """Today's fixtures with bookmaker odds; empty list when unavailable."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        return []
    try:
        response = requests.get(
            f"{ODDS_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey": key,
                "regions": "us,uk,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=timeout,
            headers=USER_AGENT,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return []
    return data if isinstance(data, list) else []


def _matches(event: dict, home: str, away: str) -> bool:
    blob = (
        f"{str(event.get('home_team', '')).lower()} "
        f"{str(event.get('away_team', '')).lower()}"
    )
    wanted = [t.lower().strip() for t in (home, away) if t]
    return bool(wanted) and all(team in blob for team in wanted)


def fixture_market(
    home_team: str, away_team: str, events: list[dict] | None = None
) -> dict | None:
    """De-vigged market view for one fixture, or None when not found.

    Returns home/draw/away probabilities, the implied favorite, and a
    `certainty` value (the market's strongest outcome probability) used to
    gauge how settled the market thinks the match is.
    """
    events = fetch_odds() if events is None else events
    for event in events or []:
        if not _matches(event, home_team, away_team):
            continue
        probabilities = implied_probabilities(_consensus(event))
        if not probabilities:
            return None
        home_name = str(event.get("home_team", ""))
        away_name = str(event.get("away_team", ""))
        favorite = max(probabilities, key=probabilities.get)
        return {
            "home_prob": probabilities.get(home_name),
            "draw_prob": probabilities.get("Draw"),
            "away_prob": probabilities.get(away_name),
            "favorite": favorite,
            "certainty": round(max(probabilities.values()), 4),
            "probabilities": probabilities,
        }
    return None


def sentiment_market_divergence(crowd_panic_score: float, certainty: float) -> float:
    """Crowd-vs-market divergence in [0, 1].

    |crowd panic| x market certainty: large when the crowd is agitated while
    the market still prices a near-settled outcome - i.e. sentiment running
    ahead of a market that has not repriced. A measured reference for what was
    previously only an internal hypothesis.
    """
    return round(abs(float(crowd_panic_score)) * float(certainty), 4)
