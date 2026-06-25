"""Plain-language labels and tooltips for the dashboard.

A single source of truth so every metric reads the same way and a general
audience (not just analysts) can understand it. The internal column names in
the data and models are unchanged - this only governs how things are *shown*,
so the validated model and the archive schema never churn.
"""
from __future__ import annotations

# internal column / concept -> (display label, one-line plain tooltip)
LABELS: dict[str, tuple[str, str]] = {
    "crowd_panic_score": (
        "Fan Mood",
        "How the fans sound right now, from -1 (calm and confident) to "
        "+1 (anxious and panicking).",
    ),
    "delta_xg_10min": (
        "Attacking Threat",
        "How much real goal danger the match has produced in the last 10 "
        "minutes, from the play-by-play.",
    ),
    "arbitrage_index": (
        "Hype-vs-Reality Gap",
        "How far fan mood has run ahead of what is actually happening on the "
        "pitch. A big gap can mean the crowd is overreacting.",
    ),
    "rolling_xg": (
        "Goal Threat (rolling)",
        "Rolling expected-goals built up over the match so far.",
    ),
    "emotional_volatility": (
        "Mood Swings",
        "How fast the crowd's emotions are flipping minute to minute.",
    ),
    "comment_volume": (
        "Reactions / min",
        "How many fan posts were captured for this minute.",
    ),
    "situation": (
        "Match Read",
        "The model's plain read of the moment (e.g. even contest, late drama, "
        "overreaction).",
    ),
    "flagged": (
        "Overreaction Moments",
        "Minutes where the crowd spiked but the match itself stayed calm.",
    ),
    "sentiment_market_divergence": (
        "Mood-vs-Odds Gap",
        "High when fans are agitated while bookmakers still price a settled "
        "result.",
    ),
    "scored_share": (
        "Reading Coverage",
        "Share of captured posts the model could actually read (across "
        "languages and emoji). Higher is more representative.",
    ),
    "attribution_coverage": (
        "Team Coverage",
        "Share of reactions we could tie to one specific team.",
    ),
}

TITLE = "⚽ World Cup Crowd Mood Engine"
SUBTITLE = (
    "Reads what football fans feel in real time, team by team, and lines it up "
    "against what is actually happening in the match."
)

# Short "what am I looking at" guide rendered in an expander.
GUIDE: list[tuple[str, str]] = [
    (
        "What this does",
        "During a live match it reads thousands of fan posts, works out the "
        "crowd's mood for each team, and compares it to the real match action.",
    ),
    (
        "Fan Mood",
        "A single number per team from calm/confident (-1) to anxious/"
        "panicking (+1), built from a model that reads emotion across languages.",
    ),
    (
        "Hype-vs-Reality Gap",
        "When fans are spiking but the match is calm, the gap is wide - a "
        "possible overreaction. That is the moment the engine flags.",
    ),
    (
        "Is it trustworthy?",
        "The emotion model is tested against human-labelled data (see the "
        "accuracy panel). It is an analytics tool, not betting advice.",
    ),
]


def label(key: str) -> str:
    """Display label for an internal column/concept (falls back to a title-cased key)."""
    if key in LABELS:
        return LABELS[key][0]
    return key.replace("_", " ").title()


def tooltip(key: str) -> str:
    """Plain-language tooltip for a metric, or empty string when none exists."""
    return LABELS[key][1] if key in LABELS else ""
