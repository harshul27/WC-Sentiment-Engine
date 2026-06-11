"""Custom emotion model: dynamic crowd-emotion classification during matches.

A dependency-free, fully vectorized classifier that scores every fan comment
against six football-specific emotion lexicons, aggregates per-minute emotion
distributions, tracks the dominant emotion and its volatility, and converts
the profile into the engine's bounded Crowd Panic Score plus plain-language
takeaways that relate crowd emotion to the actual match state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

EMOTIONS: tuple[str, ...] = ("panic", "anger", "joy", "confidence", "despair", "surprise")
EMOTION_COLUMNS: list[str] = [f"emo_{name}" for name in EMOTIONS]

_EMOTION_LEXICON: dict[str, dict[str, float]] = {
    "panic": {
        "panic": 1.0,
        "we are done": 1.0,
        "we're done": 1.0,
        "gonna lose": 0.9,
        "throwing it away": 0.9,
        "collapse": 0.9,
        "nervous": 0.7,
        "anxious": 0.7,
        "scared": 0.8,
        "terrified": 1.0,
        "oh no": 0.6,
        "can't watch": 0.8,
        "cant watch": 0.8,
        "bottle": 0.7,
        "choke": 0.8,
        "choking": 0.8,
        "heart attack": 0.7,
        "here we go again": 0.6,
        "hold on": 0.4,
    },
    "anger": {
        "robbed": 0.9,
        "disgrace": 0.9,
        "pathetic": 0.8,
        "useless": 0.7,
        "sack": 0.8,
        "fire the": 0.8,
        "trash": 0.7,
        "garbage": 0.7,
        "what is he doing": 0.7,
        "shocking": 0.6,
        "embarrassing": 0.7,
        "rubbish": 0.6,
        "terrible call": 0.9,
        "var robbery": 1.0,
        "cheat": 0.8,
        "corrupt": 0.9,
        "awful": 0.6,
        "ref is": 0.5,
    },
    "joy": {
        "what a goal": 1.0,
        "golazo": 1.0,
        "amazing": 0.7,
        "brilliant": 0.8,
        "incredible": 0.8,
        "unbelievable": 0.7,
        "yes!!": 0.9,
        "let's go": 0.7,
        "lets go": 0.7,
        "vamos": 0.7,
        "beautiful": 0.6,
        "stunning": 0.7,
        "love it": 0.6,
        "get in": 0.7,
        "screamer": 0.9,
        "masterclass": 0.8,
        "goal!!": 0.9,
        "what a save": 0.7,
    },
    "confidence": {
        "in control": 0.9,
        "comfortable": 0.8,
        "cruising": 0.9,
        "easy": 0.5,
        "dominating": 0.9,
        "dominant": 0.8,
        "got this": 0.7,
        "no worries": 0.8,
        "calm": 0.6,
        "composed": 0.7,
        "winning this": 0.7,
        "routine": 0.6,
        "professional": 0.5,
        "relax": 0.5,
    },
    "despair": {
        "it's over": 0.9,
        "its over": 0.9,
        "hopeless": 1.0,
        "no hope": 1.0,
        "give up": 0.8,
        "season over": 0.8,
        "heartbroken": 1.0,
        "gutted": 0.9,
        "devastated": 1.0,
        "crying": 0.7,
        "why do i support": 0.9,
        "never win": 0.8,
        "we lost": 0.7,
        "pain": 0.5,
    },
    "surprise": {
        "no way": 0.8,
        "can't believe": 0.8,
        "cant believe": 0.8,
        "out of nowhere": 0.9,
        "shocked": 0.8,
        "stunned": 0.8,
        "wow": 0.6,
        "omg": 0.6,
        "didn't see that": 0.8,
        "didnt see that": 0.8,
        "plot twist": 0.7,
        "what just happened": 0.9,
    },
}


def classify_comments(messages: pd.Series) -> pd.DataFrame:
    """Vectorized per-comment emotion intensities, one column per emotion.

    Intensities are raw weighted term counts (>= 0); a comment matching no
    lexicon term scores zero everywhere and is treated as neutral downstream.
    """
    text = messages.fillna("").astype(str).str.lower()
    intensities: dict[str, np.ndarray] = {}
    for emotion, lexicon in _EMOTION_LEXICON.items():
        raw = np.zeros(len(text), dtype=np.float64)
        for term, weight in lexicon.items():
            raw += text.str.count(re.escape(term)).to_numpy(dtype=np.float64) * weight
        intensities[f"emo_{emotion}"] = raw
    return pd.DataFrame(intensities, index=messages.index)


def emotion_shares(intensities: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw intensities into per-comment emotion shares (rows sum
    to 1 when any emotion fired, all-zero rows stay zero / neutral)."""
    totals = intensities.sum(axis=1)
    shares = intensities.div(totals.replace(0.0, np.nan), axis=0).fillna(0.0)
    return shares


def minute_profile(chat: pd.DataFrame) -> pd.DataFrame:
    """Minute-indexed mean emotion distribution with comment volume.

    Expects chat columns: minute, message. Minutes with no comments are
    forward-filled so the profile is continuous across the match.
    """
    empty = pd.DataFrame(
        columns=["minute", *EMOTION_COLUMNS, "comment_volume"]
    ).astype({"minute": "int64", "comment_volume": "int64"})
    if chat.empty:
        return empty
    shares = emotion_shares(classify_comments(chat["message"]))
    frame = pd.concat([chat[["minute"]].reset_index(drop=True), shares.reset_index(drop=True)], axis=1)
    last_minute = int(frame["minute"].max())
    grouped = frame.groupby("minute")[EMOTION_COLUMNS].mean()
    volume = frame.groupby("minute").size().rename("comment_volume")
    profile = (
        grouped.join(volume)
        .reindex(range(last_minute + 1))
        .ffill()
        .fillna(0.0)
        .reset_index(names="minute")
    )
    profile["comment_volume"] = profile["comment_volume"].astype("int64")
    profile["minute"] = profile["minute"].astype("int64")
    return profile


def dominant_emotion(profile: pd.DataFrame) -> pd.Series:
    """Strongest emotion per minute; 'neutral' when nothing fired."""
    if profile.empty:
        return pd.Series(dtype="str", name="dominant_emotion")
    shares = profile[EMOTION_COLUMNS]
    label = shares.idxmax(axis=1).str.removeprefix("emo_")
    label[shares.max(axis=1) <= 0.0] = "neutral"
    return label.rename("dominant_emotion")


def emotional_volatility(profile: pd.DataFrame, window: int = 5) -> pd.Series:
    """Rolling crowd-mood instability in [0, 1].

    Half the L1 distance between consecutive minute distributions, smoothed:
    0 = stable mood, 1 = the crowd's emotional mix flipping entirely.
    """
    if profile.empty:
        return pd.Series(dtype="float64", name="emotional_volatility")
    shares = profile[EMOTION_COLUMNS].to_numpy(dtype=np.float64)
    step = np.zeros(len(shares), dtype=np.float64)
    if len(shares) > 1:
        step[1:] = np.abs(np.diff(shares, axis=0)).sum(axis=1) / 2.0
    return (
        pd.Series(step, index=profile.index, name="emotional_volatility")
        .rolling(window=window, min_periods=1)
        .mean()
        .clip(0.0, 1.0)
    )


def panic_from_profile(profile: pd.DataFrame) -> pd.Series:
    """Map the emotion distribution onto the engine's bounded panic score.

    Negative emotions push toward +1 (panic), positive emotions toward -1
    (confidence); surprise is treated as direction-neutral.
    """
    if profile.empty:
        return pd.Series(dtype="float64", name="crowd_panic_score")
    negative = (
        1.2 * profile["emo_panic"]
        + 1.1 * profile["emo_despair"]
        + 0.6 * profile["emo_anger"]
    )
    positive = 1.0 * profile["emo_confidence"] + 0.9 * profile["emo_joy"]
    return pd.Series(
        np.tanh(2.0 * (negative - positive).to_numpy(dtype=np.float64)),
        index=profile.index,
        name="crowd_panic_score",
    )


@dataclass
class EmotionAgent:
    """Agent A v2: minute-indexed emotion profile + derived panic score."""

    window_minutes: int = 5

    def run(self, chat: pd.DataFrame) -> pd.DataFrame:
        profile = minute_profile(chat)
        if profile.empty:
            return pd.DataFrame(
                columns=[
                    "minute",
                    "crowd_panic_score",
                    *EMOTION_COLUMNS,
                    "dominant_emotion",
                    "emotional_volatility",
                    "comment_volume",
                ]
            ).astype({"minute": "int64", "crowd_panic_score": "float64"})
        panic = (
            panic_from_profile(profile)
            .rolling(window=self.window_minutes, min_periods=1)
            .mean()
            .clip(-1.0, 1.0)
        )
        result = profile.copy()
        result.insert(1, "crowd_panic_score", panic)
        result["dominant_emotion"] = dominant_emotion(profile)
        result["emotional_volatility"] = emotional_volatility(profile)
        return result


def generate_takeaways(
    state: pd.DataFrame,
    threshold: float,
    match_stats: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Plain-language product takeaways relating crowd emotion to the game.

    Each takeaway: {"tone": info|warning|positive, "headline": ..., "detail": ...}.
    Rules fire on the latest scored minute of the supplied state frame.
    """
    if state.empty:
        return [
            {
                "tone": "info",
                "headline": "No crowd signal yet",
                "detail": "Waiting for enough fan reactions to build an emotion profile.",
            }
        ]
    latest = state.iloc[-1]
    panic = float(latest.get("crowd_panic_score", 0.0))
    stability = float(latest.get("delta_xg_10min", 0.0))
    index = float(latest.get("arbitrage_index", 0.0))
    dominant = str(latest.get("dominant_emotion", "neutral"))
    volatility = float(latest.get("emotional_volatility", 0.0))
    anger = float(latest.get("emo_anger", 0.0))
    takeaways: list[dict[str, str]] = []
    stats_note = ""
    if match_stats:
        fragments = [
            f"{team}: {stats.get('possessionPct', '?')}% possession, "
            f"{stats.get('shotsOnTarget', '?')} on target"
            for team, stats in match_stats.items()
        ]
        stats_note = " Pitch data - " + "; ".join(fragments) + "."
    if index >= threshold and panic > 0:
        takeaways.append(
            {
                "tone": "warning",
                "headline": "Market overreaction signal",
                "detail": (
                    f"Crowd mood is dominated by {dominant} while the underlying threat "
                    f"level is stable (xG stability {stability:.2f}). Historically the spot "
                    f"where sentiment-driven prices decouple from the pitch.{stats_note}"
                ),
            }
        )
    if panic <= -0.4 and stability < 0.3:
        takeaways.append(
            {
                "tone": "warning",
                "headline": "Complacency risk",
                "detail": (
                    "The crowd is euphoric/confident but the match data shows little "
                    f"attacking control (xG stability {stability:.2f}). Sentiment may be "
                    f"lagging a genuine momentum shift.{stats_note}"
                ),
            }
        )
    if anger >= 0.35:
        takeaways.append(
            {
                "tone": "info",
                "headline": "Anger-driven sentiment",
                "detail": (
                    "A large share of reactions are about officiating or frustration, "
                    "not match state. Treat the panic score as noisy until this clears."
                ),
            }
        )
    if volatility >= 0.5:
        takeaways.append(
            {
                "tone": "info",
                "headline": "Unstable crowd mood",
                "detail": (
                    f"Emotional volatility is high ({volatility:.2f}); the crowd is reacting "
                    "to every event. Wait for the mood to settle before trusting the signal."
                ),
            }
        )
    if not takeaways:
        takeaways.append(
            {
                "tone": "positive",
                "headline": "Sentiment aligned with match state",
                "detail": (
                    f"Dominant emotion ({dominant}) is consistent with the pitch data "
                    f"(panic {panic:+.2f}, xG stability {stability:.2f}). No divergence "
                    f"to exploit right now.{stats_note}"
                ),
            }
        )
    return takeaways
