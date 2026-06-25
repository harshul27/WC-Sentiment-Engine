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

import textmodel

EMOTIONS: tuple[str, ...] = ("panic", "anger", "joy", "confidence", "despair", "surprise")
EMOTION_COLUMNS: list[str] = [f"emo_{name}" for name in EMOTIONS]

# The World Cup crowd is global: an English-only lexicon would systematically
# under-read Spanish/Portuguese/French reactions and emoji-only posts, biasing
# the panic score toward English speakers. Each emotion therefore carries
# English terms, the highest-volume non-English football terms, and the emoji
# that dominate live reaction streams. scored_share() exposes how much of a
# window actually matched, so the remaining coverage gap stays visible.
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
        # es / pt / fr
        "no puede ser": 0.8,
        "nos van a remontar": 0.9,
        "que nervios": 0.8,
        "vamos a perder": 0.9,
        "que medo": 0.8,
        "vai dar ruim": 0.8,
        "on va perdre": 0.9,
        # emoji
        "😰": 0.8,
        "😨": 0.8,
        "😱": 0.7,
        "🥶": 0.6,
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
        # es / pt / fr
        "robo": 0.9,
        "ladrones": 0.9,
        "verguenza": 0.9,
        "vergüenza": 0.9,
        "roubo": 0.9,
        "vergonha": 0.9,
        "arbitre": 0.5,
        "scandale": 0.8,
        # emoji
        "🤬": 1.0,
        "😡": 0.9,
        "😠": 0.7,
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
        # es / pt / fr
        "golaco": 1.0,
        "golaço": 1.0,
        "que golazo": 1.0,
        "golzao": 0.9,
        "gooool": 0.9,
        "que jogo": 0.6,
        "magnifique": 0.8,
        "quel but": 0.9,
        # emoji
        "⚽": 0.4,
        "🔥": 0.5,
        "🎉": 0.7,
        "🥳": 0.7,
        "🤩": 0.7,
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
        # es / pt / fr
        "tranquilo": 0.7,
        "controlado": 0.8,
        "tranquilo todo": 0.8,
        "esta ganado": 0.7,
        "facil": 0.5,
        "tranquille": 0.7,
        # emoji
        "😎": 0.7,
        "💪": 0.6,
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
        # es / pt / fr
        "se acabo": 0.9,
        "se acabó": 0.9,
        "estamos eliminados": 1.0,
        "acabou": 0.9,
        "perdemos": 0.7,
        "c'est fini": 0.9,
        "elimines": 0.9,
        # emoji
        "😭": 0.9,
        "😢": 0.7,
        "💔": 0.9,
        "😞": 0.6,
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
        # es / pt / fr
        "no me lo creo": 0.8,
        "increible": 0.7,
        "increíble": 0.7,
        "nao acredito": 0.8,
        "inacreditavel": 0.8,
        "incroyable": 0.7,
        # emoji
        "😲": 0.7,
        "🤯": 0.8,
        "😳": 0.6,
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


def scored_share(messages: pd.Series) -> float:
    """Fraction of comments that matched at least one emotion term in [0, 1].

    Makes lexicon coverage observable: a low value means most of the window
    (e.g. non-supported languages or pure media) went unscored, so the panic
    score rests on a small, possibly biased slice of the crowd.
    """
    if messages is None or len(messages) == 0:
        return 0.0
    intensities = classify_comments(messages)
    matched = (intensities.sum(axis=1) > 0.0).mean()
    return float(round(matched, 4))


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


# --- trained-model scoring (primary) -------------------------------------
# The lexicon above is retained as a transparent, dependency-free fallback and
# as the source for the `confidence` emotion (no public dataset labels it). The
# primary scorer is the offline-trained model in data/models/ (91% accuracy on
# dair-ai/emotion vs 4% for the lexicon; multilingual panic direction from the
# char-n-gram sentiment model). See data/benchmarks/emotion_benchmark.json.
_MODEL_CACHE: dict[str, object | None] = {}


def _load_model(name: str) -> object | None:
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = textmodel.try_load(name)
    return _MODEL_CACHE[name]


def models_available() -> bool:
    """True when the trained emotion model artifacts are present."""
    return _load_model("emotion_model") is not None


# How strongly the football lexicon overrides the general-domain model when it
# fires. The trained model is excellent on the public benchmark but was trained
# on general tweets, so football phrasing ("we are done", "disgrace") can be
# mislabelled (often as joy). The curated lexicon is high-precision on exactly
# that phrasing, so where it fires we trust it; where it is silent (e.g. plain
# non-English posts) the model carries the load. _LEX_HALF sets the lexicon
# intensity at which the blend reaches half its cap; _LEX_MAX caps its weight.
_LEX_HALF = 0.6
_LEX_MAX = 0.7


def model_comment_scores(messages: pd.Series) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Per-comment emotion shares (6) + multilingual panic, hybrid scored.

    Returns None when no trained model is available (callers fall back to the
    lexicon). Each comment's distribution blends the trained model (recall +
    multilingual reach) with the football lexicon (precision on football
    phrasing), weighting the lexicon by how strongly it fired. Panic direction
    additionally folds in the multilingual sentiment model so non-English posts
    genuinely move the score.
    """
    emo_model = _load_model("emotion_model")
    if emo_model is None:
        return None
    texts = messages.fillna("").astype(str).tolist()
    proba = emo_model.predict_proba(texts)
    model_frame = pd.DataFrame(
        {f"emo_{cls}": proba[:, i] for i, cls in enumerate(emo_model.classes)},
        index=messages.index,
    ).reindex(columns=EMOTION_COLUMNS, fill_value=0.0)

    lex_intensity = classify_comments(messages)
    lex_shares = emotion_shares(lex_intensity)  # 6-col, all-zero when silent
    lex_strength = lex_intensity.sum(axis=1).to_numpy(dtype=np.float64)
    weight = np.minimum(_LEX_MAX, lex_strength / (lex_strength + _LEX_HALF))
    weight = np.where(lex_strength > 0.0, weight, 0.0)[:, None]

    blended = (1.0 - weight) * model_frame.to_numpy(dtype=np.float64) + (
        weight * lex_shares.to_numpy(dtype=np.float64)
    )
    frame = pd.DataFrame(blended, index=messages.index, columns=EMOTION_COLUMNS)
    totals = frame.sum(axis=1).replace(0.0, np.nan)
    shares = frame.div(totals, axis=0).fillna(0.0)

    negative = (
        1.2 * shares["emo_panic"] + 1.1 * shares["emo_despair"] + 0.6 * shares["emo_anger"]
    )
    positive = 1.0 * shares["emo_confidence"] + 0.9 * shares["emo_joy"]
    emotion_panic = (negative - positive).to_numpy(dtype=np.float64)
    sent_model = _load_model("sentiment_model")
    sentiment_panic = np.zeros(len(texts), dtype=np.float64)
    if sent_model is not None:
        sproba = sent_model.predict_proba(texts)
        index = {cls: i for i, cls in enumerate(sent_model.classes)}
        sentiment_panic = sproba[:, index["negative"]] - sproba[:, index["positive"]]
    panic = np.tanh(emotion_panic + sentiment_panic)
    return shares, panic


def model_minute_profile(chat: pd.DataFrame) -> pd.DataFrame | None:
    """Minute-indexed model emotion profile + per-minute panic (`_panic`)."""
    scored = model_comment_scores(chat["message"])
    if scored is None:
        return None
    shares, panic = scored
    frame = chat[["minute"]].reset_index(drop=True).copy()
    frame = pd.concat([frame, shares.reset_index(drop=True)], axis=1)
    frame["_panic"] = panic
    last_minute = int(frame["minute"].max())
    grouped = frame.groupby("minute")[[*EMOTION_COLUMNS, "_panic"]].mean()
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


@dataclass
class EmotionAgent:
    """Agent A v2: minute-indexed emotion profile + derived panic score.

    Uses the trained model when its artifacts are present, otherwise the
    interpretable lexicon - identical output schema either way.
    """

    window_minutes: int = 5

    def _empty(self) -> pd.DataFrame:
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

    def run(self, chat: pd.DataFrame) -> pd.DataFrame:
        if chat.empty:
            return self._empty()
        model_profile = model_minute_profile(chat) if models_available() else None
        if model_profile is not None:
            panic = (
                model_profile["_panic"]
                .rolling(window=self.window_minutes, min_periods=1)
                .mean()
                .clip(-1.0, 1.0)
            )
            result = model_profile.drop(columns="_panic")
            result.insert(1, "crowd_panic_score", panic)
            result["dominant_emotion"] = dominant_emotion(result)
            result["emotional_volatility"] = emotional_volatility(result)
            return result
        profile = minute_profile(chat)
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


def _profile_summary(messages: pd.Series) -> dict[str, object]:
    """Dominant emotion, mean shares, and reading coverage for a message set."""
    empty = {
        "dominant": "neutral",
        "shares": {col: 0.0 for col in EMOTION_COLUMNS},
        "volume": 0,
        "coverage": 0.0,
    }
    if messages is None or len(messages) == 0:
        return empty
    scored = model_comment_scores(messages) if models_available() else None
    if scored is not None:
        shares = scored[0]
    else:
        shares = emotion_shares(classify_comments(messages))
    mean_shares = shares.mean(axis=0)
    top = mean_shares.idxmax()
    dominant = (
        str(top).removeprefix("emo_") if float(mean_shares.max()) > 0.0 else "neutral"
    )
    return {
        "dominant": dominant,
        "shares": {col: float(round(mean_shares.get(col, 0.0), 4)) for col in EMOTION_COLUMNS},
        "volume": int(len(messages)),
        "coverage": scored_share(messages),
    }


def team_emotion_summary(
    chat: pd.DataFrame, home_team: str, away_team: str
) -> dict[str, dict[str, object]]:
    """Per-team mood: {'home'|'away': {team, dominant, shares, volume, coverage}}.

    Expects a `team` column (home|away|both|neither) as produced by
    teams.tag_reactions; `both`-tagged posts count for each side. Returns an
    empty dict when the chat has no team tags (e.g. simulator/committed modes).
    """
    if chat.empty or "team" not in chat.columns:
        return {}
    summary: dict[str, dict[str, object]] = {}
    for side, name in (("home", home_team), ("away", away_team)):
        subset = chat.loc[chat["team"].isin([side, "both"]), "message"]
        entry = _profile_summary(subset)
        entry["team"] = str(name or side)
        summary[side] = entry
    return summary


def headline_outcome(
    state: pd.DataFrame,
    threshold: float,
    team_summary: dict[str, dict[str, object]] | None = None,
) -> str:
    """One plain-language sentence describing the moment for a lay audience."""
    if state is None or state.empty:
        return "Waiting for enough fan reactions to read the crowd."
    latest = state.iloc[-1]
    panic = float(latest.get("crowd_panic_score", 0.0))
    stability = float(latest.get("delta_xg_10min", 0.0))
    gap = float(latest.get("arbitrage_index", 0.0))
    mood = "anxious" if panic > 0.25 else "calm and confident" if panic < -0.25 else "split"
    match_state = (
        "the match is producing real chances"
        if stability >= 0.45
        else "the match itself is fairly quiet"
        if stability <= 0.2
        else "the match is evenly balanced"
    )
    who = ""
    if team_summary:
        moods = {
            side: str(info.get("dominant", "neutral"))
            for side, info in team_summary.items()
            if info.get("volume")
        }
        named = {
            side: str(team_summary[side].get("team", side)) for side in moods
        }
        if moods:
            who = (
                " "
                + " | ".join(
                    f"{named[s]} fans: {moods[s]}" for s in moods
                )
                + "."
            )
    if gap >= threshold and panic > 0:
        tail = "fans are spiking faster than the pitch justifies — a possible overreaction."
    elif panic <= -0.4 and stability < 0.3:
        tail = "fans look relaxed even though the game is flat — watch for a momentum swing."
    else:
        tail = "fan mood and the match are roughly in step."
    return f"Right now: fans are {mood} while {match_state} — {tail}{who}"
