"""Vectorized sentiment math, multi-agent parsing, and arbitrage calculations.

Implements the three-agent workflow described in CLAUDE.md without heavy
framework overhead:

  SocialListeningAgent  -> rolling Crowd Panic Score in [-1.0, 1.0]
  MatchProgressionAgent -> rolling Expected Goals (xG) stability index
  ArbitrageSelector     -> Arbitrage_Index = |panic| * (1 - delta_xG_10min)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

DEFAULT_CONFIG: dict[str, object] = {
    "version": "0.1.0",
    "hyperparameters": {
        "panic_score_bounds": [-1.0, 1.0],
        "xg_rolling_window_minutes": 10,
        "arbitrage_flag_threshold": 0.65,
    },
    "log_loss_history": [],
}

# Positive weights signal panic, negative weights signal calm/confidence.
_SENTIMENT_LEXICON: dict[str, float] = {
    "panic": 1.0,
    "disaster": 1.0,
    "we are done": 1.0,
    "we're done": 1.0,
    "choke": 0.9,
    "choking": 0.9,
    "bottle": 0.8,
    "terrible": 0.7,
    "awful": 0.7,
    "useless": 0.7,
    "sack": 0.8,
    "nervous": 0.6,
    "scared": 0.6,
    "worried": 0.5,
    "throwing it away": 0.9,
    "collapse": 0.9,
    "no chance": 0.7,
    "embarrassing": 0.6,
    "calm": -0.6,
    "composed": -0.6,
    "in control": -0.8,
    "control": -0.4,
    "comfortable": -0.7,
    "dominating": -0.8,
    "dominant": -0.7,
    "cruising": -0.8,
    "easy": -0.5,
    "no worries": -0.7,
    "relax": -0.5,
    "winning this": -0.6,
    "got this": -0.6,
}

# Ordered: first matching pattern wins. Values are per-event xG proxies.
_EVENT_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (r"\bgoal\b|\bscores\b", "goal", 0.40),
    (r"\bpenalty\b", "penalty", 0.76),
    (r"\bbig chance\b|\bsitter\b|\bone on one\b", "big_chance", 0.35),
    (r"\bhits the (post|crossbar|bar)\b|\bwoodwork\b", "woodwork", 0.30),
    (r"\bshot on target\b|\bforces a save\b|\bsaved\b|\bgreat save\b", "shot_on_target", 0.30),
    (r"\bshot (off target|wide)\b|\bgoes wide\b|\bover the bar\b|\bblazes\b", "shot_off_target", 0.07),
    (r"\bblocked\b", "blocked_shot", 0.05),
    (r"\bdangerous free[- ]kick\b", "free_kick", 0.08),
    (r"\bcorner\b", "corner", 0.03),
)

_COMMENTARY_LINE = re.compile(r"^\s*(\d+)(?:\+\d+)?'\s*(?:([A-Za-z .'-]+?)\s*:)?\s*(.+)$")


def load_config(path: str) -> dict[str, object]:
    """Read model_config.json, falling back to defaults on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_CONFIG))
    if "hyperparameters" not in config:
        config["hyperparameters"] = dict(DEFAULT_CONFIG["hyperparameters"])  # type: ignore[arg-type]
    config.setdefault("log_loss_history", [])
    return config


def save_config(path: str, config: dict[str, object]) -> None:
    """Persist the dynamic hyperparameter state back to disk."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def score_messages(messages: pd.Series) -> pd.Series:
    """Vectorized lexicon sentiment: one bounded panic value per message."""
    text = messages.fillna("").astype(str).str.lower()
    raw = np.zeros(len(text), dtype=np.float64)
    for term, weight in _SENTIMENT_LEXICON.items():
        raw += text.str.count(re.escape(term)).to_numpy(dtype=np.float64) * weight
    return pd.Series(np.tanh(raw), index=messages.index, name="panic")


def llm_panic_score(text_block: str, timeout: float = 20.0) -> float | None:
    """Optional free-tier LLM refinement of the panic score.

    Uses OpenRouter or Groq when an API key is present in the environment;
    returns None (caller falls back to the lexicon) on any missing key,
    network failure, or malformed response.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        url = "https://openrouter.ai/api/v1/chat/completions"
        key = os.environ["OPENROUTER_API_KEY"]
        model = "meta-llama/llama-3.1-8b-instruct:free"
    elif os.environ.get("GROQ_API_KEY"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        key = os.environ["GROQ_API_KEY"]
        model = "llama-3.1-8b-instant"
    else:
        return None
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Rate the crowd panic in these fan messages from -1.0 "
                    "(total confidence) to 1.0 (extreme panic). Reply with "
                    'JSON only: {"crowd_panic_score": <float>}\n\n' + text_block[:4000]
                ),
            }
        ],
    }
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"-?\d+(?:\.\d+)?", content)
        if match is None:
            return None
        return float(np.clip(float(match.group()), -1.0, 1.0))
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None


def parse_commentary(lines: pd.Series) -> pd.DataFrame:
    """Parse play-by-play text lines into structured match events.

    Returns columns: minute (int64), team (str), event_type (str),
    xg_value (float64). Lines that do not match the commentary format
    are dropped; lines without a recognised event contribute 0.0 xG.
    """
    records: list[tuple[int, str, str, float]] = []
    for line in lines.fillna("").astype(str):
        parsed = _COMMENTARY_LINE.match(line)
        if parsed is None:
            continue
        minute = int(parsed.group(1))
        team = (parsed.group(2) or "unknown").strip()
        body = parsed.group(3).lower()
        event_type, xg_value = "play", 0.0
        for pattern, name, value in _EVENT_PATTERNS:
            if re.search(pattern, body):
                event_type, xg_value = name, value
                break
        records.append((minute, team, event_type, xg_value))
    return pd.DataFrame(
        records, columns=["minute", "team", "event_type", "xg_value"]
    ).astype({"minute": "int64", "team": "str", "event_type": "str", "xg_value": "float64"})


def rolling_xg_stability(events: pd.DataFrame, window_minutes: int = 10) -> pd.DataFrame:
    """Minute-indexed rolling offensive-threat stability index in [0, 1].

    delta_xg_10min = tanh(rolling xG sum over the window): high values mean
    the match retains genuine attacking threat (a stable, healthy state).
    """
    if events.empty:
        return pd.DataFrame(columns=["minute", "rolling_xg", "delta_xg_10min"]).astype(
            {"minute": "int64", "rolling_xg": "float64", "delta_xg_10min": "float64"}
        )
    last_minute = int(events["minute"].max())
    per_minute = (
        events.groupby("minute")["xg_value"]
        .sum()
        .reindex(range(last_minute + 1), fill_value=0.0)
    )
    rolling = per_minute.rolling(window=window_minutes, min_periods=1).sum()
    return pd.DataFrame(
        {
            "minute": rolling.index.astype("int64"),
            "rolling_xg": rolling.to_numpy(dtype=np.float64),
            "delta_xg_10min": np.tanh(rolling.to_numpy(dtype=np.float64)),
        }
    )


def compute_arbitrage_index(
    panic_scores: np.ndarray | pd.Series, delta_xg: np.ndarray | pd.Series
) -> np.ndarray:
    """Core equation: Arbitrage_Index = |panic| * (1.0 - delta_xG_10min)."""
    panic = np.clip(np.asarray(panic_scores, dtype=np.float64), -1.0, 1.0)
    stability = np.clip(np.asarray(delta_xg, dtype=np.float64), 0.0, 1.0)
    return np.abs(panic) * (1.0 - stability)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary cross-entropy with numerical clipping."""
    prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    truth = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(truth * np.log(prob) + (1.0 - truth) * np.log(1.0 - prob)))


def grid_search_threshold(
    arbitrage_index: np.ndarray | pd.Series,
    overreaction_truth: np.ndarray | pd.Series,
    steepness: float = 8.0,
) -> dict[str, float]:
    """Self-correction routine: pick the flag threshold minimizing log-loss.

    overreaction_truth holds 1.0 where the final match outcome proved the
    crowd panic unjustified (a genuine arbitrage moment), else 0.0.
    """
    index = np.asarray(arbitrage_index, dtype=np.float64)
    truth = np.asarray(overreaction_truth, dtype=np.float64)
    best_threshold, best_loss = 0.65, float("inf")
    for threshold in np.arange(0.05, 0.96, 0.05):
        prob = 1.0 / (1.0 + np.exp(-steepness * (index - threshold)))
        loss = log_loss(truth, prob)
        if loss < best_loss:
            best_threshold, best_loss = float(round(threshold, 2)), loss
    return {"arbitrage_flag_threshold": best_threshold, "log_loss": best_loss}


@dataclass
class SocialListeningAgent:
    """Agent A: turns raw fan chat into a minute-indexed Crowd Panic Score."""

    window_minutes: int = 5
    use_llm: bool = True

    def run(self, chat: pd.DataFrame) -> pd.DataFrame:
        if chat.empty:
            return pd.DataFrame(columns=["minute", "crowd_panic_score"]).astype(
                {"minute": "int64", "crowd_panic_score": "float64"}
            )
        scored = chat.assign(panic=score_messages(chat["message"]))
        last_minute = int(scored["minute"].max())
        per_minute = (
            scored.groupby("minute")["panic"]
            .mean()
            .reindex(range(last_minute + 1))
            .ffill()
            .fillna(0.0)
        )
        rolling = per_minute.rolling(window=self.window_minutes, min_periods=1).mean()
        result = pd.DataFrame(
            {
                "minute": rolling.index.astype("int64"),
                "crowd_panic_score": np.clip(
                    rolling.to_numpy(dtype=np.float64), -1.0, 1.0
                ),
            }
        )
        if self.use_llm:
            recent = scored.loc[
                scored["minute"] >= last_minute - self.window_minutes, "message"
            ]
            refined = llm_panic_score("\n".join(recent.astype(str).tolist()))
            if refined is not None:
                score_col = result.columns.get_loc("crowd_panic_score")
                blended = 0.5 * float(result.iloc[-1, score_col]) + 0.5 * refined
                result.iloc[-1, score_col] = blended
        return result


@dataclass
class MatchProgressionAgent:
    """Agent B: converts play-by-play commentary into rolling xG metrics."""

    window_minutes: int = 10

    def run(self, commentary: pd.Series) -> pd.DataFrame:
        events = parse_commentary(commentary)
        return rolling_xg_stability(events, self.window_minutes)


@dataclass
class ArbitrageSelector:
    """Agent C: fuses both signals and flags market arbitrage moments."""

    threshold: float = 0.65

    def run(self, social: pd.DataFrame, match: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(social, match, on="minute", how="outer").sort_values("minute")
        merged["crowd_panic_score"] = merged["crowd_panic_score"].ffill().fillna(0.0)
        merged["rolling_xg"] = merged["rolling_xg"].ffill().fillna(0.0)
        merged["delta_xg_10min"] = merged["delta_xg_10min"].ffill().fillna(0.0)
        merged["arbitrage_index"] = compute_arbitrage_index(
            merged["crowd_panic_score"], merged["delta_xg_10min"]
        )
        merged["flagged"] = merged["arbitrage_index"] >= self.threshold
        return merged.reset_index(drop=True).astype(
            {
                "minute": "int64",
                "crowd_panic_score": "float64",
                "rolling_xg": "float64",
                "delta_xg_10min": "float64",
                "arbitrage_index": "float64",
                "flagged": "bool",
            }
        )
