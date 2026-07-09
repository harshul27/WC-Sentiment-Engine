"""Mood-vs-game consistency: is each team's fan mood plausible right now?

A losing side whose fans read as "joy" is either a misread window or a genuine
anomaly - both matter. This module checks each team's crowd mood against the
actual game situation (scoreline, threat, keeper workload) and returns a
verdict with a plain-language explanation, plus a `clarity` score that says
how decisive the mood reading itself is.

Design rule (deliberate): a conflict is FLAGGED AND EXPLAINED, never silently
suppressed. A consolation-goal joy spike from a trailing side is legitimate
crowd behaviour - the explanation says so and the moment is surfaced as a
possible overreaction, leaving the judgement visible instead of hidden.
"""
from __future__ import annotations

import re

POSITIVE_EMOTIONS = {"joy", "confidence"}
NEGATIVE_EMOTIONS = {"panic", "despair", "anger"}

# Clarity blend weights: how decisive the mood reading is.
#   margin   - top-1 minus top-2 emotion share (decisiveness of the label)
#   coverage - share of posts the model could read
#   volume   - saturates at VOLUME_FULL reactions
_W_MARGIN, _W_COVERAGE, _W_VOLUME = 0.5, 0.3, 0.2
VOLUME_FULL = 30


def parse_score(score: str) -> tuple[int, int] | None:
    """"2-1" / "2–1" -> (2, 1); None when unparseable."""
    parts = re.split(r"[-–—]", str(score or ""))
    if len(parts) != 2:
        return None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None


def game_context(
    score: str,
    home_team: str,
    away_team: str,
    delta_xg: float = 0.0,
    keeper: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, object]]:
    """Per-side game situation: {'home'|'away': {team, status, margin, ...}}.

    status is leading | trailing | level; threat is the match-wide xG
    stability; keeper_saves is that side's keeper workload when known.
    """
    parsed = parse_score(score)
    home_goals, away_goals = parsed if parsed else (0, 0)
    context: dict[str, dict[str, object]] = {}
    for side, team, own, other in (
        ("home", home_team, home_goals, away_goals),
        ("away", away_team, away_goals, home_goals),
    ):
        status = "leading" if own > other else "trailing" if own < other else "level"
        saves = 0.0
        if keeper and team in keeper:
            saves = float(keeper[team].get("saves", 0.0))
        context[side] = {
            "team": str(team),
            "status": status,
            "margin": abs(own - other),
            "threat": float(delta_xg),
            "keeper_saves": saves,
        }
    return context


def clarity_score(summary: dict[str, object]) -> float:
    """How decisive one team's mood reading is, in [0, 1].

    Blends the emotion-share margin (top-1 minus top-2), reading coverage,
    and reaction volume. Low clarity means "mixed/thin signal - hold the
    verdict loosely", and the dashboard shows exactly that.
    """
    shares = dict(summary.get("shares") or {})
    values = sorted((float(v) for v in shares.values()), reverse=True)
    margin = (values[0] - values[1]) if len(values) >= 2 else (values[0] if values else 0.0)
    coverage = float(summary.get("coverage", 0.0) or 0.0)
    volume = min(1.0, float(summary.get("volume", 0) or 0) / VOLUME_FULL)
    return round(
        min(1.0, _W_MARGIN * margin + _W_COVERAGE * coverage + _W_VOLUME * volume), 4
    )


def mood_consistency(
    team_summary: dict[str, dict[str, object]],
    context: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Verdict per side: does the crowd mood fit the game situation?

    Returns {'home'|'away': {team, verdict, explanation, clarity, dominant,
    status}} where verdict is consistent | conflict. Rules (documented, on
    purpose simple and auditable):
      trailing + positive mood            -> conflict (unjustified positivity;
                                             possible overreaction or a recent
                                             consolation-goal spike)
      leading  + negative mood + low own
      pressure (few keeper saves)         -> conflict (panic without cause)
      everything else                     -> consistent
    """
    verdicts: dict[str, dict[str, object]] = {}
    for side, summary in (team_summary or {}).items():
        ctx = (context or {}).get(side, {})
        team = str(summary.get("team") or ctx.get("team") or side)
        dominant = str(summary.get("dominant", "neutral"))
        status = str(ctx.get("status", "level"))
        margin = int(ctx.get("margin", 0) or 0)
        saves = float(ctx.get("keeper_saves", 0.0) or 0.0)
        clarity = clarity_score(summary)
        verdict, explanation = "consistent", (
            f"{team} fans read as {dominant}, which fits a side that is {status}."
        )
        if status == "trailing" and dominant in POSITIVE_EMOTIONS:
            verdict = "conflict"
            explanation = (
                f"{team} trail by {margin} yet fan mood reads {dominant} - "
                "conflicting with the game situation. Either a genuine anomaly "
                "(e.g. a consolation goal or pride in the performance) or a "
                "misread window; treat as a possible overreaction moment."
            )
        elif status == "leading" and dominant in NEGATIVE_EMOTIONS and saves <= 2:
            verdict = "conflict"
            explanation = (
                f"{team} lead by {margin} and their goal is not under siege "
                f"(keeper saves: {saves:.0f}), yet fan mood reads {dominant} - "
                "nervousness ahead of the game situation; classic overreaction "
                "territory."
            )
        elif status == "leading" and dominant in NEGATIVE_EMOTIONS:
            explanation = (
                f"{team} lead but the keeper is busy ({saves:.0f} saves), so a "
                f"{dominant} crowd is understandable - the lead is under pressure."
            )
        verdicts[side] = {
            "team": team,
            "verdict": verdict,
            "explanation": explanation,
            "clarity": clarity,
            "dominant": dominant,
            "status": status,
        }
    return verdicts


def conflict_moments(
    verdicts: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    """Conflicts shaped for the Overreaction Moments panel (reason + detail)."""
    moments: list[dict[str, str]] = []
    for info in (verdicts or {}).values():
        if info.get("verdict") != "conflict":
            continue
        status = str(info.get("status", ""))
        reason = (
            "positive-while-losing" if status == "trailing" else "panic-while-ahead"
        )
        moments.append(
            {
                "reason": reason,
                "team": str(info.get("team", "")),
                "detail": str(info.get("explanation", "")),
            }
        )
    return moments
