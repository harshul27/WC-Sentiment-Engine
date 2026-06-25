"""Tests for the plain-language label registry in src/glossary.py."""
from __future__ import annotations

import glossary


def test_jargon_terms_have_plain_labels_and_tooltips() -> None:
    # The terms the user flagged as confusing must be renamed with a tooltip.
    for key in ("crowd_panic_score", "arbitrage_index", "delta_xg_10min"):
        assert key in glossary.LABELS
        label, tip = glossary.LABELS[key]
        assert label and "panic" not in label.lower() and "arbitrage" not in label.lower()
        assert len(tip) > 10


def test_label_falls_back_for_unknown_key() -> None:
    assert glossary.label("crowd_panic_score") == "Fan Mood"
    assert glossary.label("some_unknown_col") == "Some Unknown Col"
    assert glossary.tooltip("some_unknown_col") == ""


def test_guide_and_titles_present() -> None:
    assert glossary.TITLE and glossary.SUBTITLE
    assert len(glossary.GUIDE) >= 3
    assert all(len(heading) > 0 and len(body) > 0 for heading, body in glossary.GUIDE)
