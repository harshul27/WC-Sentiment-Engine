"""Smoke tests for the Streamlit dashboard via streamlit.testing.AppTest.

Tests pin the mode radio before the first run so no live network calls
are made; the live connectors themselves are covered in test_live.py.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import app

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


def _make(mode: str) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["mode"] = mode
    return at


def test_simulator_mode_renders_without_exceptions() -> None:
    at = _make(app.MODE_SIM).run()
    assert not at.exception
    assert at.title
    assert len(at.metric) >= 4
    assert at.button


def test_live_stream_simulator_button() -> None:
    at = _make(app.MODE_SIM).run()
    at.sidebar.slider[0].set_value(0.01)
    at.button[0].click()
    at.run()
    assert not at.exception
    assert len(at.metric) >= 4
    assert at.session_state["sim_state"] is not None


def test_committed_state_mode_renders() -> None:
    at = _make(app.MODE_STATE).run()
    assert not at.exception
    assert len(at.metric) >= 4
