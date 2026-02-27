"""
tests/test_processor.py
Basic unit tests for the core processing logic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.processor import infer_sentiment_from_label, detect_themes, recommend_actions


def test_sentiment_negative():
    assert infer_sentiment_from_label("__label__1") == "negative"


def test_sentiment_positive():
    assert infer_sentiment_from_label("__label__2") == "positive"


def test_sentiment_unknown():
    assert infer_sentiment_from_label(None) == "unknown"
    assert infer_sentiment_from_label("__label__99") == "unknown"


def test_detect_themes_delivery():
    themes = detect_themes("My package was delayed and the delivery was late.")
    theme_names = [t.name for t in themes]
    assert "delivery" in theme_names


def test_detect_themes_empty():
    themes = detect_themes("")
    assert themes == []


def test_recommend_actions_negative():
    themes = detect_themes("The app keeps crashing and has lots of bugs.")
    actions = recommend_actions("negative", themes)
    assert len(actions) > 0
    assert actions[0].priority < 3  # negative → high priority


def test_recommend_actions_no_themes():
    actions = recommend_actions("negative", [])
    assert actions == []


if __name__ == "__main__":
    tests = [
        test_sentiment_negative,
        test_sentiment_positive,
        test_sentiment_unknown,
        test_detect_themes_delivery,
        test_detect_themes_empty,
        test_recommend_actions_negative,
        test_recommend_actions_no_themes,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
