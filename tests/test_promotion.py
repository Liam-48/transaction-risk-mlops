
import numpy as np
import pytest

from src.promotion import decide_promotion, evaluate_candidate


def test_better_challenger_is_eligible():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])

    champion = np.array([
        0.1, 0.9, 0.2, 0.8, 0.6, 0.4, 0.5, 0.3
    ])

    challenger = np.array([
        0.95, 0.1, 0.9, 0.2, 0.05, 0.8, 0.3, 0.0
    ])

    result = decide_promotion(
        y, champion, challenger, capacity=0.25
    )

    assert result["eligible_for_promotion"] is True
    assert result["frauds_gained"] == 2


def test_identical_model_is_not_promoted():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1])

    result = decide_promotion(
        y, scores, scores, capacity=0.5
    )

    assert result["eligible_for_promotion"] is False
    assert result["frauds_gained"] == 0


def test_invalid_capacity():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])

    with pytest.raises(ValueError):
        evaluate_candidate(y, scores, capacity=0)
