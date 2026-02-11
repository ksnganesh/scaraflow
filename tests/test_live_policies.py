from datetime import timedelta

import pytest

from scara_live.policies import LiveRetrievalPolicy


def test_live_policy_valid_defaults():
    policy = LiveRetrievalPolicy(window=timedelta(minutes=5))
    assert policy.top_k == 5
    assert policy.min_score == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": timedelta(seconds=0)},
        {"window": timedelta(seconds=-1)},
        {"window": timedelta(seconds=1), "top_k": 0},
        {"window": timedelta(seconds=1), "min_score": -0.1},
        {"window": timedelta(seconds=1), "max_context_blocks": 0},
        {"window": timedelta(seconds=1), "max_context_chars": 0},
        {"window": timedelta(seconds=1), "recency_weights": (-0.1, 0.1)},
        {"window": timedelta(seconds=1), "recency_weights": (0.0, 0.0)},
    ],
)
def test_live_policy_validation_errors(kwargs):
    with pytest.raises(ValueError):
        LiveRetrievalPolicy(**kwargs)
