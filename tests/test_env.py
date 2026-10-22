"""
test_env.py - Unit tests for Gymnasium IPSEnvironment and Reward Functions
"""

import numpy as np
import pytest
from environment.ips_env import IPSEnvironment
from environment.reward_functions import (
    ACTION_DEEP_INSPECT,
    ACTION_DROP,
    ACTION_PASS,
    ACTION_RATE_LIMIT,
    IPSRewardFunction,
)


def test_reward_function():
    reward_fn = IPSRewardFunction(
        reward_tp_drop=10.0,
        reward_tn_pass=2.0,
        penalty_fp_drop=-50.0,
        penalty_fn_pass=-30.0,
    )

    # True Positive: Dropping Attack
    r_tp, outcome_tp = reward_fn.compute(action=ACTION_DROP, is_attack=True)
    assert r_tp == 10.0
    assert outcome_tp["tp"] is True

    # True Negative: Passing Benign
    r_tn, outcome_tn = reward_fn.compute(action=ACTION_PASS, is_attack=False)
    assert r_tn == 2.0
    assert outcome_tn["tn"] is True

    # False Positive: Dropping Benign (Severe Penalty)
    r_fp, outcome_fp = reward_fn.compute(action=ACTION_DROP, is_attack=False)
    assert r_fp == -50.0
    assert outcome_fp["fp"] is True

    # False Negative: Passing Attack
    r_fn, outcome_fn = reward_fn.compute(action=ACTION_PASS, is_attack=True)
    assert r_fn == -30.0
    assert outcome_fn["fn"] is True


def test_gym_environment_step_reset():
    X_seq = np.random.randn(100, 10, 78).astype(np.float32)
    y_multi = np.random.choice([0, 1, 2], size=100)
    y_binary = (y_multi > 0).astype(int)

    env = IPSEnvironment(
        X_seq=X_seq,
        y_multi=y_multi,
        y_binary=y_binary,
        max_steps=20,
        state_dim=40,
    )

    obs, info = env.reset()
    assert obs.shape == (40,)
    assert isinstance(obs, np.ndarray)

    for i in range(20):
        next_obs, reward, terminated, truncated, step_info = env.step(i % 4)
        assert next_obs.shape == (40,)
        assert isinstance(reward, float)
        assert "accuracy" in step_info

    assert terminated is True
