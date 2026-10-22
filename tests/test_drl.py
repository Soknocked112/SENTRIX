"""
test_drl.py - Unit tests for PPO Actor-Critic Agent and Rollout Buffer
"""

import numpy as np
import pytest
import torch
from models.drl_agent import ActorNetwork, CriticNetwork, PPOAgent, RolloutBuffer


def test_actor_critic_networks():
    state_dim = 40
    action_dim = 4
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim)

    state = torch.randn(2, state_dim)
    dist = actor(state)
    value = critic(state)

    assert dist.probs.shape == (2, action_dim)
    assert value.shape == (2, 1)


def test_ppo_agent_rollout_and_update():
    agent = PPOAgent(
        state_dim=40,
        action_dim=4,
        lr_actor=0.001,
        lr_critic=0.001,
        k_epochs=2,
        batch_size=16,
    )

    # Collect 32 dummy transitions
    for _ in range(32):
        state = np.random.randn(40).astype(np.float32)
        action, log_prob, val = agent.select_action(state)
        reward = 1.0
        done = False
        agent.record_transition(state, action, log_prob, reward, val, done)

    assert len(agent.buffer) == 32

    losses = agent.update()
    assert "actor_loss" in losses
    assert "critic_loss" in losses
    assert len(agent.buffer) == 0
