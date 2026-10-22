"""
drl_agent.py - Deep Reinforcement Learning (PPO) Actor-Critic Agent for IPS

Learns active mitigation policies:
- PASS: Allow flow through without penalty.
- DROP: Instant in-kernel eBPF/XDP drop rule insertion.
- RATE_LIMIT: Token bucket traffic throttling.
- DEEP_INSPECT: Buffer flow for deep payload inspection.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """Policy Network mapping state observations to action probabilities."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> Categorical:
        logits = self.net(state)
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """Value Network estimating the expected cumulative return V(s)."""

    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class RolloutBuffer:
    """Stores on-policy trajectory transitions during an episode rollout."""

    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.state_values: List[torch.Tensor] = []
        self.is_terminals: List[bool] = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent with Generalized Advantage Estimation (GAE).
    """

    def __init__(
        self,
        state_dim: int = 40,
        action_dim: int = 4,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.20,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.50,
        k_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Choose action from current policy given state observation.
        Returns: (action, log_prob, state_value)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.actor(state_t)
            value = self.critic(state_t)

            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def record_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        state_value: float,
        done: bool,
    ):
        """Append step transition into the rollout buffer."""
        self.buffer.states.append(torch.FloatTensor(state).to(self.device))
        self.buffer.actions.append(torch.tensor(action, dtype=torch.long).to(self.device))
        self.buffer.logprobs.append(torch.tensor(log_prob, dtype=torch.float32).to(self.device))
        self.buffer.rewards.append(reward)
        self.buffer.state_values.append(torch.tensor(state_value, dtype=torch.float32).to(self.device))
        self.buffer.is_terminals.append(done)

    def compute_gae(self, last_val: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        rewards = self.buffer.rewards
        values = [v.item() for v in self.buffer.state_values] + [last_val]
        dones = self.buffer.is_terminals

        advantages = []
        gae = 0.0
        n_steps = len(rewards)

        for step in reversed(range(n_steps)):
            delta = rewards[step] + self.gamma * values[step + 1] * (1.0 - float(dones[step])) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[step])) * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = advantages_t + torch.tensor(values[:-1], dtype=torch.float32, device=self.device)

        # Normalize advantages for training stability
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        return advantages_t, returns_t

    def update(self) -> Dict[str, float]:
        """Execute PPO policy and value network updates over rollout trajectories."""
        if len(self.buffer) == 0:
            return {}

        advantages, returns = self.compute_gae()

        old_states = torch.stack(self.buffer.states)
        old_actions = torch.stack(self.buffer.actions)
        old_logprobs = torch.stack(self.buffer.logprobs)

        dataset_size = len(self.buffer)
        indices = np.arange(dataset_size)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        updates_count = 0

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_states = old_states[batch_idx]
                b_actions = old_actions[batch_idx]
                b_logprobs = old_logprobs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Evaluate current policy
                dist = self.actor(b_states)
                new_logprobs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Ratio: r(theta) = pi_theta(a|s) / pi_theta_old(a|s)
                ratios = torch.exp(new_logprobs - b_logprobs)

                # Clipped surrogate objective
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - (self.entropy_coef * entropy)

                # Update Actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.optimizer_actor.step()

                # Evaluate current critic value
                state_values = self.critic(b_states).squeeze(-1)
                critic_loss = nn.MSELoss()(state_values, b_returns)

                # Update Critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_critic.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                updates_count += 1

        self.buffer.clear()

        return {
            "actor_loss": total_actor_loss / max(1, updates_count),
            "critic_loss": total_critic_loss / max(1, updates_count),
            "entropy": total_entropy / max(1, updates_count),
        }

    def save(self, filepath: str):
        """Save model weights to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model weights from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
