"""
ips_env.py - Gymnasium-compliant Intrusion Prevention Reinforcement Learning Environment

Simulates sequential network flow arrivals, TCN temporal state embedding,
eBPF/XDP fast-path verdicts, and reward calculation based on mitigation accuracy.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from ebpf.bpf_maps import BPFMapInterface
from ebpf.xdp_simulator import XDPSimulator
from environment.reward_functions import (
    ACTION_DEEP_INSPECT,
    ACTION_DROP,
    ACTION_PASS,
    ACTION_RATE_LIMIT,
    IPSRewardFunction,
)


class IPSEnvironment(gym.Env):
    """
    Gymnasium Environment for Deep Reinforcement Learning IPS.
    
    Observation Space: Box(40,)
      - 32-dim latent embedding from TCN
      - 8-dim dynamic flow and eBPF state metrics
      
    Action Space: Discrete(4)
      - 0: PASS
      - 1: DROP
      - 2: RATE_LIMIT
      - 3: DEEP_INSPECT
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        X_seq: Optional[np.ndarray] = None,
        y_multi: Optional[np.ndarray] = None,
        y_binary: Optional[np.ndarray] = None,
        tcn_model: Optional[torch.nn.Module] = None,
        reward_fn: Optional[IPSRewardFunction] = None,
        ebpf_backend: Optional[BPFMapInterface] = None,
        max_steps: int = 500,
        state_dim: int = 40,
        latent_dim: int = 32,
        device: str = "cpu",
    ):
        super().__init__()

        self.X_seq = X_seq
        self.y_multi = y_multi
        self.y_binary = y_binary
        self.tcn_model = tcn_model
        self.reward_fn = reward_fn or IPSRewardFunction()
        self.ebpf_backend = ebpf_backend or XDPSimulator()
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)

        if self.tcn_model is not None:
            self.tcn_model.eval()
            self.tcn_model.to(self.device)

        # Action: 0=PASS, 1=DROP, 2=RATE_LIMIT, 3=DEEP_INSPECT
        self.action_space = spaces.Discrete(4)

        # Observation: Latent TCN embedding (32) + Flow Metrics (8) = 40
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_idx = 0
        self.total_samples = len(self.X_seq) if self.X_seq is not None else 1000

        # Episode Metrics Tracking
        self.ep_reward = 0.0
        self.ep_tp = 0
        self.ep_tn = 0
        self.ep_fp = 0
        self.ep_fn = 0
        self.ep_rate_limited = 0
        self.ep_inspected = 0

    def _extract_tcn_embedding(self, x_seq_sample: np.ndarray) -> np.ndarray:
        """Extract latent representation from TCN model or fallback projection."""
        if self.tcn_model is not None:
            with torch.no_grad():
                tensor_in = torch.FloatTensor(x_seq_sample).unsqueeze(0).to(self.device)
                _, latent = self.tcn_model(tensor_in)
                return latent.cpu().numpy().squeeze(0)
        else:
            # Deterministic projection fallback for initial setup / dry-runs
            flat = x_seq_sample[-1, : self.latent_dim]
            if len(flat) < self.latent_dim:
                flat = np.pad(flat, (0, self.latent_dim - len(flat)))
            return flat.astype(np.float32)

    def _get_flow_dynamics_features(self, x_seq_sample: np.ndarray) -> np.ndarray:
        """Extract 8-dimensional operational network and eBPF status vector."""
        metrics = self.ebpf_backend.get_metrics()
        last_step = x_seq_sample[-1]

        # Selected representative flow dynamics from last flow record
        f_flow_dur = float(last_step[0]) if len(last_step) > 0 else 0.0
        f_fwd_pkts = float(last_step[1]) if len(last_step) > 1 else 0.0
        f_bwd_pkts = float(last_step[2]) if len(last_step) > 2 else 0.0
        f_fwd_len  = float(last_step[3]) if len(last_step) > 3 else 0.0

        # System state
        drop_ratio = float(metrics.drop_ratio)
        active_blocks = float(len(self.ebpf_backend.get_active_blocks())) / 1000.0
        rx_rate = float(metrics.rx_packets % 100) / 100.0
        norm_step = float(self.current_step) / float(self.max_steps)

        return np.array([
            f_flow_dur,
            f_fwd_pkts,
            f_bwd_pkts,
            f_fwd_len,
            drop_ratio,
            active_blocks,
            rx_rate,
            norm_step,
        ], dtype=np.float32)

    def _build_observation(self, idx: int) -> np.ndarray:
        """Construct full 40-dim observation vector."""
        if self.X_seq is not None:
            sample_seq = self.X_seq[idx]
        else:
            sample_seq = np.zeros((10, 78), dtype=np.float32)

        latent = self._extract_tcn_embedding(sample_seq)
        dynamics = self._get_flow_dynamics_features(sample_seq)
        obs = np.concatenate([latent, dynamics]).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)

        self.current_step = 0
        # Choose a random starting offset in dataset
        if self.X_seq is not None and len(self.X_seq) > self.max_steps:
            self.current_idx = np.random.randint(0, len(self.X_seq) - self.max_steps)
        else:
            self.current_idx = 0

        self.ep_reward = 0.0
        self.ep_tp = 0
        self.ep_tn = 0
        self.ep_fp = 0
        self.ep_fn = 0
        self.ep_rate_limited = 0
        self.ep_inspected = 0
        self.ebpf_backend.clear()

        obs = self._build_observation(self.current_idx)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action, update eBPF rules, calculate reward, advance state."""
        idx = self.current_idx + self.current_step

        if self.y_binary is not None and idx < len(self.y_binary):
            is_attack = bool(self.y_binary[idx] == 1)
            threat_class = int(self.y_multi[idx])
        else:
            is_attack = bool(np.random.rand() > 0.5)
            threat_class = 1 if is_attack else 0

        # Generate synthetic IP for eBPF rule tracking
        pseudo_ip = f"10.0.{threat_class}.{idx % 254 + 1}"

        # Apply Action to eBPF Data Plane
        if action == ACTION_DROP:
            self.ebpf_backend.block_ip(pseudo_ip, duration_sec=300.0, reason="DRL DROP")
        elif action == ACTION_RATE_LIMIT:
            self.ebpf_backend.set_rate_limit(pseudo_ip, rate_pps=50.0, burst=25.0)

        # Simulate packet processing in eBPF backend
        self.ebpf_backend.process_packet(
            src_ip=pseudo_ip,
            dst_ip="192.168.1.1",
            src_port=10000 + (idx % 5000),
            dst_port=80,
            proto=6,
        )

        # Calculate reward
        reward, outcome = self.reward_fn.compute(action, is_attack, threat_class)
        self.ep_reward += reward

        if outcome["tp"]:
            self.ep_tp += 1
        if outcome["tn"]:
            self.ep_tn += 1
        if outcome["fp"]:
            self.ep_fp += 1
        if outcome["fn"]:
            self.ep_fn += 1
        if outcome["rate_limited_attack"] or outcome["rate_limited_benign"]:
            self.ep_rate_limited += 1
        if outcome["inspected"]:
            self.ep_inspected += 1

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        next_idx = self.current_idx + self.current_step
        if next_idx >= self.total_samples:
            terminated = True
            next_idx = self.total_samples - 1

        next_obs = self._build_observation(next_idx)

        total_decisions = max(1, self.ep_tp + self.ep_tn + self.ep_fp + self.ep_fn)
        info = {
            "step": self.current_step,
            "cumulative_reward": self.ep_reward,
            "tp": self.ep_tp,
            "tn": self.ep_tn,
            "fp": self.ep_fp,
            "fn": self.ep_fn,
            "accuracy": (self.ep_tp + self.ep_tn) / total_decisions,
            "false_positive_rate": self.ep_fp / max(1, self.ep_fp + self.ep_tn),
            "threat_class": threat_class,
            "is_attack": is_attack,
            "action": action,
        }

        return next_obs, reward, terminated, truncated, info
