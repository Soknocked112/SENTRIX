from collections import deque
import logging
import os
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from ebpf.bpf_maps import BPFMapInterface
from ebpf.xdp_manager import XDPManager
from ebpf.xdp_simulator import XDPSimulator
from environment.reward_functions import (
    ACTION_DEEP_INSPECT,
    ACTION_DROP,
    ACTION_PASS,
    ACTION_RATE_LIMIT,
)
from models.drl_agent import PPOAgent
from models.tcn import TCNThreatDetector

logger = logging.getLogger("ips_orchestrator")


class IPSOrchestrator:
    """Coordinates packet buffer, model inference, and eBPF rule updates."""

    def __init__(
        self,
        tcn_model: Optional[TCNThreatDetector] = None,
        drl_agent: Optional[PPOAgent] = None,
        ebpf_backend: Optional[BPFMapInterface] = None,
        sequence_length: int = 10,
        num_features: int = 78,
        latent_dim: int = 32,
        state_dim: int = 40,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.tcn = tcn_model
        if self.tcn is not None:
            self.tcn.eval()
            self.tcn.to(self.device)

        self.agent = drl_agent
        self.ebpf = ebpf_backend or XDPManager(force_simulator=True)
        self.seq_len = sequence_length
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.flow_buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        self.processed_flows = 0
        self.decision_latencies_us: List[float] = []
        self.action_counts = {
            ACTION_PASS: 0,
            ACTION_DROP: 0,
            ACTION_RATE_LIMIT: 0,
            ACTION_DEEP_INSPECT: 0,
        }

    def reset_buffer(self):
        self.flow_buffer.clear()

    def process_flow_event(
        self,
        features: np.ndarray,
        src_ip: str = "192.168.1.100",
        dst_ip: str = "10.0.0.1",
        src_port: int = 54321,
        dst_port: int = 80,
        proto: int = 6,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter_ns()
        self.processed_flows += 1

        # Check existing kernel/simulator block
        if self.ebpf.is_ip_blocked(src_ip):
            t_end = time.perf_counter_ns()
            latency_us = (t_end - t_start) / 1000.0
            self.decision_latencies_us.append(latency_us)
            self.action_counts[ACTION_DROP] += 1
            return {
                "action": ACTION_DROP,
                "action_name": "DROP",
                "fast_path": True,
                "src_ip": src_ip,
                "threat_pred": 1,
                "threat_prob": 1.0,
                "latency_us": latency_us,
                "verdict": "XDP_DROP",
            }

        # Update sliding window buffer
        self.flow_buffer.append(features)
        while len(self.flow_buffer) < self.seq_len:
            self.flow_buffer.appendleft(features)

        window_arr = np.array(self.flow_buffer, dtype=np.float32)

        # Run TCN inference for threat detection and embedding
        with torch.no_grad():
            tensor_in = torch.FloatTensor(window_arr).unsqueeze(0).to(self.device)
            if self.tcn is not None:
                logits, latent = self.tcn(tensor_in)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_class = int(np.argmax(probs))
                pred_prob = float(probs[pred_class])
                latent_vec = latent.squeeze(0).cpu().numpy()
            else:
                pred_class = 0
                pred_prob = 1.0
                latent_vec = features[: self.latent_dim]
                if len(latent_vec) < self.latent_dim:
                    latent_vec = np.pad(latent_vec, (0, self.latent_dim - len(latent_vec)))

        # Build state observation (latent + flow statistics)
        metrics = self.ebpf.get_metrics()
        flow_dynamics = np.array([
            float(features[0]) if len(features) > 0 else 0.0,
            float(features[1]) if len(features) > 1 else 0.0,
            float(features[2]) if len(features) > 2 else 0.0,
            float(features[3]) if len(features) > 3 else 0.0,
            float(metrics.drop_ratio),
            float(len(self.ebpf.get_active_blocks())) / 1000.0,
            float(pred_prob),
            float(pred_class) / 5.0,
        ], dtype=np.float32)

        state_obs = np.concatenate([latent_vec, flow_dynamics]).astype(np.float32)

        # Decide policy action
        if self.agent is not None:
            action, _, _ = self.agent.select_action(state_obs, deterministic=True)
        else:
            action = ACTION_DROP if pred_class > 0 else ACTION_PASS

        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        action_names = {
            ACTION_PASS: "PASS",
            ACTION_DROP: "DROP",
            ACTION_RATE_LIMIT: "RATE_LIMIT",
            ACTION_DEEP_INSPECT: "DEEP_INSPECT",
        }
        action_name = action_names.get(action, "UNKNOWN")

        # Apply mitigation rule
        if action == ACTION_DROP:
            self.ebpf.block_ip(src_ip, duration_sec=300.0, reason=f"TCN-DRL Class {pred_class}")
            verdict = "XDP_DROP"
        elif action == ACTION_RATE_LIMIT:
            self.ebpf.set_rate_limit(src_ip, rate_pps=100.0, burst=50.0)
            verdict = "XDP_RATE_LIMITED"
        elif action == ACTION_DEEP_INSPECT:
            verdict = "FORWARD_TO_INSPECT_BUFFER"
        else:
            verdict = "XDP_PASS"

        t_end = time.perf_counter_ns()
        latency_us = (t_end - t_start) / 1000.0
        self.decision_latencies_us.append(latency_us)

        return {
            "action": action,
            "action_name": action_name,
            "fast_path": False,
            "src_ip": src_ip,
            "threat_pred": pred_class,
            "threat_prob": pred_prob,
            "latency_us": latency_us,
            "verdict": verdict,
        }

    def get_latency_stats(self) -> Dict[str, float]:
        """Compute latency statistics in microseconds."""
        if not self.decision_latencies_us:
            return {"mean_us": 0.0, "p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0}
        arr = np.array(self.decision_latencies_us)
        return {
            "mean_us": float(np.mean(arr)),
            "p50_us": float(np.percentile(arr, 50)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
        }
