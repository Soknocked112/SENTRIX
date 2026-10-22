"""
test_orchestrator.py - Unit tests for end-to-end IPS Orchestrator
"""

import numpy as np
import pytest
from ebpf.xdp_simulator import XDPSimulator
from engine.ips_orchestrator import IPSOrchestrator
from environment.reward_functions import ACTION_DROP, ACTION_PASS
from models.drl_agent import PPOAgent
from models.tcn import TCNThreatDetector


def test_orchestrator_pipeline():
    tcn = TCNThreatDetector(input_features=78, num_channels=[16, 16], latent_dim=32, num_classes=6)
    agent = PPOAgent(state_dim=40, action_dim=4)
    ebpf_sim = XDPSimulator()

    orch = IPSOrchestrator(
        tcn_model=tcn,
        drl_agent=agent,
        ebpf_backend=ebpf_sim,
        sequence_length=10,
        num_features=78,
        latent_dim=32,
        state_dim=40,
    )

    dummy_features = np.zeros(78, dtype=np.float32)

    # Process 5 flows
    for i in range(5):
        event = orch.process_flow_event(
            features=dummy_features,
            src_ip=f"192.168.1.{10 + i}",
            dst_ip="10.0.0.1",
        )
        assert "action" in event
        assert "verdict" in event
        assert event["latency_us"] > 0

    assert orch.processed_flows == 5
    stats = orch.get_latency_stats()
    assert "mean_us" in stats
    assert stats["mean_us"] > 0
