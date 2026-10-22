#!/usr/bin/env python3
"""
04_run_ips.py - Live Execution and Demonstration of eBPF-RL-TCN IPS
"""

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import yaml

from ebpf.xdp_manager import XDPManager
from engine.cli_monitor import CLIMonitor
from engine.ips_orchestrator import IPSOrchestrator
from models.drl_agent import PPOAgent
from models.tcn import TCNThreatDetector


def main():
    parser = argparse.ArgumentParser(description="Run Live eBPF-RL-TCN IPS Execution")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--flows", type=int, default=100, help="Number of flow events to simulate")
    parser.add_argument("--delay", type=float, default=0.01, help="Simulated arrival delay between flows (s)")
    parser.add_argument("--kernel", action="store_true", help="Force real kernel XDP attachment (requires root)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["processed_data_dir"]
    test_npz = os.path.join(data_dir, "test_data.npz")
    ckpt_dir = cfg["paths"]["checkpoints_dir"]

    # 1. Load Pretrained TCN
    tcn_ckpt_path = os.path.join(ckpt_dir, "tcn_best.pth")
    tcn_model = None
    if os.path.exists(tcn_ckpt_path):
        ckpt = torch.load(tcn_ckpt_path, map_location="cpu")
        tcn_model = TCNThreatDetector(
            input_features=ckpt["input_features"],
            num_channels=ckpt["num_channels"],
            latent_dim=ckpt["latent_dim"],
            num_classes=ckpt["num_classes"],
        )
        tcn_model.load_state_dict(ckpt["model_state_dict"])
        tcn_model.eval()

    # 2. Load Pretrained DRL Policy
    drl_ckpt_path = os.path.join(ckpt_dir, "drl_ppo_best.pth")
    drl_agent = None
    if os.path.exists(drl_ckpt_path):
        drl_agent = PPOAgent(
            state_dim=int(cfg["drl"]["state_dim"]),
            action_dim=int(cfg["drl"]["action_dim"]),
            device="cpu",
        )
        drl_agent.load(drl_ckpt_path)

    # 3. Initialize eBPF Backend
    ebpf_backend = XDPManager(
        interface=cfg["ebpf"]["interface"],
        c_source_path=cfg["ebpf"]["c_source_path"],
        force_simulator=not args.kernel,
    )

    # 4. Initialize Orchestrator
    orchestrator = IPSOrchestrator(
        tcn_model=tcn_model,
        drl_agent=drl_agent,
        ebpf_backend=ebpf_backend,
        sequence_length=int(cfg["data"]["sequence_length"]),
        latent_dim=int(cfg["tcn"]["latent_dim"]),
        state_dim=int(cfg["drl"]["state_dim"]),
    )

    # 5. Load Stream Data
    if os.path.exists(test_npz):
        test_data = np.load(test_npz)
        X_test = test_data["X"]
        y_test = test_data["y"]
    else:
        # Fallback synthetic traffic stream
        X_test = np.random.randn(args.flows + 20, 10, 78).astype(np.float32)
        y_test = np.random.choice([0, 1, 2, 3], size=len(X_test))

    monitor = CLIMonitor()
    monitor.render_header()

    num_flows = min(args.flows, len(X_test))

    for i in range(num_flows):
        features = X_test[i, -1]  # Most recent feature vector
        threat_class = int(y_test[i])
        src_ip = f"192.168.{threat_class}.{10 + (i % 20)}"

        event = orchestrator.process_flow_event(
            features=features,
            src_ip=src_ip,
            dst_ip="10.0.0.1",
            src_port=20000 + (i % 1000),
            dst_port=80,
            proto=6,
        )

        metrics = ebpf_backend.get_metrics()
        monitor.render_step(i + 1, event, metrics.to_dict())

        if args.delay > 0:
            time.sleep(args.delay)

    latencies = orchestrator.get_latency_stats()
    monitor.render_summary(num_flows, latencies, orchestrator.action_counts)


if __name__ == "__main__":
    main()
