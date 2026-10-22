"""
app.py - Streamlit Web Dashboard for eBPF-RL-TCN Intrusion Prevention System
"""

import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

from ebpf.xdp_manager import XDPManager
from ebpf.xdp_simulator import XDPSimulator
from engine.ips_orchestrator import IPSOrchestrator
from models.drl_agent import PPOAgent
from models.tcn import TCNThreatDetector

st.set_page_config(
    page_title="eBPF-RL-TCN IPS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack"]
ACTION_NAMES = ["PASS", "DROP", "RATE_LIMIT", "DEEP_INSPECT"]


@st.cache_resource
def load_system():
    config_path = "configs/default_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    tcn_ckpt_path = os.path.join(ckpt_dir, "tcn_best.pth")
    drl_ckpt_path = os.path.join(ckpt_dir, "drl_ppo_best.pth")

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

    drl_agent = None
    if os.path.exists(drl_ckpt_path):
        drl_agent = PPOAgent(
            state_dim=int(cfg["drl"]["state_dim"]),
            action_dim=int(cfg["drl"]["action_dim"]),
            device="cpu",
        )
        drl_agent.load(drl_ckpt_path)

    ebpf_backend = XDPSimulator()

    orchestrator = IPSOrchestrator(
        tcn_model=tcn_model,
        drl_agent=drl_agent,
        ebpf_backend=ebpf_backend,
        sequence_length=int(cfg["data"]["sequence_length"]),
        latent_dim=int(cfg["tcn"]["latent_dim"]),
        state_dim=int(cfg["drl"]["state_dim"]),
    )

    return cfg, orchestrator, ebpf_backend


cfg, orchestrator, ebpf_backend = load_system()

# Sidebar
st.sidebar.title("🛡️ eBPF-RL-TCN IPS")
st.sidebar.markdown("**Intrusion Prevention Control Plane**")
st.sidebar.info(
    "Architecture:\n"
    "- **Data Plane:** eBPF / XDP\n"
    "- **Feature Extractor:** Dilated Causal TCN\n"
    "- **Mitigation Agent:** PPO Actor-Critic\n"
    "- **Dataset:** CIC-IDS 2017"
)

mode = st.sidebar.radio("Navigation", ["Live Traffic Replay", "Model & Benchmark Metrics", "Active eBPF Rules"])

if mode == "Live Traffic Replay":
    st.header("⚡ Real-Time Packet Stream & Mitigation Replay")

    col1, col2, col3, col4 = st.columns(4)
    metrics = ebpf_backend.get_metrics()
    col1.metric("Total RX Packets", f"{metrics.rx_packets:,}")
    col2.metric("Dropped Packets", f"{metrics.dropped_packets:,}", delta=f"{metrics.drop_ratio*100:.1f}%")
    col3.metric("Passed Packets", f"{metrics.passed_packets:,}")
    col4.metric("Active Block Rules", len(ebpf_backend.get_active_blocks()))

    st.subheader("Stream Simulation Controls")
    c1, c2, c3 = st.columns([2, 1, 1])
    attack_filter = c1.selectbox("Select Traffic Profile to Inject", ["Mixed Realistic Traffic"] + CLASS_NAMES)
    num_packets = c2.slider("Packets to Inject", min_value=10, max_value=200, value=30, step=10)
    trigger = c3.button("🚀 Replay Traffic Stream", use_container_width=True)

    if trigger:
        test_npz = os.path.join(cfg["data"]["processed_data_dir"], "test_data.npz")
        if os.path.exists(test_npz):
            data = np.load(test_npz)
            X_test, y_test = data["X"], data["y"]
            if attack_filter != "Mixed Realistic Traffic":
                cls_idx = CLASS_NAMES.index(attack_filter)
                mask = (y_test == cls_idx)
                if np.any(mask):
                    X_test = X_test[mask]
                    y_test = y_test[mask]
        else:
            X_test = np.random.randn(num_packets + 10, 10, 78).astype(np.float32)
            y_test = np.random.choice([0, 1, 2, 3], size=len(X_test))

        log_rows = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(min(num_packets, len(X_test))):
            feat = X_test[i, -1]
            true_cls = int(y_test[i])
            src_ip = f"192.168.{true_cls}.{10 + (i % 5)}"

            event = orchestrator.process_flow_event(
                features=feat,
                src_ip=src_ip,
                dst_ip="10.0.0.1",
                src_port=30000 + i,
                dst_port=80,
            )

            log_rows.append({
                "Flow #": i + 1,
                "Source IP": event["src_ip"],
                "True Label": CLASS_NAMES[true_cls] if true_cls < len(CLASS_NAMES) else str(true_cls),
                "TCN Threat Class": CLASS_NAMES[event["threat_pred"]],
                "Confidence": f"{event['threat_prob']*100:.1f}%",
                "DRL Action": event["action_name"],
                "eBPF Verdict": event["verdict"],
                "Latency (µs)": f"{event['latency_us']:.2f}",
            })

            progress_bar.progress((i + 1) / min(num_packets, len(X_test)))
            time.sleep(0.01)

        progress_bar.empty()
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)

        l_stats = orchestrator.get_latency_stats()
        st.success(
            f"Stream complete! Processed {len(log_rows)} flows. "
            f"Mean Decision Latency: {l_stats['mean_us']:.2f} µs (P50: {l_stats['p50_us']:.2f} µs, P99: {l_stats['p99_us']:.2f} µs)"
        )

elif mode == "Model & Benchmark Metrics":
    st.header("📊 Detection & Prevention Benchmarks")
    results_dir = cfg["paths"]["results_dir"]

    col1, col2 = st.columns(2)
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    if os.path.exists(cm_path):
        col1.image(cm_path, caption="TCN Confusion Matrix on CICIDS 2017")
    else:
        col1.info("Run scripts/05_evaluate.py to generate confusion matrix.")

    lat_path = os.path.join(results_dir, "latency_distribution.png")
    if os.path.exists(lat_path):
        col2.image(lat_path, caption="eBPF-RL-TCN Latency Distribution Profile")
    else:
        col2.info("Run scripts/05_evaluate.py to generate latency distribution plot.")

    metrics_json = os.path.join(results_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_json):
        with open(metrics_json, "r") as f:
            summary = json.load(f)
        st.subheader("Quantitative Evaluation Summary")
        st.json(summary)

elif mode == "Active eBPF Rules":
    st.header("🛡️ In-Kernel / Simulated eBPF Map State")

    blocks = ebpf_backend.get_active_blocks()
    st.subheader(f"Blocked IP Table ({len(blocks)} active rules)")

    if blocks:
        block_data = [
            {
                "Blocked IP": b.ip,
                "Reason": b.reason,
                "Packets Dropped": b.drop_count,
                "Expires (ns)": b.expiry_ts_ns,
            }
            for b in blocks
        ]
        st.dataframe(pd.DataFrame(block_data), use_container_width=True)
    else:
        st.info("No active blocked IPs in eBPF map. All legitimate traffic passing.")

    st.subheader("Manual IP Filter Injection")
    c1, c2, c3 = st.columns([2, 1, 1])
    manual_ip = c1.text_input("IPv4 Address", "192.168.1.50")
    duration = c2.number_input("Duration (seconds)", value=300, min_value=10, max_value=86400)
    add_btn = c3.button("Add eBPF Drop Rule")

    if add_btn:
        ebpf_backend.block_ip(manual_ip, duration_sec=duration, reason="Manual Operator Rule")
        st.success(f"Added drop rule for {manual_ip} into eBPF blocked_ips_map for {duration} seconds.")
        st.rerun()
