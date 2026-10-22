#!/usr/bin/env python3
"""
05_evaluate.py - Benchmark and Quantitative Evaluation Suite for eBPF-RL-TCN IPS
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch
import yaml

from ebpf.xdp_simulator import XDPSimulator
from engine.ips_orchestrator import IPSOrchestrator
from models.drl_agent import PPOAgent
from models.tcn import TCNThreatDetector

CLASS_NAMES = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack"]
ACTION_NAMES = ["PASS", "DROP", "RATE_LIMIT", "DEEP_INSPECT"]


def evaluate_tcn_model(model, X_test, y_test, results_dir):
    """Evaluate TCN threat classifier."""
    print("\n" + "=" * 78)
    print(" 1. TCN INTRUSION DETECTION EVALUATION")
    print("=" * 78)

    model.eval()
    batch_size = 256
    num_samples = len(X_test)
    all_preds = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x = torch.FloatTensor(X_test[i : i + batch_size])
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    accuracy = (all_preds == y_test).mean()
    macro_f1 = f1_score(y_test, all_preds, average="macro", zero_division=0)
    macro_prec = precision_score(y_test, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, all_preds, average="macro", zero_division=0)

    print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro Precision:   {macro_prec * 100:.2f}%")
    print(f"Macro Recall:      {macro_rec * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")

    print("\nDetailed Per-Class Performance:")
    unique_classes = sorted(np.unique(np.concatenate([y_test, all_preds])))
    target_names = [CLASS_NAMES[c] for c in unique_classes]
    print(classification_report(y_test, all_preds, target_names=target_names, digits=4, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, all_preds, labels=unique_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("TCN Confusion Matrix on CICIDS 2017 Test Set", fontsize=14)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to {cm_path}")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
    }


def evaluate_prevention_pipeline(orchestrator, X_test, y_test, results_dir):
    """Evaluate end-to-end eBPF-RL mitigation actions and latency."""
    print("\n" + "=" * 78)
    print(" 2. END-TO-END eBPF-RL-TCN PREVENTION PERFORMANCE")
    print("=" * 78)

    num_samples = min(len(X_test), 5000)
    action_matrix = np.zeros((len(CLASS_NAMES), len(ACTION_NAMES)), dtype=int)
    latencies = []

    orchestrator.reset_buffer()

    for i in range(num_samples):
        features = X_test[i, -1]
        true_cls = int(y_test[i])
        src_ip = f"10.{true_cls}.{(i // 254) % 254}.{i % 254 + 1}"

        event = orchestrator.process_flow_event(features, src_ip=src_ip)
        act = event["action"]
        action_matrix[true_cls, act] += 1
        latencies.append(event["latency_us"])

    # Compute IPS operational metrics
    benign_total = max(1, action_matrix[0].sum())
    benign_passed = action_matrix[0, 0]
    benign_dropped = action_matrix[0, 1]
    fpr = benign_dropped / benign_total

    attack_total = max(1, action_matrix[1:].sum())
    attack_mitigated = action_matrix[1:, 1].sum() + action_matrix[1:, 2].sum()
    mitigation_rate = attack_mitigated / attack_total

    print(f"Total Evaluated Flows:        {num_samples:,}")
    print(f"Benign Traffic Pass Rate:     {(benign_passed / benign_total) * 100:.2f}%")
    print(f"False Positive Rate (FPR):    {fpr * 100:.2f}% (Benign Dropped)")
    print(f"Attack Mitigation Rate:       {mitigation_rate * 100:.2f}% (Dropped or Throttled)")

    # Latency profile
    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))

    print(f"\nDecision Latency Benchmark (End-to-End Control + eBPF Data Plane):")
    print(f"  - Mean Latency: {mean_lat:6.2f} µs")
    print(f"  - Median (P50): {p50_lat:6.2f} µs")
    print(f"  - 95th Percentile: {p95_lat:6.2f} µs")
    print(f"  - 99th Percentile: {p99_lat:6.2f} µs")

    # Plot Latency Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(lat_arr, bins=40, color="teal", edgecolor="black", alpha=0.7)
    plt.axvline(p50_lat, color="navy", linestyle="--", label=f"P50: {p50_lat:.1f} µs")
    plt.axvline(p95_lat, color="darkorange", linestyle="--", label=f"P95: {p95_lat:.1f} µs")
    plt.axvline(p99_lat, color="crimson", linestyle="--", label=f"P99: {p99_lat:.1f} µs")
    plt.xlabel("End-to-End Decision Latency (Microseconds)")
    plt.ylabel("Flow Count")
    plt.title("eBPF-RL-TCN Latency Distribution Profile")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    lat_path = os.path.join(results_dir, "latency_distribution.png")
    plt.tight_layout()
    plt.savefig(lat_path, dpi=300)
    plt.close()
    print(f"[+] Latency distribution plot saved to {lat_path}")

    # Plot Policy Action Heatmap
    plt.figure(figsize=(8, 6))
    action_pct = action_matrix / np.maximum(1, action_matrix.sum(axis=1, keepdims=True)) * 100.0
    sns.heatmap(action_pct, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=ACTION_NAMES, yticklabels=CLASS_NAMES)
    plt.title("DRL Action Policy Distribution (%) Across Threat Classes")
    plt.xlabel("RL Enforced Action")
    plt.ylabel("True Flow Class")
    plt.tight_layout()
    act_path = os.path.join(results_dir, "action_policy_distribution.png")
    plt.savefig(act_path, dpi=300)
    plt.close()
    print(f"[+] Action policy distribution heatmap saved to {act_path}")

    return {
        "benign_pass_rate": float(benign_passed / benign_total),
        "false_positive_rate": float(fpr),
        "attack_mitigation_rate": float(mitigation_rate),
        "latency_mean_us": mean_lat,
        "latency_p50_us": p50_lat,
        "latency_p95_us": p95_lat,
        "latency_p99_us": p99_lat,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate eBPF-RL-TCN IPS Framework")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["processed_data_dir"]
    test_npz = os.path.join(data_dir, "test_data.npz")
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(test_npz):
        raise FileNotFoundError(f"Test dataset not found at {test_npz}. Run 01_preprocess_data.py first.")

    test_data = np.load(test_npz)
    X_test = test_data["X"]
    y_test = test_data["y"]

    # Load TCN Model
    tcn_ckpt_path = os.path.join(ckpt_dir, "tcn_best.pth")
    if not os.path.exists(tcn_ckpt_path):
        raise FileNotFoundError(f"TCN checkpoint not found at {tcn_ckpt_path}. Run 02_train_tcn.py first.")

    tcn_ckpt = torch.load(tcn_ckpt_path, map_location="cpu")
    tcn_model = TCNThreatDetector(
        input_features=tcn_ckpt["input_features"],
        num_channels=tcn_ckpt["num_channels"],
        latent_dim=tcn_ckpt["latent_dim"],
        num_classes=tcn_ckpt["num_classes"],
    )
    tcn_model.load_state_dict(tcn_ckpt["model_state_dict"])

    # Load DRL Agent
    drl_ckpt_path = os.path.join(ckpt_dir, "drl_ppo_best.pth")
    drl_agent = None
    if os.path.exists(drl_ckpt_path):
        drl_agent = PPOAgent(
            state_dim=int(cfg["drl"]["state_dim"]),
            action_dim=int(cfg["drl"]["action_dim"]),
            device="cpu",
        )
        drl_agent.load(drl_ckpt_path)

    # 1. Evaluate TCN
    tcn_metrics = evaluate_tcn_model(tcn_model, X_test, y_test, results_dir)

    # 2. Evaluate Prevention Pipeline
    ebpf_sim = XDPSimulator()
    orchestrator = IPSOrchestrator(
        tcn_model=tcn_model,
        drl_agent=drl_agent,
        ebpf_backend=ebpf_sim,
        sequence_length=int(cfg["data"]["sequence_length"]),
        latent_dim=int(cfg["tcn"]["latent_dim"]),
        state_dim=int(cfg["drl"]["state_dim"]),
    )
    prevention_metrics = evaluate_prevention_pipeline(orchestrator, X_test, y_test, results_dir)

    # Save summary JSON
    summary = {
        "tcn_detection": tcn_metrics,
        "prevention_pipeline": prevention_metrics,
    }
    metrics_json_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Full evaluation metrics saved to {metrics_json_path}")


if __name__ == "__main__":
    main()
