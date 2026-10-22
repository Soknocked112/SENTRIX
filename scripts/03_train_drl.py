#!/usr/bin/env python3
"""
03_train_drl.py - Train Deep Reinforcement Learning (PPO) IPS Agent
"""

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ebpf.xdp_simulator import XDPSimulator
from environment.ips_env import IPSEnvironment
from environment.reward_functions import IPSRewardFunction
from models.drl_agent import PPOAgent
from models.tcn import TCNThreatDetector


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Active Intrusion Prevention")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    parser.add_argument("--steps_per_episode", type=int, default=None, help="Override steps per episode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["processed_data_dir"]
    train_npz = os.path.join(data_dir, "train_data.npz")
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    max_episodes = args.episodes or int(cfg["drl"]["max_episodes"])
    steps_per_episode = args.steps_per_episode or int(cfg["drl"]["steps_per_episode"])
    state_dim = int(cfg["drl"]["state_dim"])
    action_dim = int(cfg["drl"]["action_dim"])
    lr_actor = float(cfg["drl"]["learning_rate_actor"])
    lr_critic = float(cfg["drl"]["learning_rate_critic"])
    gamma = float(cfg["drl"]["gamma"])
    gae_lambda = float(cfg["drl"]["gae_lambda"])
    clip_eps = float(cfg["drl"]["clip_eps"])
    k_epochs = int(cfg["drl"]["k_epochs"])
    device = cfg["drl"].get("device", "cpu")

    # Load preprocessed training sequences
    print(f"[*] Loading training data from {train_npz}...")
    if not os.path.exists(train_npz):
        raise FileNotFoundError(f"Training data not found at {train_npz}. Run 01_preprocess_data.py first.")

    train_data = np.load(train_npz)
    X_train = train_data["X"]
    y_train = train_data["y"]
    y_bin_train = train_data["y_binary"]
    print(f"[+] Loaded {len(X_train):,} training sequences of shape {X_train.shape[1:]}")

    # Load pretrained TCN feature extractor if available
    tcn_ckpt_path = os.path.join(ckpt_dir, "tcn_best.pth")
    tcn_model = None
    if os.path.exists(tcn_ckpt_path):
        print(f"[*] Loading pretrained TCN feature extractor from {tcn_ckpt_path}...")
        tcn_ckpt = torch.load(tcn_ckpt_path, map_location=device)
        tcn_model = TCNThreatDetector(
            input_features=tcn_ckpt["input_features"],
            num_channels=tcn_ckpt["num_channels"],
            latent_dim=tcn_ckpt["latent_dim"],
            num_classes=tcn_ckpt["num_classes"],
        )
        tcn_model.load_state_dict(tcn_ckpt["model_state_dict"])
        tcn_model.eval()
        print("[+] Pretrained TCN feature extractor ready.")
    else:
        print("[!] No pretrained TCN checkpoint found. Using temporal projection adapter.")

    # Instantiate Reward Function & Simulator
    reward_fn = IPSRewardFunction(
        reward_tp_drop=float(cfg["environment"]["reward_tp_drop"]),
        reward_tn_pass=float(cfg["environment"]["reward_tn_pass"]),
        penalty_fp_drop=float(cfg["environment"]["penalty_fp_drop"]),
        penalty_fn_pass=float(cfg["environment"]["penalty_fn_pass"]),
        reward_rate_limit_attack=float(cfg["environment"]["reward_rate_limit_attack"]),
        penalty_rate_limit_benign=float(cfg["environment"]["penalty_rate_limit_benign"]),
        cost_deep_inspect=float(cfg["environment"]["cost_deep_inspect"]),
    )

    ebpf_sim = XDPSimulator()

    # Create Environment
    env = IPSEnvironment(
        X_seq=X_train,
        y_multi=y_train,
        y_binary=y_bin_train,
        tcn_model=tcn_model,
        reward_fn=reward_fn,
        ebpf_backend=ebpf_sim,
        max_steps=steps_per_episode,
        state_dim=state_dim,
        latent_dim=int(cfg["tcn"]["latent_dim"]),
        device=device,
    )

    # Initialize PPO Agent
    print(f"[*] Initializing PPO Agent (State Dim: {state_dim}, Action Dim: {action_dim})...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        k_epochs=k_epochs,
        device=device,
    )

    best_reward = -float("inf")
    history = {"rewards": [], "accuracies": [], "fprs": [], "actor_losses": [], "critic_losses": []}

    print(f"[*] Starting DRL Training for {max_episodes} episodes...")
    start_time = time.time()

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action, log_prob, val = agent.select_action(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.record_transition(state, action, log_prob, reward, val, done)
            state = next_state
            ep_reward += reward

        # PPO Policy & Value updates
        loss_dict = agent.update()

        history["rewards"].append(ep_reward)
        history["accuracies"].append(info["accuracy"])
        history["fprs"].append(info["false_positive_rate"])
        history["actor_losses"].append(loss_dict.get("actor_loss", 0.0))
        history["critic_losses"].append(loss_dict.get("critic_loss", 0.0))

        if episode % 10 == 0 or episode == 1:
            mean_rew = np.mean(history["rewards"][-10:])
            mean_acc = np.mean(history["accuracies"][-10:])
            mean_fpr = np.mean(history["fprs"][-10:])
            print(f"Episode [{episode:03d}/{max_episodes:03d}] - "
                  f"Reward: {ep_reward:7.1f} (Avg: {mean_rew:7.1f}) | "
                  f"Mitigation Acc: {info['accuracy']*100:5.1f}% | "
                  f"FPR: {info['false_positive_rate']*100:4.1f}% | "
                  f"TP: {info['tp']}, FP: {info['fp']}, Drops: {ebpf_sim.get_metrics().dropped_packets}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            save_path = os.path.join(ckpt_dir, "drl_ppo_best.pth")
            agent.save(save_path)

    total_time = time.time() - start_time
    print(f"[+] DRL Training finished in {total_time:.1f}s. Best Single-Episode Reward: {best_reward:.1f}")

    # Plot & Save Training Curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history["rewards"], color="steelblue", alpha=0.3)
    # Moving average
    ma_rewards = np.convolve(history["rewards"], np.ones(10)/10, mode="valid")
    plt.plot(np.arange(9, len(history["rewards"])), ma_rewards, color="navy", label="10-Ep Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("PPO Cumulative Episode Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 2)
    plt.plot(history["accuracies"], color="teal", label="Mitigation Accuracy")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Mitigation Decision Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 3)
    plt.plot(history["fprs"], color="crimson", label="False Positive Rate")
    plt.xlabel("Episode")
    plt.ylabel("FPR")
    plt.title("False Positive Rate (Benign Drops)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plot_path = os.path.join(results_dir, "drl_training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] DRL Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()
