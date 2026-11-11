#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from environment.simulator import XDPSimulator
from engine.platform_utils import (
    get_device,
    get_profile_settings,
    optimize_cpu_threads,
    print_platform_diagnostics,
    resolve_data_dir,
    resolve_output_dir,
)
from environment.ips_env import IPSEnvironment
from environment.reward_functions import IPSRewardFunction
from models.drl_agent import PPOAgent
from models.tcn import TCNThreatDetector


def precompute_tcn_latents(
    tcn_model: TCNThreatDetector,
    X_seq: np.ndarray,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Precompute TCN latent embeddings in vectorized batches.
    Replaces 100,000 single-step forward passes inside the Gym loop with O(1) lookups,
    speeding up DRL training on Edge PCs by 50x-100x and on Cloud GPUs by 10x-20x.
    """
    tcn_model.eval()
    tcn_model.to(device)
    embeddings = []
    num_samples = len(X_seq)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = torch.from_numpy(X_seq[i : i + batch_size]).float().to(device)
            _, latent = tcn_model(batch)
            embeddings.append(latent.cpu().numpy())
    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for Active Intrusion Prevention")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", default=None, help="Path to preprocessed data directory")
    parser.add_argument("--ckpt_dir", default=None, help="Directory for checkpoints")
    parser.add_argument("--results_dir", default=None, help="Directory for plots & logs")
    parser.add_argument("--tcn_ckpt", default=None, help="Explicit path to pretrained TCN checkpoint")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    parser.add_argument("--steps_per_episode", type=int, default=None, help="Override steps per episode")
    parser.add_argument("--batch_size", type=int, default=None, help="Override PPO update batch size")
    parser.add_argument("--device", default=None, help="Device to train on ('cpu', 'cuda', 'mps', 'auto')")
    parser.add_argument("--profile", choices=["standard", "edge", "quick"], default=None, help="Hardware profile")
    parser.add_argument("--edge", action="store_true", help="Shortcut for --profile edge (low RAM/compute)")
    parser.add_argument("--max_samples", type=int, default=None, help="Subsample dataset for low-resource training")
    parser.add_argument("--no_precompute", action="store_true", help="Disable vectorized latent precomputation")
    args = parser.parse_args()

    print_platform_diagnostics()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    profile_name = "edge" if args.edge else args.profile
    profile_cfg = get_profile_settings(profile_name)

    data_dir = resolve_data_dir(args.data_dir, cfg["data"].get("processed_data_dir", "data/processed"))
    train_npz = os.path.join(data_dir, "train_data.npz")
    ckpt_dir = resolve_output_dir(args.ckpt_dir, cfg["paths"].get("checkpoints_dir", "checkpoints"))
    results_dir = resolve_output_dir(args.results_dir, cfg["paths"].get("results_dir", "results"))

    max_episodes = args.episodes or profile_cfg.get("drl_episodes") or int(cfg["drl"]["max_episodes"])
    steps_per_episode = args.steps_per_episode or profile_cfg.get("drl_steps_per_episode") or int(cfg["drl"]["steps_per_episode"])
    ppo_batch_size = args.batch_size or profile_cfg.get("drl_batch_size") or int(cfg["drl"]["batch_size"])
    max_samples = args.max_samples or profile_cfg.get("max_samples")

    state_dim = int(cfg["drl"]["state_dim"])
    action_dim = int(cfg["drl"]["action_dim"])
    lr_actor = float(cfg["drl"]["learning_rate_actor"])
    lr_critic = float(cfg["drl"]["learning_rate_critic"])
    gamma = float(cfg["drl"]["gamma"])
    gae_lambda = float(cfg["drl"]["gae_lambda"])
    clip_eps = float(cfg["drl"]["clip_eps"])
    k_epochs = int(cfg["drl"]["k_epochs"])

    device = get_device(args.device or cfg["drl"].get("device", "auto"))
    optimize_cpu_threads(device)

    print(f"Profile: {profile_cfg['profile'].upper()} | Target Device: {device} | Episodes: {max_episodes} | Steps/Ep: {steps_per_episode}")

    # Load preprocessed training sequences
    print(f"Loading training data from {train_npz}...")
    if not os.path.exists(train_npz):
        raise FileNotFoundError(f"Training data not found at {train_npz}. Run 01_preprocess_data.py first.")

    train_data = np.load(train_npz)
    X_train = train_data["X"]
    y_train = train_data["y"]
    y_bin_train = train_data["y_binary"]

    if max_samples is not None and max_samples > 0 and max_samples < len(X_train):
        print(f"Subsampling dataset for resource optimization: {max_samples:,} of {len(X_train):,} samples")
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        y_bin_train = y_bin_train[:max_samples]

    print(f"Loaded {len(X_train):,} training sequences of shape {X_train.shape[1:]}")

    # Load pretrained TCN feature extractor if available
    tcn_ckpt_path = args.tcn_ckpt or os.path.join(ckpt_dir, "tcn_best.pth")
    tcn_model = None
    if os.path.exists(tcn_ckpt_path):
        print(f"Loading pretrained TCN feature extractor from {tcn_ckpt_path}...")
        tcn_ckpt = torch.load(tcn_ckpt_path, map_location=device, weights_only=False)
        tcn_model = TCNThreatDetector(
            input_features=tcn_ckpt["input_features"],
            num_channels=tcn_ckpt["num_channels"],
            latent_dim=tcn_ckpt["latent_dim"],
            num_classes=tcn_ckpt["num_classes"],
        )
        tcn_model.load_state_dict(tcn_ckpt["model_state_dict"])
        tcn_model.eval()
        print("Pretrained TCN feature extractor ready.")
    else:
        print(f"Note: No pretrained TCN checkpoint found at {tcn_ckpt_path}. Using temporal projection adapter.")

    # Vectorized latent precomputation
    precomputed_latents = None
    if tcn_model is not None and not args.no_precompute:
        print("Vectorizing TCN latent representations across sequences for ultra-fast rollouts...")
        t_pre = time.time()
        precomputed_latents = precompute_tcn_latents(
            tcn_model=tcn_model,
            X_seq=X_train,
            batch_size=min(512, len(X_train)),
            device=device,
        )
        print(f"Precomputed {len(precomputed_latents):,} embeddings in {time.time() - t_pre:.2f}s (RAM: {precomputed_latents.nbytes / (1024*1024):.1f} MB)")

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
        precomputed_latents=precomputed_latents,
        reward_fn=reward_fn,
        ebpf_backend=ebpf_sim,
        max_steps=steps_per_episode,
        state_dim=state_dim,
        latent_dim=int(cfg["tcn"]["latent_dim"]),
        device=str(device),
    )

    print(f"Initializing PPO agent (state_dim={state_dim}, action_dim={action_dim}, batch_size={ppo_batch_size})...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        k_epochs=k_epochs,
        batch_size=ppo_batch_size,
        device=str(device),
    )

    best_reward = -float("inf")
    history = {"rewards": [], "accuracies": [], "fprs": [], "actor_losses": [], "critic_losses": []}

    print(f"Starting training for {max_episodes} episodes...")
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
    print(f"Training finished in {total_time:.1f}s. Best single-episode reward: {best_reward:.1f}")

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
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()
