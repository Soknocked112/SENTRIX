#!/usr/bin/env python3
"""
SENTRIX: Main Unified Training Runner
Train the Temporal Convolutional Network (TCN) and/or Proximal Policy Optimization (PPO) DRL agent.

Usage:
  python train.py                     # Train both TCN and DRL sequentially
  python train.py --model tcn         # Train TCN threat detector only
  python train.py --model drl         # Train PPO mitigation policy only
  python train.py --edge              # Train with low-resource Edge PC profile
  python train.py --amp               # Train with mixed precision on GPU (Kaggle/Colab)
"""

import argparse
import importlib
import sys
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.platform_utils import print_platform_diagnostics

train_tcn_mod = importlib.import_module("scripts.02_train_tcn")
train_drl_mod = importlib.import_module("scripts.03_train_drl")


def main():
    parser = argparse.ArgumentParser(description="SENTRIX IPS: Unified Model Training")
    parser.add_argument("--model", choices=["all", "tcn", "drl"], default="all", help="Model to train")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--data_dir", default=None, help="Path to preprocessed data directory")
    parser.add_argument("--ckpt_dir", default=None, help="Directory to save checkpoints")
    parser.add_argument("--results_dir", default=None, help="Directory to save plots & logs")
    parser.add_argument("--device", default=None, help="Compute device ('cpu', 'cuda', 'mps', 'auto')")
    parser.add_argument("--profile", choices=["standard", "edge", "quick"], default=None, help="Hardware profile")
    parser.add_argument("--edge", action="store_true", help="Shortcut for --profile edge (low RAM/compute)")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision for CUDA")
    parser.add_argument("--max_samples", type=int, default=None, help="Subsample dataset size")
    # Model-specific overrides
    parser.add_argument("--epochs", type=int, default=None, help="TCN training epochs")
    parser.add_argument("--episodes", type=int, default=None, help="DRL training episodes")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    args = parser.parse_args()

    print_platform_diagnostics()

    # Train TCN
    if args.model in ("all", "tcn"):
        print("\n" + "=" * 70)
        print(" [PHASE 1] TRAINING TEMPORAL CONVOLUTIONAL NETWORK (TCN)")
        print("=" * 70)
        # Build arguments for TCN
        tcn_argv = ["02_train_tcn.py", "--config", args.config]
        if args.data_dir:
            tcn_argv.extend(["--data_dir", args.data_dir])
        if args.ckpt_dir:
            tcn_argv.extend(["--ckpt_dir", args.ckpt_dir])
        if args.results_dir:
            tcn_argv.extend(["--results_dir", args.results_dir])
        if args.device:
            tcn_argv.extend(["--device", args.device])
        if args.profile:
            tcn_argv.extend(["--profile", args.profile])
        if args.edge:
            tcn_argv.append("--edge")
        if args.amp:
            tcn_argv.append("--amp")
        if args.max_samples:
            tcn_argv.extend(["--max_samples", str(args.max_samples)])
        if args.epochs:
            tcn_argv.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            tcn_argv.extend(["--batch_size", str(args.batch_size)])

        old_argv = sys.argv
        sys.argv = tcn_argv
        try:
            train_tcn_mod.main()
        finally:
            sys.argv = old_argv

    # Train DRL
    if args.model in ("all", "drl"):
        print("\n" + "=" * 70)
        print(" [PHASE 2] TRAINING DEEP REINFORCEMENT LEARNING POLICY (PPO)")
        print("=" * 70)
        drl_argv = ["03_train_drl.py", "--config", args.config]
        if args.data_dir:
            drl_argv.extend(["--data_dir", args.data_dir])
        if args.ckpt_dir:
            drl_argv.extend(["--ckpt_dir", args.ckpt_dir])
        if args.results_dir:
            drl_argv.extend(["--results_dir", args.results_dir])
        if args.device:
            drl_argv.extend(["--device", args.device])
        if args.profile:
            drl_argv.extend(["--profile", args.profile])
        if args.edge:
            drl_argv.append("--edge")
        if args.max_samples:
            drl_argv.extend(["--max_samples", str(args.max_samples)])
        if args.episodes:
            drl_argv.extend(["--episodes", str(args.episodes)])
        if args.batch_size:
            drl_argv.extend(["--batch_size", str(args.batch_size)])

        old_argv = sys.argv
        sys.argv = drl_argv
        try:
            train_drl_mod.main()
        finally:
            sys.argv = old_argv

    print("\n" + "=" * 70)
    print(" ALL REQUESTED TRAINING PHASES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
