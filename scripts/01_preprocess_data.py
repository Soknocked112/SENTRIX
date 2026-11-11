#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import run_pipeline
from engine.platform_utils import get_profile_settings, print_platform_diagnostics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS 2017 & 2018 flow data")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--full", action="store_true", help="Process full dataset without class balancing")
    parser.add_argument("--sample_per_class", type=int, default=None, help="Override sample size per class")
    parser.add_argument("--dirs", nargs="+", default=None, help="Override raw data directories")
    parser.add_argument("--out_dir", default=None, help="Override processed data output directory")
    parser.add_argument("--artifacts_dir", default=None, help="Override artifacts directory")
    parser.add_argument("--profile", choices=["standard", "edge", "quick"], default=None, help="Hardware/execution profile")
    parser.add_argument("--edge", action="store_true", help="Shortcut for --profile edge (low RAM, compact dataset)")
    args = parser.parse_args()

    print_platform_diagnostics()

    profile_name = "edge" if args.edge else args.profile
    profile_cfg = get_profile_settings(profile_name)

    sample_size = args.sample_per_class
    if sample_size is None and not args.full:
        sample_size = profile_cfg["sample_per_class"]

    print(f"Executing preprocessing pipeline [Profile: {profile_cfg['profile']}]...")
    run_pipeline(
        config_path=args.config,
        full_dataset=args.full,
        override_dirs=args.dirs,
        override_sample_per_class=sample_size,
        override_out_dir=args.out_dir,
        override_artifacts_dir=args.artifacts_dir,
    )
    print("Preprocessing complete.")
