#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS 2017 flow data")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--full", action="store_true", help="Process full dataset without class balancing")
    args = parser.parse_args()

    print("Starting CIC-IDS 2017 preprocessing...")
    run_pipeline(args.config, args.full)
    print("Preprocessing complete.")
