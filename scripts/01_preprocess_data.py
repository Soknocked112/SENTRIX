#!/usr/bin/env python3
"""
01_preprocess_data.py - Run CICIDS 2017 Data Cleaning and Sequence Generation
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CICIDS 2017 Data for eBPF-RL-TCN IPS")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--full", action="store_true", help="Process full 2.8M rows without balanced sampling")
    args = parser.parse_args()

    print("[*] Starting CICIDS 2017 Data Preprocessing Pipeline...")
    run_pipeline(args.config, args.full)
    print("[+] Preprocessing successfully completed!")
