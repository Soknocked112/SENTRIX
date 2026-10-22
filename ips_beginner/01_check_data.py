"""
Step 1: look at your data before doing anything else.
Run:  python 01_check_data.py            (uses the folder called  data)
  or: python 01_check_data.py --data-dir "/path/to/folder"
"""
import argparse
import os

import pandas as pd

from common import find_cic2017_files, clean_label

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
args = parser.parse_args()

files = find_cic2017_files(args.data_dir)
if not files:
    raise SystemExit(f"No CIC-IDS2017 CSV files found in '{args.data_dir}'. Check the folder path (see the guide, Part 3).")

print(f"Found {len(files)} CIC-IDS2017 file(s):")
total = 0
all_labels = []
for f in files:
    labels = clean_label(pd.read_csv(f, usecols=lambda c: c.strip() == "Label", encoding="latin1").iloc[:, 0])
    total += len(labels)
    all_labels.append(labels)
    print(f"  {os.path.basename(f):<65} {len(labels):>9,} rows")

print(f"\nTotal rows: {total:,}")
print("\nHow many flows of each type (BENIGN = normal traffic, everything else = attack):")
print(pd.concat(all_labels).value_counts().to_string())
print("\nData check finished. If you see about 8 files and about 2.8 million rows, continue to step 2.")
