"""
Step 2: clean the data and split it into a training part (80%) and a testing part (20%).
Run:  python 02_prepare_data.py
Slow computer?  python 02_prepare_data.py --rows-per-file 30000
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from common import find_cic2017_files, load_clean_file, downsample

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
parser.add_argument("--rows-per-file", type=int, default=100000,
                    help="keep about this many rows from each file (0 = keep everything, needs lots of RAM)")
args = parser.parse_args()

files = find_cic2017_files(args.data_dir)
if not files:
    raise SystemExit(f"No CIC-IDS2017 CSV files found in '{args.data_dir}'.")

parts_X, parts_y = [], []
for number, f in enumerate(files, 1):
    print(f"[{number}/{len(files)}] reading {os.path.basename(f)} ...")
    X, y = load_clean_file(f)
    X, y = downsample(X, y, args.rows_per_file)
    print(f"      kept {len(y):,} rows")
    parts_X.append(X)
    parts_y.append(y)

# Use only columns that exist in every file, in the same order
common_columns = list(parts_X[0].columns)
for X in parts_X[1:]:
    common_columns = [c for c in common_columns if c in X.columns]
X_all = pd.concat([X[common_columns] for X in parts_X], ignore_index=True)
y_all = pd.concat(parts_y, ignore_index=True)

# Very rare classes (fewer than 5 rows) cannot be split, so remove them
counts = y_all.value_counts()
too_rare = counts[counts < 5].index
if len(too_rare):
    print("Removing classes with fewer than 5 rows:", list(too_rare))
    keep = ~y_all.isin(too_rare)
    X_all, y_all = X_all[keep].reset_index(drop=True), y_all[keep].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

os.makedirs("prepared", exist_ok=True)
joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
             "feature_names": common_columns}, "prepared/data.joblib", compress=3)

print(f"\nFeatures: {len(common_columns)}")
print(f"Training rows: {len(y_train):,}   Testing rows: {len(y_test):,}")
print(f"Attack share in training data: {(y_train != 'BENIGN').mean() * 100:.1f}%")
print("Saved to prepared/data.joblib. Continue to step 3.")
