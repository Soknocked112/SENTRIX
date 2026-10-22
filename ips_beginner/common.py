"""
common.py - helper functions shared by all scripts.
Nothing here needs to be run directly.
"""
import glob
import os

import numpy as np
import pandas as pd

# Columns that identify a flow but are not useful (or are "cheating") for learning.
DROP_COLUMNS = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Source Port", "SimillarHTTP"}


def find_cic2017_files(data_dir):
    """Return every CIC-IDS2017 CSV found inside data_dir (sub-folders included).
    Files from other datasets (for example the 2018 ones) are skipped automatically,
    because we check the column names inside each file."""
    files = sorted(set(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)))
    good = []
    for f in files:
        try:
            columns = pd.read_csv(f, nrows=0, encoding="latin1").columns
        except Exception:
            continue
        names = {str(c).strip().lower() for c in columns}
        if {"destination port", "flow duration", "label"} <= names:
            good.append(f)
    return good


def clean_label(series):
    """Fix the broken dash in names like 'Web Attack - Brute Force' and remove extra spaces."""
    s = series.astype(str).str.replace(r"[^\x20-\x7E]|\?", " - ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def load_clean_file(path):
    """Read one CSV and return (features, labels) with only clean numeric data."""
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if not c.endswith(".1")]]       # duplicated column

    labels = clean_label(df["Label"])
    X = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == "Label" or col in DROP_COLUMNS:
            continue
        numbers = pd.to_numeric(df[col], errors="coerce")
        if numbers.notna().mean() > 0.5:                              # keep only real number columns
            X[col] = numbers
    X = X.replace([np.inf, -np.inf], np.nan)

    good_rows = X.notna().all(axis=1) & (labels != "") & (labels.str.lower() != "label") & (labels.str.lower() != "nan")
    X = X[good_rows].astype("float32").reset_index(drop=True)
    labels = labels[good_rows].reset_index(drop=True)
    return X, labels


def downsample(X, y, max_rows, min_keep=2000, seed=42):
    """Shrink a big file to roughly max_rows rows but keep rare attack types."""
    if not max_rows or len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    fraction = max_rows / len(y)
    chosen = []
    values = y.values
    for label in np.unique(values):
        where = np.flatnonzero(values == label)
        want = max(min(len(where), min_keep), int(len(where) * fraction))
        chosen.append(rng.choice(where, size=min(want, len(where)), replace=False))
    chosen = np.sort(np.concatenate(chosen))
    return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)
