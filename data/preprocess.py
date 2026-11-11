import argparse
import glob
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")

# Non-generalizable or identifying columns to remove
DROP_COLUMNS = {
    "flow id",
    "source ip",
    "destination ip",
    "timestamp",
    "source port",
    "simillarhttp",
    "unnamed: 0",
}

# Mapping raw CICIDS 2017 labels to 6 canonical categories
LABEL_MAPPING = {
    "benign": 0,
    # DoS
    "dos hulk": 1,
    "dos goldeneye": 1,
    "dos slowloris": 1,
    "dos slowhttptest": 1,
    "heartbleed": 1,
    # DDoS
    "ddos": 2,
    # PortScan
    "portscan": 3,
    # Brute Force
    "ftp-patator": 4,
    "ssh-patator": 4,
    "web attack - brute force": 4,
    "web attack brute force": 4,
    # Web & Infiltration Attacks
    "web attack - xss": 5,
    "web attack xss": 5,
    "web attack - sql injection": 5,
    "web attack sql injection": 5,
    "bot": 5,
    "infiltration": 5,
}

CLASS_NAMES = {
    0: "BENIGN",
    1: "DoS",
    2: "DDoS",
    3: "PortScan",
    4: "BruteForce",
    5: "WebAttack",
}


def clean_label_string(val: str) -> str:
    """Normalize label strings by fixing non-ASCII dashes and spaces."""
    s = str(val).strip()
    # Replace non-ASCII dashes or question marks with standard hyphen
    s = re.sub(r"[^\x20-\x7E]|\?", "-", s)
    s = " ".join(s.split())
    return s.lower()


def load_raw_cicids(data_dir: str) -> pd.DataFrame:
    """Find and concatenate all CICIDS 2017 CSV files."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info(f"Discovered {len(csv_files)} CSV files in {data_dir}:")
    for f in csv_files:
        logger.info(f"  - {os.path.basename(f)}")

    dataframes = []
    for f in csv_files:
        logger.info(f"Loading {os.path.basename(f)}...")
        df_part = pd.read_csv(f, encoding="latin1", low_memory=False)
        # Strip whitespace from column names
        df_part.columns = [c.strip() for c in df_part.columns]
        # Remove duplicate columns if any (e.g. 'Fwd Header Length.1')
        df_part = df_part.loc[:, ~df_part.columns.duplicated()]
        dataframes.append(df_part)

    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total raw rows loaded: {len(df):,}")
    return df


def clean_and_normalize_dataframe(
    df: pd.DataFrame,
    sample_per_class: Optional[int] = 15000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Cleans infinite/NaN values, encodes labels, balances classes if requested,
    and applies standard scaling.
    """
    # Find label column
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break
    if not label_col:
        raise ValueError("No 'Label' column found in dataset.")

    # Clean labels and map to target classes
    cleaned_labels = df[label_col].apply(clean_label_string)
    mapped_labels = cleaned_labels.map(LABEL_MAPPING)
    
    # Drop unknown / unmapped rows if any
    valid_mask = mapped_labels.notna()
    df = df[valid_mask].copy()
    mapped_labels = mapped_labels[valid_mask].astype(int)

    logger.info("Class distribution in raw data:")
    for cls_idx, cls_name in CLASS_NAMES.items():
        count = (mapped_labels == cls_idx).sum()
        logger.info(f"  Class {cls_idx} ({cls_name:<11}): {count:,} records")

    # Filter out drop columns and label column
    feature_cols = [
        c for c in df.columns
        if c.lower() != "label" and c.lower() not in DROP_COLUMNS
    ]

    # Convert all feature columns to numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace inf and -inf with NaN, then impute with median
    features_df = df[feature_cols].copy()
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns that have all NaNs
    features_df = features_df.dropna(axis=1, how="all")
    feature_names = list(features_df.columns)

    # Impute remaining NaNs with column medians
    medians = features_df.median()
    features_df = features_df.fillna(medians)

    df_cleaned = features_df.copy()
    df_cleaned["target"] = mapped_labels.values

    # Balanced sampling if requested
    if sample_per_class is not None and sample_per_class > 0:
        logger.info(f"Balancing dataset with max {sample_per_class:,} samples per class...")
        sampled_dfs = []
        for cls_idx in sorted(CLASS_NAMES.keys()):
            cls_df = df_cleaned[df_cleaned["target"] == cls_idx]
            if len(cls_df) > sample_per_class:
                sampled_dfs.append(cls_df.sample(n=sample_per_class, random_state=random_state))
            else:
                sampled_dfs.append(cls_df)
        df_cleaned = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1.0, random_state=random_state)
        logger.info(f"Balanced dataset total rows: {len(df_cleaned):,}")

    X_raw = df_cleaned[feature_names].values
    y_raw = df_cleaned["target"].values
    y_binary = (y_raw > 0).astype(int)

    # Fit StandardScaler
    logger.info(f"Scaling {X_raw.shape[1]} features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, y_raw, y_binary, feature_names, scaler


def load_and_sample_dataset(
    data_dirs: Union[str, List[str]],
    sample_per_class: Optional[int] = 30000,
    chunksize: int = 150000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Streaming reservoir sampler across multiple dataset directories (e.g. CICIDS 2017 & 2018).
    Maintains bounded memory (<500 MB RAM) regardless of combined dataset size.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    csv_files = []
    for d in data_dirs:
        found = []
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.lower().endswith(".csv"):
                        found.append(os.path.join(root, f))
        if not found:
            logger.warning(f"No CSV files found in {d}")
        csv_files.extend(found)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in specified directories: {data_dirs}")

    logger.info(f"Discovered {len(csv_files)} CSV files across {len(data_dirs)} directory/directories.")
    for f in csv_files:
        logger.info(f"  - {f}")

    rng = np.random.default_rng(random_state)
    num_classes = len(CLASS_NAMES)
    counts_seen = {c: 0 for c in range(num_classes)}
    reservoirs = {c: [] for c in range(num_classes)}
    canonical_features: Optional[List[str]] = None

    k_cap = sample_per_class if (sample_per_class is not None and sample_per_class > 0) else None
    logger.info(f"Starting streaming ingestion (chunksize={chunksize:,}, quota_per_class={k_cap})...")

    for file_idx, fpath in enumerate(csv_files, 1):
        fname = os.path.basename(fpath)
        logger.info(f"[{file_idx}/{len(csv_files)}] Streaming {fname}...")

        try:
            reader = pd.read_csv(fpath, chunksize=chunksize, encoding="latin1", low_memory=False)
        except Exception as e:
            logger.error(f"Error opening {fpath}: {e}")
            continue

        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk.loc[:, ~chunk.columns.duplicated()]

            # Find label column
            label_col = None
            for c in chunk.columns:
                if c.lower() == "label":
                    label_col = c
                    break
            if not label_col:
                continue

            cleaned = chunk[label_col].apply(clean_label_string)
            mapped = cleaned.map(LABEL_MAPPING)
            valid_mask = mapped.notna()
            if not valid_mask.any():
                continue

            chunk = chunk[valid_mask]
            mapped = mapped[valid_mask].astype(int)

            if canonical_features is None:
                canonical_features = [
                    c for c in chunk.columns
                    if c.lower() != "label" and c.lower() not in DROP_COLUMNS
                ]
                logger.info(f"Established {len(canonical_features)} canonical feature columns.")

            # Filter and convert numeric features
            feature_df = chunk[canonical_features].apply(pd.to_numeric, errors="coerce")
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            chunk_data = feature_df.to_numpy(dtype=np.float32)
            chunk_labels = mapped.to_numpy(dtype=np.int64)

            # Class routing & reservoir insertion
            for c in range(num_classes):
                c_mask = (chunk_labels == c)
                if not np.any(c_mask):
                    continue

                c_data = chunk_data[c_mask]
                m = len(c_data)
                counts_seen[c] += m

                if k_cap is None:
                    reservoirs[c].append(c_data)
                else:
                    curr_len = len(reservoirs[c]) if isinstance(reservoirs[c], list) else len(reservoirs[c])
                    if isinstance(reservoirs[c], list) and curr_len < k_cap:
                        take = min(k_cap - curr_len, m)
                        reservoirs[c].append(c_data[:take])
                        c_data = c_data[take:]
                        m = len(c_data)

                    if m > 0:
                        if isinstance(reservoirs[c], list):
                            reservoirs[c] = np.vstack(reservoirs[c])

                        t_indices = np.arange(counts_seen[c] - m + 1, counts_seen[c] + 1)
                        r = rng.uniform(0.0, 1.0, size=m)
                        keep_mask = r < (k_cap / t_indices)
                        keep_indices = np.where(keep_mask)[0]

                        if len(keep_indices) > 0:
                            slots = rng.integers(0, len(reservoirs[c]), size=len(keep_indices))
                            reservoirs[c][slots] = c_data[keep_indices]

    # Consolidate collected arrays
    logger.info("Streaming complete. Class distribution in sampled reservoir:")
    X_parts = []
    y_parts = []

    for c in range(num_classes):
        cls_name = CLASS_NAMES[c]
        total_seen = counts_seen[c]
        if isinstance(reservoirs[c], list):
            res_arr = np.vstack(reservoirs[c]) if len(reservoirs[c]) > 0 else np.empty((0, len(canonical_features)), dtype=np.float32)
        else:
            res_arr = reservoirs[c]

        retained = len(res_arr)
        logger.info(f"  Class {c} ({cls_name:<11}): {retained:7,} sampled / {total_seen:10,} seen")
        if retained > 0:
            X_parts.append(res_arr)
            y_parts.append(np.full(retained, c, dtype=np.int64))

    if not X_parts:
        raise ValueError("No valid flow records extracted.")

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    y_bin_all = (y_all > 0).astype(np.int64)

    # Impute remaining NaNs with column medians
    logger.info(f"Imputing missing values across {X_all.shape[0]:,} sampled records...")
    col_medians = np.nanmedian(X_all, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    nan_mask = np.isnan(X_all)
    if np.any(nan_mask):
        indices = np.where(nan_mask)
        X_all[indices] = np.take(col_medians, indices[1])

    # Uniform random shuffle across classes
    logger.info("Shuffling balanced dataset...")
    shuffle_perm = rng.permutation(len(X_all))
    X_all = X_all[shuffle_perm]
    y_all = y_all[shuffle_perm]
    y_bin_all = y_bin_all[shuffle_perm]

    # Fit StandardScaler
    logger.info(f"Fitting StandardScaler on {len(X_all):,} records with {len(canonical_features)} features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    return X_scaled, y_all, y_bin_all, canonical_features, scaler


def create_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    y_binary: np.ndarray,
    seq_len: int = 10,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs temporal sliding windows of shape (N_windows, seq_len, num_features).
    The label corresponds to the last time-step of each window.
    """
    num_samples = (len(X) - seq_len) // stride + 1
    if num_samples <= 0:
        raise ValueError(f"Sequence length ({seq_len}) exceeds dataset length ({len(X)}).")

    num_features = X.shape[1]
    X_seq = np.empty((num_samples, seq_len, num_features), dtype=np.float32)
    y_seq = np.empty(num_samples, dtype=np.int64)
    y_bin_seq = np.empty(num_samples, dtype=np.int64)

    for i in range(num_samples):
        idx = i * stride
        X_seq[i] = X[idx : idx + seq_len]
        y_seq[i] = y[idx + seq_len - 1]
        y_bin_seq[i] = y_binary[idx + seq_len - 1]

    return X_seq, y_seq, y_bin_seq


from engine.platform_utils import resolve_output_dir, resolve_raw_dirs

def run_pipeline(
    config_path: str = "configs/default_config.yaml",
    full_dataset: bool = False,
    override_dirs: Optional[List[str]] = None,
    override_sample_per_class: Optional[int] = None,
    override_out_dir: Optional[str] = None,
    override_artifacts_dir: Optional[str] = None,
) -> None:
    """Execute end-to-end preprocessing pipeline."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_raw = cfg["data"].get("raw_data_dirs", cfg["data"].get("raw_data_dir", "CICIDS 2017"))
    raw_dirs = resolve_raw_dirs(override_dirs=override_dirs, cfg_dirs=cfg_raw)

    out_dir = override_out_dir or cfg["data"].get("processed_data_dir", "data/processed")
    out_dir = resolve_output_dir(out_dir, default_dir="data/processed")

    artifacts_dir = override_artifacts_dir or cfg.get("paths", {}).get("artifacts_dir", "artifacts")
    artifacts_dir = resolve_output_dir(artifacts_dir, default_dir="artifacts")

    seq_len = int(cfg["data"]["sequence_length"])
    train_ratio = float(cfg["data"]["train_ratio"])
    val_ratio = float(cfg["data"]["val_ratio"])
    test_ratio = float(cfg["data"]["test_ratio"])
    random_seed = int(cfg["project"]["random_seed"])

    if override_sample_per_class is not None:
        sample_per_class = override_sample_per_class
    elif full_dataset:
        sample_per_class = None
    else:
        sample_per_class = cfg["data"].get("sample_size_per_class", 30000)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load & sample data via streaming (bounded memory)
    X_scaled, y, y_bin, feature_names, scaler = load_and_sample_dataset(
        data_dirs=raw_dirs,
        sample_per_class=sample_per_class,
        random_state=random_seed,
    )

    # 3. Create temporal sequences
    logger.info(f"Generating temporal sequence windows (seq_len={seq_len})...")
    X_seq, y_seq, y_bin_seq = create_sliding_windows(X_scaled, y, y_bin, seq_len=seq_len, stride=1)
    logger.info(f"Sequence dataset shape: {X_seq.shape}")

    # 4. Train / Val / Test Splits
    logger.info("Splitting dataset into train / val / test sets...")
    val_test_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp, y_bin_train, y_bin_temp = train_test_split(
        X_seq, y_seq, y_bin_seq,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=y_seq,
    )

    test_fraction_of_temp = test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test, y_bin_val, y_bin_test = train_test_split(
        X_temp, y_temp, y_bin_temp,
        test_size=test_fraction_of_temp,
        random_state=random_seed,
        stratify=y_temp,
    )

    logger.info(f"Train split: {X_train.shape[0]:,} sequences")
    logger.info(f"Val split:   {X_val.shape[0]:,} sequences")
    logger.info(f"Test split:  {X_test.shape[0]:,} sequences")

    # 5. Save compressed NPZ files
    train_file = os.path.join(out_dir, "train_data.npz")
    val_file = os.path.join(out_dir, "val_data.npz")
    test_file = os.path.join(out_dir, "test_data.npz")

    logger.info(f"Saving train data to {train_file}...")
    np.savez_compressed(train_file, X=X_train, y=y_train, y_binary=y_bin_train)

    logger.info(f"Saving val data to {val_file}...")
    np.savez_compressed(val_file, X=X_val, y=y_val, y_binary=y_bin_val)

    logger.info(f"Saving test data to {test_file}...")
    np.savez_compressed(test_file, X=X_test, y=y_test, y_binary=y_bin_test)

    # 6. Save scaler & metadata
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved feature scaler to {scaler_path}")

    metadata = {
        "num_features": len(feature_names),
        "sequence_length": seq_len,
        "feature_names": feature_names,
        "class_mapping": CLASS_NAMES,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved dataset metadata to {meta_path}")
    logger.info("Data preprocessing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS 2017 & 2018 datasets for TCN-DRL IPS")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--full", action="store_true", help="Process full dataset without downsampling")
    parser.add_argument("--sample_per_class", type=int, default=None, help="Override sample size per class")
    parser.add_argument("--dirs", nargs="+", default=None, help="Override raw data directories")
    parser.add_argument("--out_dir", default=None, help="Override processed data output directory")
    parser.add_argument("--artifacts_dir", default=None, help="Override artifacts directory")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        full_dataset=args.full,
        override_dirs=args.dirs,
        override_sample_per_class=args.sample_per_class,
        override_out_dir=args.out_dir,
        override_artifacts_dir=args.artifacts_dir,
    )
