#!/usr/bin/env python3
"""
Preprocess CIC-IDS 2018 Dataset to match the CIC-IDS 2017 schema.

Key operations:
1. Standardizes abbreviated 2018 column headers to match 2017 feature names.
2. Removes non-feature / identifier metadata (Src/Dst IP, Port, Timestamp, Protocol).
3. Strips repeated header artifacts (rows where Label == 'Label').
4. Adds duplicate 'Fwd Header Length.1' column for exact 79-column parity with 2017.
5. Standardizes attack labels to canonical naming conventions compatible with SENTRIX.
6. Streams in chunks to operate safely within memory limits (<600 MB RAM).
7. Outputs processed CSVs into 'CICIDS 2018/'.
"""

import glob
import json
import logging
import os
import sys
import time
from typing import Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess_2018")

# 79 columns in exact order from CICIDS 2017
COLUMNS_2017 = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
]

# Rename map for CIC-IDS 2018 abbreviations -> CIC-IDS 2017 standard names
RENAME_2018_TO_2017 = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

# Metadata columns to discard
DROP_METADATA = {"protocol", "timestamp", "src ip", "dst ip", "flow id", "src port"}

# Label standardization from 2018 naming to 2017 compatible naming
LABEL_STANDARDIZATION = {
    "Benign": "BENIGN",
    "DoS attacks-Hulk": "DoS Hulk",
    "DoS attacks-SlowHTTPTest": "DoS Slowhttptest",
    "DoS attacks-GoldenEye": "DoS GoldenEye",
    "DoS attacks-Slowloris": "DoS slowloris",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "FTP-BruteForce": "FTP-Patator",
    "SSH-Bruteforce": "SSH-Patator",
    "Brute Force -Web": "Web Attack - Brute Force",
    "Brute Force -XSS": "Web Attack - XSS",
    "SQL Injection": "Web Attack - SQL Injection",
    "Infilteration": "Infiltration",
    "Bot": "Bot",
}


def clean_filename(raw_name: str) -> str:
    """Produce clean standardized filename from raw 2018 filename."""
    name = raw_name.replace("_TrafficForML_CICFlowMeter.csv", ".csv")
    if name.startswith("Thuesday-"):
        name = name.replace("Thuesday-", "Tuesday-")
    return name


def preprocess_file(
    src_file: str,
    dst_file: str,
    chunksize: int = 250_000,
) -> Dict[str, any]:
    """Process a single 2018 CSV in streaming chunks."""
    logger.info(f"Processing: {os.path.basename(src_file)} -> {os.path.basename(dst_file)}")
    t0 = time.time()
    
    total_raw_rows = 0
    total_valid_rows = 0
    label_counts: Dict[str, int] = {}
    is_first_chunk = True

    # Read in chunks for bounded memory footprint
    for chunk_idx, chunk in enumerate(pd.read_csv(src_file, chunksize=chunksize, low_memory=False)):
        total_raw_rows += len(chunk)
        
        # Clean column names
        chunk.columns = [c.strip() for c in chunk.columns]

        # 1. Filter out repeated headers and null labels
        if "Label" in chunk.columns:
            valid_mask = (chunk["Label"] != "Label") & chunk["Label"].notna()
            chunk = chunk[valid_mask]
        
        # Also ensure 'Dst Port' != 'Dst Port' if present
        if "Dst Port" in chunk.columns:
            chunk = chunk[chunk["Dst Port"] != "Dst Port"]

        if len(chunk) == 0:
            continue

        # 2. Drop non-feature metadata columns
        drop_cols = [c for c in chunk.columns if c.lower() in DROP_METADATA]
        if drop_cols:
            chunk = chunk.drop(columns=drop_cols)

        # 3. Rename columns
        chunk = chunk.rename(columns=RENAME_2018_TO_2017)

        # 4. Duplicate Fwd Header Length to create Fwd Header Length.1
        if "Fwd Header Length" in chunk.columns:
            chunk["Fwd Header Length.1"] = chunk["Fwd Header Length"]

        # 5. Standardize Label values
        if "Label" in chunk.columns:
            chunk["Label"] = chunk["Label"].map(LABEL_STANDARDIZATION).fillna(chunk["Label"])
            for lbl, cnt in chunk["Label"].value_counts().items():
                label_counts[str(lbl)] = label_counts.get(str(lbl), 0) + int(cnt)

        # 6. Reorder strictly according to 2017 schema
        chunk = chunk[COLUMNS_2017]

        # 7. Append to destination CSV
        chunk.to_csv(
            dst_file,
            mode="w" if is_first_chunk else "a",
            header=is_first_chunk,
            index=False,
        )
        is_first_chunk = False
        total_valid_rows += len(chunk)

    elapsed = time.time() - t0
    logger.info(
        f"Finished {os.path.basename(dst_file)}: "
        f"{total_valid_rows:,} / {total_raw_rows:,} rows in {elapsed:.1f}s"
    )

    return {
        "source": os.path.basename(src_file),
        "destination": os.path.basename(dst_file),
        "raw_rows": total_raw_rows,
        "valid_rows": total_valid_rows,
        "dropped_header_rows": total_raw_rows - total_valid_rows,
        "label_distribution": label_counts,
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    raw_dir = "/home/sounak/Documents/CIC-IDS-2018-Dataset"
    output_dir = "/home/sounak/IPS project/CICIDS 2018"
    docs_link_dir = "/home/sounak/Documents/CICIDS 2018"

    os.makedirs(output_dir, exist_ok=True)

    # Create convenience symlink in Documents if not present
    if not os.path.exists(docs_link_dir):
        try:
            os.symlink(output_dir, docs_link_dir)
            logger.info(f"Created symlink: {docs_link_dir} -> {output_dir}")
        except Exception as e:
            logger.warning(f"Could not create symlink in Documents: {e}")

    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not raw_files:
        logger.error(f"No CSV files found in {raw_dir}")
        sys.exit(1)

    logger.info(f"Discovered {len(raw_files)} files to preprocess in {raw_dir}")
    logger.info(f"Output directory: {output_dir}")

    total_start = time.time()
    file_summaries = []
    global_labels: Dict[str, int] = {}

    for src in raw_files:
        clean_name = clean_filename(os.path.basename(src))
        dst = os.path.join(output_dir, clean_name)
        summary = preprocess_file(src, dst)
        file_summaries.append(summary)

        for lbl, cnt in summary["label_distribution"].items():
            global_labels[lbl] = global_labels.get(lbl, 0) + cnt

    total_time = time.time() - total_start
    total_records = sum(s["valid_rows"] for s in file_summaries)

    dataset_summary = {
        "dataset_name": "CIC-IDS-2018 Preprocessed (Harmonized with CIC-IDS-2017)",
        "source_dir": raw_dir,
        "output_dir": output_dir,
        "total_files": len(file_summaries),
        "total_records": total_records,
        "total_features": len(COLUMNS_2017) - 1,
        "columns": COLUMNS_2017,
        "label_distribution": global_labels,
        "processing_time_seconds": round(total_time, 2),
        "file_details": file_summaries,
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dataset_summary, f, indent=2)

    logger.info(f"All {len(raw_files)} files preprocessed successfully in {total_time:.1f}s!")
    logger.info(f"Total harmonized records: {total_records:,}")
    logger.info(f"Dataset summary written to: {summary_path}")
    for lbl, count in sorted(global_labels.items(), key=lambda x: -x[1]):
        logger.info(f"  {lbl:<26}: {count:,}")


if __name__ == "__main__":
    main()
