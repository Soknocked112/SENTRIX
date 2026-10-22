"""
test_data.py - Unit tests for data preprocessing and sliding window sequence generation.
"""

import numpy as np
import pytest
from data.preprocess import clean_label_string, create_sliding_windows, LABEL_MAPPING


def test_clean_label_string():
    assert clean_label_string("BENIGN") == "benign"
    assert clean_label_string("Web Attack \x96 Brute Force") == "web attack - brute force"
    assert clean_label_string("DoS Hulk  ") == "dos hulk"


def test_label_mapping():
    assert LABEL_MAPPING["benign"] == 0
    assert LABEL_MAPPING["dos hulk"] == 1
    assert LABEL_MAPPING["ddos"] == 2
    assert LABEL_MAPPING["portscan"] == 3
    assert LABEL_MAPPING["ftp-patator"] == 4
    assert LABEL_MAPPING["web attack - xss"] == 5


def test_create_sliding_windows():
    n_records = 50
    n_features = 10
    seq_len = 5

    X = np.arange(n_records * n_features).reshape(n_records, n_features).astype(np.float32)
    y = np.arange(n_records)
    y_bin = (y > 25).astype(int)

    X_seq, y_seq, y_bin_seq = create_sliding_windows(X, y, y_bin, seq_len=seq_len, stride=1)

    expected_samples = n_records - seq_len + 1
    assert X_seq.shape == (expected_samples, seq_len, n_features)
    assert len(y_seq) == expected_samples
    assert len(y_bin_seq) == expected_samples

    # Check that sequence 0 has last record at index seq_len - 1
    assert y_seq[0] == seq_len - 1
    assert np.allclose(X_seq[0, -1], X[seq_len - 1])
