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


def test_load_and_sample_dataset(tmp_path):
    import pandas as pd
    from data.preprocess import load_and_sample_dataset

    # Create two synthetic data directories simulating 2017 & 2018
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()

    # Create synthetic CSV in dir1
    df1 = pd.DataFrame({
        " Destination Port": [80, 443, 22, 8080],
        " Flow Duration": [1000, 2000, 3000, 4000],
        "Label": ["BENIGN", "DoS Hulk", "PortScan", "BENIGN"],
    })
    df1.to_csv(dir1 / "data1.csv", index=False)

    # Create synthetic CSV in dir2
    df2 = pd.DataFrame({
        "Destination Port": [21, 80, 443, 80],
        "Flow Duration": [5000, 6000, 7000, 8000],
        "Label": ["FTP-Patator", "DDoS", "Bot", "BENIGN"],
    })
    df2.to_csv(dir2 / "data2.csv", index=False)

    X_scaled, y, y_bin, feature_names, scaler = load_and_sample_dataset(
        data_dirs=[str(dir1), str(dir2)],
        sample_per_class=2,
        chunksize=2,
        random_state=42,
    )

    assert len(feature_names) == 2
    assert "Destination Port" in feature_names
    assert "Flow Duration" in feature_names
    assert len(X_scaled) == len(y)
    assert len(y) == len(y_bin)
    # Scaled output should be float32 or float64 numpy array
    assert isinstance(X_scaled, np.ndarray)
    assert scaler is not None

