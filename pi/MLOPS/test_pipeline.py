"""
tests/test_pipeline.py — 6G Smart City IDS
Atelier 3: Unit tests for the ML pipeline functions
"""

import numpy as np
import pandas as pd
import pytest

from model_pipeline import (
    build_preprocessor,
    classify_attack_type,
    make_xy,
    _build_model,
    SMOTE_THRESHOLD,
    FEATURE_MAP,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_embb_df():
    """Minimal synthetic eMBB-style DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "Dur":     rng.uniform(0, 5, n),
        "TotPkts": rng.integers(1, 500, n),
        "TotBytes":rng.integers(64, 100_000, n),
        "Rate":    rng.uniform(0, 1000, n),
        "Load":    rng.uniform(0, 50_000, n),
        "Loss":    rng.uniform(0, 0.1, n),
        "pLoss":   rng.uniform(0, 0.05, n),
        "TcpRtt":  rng.uniform(0, 0.2, n),
        "Label":   rng.choice(["Benign", "Malicious"], n),
    })
    return df


@pytest.fixture
def sample_toniot_df():
    """Minimal synthetic TON_IoT-style DataFrame for testing."""
    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame({
        "src_bytes": rng.integers(100, 200_000, n),
        "dst_bytes": rng.integers(100, 200_000, n),
        "src_pkts":  rng.integers(1, 2000, n),
        "dst_pkts":  rng.integers(1, 500, n),
        "duration":  rng.uniform(0, 120, n),
        "proto":     rng.choice(["tcp", "udp", "icmp"], n),
        "conn_state":rng.choice(["SF", "REJ", "RSTO", "S0"], n),
        "service":   rng.choice(["http", "ssh", "-", "ftp"], n),
        "Label":     rng.choice(["Benign", "Malicious"], n),
    })
    return df


# ── make_xy ───────────────────────────────────────────────────────────────────

def test_make_xy_returns_correct_shapes(sample_embb_df):
    X, y = make_xy(sample_embb_df)
    assert len(X) == len(y) == len(sample_embb_df)
    assert "Label" not in X.columns


def test_make_xy_drops_metadata_columns():
    df = pd.DataFrame({
        "Dur": [1.0], "TotPkts": [10],
        "Label": ["Benign"], "UniqueID": [99], "timestamp": ["2024-01-01"],
    })
    X, _ = make_xy(df)
    assert "UniqueID" not in X.columns
    assert "timestamp" not in X.columns


# ── build_preprocessor ────────────────────────────────────────────────────────

def test_build_preprocessor_numeric(sample_embb_df):
    X, _ = make_xy(sample_embb_df)
    pre = build_preprocessor(X)
    X_proc = pre.fit_transform(X)
    assert X_proc.shape[0] == len(X)
    assert not np.any(np.isnan(X_proc))


def test_build_preprocessor_mixed(sample_toniot_df):
    X, _ = make_xy(sample_toniot_df)
    pre = build_preprocessor(X)
    X_proc = pre.fit_transform(X)
    assert X_proc.shape[0] == len(X)
    assert not np.any(np.isnan(X_proc))


# ── classify_attack_type ──────────────────────────────────────────────────────

def test_classify_embb_syn_flood():
    row = {"Dur": 0.3, "SynAck": 0.003, "TotPkts": 20, "pLoss": 0.0}
    assert classify_attack_type("eMBB", row) == "TCP SYN Flood"


def test_classify_embb_bandwidth_saturation():
    row = {"Dur": 2.0, "SynAck": 0.01, "TotPkts": 200, "pLoss": 0.05}
    assert classify_attack_type("eMBB", row) == "Bandwidth Saturation"


def test_classify_toniot_ddos():
    row = {"src_pkts": 5000, "duration": 1.0, "src_bytes": 10_000,
           "dst_bytes": 5_000, "conn_state": "SF", "service": "-", "proto": "udp"}
    assert classify_attack_type("TON_IoT", row) == "DDoS"


def test_classify_toniot_scanning():
    row = {"src_pkts": 5, "duration": 0.5, "src_bytes": 200,
           "dst_bytes": 0, "conn_state": "REJ", "service": "-", "proto": "tcp"}
    assert classify_attack_type("TON_IoT", row) == "Scanning"


def test_classify_urllc_udp_flood():
    row = {"Dur": 0, "TotPkts": 1, "TcpRtt": 0.0}
    assert classify_attack_type("URLLC", row) == "UDP DDoS Flood"


def test_classify_unknown_dataset():
    assert classify_attack_type("UNKNOWN", {}) == "Unknown"


# ── _build_model ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", [
    "RandomForest", "LogisticRegression", "ExtraTrees", "MLP",
])
def test_build_model_fits_and_predicts(model_name, sample_embb_df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    X, y = make_xy(sample_embb_df)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=0, stratify=y_enc
    )
    pre = build_preprocessor(X_train)
    X_tr = pre.fit_transform(X_train)
    X_te = pre.transform(X_test)

    model = _build_model(model_name)
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        _build_model("UNKNOWN_MODEL")


# ── FEATURE_MAP sanity ────────────────────────────────────────────────────────

def test_feature_map_all_datasets():
    for ds in ["mMTC", "URLLC", "eMBB", "TON_IoT"]:
        assert ds in FEATURE_MAP
        assert len(FEATURE_MAP[ds]) > 0
